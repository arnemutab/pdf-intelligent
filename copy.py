
import hashlib

import streamlit as st
import pymupdf
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Intelligent",
    page_icon="📚",
    layout="wide"
)

MODEL = "openai/gpt-oss-20b"
MAX_CONTEXT_CHARS = 10000
MAX_HISTORY_MESSAGES = 4
MAX_OUTPUT_TOKENS = 1000

# Paramètres de l'analyse scientifique complète
ANALYSIS_CHUNK_SIZE = 4500
ANALYSIS_CHUNK_OVERLAP = 300
ANALYSIS_CHUNK_MAX_OUTPUT_TOKENS = 2000
ANALYSIS_FINAL_MAX_OUTPUT_TOKENS = 2500

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_RETRIEVED_CHUNKS = 4
MIN_SIMILARITY = 0.30

st.title("📚 PDF Intelligent")
st.subheader("Comprendre un article scientifique plus rapidement")
st.write(
    "Importez un article PDF pour obtenir une analyse structurée "
    "et poser vos questions sur son contenu."
)

api_key = st.secrets.get("GROQ_API_KEY", "")

# Conserver le résumé scientifique
if "resume_scientifique" not in st.session_state:
    st.session_state["resume_scientifique"] = None
# --------------------------------------------------
# Modèle d'embeddings pour le RAG sémantique
# --------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def charger_modele_embeddings():
    modele = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    return modele


@st.cache_data(show_spinner="Création des embeddings du PDF...")
def creer_embeddings(textes, document_id):
    modele = charger_modele_embeddings()

    vecteurs = modele.encode(
        list(textes),
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vecteurs.astype("float32")

# --------------------------------------------------
# Construction de l'index vectoriel FAISS
# --------------------------------------------------

@st.cache_resource(
    show_spinner="Construction de l'index FAISS..."
)
def construire_index_faiss(vecteurs, document_id):

    # Vérifier que les vecteurs existent
    if vecteurs is None or len(vecteurs) == 0:
        return None

    # Déterminer la dimension des embeddings
    dimension = vecteurs.shape[1]

    # Créer un index utilisant la similarité
    # par produit scalaire
    index = faiss.IndexFlatIP(dimension)

    # Ajouter les vecteurs à l'index
    index.add(vecteurs.copy())

    return index


# --------------------------------------------------
# Contextualisation des questions de suivi
# --------------------------------------------------


def contextualiser_question(question, historique):
    question = question.strip()

    if not historique:
        return question

    question_min = question.lower()

    marqueurs_suivi = [
         # Français
    "et ", "ses ", "son ", "sa ", "leurs ", "leur ",
    "cette ", "ce ", "ces ", "celle", "celui",
    "qu'en est-il", "pourquoi", "peux-tu préciser",
    "peux tu préciser", "peux-tu détailler",
    "peux tu détailler", "que viens-tu de mentionner",
    "que tu viens de mentionner", "ces résultats",
    "les résultats que tu viens",

    # Anglais
    "and ", "its ", "their ", "this ", "that ",
    "these ", "those ", "it ", "they ",
    "why ", "what about", "can you clarify",
    "could you clarify", "can you elaborate",
    "could you elaborate", "explain further",
    "the results you mentioned",
    "those results", "what did you mean",
    "tell me more", "how about"
    ]

    
    est_suivi = (
        any(
            question_min.startswith(m)
            for m in marqueurs_suivi
        )
        or "que tu viens de mentionner" in question_min
    )


    if not est_suivi:
        return question

    dernieres_questions = [
        message["content"]
        for message in historique
        if message["role"] == "user"
    ]

    dernieres_reponses = [
        message["content"]
        for message in historique
        if message["role"] == "assistant"
    ]

    contexte = []

    if dernieres_questions:
        contexte.append(
            f"Question précédente : {dernieres_questions[-1]}"
        )

    if dernieres_reponses:
        contexte.append(
            f"Dernière réponse : {dernieres_reponses[-1][:1200]}"
        )

    contexte.append(f"Question actuelle : {question}")

    return "\n".join(contexte)

# --------------------------------------------------
# Recherche hybride FAISS + BM25
# --------------------------------------------------

def rechercher_passages(
    question,
    chunks_pdf,
    index_faiss,
    nombre=MAX_RETRIEVED_CHUNKS
):
    if (
        not chunks_pdf
        or not question.strip()
        or index_faiss is None
    ):
        return []

    documents = [
        chunk["text"] for chunk in chunks_pdf
    ]

    if index_faiss.ntotal != len(documents):
        raise ValueError(
            "L'index FAISS ne correspond pas aux passages."
        )

    # 1. Recherche sémantique FAISS
    modele = charger_modele_embeddings()

    vecteur_question = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(10, index_faiss.ntotal)

    distances, indices = index_faiss.search(
        vecteur_question, k
    )

    
    # Conserver les scores de similarité FAISS
    similarites = {
        int(i): float(score)
        for i, score in zip(indices[0], distances[0])
        if i >= 0
    }

    # Garder les passages suffisamment similaires
    rang_faiss = [
        i for i in similarites
        if similarites[i] >= MIN_SIMILARITY
    ]


    # 2. Recherche lexicale BM25
    def tokeniser(texte):
        return re.findall(
            r"\b\w+\b", texte.lower()
        )

    corpus = [
        tokeniser(doc) for doc in documents
    ]

    rang_bm25 = []

    if any(corpus):
        bm25 = BM25Okapi(corpus)

        scores_bm25 = bm25.get_scores(
            tokeniser(question)
        )

        rang_bm25 = sorted(
            range(len(documents)),
            key=lambda i: scores_bm25[i],
            reverse=True
        )[:10]

    # 3. Fusion des rangs RRF
    scores_rrf = {}

    for rang, index in enumerate(rang_faiss):
        scores_rrf[index] = (
            scores_rrf.get(index, 0)
            + 1 / (60 + rang + 1)
        )

    for rang, index in enumerate(rang_bm25):
        scores_rrf[index] = (
            scores_rrf.get(index, 0)
            + 1 / (60 + rang + 1)
        )

    
    # Filtrer les candidats avec leur similarité sémantique
    candidats_pertinents = []

    # Évaluer aussi les candidats provenant de BM25
    indices_a_verifier = list(scores_rrf.keys())

    if indices_a_verifier:
        vecteurs_candidats = index_faiss.reconstruct_batch(
            __import__("numpy").asarray(
                indices_a_verifier,
                dtype="int64"
           )
        )

        scores_semantiques = (
            vecteurs_candidats @ vecteur_question[0]
        )

        for index, score in zip(
            indices_a_verifier,
            scores_semantiques
        ):
            if float(score) >= MIN_SIMILARITY:
                candidats_pertinents.append(index)

    meilleurs_indices = sorted(
        candidats_pertinents,
        key=lambda i: scores_rrf[i],
        reverse=True
    )[:nombre]


    # 4. Conserver les pages et les textes
    passages = []

    for index in meilleurs_indices:
        passage = dict(chunks_pdf[index])

        passage["score_pertinence"] = round(
            scores_rrf[index], 5
        )

        passages.append(passage)

    return passages

# --------------------------------------------------
# Découpage de l'article pour l'analyse scientifique
# --------------------------------------------------

def decouper_article(
    pages_texte,
    taille=ANALYSIS_CHUNK_SIZE,
    chevauchement=ANALYSIS_CHUNK_OVERLAP
):
    blocs = []

    for numero_page, contenu in enumerate(pages_texte, start=1):

        contenu = contenu.strip()

        if not contenu:
            continue

        debut = 0

        while debut < len(contenu):

            fin = debut + taille

            extrait = contenu[debut:fin].strip()

            if extrait:
                blocs.append({
                    "page": numero_page,
                    "text": extrait
                })

            if fin >= len(contenu):
                break

            debut = fin - chevauchement

    return blocs
# --------------------------------------------------
# Analyse individuelle de chaque bloc avec Groq
# --------------------------------------------------

def analyser_bloc(client, bloc, numero_bloc, total_blocs):

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse scientifique.

Analyse attentivement cet extrait d'un article scientifique.

Numéro du bloc : {numero_bloc}/{total_blocs}
Page du document : {bloc["page"]}

Extrais uniquement les informations présentes dans cet extrait.

Organise ton analyse selon les rubriques suivantes :

1. Sujet et objectif
2. Problématique
3. Méthodologie
4. Résultats principaux
5. Limites de l'étude
6. Mots-clés importants

Règles impératives :
- N'invente aucune information.
- Ne présente pas une hypothèse comme un résultat.
- Si une information n'est pas présente, indique "Non mentionné dans cet extrait".
- Conserve les termes scientifiques importants.
- Ne tire pas de conclusion sur l'ensemble de l'article à partir de ce seul extrait.
- Indique la page source : {bloc["page"]}.

Texte à analyser :

{bloc["text"]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant d'analyse scientifique "
                    "rigoureux et précis."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=ANALYSIS_CHUNK_MAX_OUTPUT_TOKENS
    )

    choix = response.choices[0]
    message = choix.message
    resultat = message.content

    if not resultat:
        raise ValueError(
            f"Réponse vide pour le bloc {numero_bloc}. "
            f"finish_reason={choix.finish_reason}"
        )

    return resultat

# --------------------------------------------------
# Regroupement des analyses intermédiaires
# --------------------------------------------------


def regrouper_analyses(client, analyses):

    analyses_actuelles = analyses[:]

    while len(analyses_actuelles) > 4:

        analyses_regroupees = []

        taille_lot = 4

        for debut in range(0, len(analyses_actuelles), taille_lot):

            lot = analyses_actuelles[debut:debut + taille_lot]

            texte_analyses = "\n\n".join(lot)

            prompt = f"""
Tu es un assistant spécialisé dans l'analyse scientifique.

Voici plusieurs analyses intermédiaires d'un même article.

Produis une synthèse structurée et concise.

Consignes :
- Conserve les informations scientifiques importantes.
- Regroupe les informations similaires.
- Évite les répétitions.
- Conserve les numéros de page.
- Ne crée aucune information absente des analyses.
- Signale les contradictions éventuelles.

Analyses à regrouper :

{texte_analyses}
"""

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant rigoureux "
                            "spécialisé dans la synthèse scientifique."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_completion_tokens=600
            )

            resultat = response.choices[0].message.content

            if not resultat:
                raise ValueError(
                    "Une synthèse intermédiaire n'a pas été retournée."
                )

            analyses_regroupees.append(resultat)

        analyses_actuelles = analyses_regroupees

    return analyses_actuelles


# --------------------------------------------------
# Génération de l'analyse scientifique finale
# --------------------------------------------------

def generer_analyse_finale(client, analyses_regroupees):

    texte_analyses = "\n\n".join(analyses_regroupees)

    prompt = f"""
Tu es un expert en analyse d'articles scientifiques.

À partir des synthèses intermédiaires suivantes,
produis une analyse scientifique complète et détaillée
de l'article.

Structure obligatoirement ta réponse selon les six rubriques :

## 1. Sujet et objectif
Présente le domaine scientifique, le sujet étudié
et les objectifs de l'article.

## 2. Problématique
Explique le problème scientifique traité,
les questions de recherche et les hypothèses éventuelles.

## 3. Méthodologie
Décris les méthodes utilisées, les données,
les outils, les modèles et le protocole expérimental
lorsqu'ils sont mentionnés.

## 4. Résultats principaux
Présente les résultats importants, les observations
et les contributions scientifiques de l'étude.

## 5. Limites de l'étude
Identifie les limites explicitement mentionnées
ainsi que les éventuelles limites déductibles des informations
disponibles. Dans ce dernier cas, précise qu'il s'agit
d'une interprétation.

## 6. Cinq mots-clés
Propose cinq mots-clés représentatifs de l'article.

Règles importantes :
- Produis une analyse détaillée et claire.
- Ne répète pas inutilement les mêmes informations.
- N'invente aucun résultat, chiffre ou référence.
- Distingue les résultats des hypothèses.
- Conserve les numéros de page lorsqu'ils sont disponibles.
- Si une information est absente, indique-le clairement.
- Base-toi uniquement sur les synthèses fournies.

Synthèses intermédiaires :

{texte_analyses}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un expert rigoureux en analyse "
                    "et synthèse d'articles scientifiques."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=ANALYSIS_FINAL_MAX_OUTPUT_TOKENS
    )

    resultat = response.choices[0].message.content

    if not resultat:
        raise ValueError(
            "L'analyse scientifique finale n'a pas été générée."
        )

    return resultat

# --------------------------------------------------
# 2. Importation et extraction du PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choisissez un article scientifique",
    type=["pdf"]
)

if uploaded_file:
    try:
        pdf_bytes = uploaded_file.getvalue()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf:
            pages_texte = [
                page.get_text()
                for page in pdf
            ]

        nb_pages = len(pages_texte)
        texte = "\n".join(pages_texte)

        st.success(
            f"Document chargé : {nb_pages} page(s)"
        )

        if not texte.strip():
            st.warning(
                "Aucun texte détecté. Ce PDF est peut-être scanné. "
                "La reconnaissance OCR n'est pas encore disponible."
            )
            st.stop()

        # Réinitialiser la conversation si le PDF change.
        if st.session_state.get("document_id") != document_id:
            st.session_state["document_id"] = document_id
            st.session_state["chat_messages"] = []
            st.session_state["resume_scientifique"] = None

        # Préparer le contexte avec les numéros de page.
        # Préparer les passages du PDF en conservant les pages.
        chunks_pdf = []

        for numero, contenu in enumerate(pages_texte, start=1):
            contenu = contenu.strip()
            if not contenu:
                continue

            debut = 0
            while debut < len(contenu):
                fin = debut + CHUNK_SIZE
                extrait = contenu[debut:fin].strip()

                if extrait:
                    chunks_pdf.append({
                       "page": numero,
                       "text": extrait
                    })

                if fin >= len(contenu):
                    break
                debut = fin - CHUNK_OVERLAP

        st.success(
           f"Document prêt : {len(chunks_pdf)} passages indexés."
        )
        

        
# --------------------------------------------------
# Préparation de l'index vectoriel du PDF
# --------------------------------------------------

        textes_chunks = tuple(
            chunk["text"] for chunk in chunks_pdf
        )

        if textes_chunks:

            embeddings_pdf = creer_embeddings(
                textes_chunks,
                document_id
            )

            index_faiss = construire_index_faiss(
                embeddings_pdf,
                document_id
            )

            if index_faiss is not None:
               st.success(
                    f"Index FAISS créé : "
                    f"{index_faiss.ntotal} vecteurs indexés."
                )

        else:
            index_faiss = None
            st.warning(
                "Aucun passage disponible pour l'indexation."
            )

        # --------------------------------------------------
        # 4. Analyse scientifique existante
        # --------------------------------------------------

        st.markdown("---")
        st.subheader("📝 Analyse scientifique")

        if st.button("Analyser mon article", type="primary"):
            if not api_key:
                st.error("Clé API Groq manquante.")

            else:
                try:
                    client = Groq(api_key=api_key)

                    with st.spinner("Préparation de l'article..."):
                        blocs = decouper_article(pages_texte)

                    if not blocs:
                        st.warning(
                            "Aucun texte exploitable n'a été trouvé dans le PDF."
                        )

                    else:
                        analyses = []
                        progression = st.progress(0)
                        statut = st.empty()

                        total_blocs = len(blocs)

                        for i, bloc in enumerate(blocs, start=1):
                            statut.text(
                                f"Analyse du bloc {i}/{total_blocs} "
                                f"(page {bloc['page']})..."
                            )

                            resultat = analyser_bloc(
                                client,
                                bloc,
                                i,
                                total_blocs
                            )

                            analyses.append(resultat)
                            progression.progress(i / total_blocs)

                        statut.text(
                            "Regroupement des analyses intermédiaires..."
                        )

                        analyses_regroupees = regrouper_analyses(
                            client,
                            analyses
                        )

                        statut.text(
                            "Génération de l'analyse scientifique finale..."
                        )

                        resume = generer_analyse_finale(
                            client,
                            analyses_regroupees
                        )

                        st.session_state["resume_scientifique"] = resume

                        progression.empty()
                        statut.empty()

                        st.success("Analyse scientifique terminée !")

                except Exception as e:
                    st.error(
                        f"L'analyse a échoué : {str(e)}"
                    )

                    
        # Afficher le résumé enregistré, même après une interaction
        if st.session_state.get("resume_scientifique"):

            resume = st.session_state["resume_scientifique"]

            st.markdown("### Résultat de l'analyse")
            st.markdown(resume)

            st.download_button(
                "Télécharger le résumé",
                data=resume,
                file_name="resume_scientifique.md",
                mime="text/markdown"
            )


        # --------------------------------------------------
        # 3. Chat avec le PDF
        # --------------------------------------------------

        st.markdown("---")
        st.subheader("💬 Discutez avec votre PDF")
        st.caption(
            "Posez directement votre question. "
            "La réponse est générée uniquement après l'envoi."
        )

        # Afficher la conversation existante.
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Formulaire : un envoi explicite déclenche l'appel API.
        
        # Zone de saisie du chat
        question = st.chat_input(
            "Posez une question sur votre PDF..."
        )

        envoyer = question is not None



        if envoyer:
            if not question.strip():
                st.warning("Veuillez saisir une question.")

            elif not api_key:
                st.error(
                    "Clé API Groq manquante. "
                    "Vérifiez le secret GROQ_API_KEY dans Streamlit."
                )

            else:
                question_utilisateur = question.strip()

                # Enregistrer et afficher la question
                st.session_state["chat_messages"].append({
                    "role": "user",
                    "content": question_utilisateur
                })

                with st.chat_message("user"):
                    st.markdown(question_utilisateur)

                
                # Récupérer l'historique avant la question actuelle
                historique_recherche = (
                    st.session_state["chat_messages"][:-1]
                )

                # Contextualiser la question pour la recherche
                question_recherche = contextualiser_question(
                    question_utilisateur,
                    historique_recherche
                )

                # Recherche hybride FAISS + BM25
                passages = rechercher_passages(
                    question_recherche,
                    chunks_pdf,
                    index_faiss
                )

                if not passages:
                    reponse = (
                        "Je n'ai pas trouvé de passage suffisamment pertinent "
                        "pour répondre à cette question dans le document. "
                        "Essayez de reformuler votre question."
                    )

                    st.warning(reponse)

                    # Conserver le message dans l'historique
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": reponse
                    })

                else:
                    # Afficher les sources utilisées pour la réponse
                    with st.expander(
                        f"🔎 Voir les {len(passages)} passages utilisés"
                    ):
                        for numero, passage in enumerate(passages, start=1):
                            st.markdown(
                                f"**Passage {numero} — Page {passage['page']}**"
                            )
                            st.write(passage["text"])
                            st.divider()

                    # Construire le contexte avec les numéros de page
                    contexte_retrouve = "\n\n".join(
                        f"[Page {passage['page']}]\n{passage['text']}"
                        for passage in passages
                    )

                    system_prompt = """
Tu es Smart PDF, un assistant d'analyse scientifique rigoureux.

Tu réponds dans la même langue que celle utilisée dans la question de l'utilisateur :
- Si la question est en français, réponds en français.
- If the question is in English, answer in English.

Règles impératives :
- Utilise uniquement les informations disponibles dans le document.
- Cite les pages qui soutiennent tes affirmations sous la forme [Page 3].
- Les numéros de page correspondent aux marqueurs [Page N] du contexte.
- Ne fabrique jamais une citation ni un numéro de page.
- Si le document ne permet pas de répondre, dis-le clairement.
- Si une information est absente du contexte fourni, précise-le.
- Distingue les résultats rapportés par les auteurs de tes explications.
- Réponds clairement et de façon structurée.

Règles supplémentaires de citation :
- Chaque affirmation scientifique importante doit être
  accompagnée d'une référence de page.
- Cite uniquement les pages présentes dans les extraits fournis.
- N'attribue pas une affirmation à une page qui ne la soutient pas.
- Si les passages sont insuffisants, indique clairement
  que la réponse ne peut pas être établie.
- Ne présente pas une hypothèse comme un résultat démontré.


Règles de citation pour chaque nouvelle réponse :
- Cite uniquement les numéros de page présents
  dans les passages fournis pour la question actuelle.
- N'utilise pas les citations des réponses précédentes
  si elles ne figurent pas dans les passages actuels.
- L'historique sert uniquement à comprendre la conversation.
- Si aucune source actuelle ne soutient une affirmation,
  indique que l'information n'est pas disponible.
- Ne crée jamais de numéro de page.


"""

                    # Préparer l'historique des échanges précédents
                    historique = (
                        st.session_state["chat_messages"][:-1]
                        [-MAX_HISTORY_MESSAGES:]
                    )

                    historique = [
                        {
                            "role": message["role"],
                            "content": message["content"][:1500]
                        }
                        for message in historique
                    ]
                    
                    messages_api = [
                        {"role": "system", "content": system_prompt}
                    ]

                    # Ajouter les échanges précédents
                    messages_api.extend(historique)

                    # Ajouter le contexte du PDF et la question actuelle
                    messages_api.append({
                        "role": "user",
                        "content": (
                            "Voici les passages sélectionnés dans l'article "
                            "pour cette question.\n"
                            "Réponds uniquement à partir de ces extraits. "
                            "Cite les pages sous la forme [Page N]. "
                            "Si les extraits ne suffisent pas, indique-le clairement.\n\n"
                            f"Passages du document :\n{contexte_retrouve}\n\n"
                            f"Question actuelle :\n{question_utilisateur}"
                        )
                    })


                    # Appeler Groq uniquement si des passages existent
                    try:
                        client = Groq(api_key=api_key)

                        with st.chat_message("assistant"):
                            with st.spinner("Analyse du document en cours..."):
                                response = client.chat.completions.create(
                                    model=MODEL,
                                    messages=messages_api,
                                    temperature=0.2,
                                    max_completion_tokens=MAX_OUTPUT_TOKENS
                                )

                            reponse = response.choices[0].message.content

                            if not reponse:
                                reponse = (
                                    "Aucune réponse n'a été retournée. "
                                    "Veuillez réessayer."
                                )

                            
                            # Vérifier les numéros de page cités
                            pages_autorisees = {
                                passage["page"] for passage in passages
                            }

                            pages_citees = {
                                int(numero)
                                for numero in re.findall(
                                    r"\[Page\s+(\d+)\]",
                                    reponse,
                                    flags=re.IGNORECASE
                                )
                            }

                            pages_invalides = (
                                pages_citees - pages_autorisees
                            )

                            if pages_invalides:
                                st.warning(
                                    "Attention : certaines références de page "
                                    "n'ont pas été retrouvées dans les passages "
                                    "utilisés pour générer la réponse."
                                )

                                reponse = re.sub(
                                    r"\[Page\s+(\d+)\]",
                                    lambda match: (
                                        match.group(0)
                                        if int(match.group(1)) in pages_autorisees
                                        else "[Référence non vérifiée]"
                                    ),
                                    reponse,
                                    flags=re.IGNORECASE
                                )

                            st.markdown(reponse)

                        # Enregistrer la réponse dans l'historique
                        st.session_state["chat_messages"].append({
                            "role": "assistant",
                            "content": reponse
                        })

                    
                    
                    except Exception as e:
                        st.error(
                            "Je n’ai pas pu générer la réponse. "
                            "Vous pouvez réessayer."
                        )

                        # Retirer la question si elle est
                        # toujours le dernier message
                        if (
                            st.session_state["chat_messages"]
                            and st.session_state["chat_messages"][-1]["role"] == "user"
                            and st.session_state["chat_messages"][-1]["content"]
                            == question_utilisateur
                        ):
                            st.session_state["chat_messages"].pop()

                        # Conserver les détails pour le débogage
                        # dans les journaux de l'application
                        print(
                            f"Erreur du chat : "
                            f"{type(e).__name__}: {e}"
                        )

                        code_http = getattr(e, "status_code", None)

                        if code_http == 413:
                            st.info(
                                "La demande est trop volumineuse. "
                                "Essayez une question plus courte "
                                "ou effacez la conversation."
                            )
                        elif code_http == 429:
                            st.info(
                                "Le service IA est temporairement "
                                "limité. Attendez un peu avant de réessayer."
                            )
                        elif code_http in (401, 403):
                            st.info(
                                "Le service IA n’a pas autorisé la demande. "
                                "La configuration doit être vérifiée."
                            )


        if st.button("Effacer la conversation"):
            st.session_state["chat_messages"] = []
            st.rerun()

        
     
    except Exception as e:
        st.error("Erreur pendant le traitement du document.")
        st.exception(e)
