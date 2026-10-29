
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
        len(question.split()) <= 12
        or any(question_min.startswith(m) for m in marqueurs_suivi)
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
                prompt = f"""
Analyse cet article scientifique en français.

Présente les rubriques suivantes :
1. Sujet et objectif
2. Problématique
3. Méthodologie
4. Résultats principaux
5. Limites de l'étude
6. Cinq mots-clés

N'invente aucune information.
Pour toute information absente, écris :
"Non précisé dans le texte".

Cite les pages lorsque cela est possible.
Le texte ci-dessous peut être incomplet si le PDF est long.

Texte de l'article :
{texte[:12000]}
"""

                try:
                    client = Groq(api_key=api_key)

                    with st.spinner("Analyse en cours..."):
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Tu es un assistant d'analyse "
                                        "scientifique rigoureux."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.2
                        )

                    resume = response.choices[0].message.content
                    st.session_state["resume_scientifique"] = resume

                   

                except Exception:
                    st.error(
                        "L'analyse a échoué. Vérifiez la clé API, "
                        "le modèle et les limites du service."
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
                        st.error("Échec de la génération de la réponse.")
                        st.write("Type d'erreur :", type(e).__name__)

                        if hasattr(e, "status_code"):
                            st.write("Code HTTP :", e.status_code)

                        st.code(str(e)[:1500])


        if st.button("Effacer la conversation"):
            st.session_state["chat_messages"] = []
            st.rerun()

        
     
    except Exception as e:
        st.error("Erreur pendant le traitement du document.")
        st.exception(e)

