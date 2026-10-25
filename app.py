
import hashlib

import streamlit as st
import pymupdf
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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

st.title("📚 PDF Intelligent")
st.subheader("Comprendre un article scientifique plus rapidement")
st.write(
    "Importez un article PDF pour obtenir une analyse structurée "
    "et poser vos questions sur son contenu."
)

api_key = st.secrets.get("GROQ_API_KEY", "")


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


def rechercher_passages(question, chunks_pdf):
    if not chunks_pdf or not question.strip():
        return []

    # Garder les passages et leurs numéros de page.
    chunks_valides = [
        chunk for chunk in chunks_pdf
        if chunk.get("text", "").strip()
    ]

    if not chunks_valides:
        return []

    documents = [
        chunk["text"].strip()
        for chunk in chunks_valides
    ]

    # Découper le texte en termes pour BM25.
    def tokeniser(texte):
        return re.findall(r"\b\w+\b", texte.lower())

    question_tokens = tokeniser(question)

    if not question_tokens:
        return []

    corpus_tokenise = [
        tokeniser(document)
        for document in documents
    ]

    # Ne pas poursuivre si aucun terme de la question
    # n'apparaît dans les passages.
    termes_question = set(question_tokens)

    candidats = [
        i for i, tokens in enumerate(corpus_tokenise)
        if termes_question.intersection(tokens)
    ]

    if not candidats:
        return []

    # 1. Recherche BM25.
    bm25 = BM25Okapi(corpus_tokenise)
    scores_bm25 = bm25.get_scores(question_tokens)

    # 2. Recherche TF-IDF.
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        matrice = vectorizer.fit_transform(
            documents + [question]
        )

        scores_tfidf = cosine_similarity(
            matrice[-1],
            matrice[:-1]
        ).ravel()

    except ValueError:
        scores_tfidf = [0.0] * len(documents)

    # 3. Classement combiné par fusion des rangs.
    # Un passage bien classé par les deux méthodes
    # remonte dans les résultats.
    rang_bm25 = sorted(
        candidats,
        key=lambda i: scores_bm25[i],
        reverse=True,
    )

    rang_tfidf = sorted(
        candidats,
        key=lambda i: scores_tfidf[i],
        reverse=True,
    )

    position_bm25 = {
        index: rang + 1
        for rang, index in enumerate(rang_bm25)
    }

    position_tfidf = {
        index: rang + 1
        for rang, index in enumerate(rang_tfidf)
    }

    scores_combines = {}

    for index in candidats:
        score = (
            1 / (60 + position_bm25[index])
            + 1 / (60 + position_tfidf[index])
        )

        scores_combines[index] = score

    meilleurs_indices = sorted(
        candidats,
        key=lambda i: scores_combines[i],
        reverse=True,
    )

    # 4. Retourner au maximum quatre passages.
    passages = []

    for index in meilleurs_indices[:MAX_RETRIEVED_CHUNKS]:
        passage = dict(chunks_valides[index])
        passage["score_pertinence"] = round(
            scores_combines[index], 5
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
        with st.form("pdf_chat_form", clear_on_submit=True):
            question = st.text_input(
                "Votre question",
                placeholder=(
                    "Ex. : Quelle méthodologie les auteurs utilisent-ils ?"
                )
            )

            envoyer = st.form_submit_button(
                "Envoyer",
                type="primary"
            )


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

                # Rechercher les passages pertinents du PDF
                passages = rechercher_passages(
                    question_utilisateur,
                    chunks_pdf
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

Tu réponds en français en t'appuyant sur le texte du PDF fourni.

Règles impératives :
- Utilise uniquement les informations disponibles dans le document.
- Cite les pages qui soutiennent tes affirmations sous la forme [Page 3].
- Les numéros de page correspondent aux marqueurs [Page N] du contexte.
- Ne fabrique jamais une citation ni un numéro de page.
- Si le document ne permet pas de répondre, dis-le clairement.
- Si une information est absente du contexte fourni, précise-le.
- Distingue les résultats rapportés par les auteurs de tes explications.
- Réponds clairement et de façon structurée.
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
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": (
                                "Voici les passages sélectionnés dans l'article. "
                                "Réponds uniquement à partir de ces extraits. "
                                "Cite les pages sous la forme [Page N]. "
                                "Si les extraits ne suffisent pas, indique-le clairement.\n\n"
                                + contexte_retrouve
                            )
                        }
                    ]

                    messages_api.extend(historique)

                    messages_api.append({
                        "role": "user",
                        "content": question_utilisateur
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

                    st.markdown("### Résultat de l'analyse")
                    st.markdown(resume)

                    st.download_button(
                        "Télécharger le résumé",
                        data=resume,
                        file_name="resume_scientifique.md",
                        mime="text/markdown"
                    )

                except Exception:
                    st.error(
                        "L'analyse a échoué. Vérifiez la clé API, "
                        "le modèle et les limites du service."
                    )

    except Exception:
        st.error(
            "Impossible de lire ce fichier PDF. "
            "Vérifiez qu'il s'agit d'un document PDF valide."
        )
