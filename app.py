
import hashlib

import streamlit as st
import pymupdf
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def rechercher_passages(question, chunks_pdf):
    if not chunks_pdf:
        return []

    documents = [
        chunk["text"] for chunk in chunks_pdf
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrice = vectorizer.fit_transform(
        documents + [question]
    )

    scores = cosine_similarity(
        matrice[-1],
        matrice[:-1]
    ).ravel()

    meilleurs_indices = scores.argsort()[::-1]

    passages = []

    for index in meilleurs_indices:
        if scores[index] <= 0:
            break

        passages.append(chunks_pdf[index])

        if len(passages) >= MAX_RETRIEVED_CHUNKS:
            break

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
                # Afficher la question immédiatement.
                st.session_state["chat_messages"].append(
                    {
                        "role": "user",
                        "content": question.strip()
                    }
                )

                with st.chat_message("user"):
                    st.markdown(question.strip())

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

                # Conserver au maximum les derniers échanges.
                historique = st.session_state["chat_messages"][:-1][-MAX_HISTORY_MESSAGES:]

                historique = [
                    {
                        "role": message["role"],
                         "content": message["content"][:1500]
                    }
                    for message in historique
                ]

                passages = rechercher_passages(
                    question.strip(),
                    chunks_pdf
                )

                if not passages:
                    st.warning(
                       "Je n'ai pas trouvé de passage suffisamment pertinent "
                        "pour répondre à cette question dans le document."
                    )
                    st.stop()

                contexte_retrouve = "\n\n".join(
                    f"[Page {passage['page']}]\n{passage['text']}"
                    for passage in passages
                )

                historique = st.session_state["chat_messages"][:-1][
                    -MAX_HISTORY_MESSAGES:
                ]

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
                    "content": question.strip()
                })
                
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

                    st.session_state["chat_messages"].append(
                        {
                            "role": "assistant",
                            "content": reponse
                        }
                    )

                    
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
