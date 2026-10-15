
import hashlib

import streamlit as st
import pymupdf
from groq import Groq


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Intelligent",
    page_icon="📚",
    layout="wide"
)

MODEL = "openai/gpt-oss-20b"
MAX_CONTEXT_CHARS = 24000

st.title("📚 PDF Intelligent")
st.subheader("Comprendre un article scientifique plus rapidement")
st.write(
    "Importez un article PDF pour obtenir une analyse structurée "
    "et poser vos questions sur son contenu."
)

api_key = st.secrets.get("GROQ_API_KEY", "")


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
        pages_incluses = []
        caracteres_restants = MAX_CONTEXT_CHARS

        for numero, contenu in enumerate(pages_texte, start=1):
            contenu = contenu.strip()

            if not contenu:
                continue

            if caracteres_restants <= 0:
                break

            extrait = contenu[:caracteres_restants]

            pages_incluses.append(
                f"[Page {numero}]\n{extrait}"
            )

            caracteres_restants -= len(extrait)

        contexte_pdf = "\n\n".join(pages_incluses)

        if len(texte) > MAX_CONTEXT_CHARS:
            st.info(
                "Le document est long : le chat utilise uniquement "
                "la partie du texte incluse dans son contexte actuel. "
                "Certaines pages peuvent ne pas être prises en compte."
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
                historique = st.session_state["chat_messages"][:-1][-6:]

                messages_api = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            "Voici le texte du document à analyser. "
                            "Conserve les marqueurs de pages pour les citations.\n\n"
                            + contexte_pdf
                        )
                    }
                ]

                messages_api.extend(historique)

                messages_api.append(
                    {
                        "role": "user",
                        "content": question.strip()
                    }
                )

                try:
                    client = Groq(api_key=api_key)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyse du document en cours..."):
                            response = client.chat.completions.create(
                                model=MODEL,
                                messages=messages_api,
                                temperature=0.2
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

                except Exception:
                    # Éviter d'afficher des détails sensibles de l'API.
                    st.error(
                        "La réponse n'a pas pu être générée. "
                        "Vérifiez la configuration Groq, le modèle "
                        "et les limites du service, puis réessayez."
                    )

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
