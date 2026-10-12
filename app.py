import streamlit as st
import pymupdf
from groq import Groq

st.set_page_config(
    page_title="PDF Intelligent",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Intelligent")
st.subheader("Comprendre un article scientifique plus rapidement")
st.write(
    "Importez un article PDF pour obtenir une analyse structurée "
    "de son contenu."
)

api_key = st.text_input(
    "Clé API Groq",
    type="password",
    help="Votre clé reste dans la session de cette application."
)

uploaded_file = st.file_uploader(
    "Choisissez un article scientifique",
    type=["pdf"]
)

if uploaded_file:
    try:
        with pymupdf.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            texte = "\n".join(page.get_text() for page in pdf)
            nb_pages = len(pdf)

        st.success(
            f"Document chargé : {nb_pages} page(s)"
        )

        if not texte.strip():
            st.warning(
                "Aucun texte détecté. Ce PDF est peut-être scanné."
            )

        if st.button("Analyser mon article", type="primary"):
            if not api_key:
                st.error("Veuillez saisir votre clé API Groq.")
            elif not texte.strip():
                st.error("Impossible d'analyser ce PDF sans texte.")
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

Texte de l'article :
{texte[:12000]}
"""

                try:
                    client = Groq(api_key=api_key)

                    with st.spinner("Analyse en cours..."):
                        response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
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
                    st.markdown("## Résultat de l'analyse")
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
        st.error("Impossible de lire ce fichier PDF.")
