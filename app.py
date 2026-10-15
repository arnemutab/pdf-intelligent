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

api_key = st.secrets["GROQ_API_KEY"]

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
            pages_texte = [page.get_text() for page in pdf]
            texte = "\n".join(pages_texte)
            nb_pages = len(pdf)

        st.success(
            f"Document chargé : {nb_pages} page(s)"
        )

        if not texte.strip():
            st.warning(
                "Aucun texte détecté. Ce PDF est peut-être scanné."
            )
        
        st.markdown("---")
        st.subheader("💬 Poser une question sur mon PDF")
        st.caption(
            "Recherche locale par mots-clés, sans appel à l'API."
        )

        question = st.text_input(
            "Votre question",
            placeholder="Ex. : Quelle méthodologie les auteurs utilisent-ils ?"
        )

        if st.button("Rechercher dans le PDF"):
            if not texte.strip():
                st.warning("Ce PDF ne contient pas de texte exploitable.")
            elif not question.strip():
                st.warning("Veuillez saisir une question.")
            else:
                import re

                mots = set(
                    re.findall(r"\b[\w-]{3,}\b", question.lower())
                )

                mots_exclus = {
                    "quelle", "quelles", "quels", "quel",
                    "dans", "pour", "avec", "sont", "leur",
                    "leurs", "elle", "elles", "nous", "vous",
                    "est-ce", "comment", "pourquoi"
                }
                mots -= mots_exclus

                passages = []

                for numero_page, contenu_page in enumerate(
                    pages_texte, start=1
                ):
                    paragraphes = [
                        p.strip()
                        for p in contenu_page.split("\n")
                        if p.strip()
                    ]

                    for paragraphe in paragraphes:
                        correspondances = sum(
                            1 for mot in mots
                            if mot in paragraphe.lower()
                        )

                        if correspondances > 0:
                            passages.append({
                                "page": numero_page,
                                "score": correspondances,
                                "texte": paragraphe
                            })

                passages.sort(
                    key=lambda p: p["score"],
                    reverse=True
                )

                                   
                if not passages:
                    st.info(
                        "Aucun passage correspondant trouvé. "
                        "Essayez avec d'autres mots-clés."
                    )
                else:
                    passages_retenus = passages[:5]

                    st.success(
                        f"{len(passages_retenus)} passage(s) trouvé(s)."
                    )

                    for passage in passages_retenus:
                        st.markdown(
                            f"**📄 Page {passage['page']}**"
                        )
                        st.write(passage["texte"])
                        st.divider()

                    st.session_state["passages_chat"] = [
                        {
                            "page": p["page"],
                            "texte": p["texte"]
                        }
                        for p in passages_retenus
                    ]
                    st.session_state["question_chat"] = question
         
        passages_chat = st.session_state.get("passages_chat", [])
        question_chat = st.session_state.get("question_chat", "")

        if passages_chat:
            if st.button("🤖 Générer la réponse IA", type="primary"):
                if not api_key:
                    st.error("Clé API Groq manquante.")
                else:
                    contexte = "\n\n".join(
                        f"Page {p['page']} :\n{p['texte']}"
                        for p in passages_chat
                    )

                    prompt_chat = f"""
Réponds en français à la question en utilisant uniquement
les extraits du document ci-dessous.

Question : {question_chat}

Extraits :
{contexte}

Consignes :
- N'invente aucune information.
- Cite les pages sous la forme [Page 2].
- Si les extraits ne suffisent pas, indique-le clairement.
- Distingue les faits des éventuelles incertitudes.
"""

                    try:
                        client = Groq(api_key=api_key)

                        with st.spinner("Génération de la réponse..."):
                            response = client.chat.completions.create(
                                model="openai/gpt-oss-20b",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "Tu es un assistant scientifique "
                                            "rigoureux. Tu n'inventes pas "
                                            "de sources ni de résultats."
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt_chat
                                    }
                                ],
                                temperature=0.2
                            )

                        reponse_chat = (
                            response.choices[0].message.content
                        )

                        st.markdown("### 💡 Réponse")
                        st.markdown(reponse_chat)

                    except Exception as e:
                        st.error(
                            "La génération a échoué. Vérifie les limites "
                            "du service Groq et le nom du modèle."
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
