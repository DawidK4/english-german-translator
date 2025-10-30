import streamlit as st
import pandas as pd
import os
from transformers import pipeline

# -------------------------------------------------
# Ustawienia aplikacji
# -------------------------------------------------
st.set_page_config(page_title="NLP App – Analiza i Tłumaczenie", page_icon="🤖")

st.title("🤖 Aplikacja NLP – Analiza emocji i tłumaczenie tekstu 🇬🇧➡️🇩🇪")

st.markdown("""
### ℹ️ O aplikacji
Ta aplikacja demonstruje wykorzystanie modeli językowych z biblioteki **Hugging Face Transformers**.

**Funkcje:**
1. Analiza wydźwięku emocjonalnego tekstu (sentiment analysis)  
2. Tłumaczenie tekstu z **języka angielskiego na niemiecki**

Wybierz jedną z opcji poniżej, wprowadź tekst i kliknij **Start**, aby zobaczyć wynik.
""")

# -------------------------------------------------
# Wybór funkcji
# -------------------------------------------------
option = st.selectbox(
    "🧠 Wybierz funkcję:",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie EN → DE",
    ],
)

# -------------------------------------------------
# Analiza emocji
# -------------------------------------------------
if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

    if st.button("Start"):
        if not text.strip():
            st.warning("⚠️ Wpisz tekst, aby rozpocząć analizę.")
        else:
            with st.spinner("⏳ Analizuję wydźwięk emocjonalny..."):
                try:
                    classifier = pipeline("sentiment-analysis")
                    answer = classifier(text)
                    st.success("✅ Analiza zakończona sukcesem!")
                    st.write(answer)
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Wystąpił błąd podczas analizy: {e}")

# -------------------------------------------------
# Tłumaczenie EN → DE
# -------------------------------------------------
elif option == "Tłumaczenie EN → DE":
    text = st.text_area("✏️ Wpisz tekst po angielsku:", height=150)

    if st.button("Tłumacz"):
        if not text.strip():
            st.warning("⚠️ Wpisz tekst, który chcesz przetłumaczyć.")
        else:
            with st.spinner("🌍 Tłumaczę tekst..."):
                try:
                    translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
                    result = translator(text)
                    translated_text = result[0]['translation_text']
                    st.success("✅ Tłumaczenie zakończone sukcesem!")
                    st.text_area("🇩🇪 Tekst po niemiecku:", translated_text, height=150)
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Wystąpił błąd podczas tłumaczenia: {e}")

# -------------------------------------------------
# Stopka
# -------------------------------------------------
st.markdown("---")
st.caption("Autor: Dawid Kucharski | Nr indeksu: s27637 | Lab 05 – Streamlit + Hugging Face")