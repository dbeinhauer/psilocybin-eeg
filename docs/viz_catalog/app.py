import streamlit as st

st.set_page_config(
    page_title="Psilocybin-EEG Viz Catalog",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Psilocybin-EEG — Visualization Reference")
st.markdown("""
Welcome to the interactive visualization reference for the **psilocybin-EEG** project.

Use the sidebar to navigate:

| Page | Description |
|------|-------------|
| 📋 **Catalog** | Browse all analysis types, data shapes, operations, and plot sketches |
| 🔬 **Results Browser** | Browse actual computed plot files from your results folder |
""")

st.info("Select a page from the sidebar to get started.")
