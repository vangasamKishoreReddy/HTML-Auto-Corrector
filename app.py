import streamlit as st
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Function to load model and vectorizer safely
@st.cache_resource
def load_model():
    try:
        model_path = os.path.abspath(r"C:\Users\vkr20\Documents\INNOMATICS_Main\Greenie Web\html_autocorrector.pkl")
        vectorizer_path = os.path.abspath(r"C:\Users\vkr20\Documents\INNOMATICS_Main\Greenie Web\vectorizer.pkl")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load Model & Vectorizer
model, vectorizer = load_model()

# Streamlit UI
st.title("🔧 HTML Auto-Corrector")
st.write("Enter incorrect HTML, and get the corrected version!")

# Text input for incorrect HTML
bad_html = st.text_area("Paste your incorrect HTML here:", height=200)

# Prediction button
if st.button("🔍 Correct HTML"):
    if model is None or vectorizer is None:
        st.error("🚫 Model failed to load. Please check the logs.")
    elif bad_html.strip():
        try:
            # Transform input using vectorizer
            input_vector = vectorizer.transform([bad_html])

            # Predict the corrected HTML
            corrected_html = model.predict(input_vector)[0]

            # Display corrected HTML
            st.subheader("✅ Corrected HTML:")
            st.code(corrected_html, language="html")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
    else:
        st.warning("⚠️ Please enter some HTML to correct!")

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Developed by Kishore Reddy V")
