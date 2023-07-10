import streamlit as st
import streamlit.components.v1 as components
from xai import limeExplainer, tokenizer, METHODS, shapExplainer
from streamlit_shap import st_shap 
import shap
def format_dropdown_labels(val):
    return METHODS[val]['name']

# Define page settings
st.set_page_config(
    page_title='LIME explainer app for classification models',
    # layout="wide"
)

# Build app
title_text = 'LIME - SHAP Explainer Dashboard for Fine-grained Sentiment'
subheader_text = '''0: anger  &nbsp 1: fear &nbsp 2: joy &nbsp 3: love &nbsp 4: sadness &nbsp 5: surprise '''

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)
st.markdown(f"<h5 style='text-align: center;'>{subheader_text}</h5>", unsafe_allow_html=True)
st.text("")
input_text = st.text_input('Enter your text:', "")
method_list = tuple(label for label, val in METHODS.items())
method = st.selectbox(
    'Choose classifier:',
    method_list,
    index=0,
    format_func=format_dropdown_labels,
)
option = st.selectbox(
    'Choose option Explainer:',
    ('LIME', 'SHAP'))
if st.button("Explain Results"):
    if option=='SHAP':
         with st.spinner('Calculating...'):
            text = tokenizer(input_text)
            shap_values=shapExplainer(method,
                                    path_to_file=METHODS[method]['file'],
                                    text=text,
                                    lowercase=METHODS[method]['lowercase'],)
            st_shap(shap.plots.text(shap_values),800)
            st_shap(shap.plots.bar(shap_values.abs.sum(0)))
    if option == 'LIME':       
        with st.spinner('Calculating...'):
            text = tokenizer(input_text)
            exp = limeExplainer(method,
                            path_to_file=METHODS[method]['file'],
                            text=text,
                            lowercase=METHODS[method]['lowercase'],
                            num_samples=int(1000))
            # Display explainer HTML object
            components.html(exp.as_html(), height=800)
