import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
# Import custom preprocessor module before deserializing the pipeline.
from sentiment_utils import clean_text  # noqa: F401

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'sentiment_pipeline.joblib'

st.set_page_config(page_title='AI Sentiment Analyzer', page_icon='💬', layout='centered')
st.title('💬 AI-Powered Sentiment Analysis')
st.write('Enter a customer review to classify its sentiment as Positive, Negative or Neutral.')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error('Trained model not found. Run `python train_model.py` first.')
    st.stop()
model = load_model()

review = st.text_area('Customer review', placeholder='Example: The product is easy to use and works perfectly.', height=140)
if st.button('Analyze sentiment', type='primary', use_container_width=True):
    if not review.strip():
        st.warning('Please enter a review before analyzing.')
    else:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        classes = model.named_steps['clf'].classes_
        probability_table = pd.DataFrame({'Sentiment': classes, 'Estimated probability': probabilities})
        confidence = float(probabilities.max())
        st.subheader('Prediction')
        if prediction == 'Positive': st.success(f'Positive sentiment · {confidence:.1%} model confidence')
        elif prediction == 'Negative': st.error(f'Negative sentiment · {confidence:.1%} model confidence')
        else: st.info(f'Neutral sentiment · {confidence:.1%} model confidence')
        st.caption('The confidence value is the model probability for its predicted class; it is not a guarantee of correctness.')
        st.subheader('Class probabilities')
        st.bar_chart(probability_table.set_index('Sentiment'))
        st.dataframe(probability_table.sort_values('Estimated probability', ascending=False).style.format({'Estimated probability':'{:.2%}'}), use_container_width=True, hide_index=True)

with st.expander('About this model'):
    st.write('The saved scikit-learn pipeline applies conservative text normalization, TF-IDF feature extraction and tuned Logistic Regression. The model was trained on the supplied dataset.')
    st.warning('Dataset limitation: the supplied 10,000 rows contain only 300 distinct review texts. Evaluation grouped identical texts to avoid train/test leakage, but results on new real-world writing may be lower. Test with diverse, independently collected reviews before production use.')

st.divider()
st.caption('Academic project demonstration · NLP + Machine Learning + Streamlit')
