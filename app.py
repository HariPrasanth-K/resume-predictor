import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Apply light theme and styling
st.markdown("""
    <style>
    body, .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title and headers */
    h1, h2, h3, h4, h5 {
        color: #2c3e50;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 180px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #357ab8;
        color: #fff;
    }

    /* File uploader container */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4a90e2;
        border-radius: 12px;
        padding: 25px;
        background-color: #ffffff;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #357ab8;
        background-color: #f1f5ff;
    }

    /* File uploader label */
    label[for^="stFileUploader"] {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        margin-bottom: 10px;
        display: block;
    }

    /* Drag and drop text */
    div[data-testid="stFileUploader"] > div {
        color: #495057 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Browse files button */
    div[data-testid="stFileUploader"] button {
        background-color: #4a90e2 !important;
        color: #fff !important;
        border-radius: 8px !important;
        padding: 8px 15px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #357ab8 !important;
        color: #ffffff !important;
    }

    /* Textarea */
    .stTextArea textarea {
        background-color: #ffffff;
        color: #212529;
        border-radius: 10px;
        border: 1px solid #ced4da;
    }

    hr {
        border-top: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)


# Cleaning function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Handle file upload
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Predict category
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# Main Streamlit app
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📝", layout="wide")

    st.title("📝 Resume Category Prediction App")
    st.markdown("📤 **Upload your resume (PDF, DOCX, or TXT) and get the predicted job category instantly!**")

    # File uploader
    uploaded_file = st.file_uploader("📂 Upload Your Resume Here", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Text extracted successfully!")

            if st.checkbox("🧐 Show extracted resume text", False):
                st.text_area("🧾 Extracted Resume Text", resume_text, height=300)

            st.subheader("🔍 Predicted Job Category")
            category = pred(resume_text)
            st.success(f"🎯 The resume category is likely: **{category}**")

        except Exception as e:
            st.error(f"⚠️ Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()

