import streamlit as st
import pickle
import PyPDF2
import re
import spacy

# Load model and NLP
model_path = "resume_classifier.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

nlp = spacy.load("en_core_web_sm")

# Clean text for model prediction
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Extract detailed resume structure
def extract_details(text):
    doc = nlp(text)
    details = {
        "Name": "Not identified",
        "Professional Summary": "Not identified",
        "Skills": [],
        "Education": [],
        "Work Experience": [],
        "Projects": [],
        "Certificates": [],
        "Languages": [],
        "Additional Details": []
    }

    lines = text.split("\n")
    # Name: First likely PERSON entity
    for line in lines[:5]:
        line_doc = nlp(line)
        for ent in line_doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                details["Name"] = ent.text
                break
        if details["Name"] != "Not identified":
            break

    # Professional Summary: After name, before sections
    summary_lines = []
    start_idx = lines.index(details["Name"].split("\n")[0]) + 1 if details["Name"] != "Not identified" else 1
    for line in lines[start_idx:start_idx + 10]:
        if any(kw in line.lower() for kw in ["skills", "education", "experience", "projects"]):
            break
        if line.strip() and len(line.split()) > 5:
            summary_lines.append(line.strip())
    if summary_lines:
        details["Professional Summary"] = " ".join(summary_lines)

    # Skills
    skill_keywords = [
        "python", "java", "sql", "machine learning", "excel", "communication", "leadership",
        "javascript", "html", "css", "project management", "data analysis", "cloud", "aws"
    ]
    text_lower = text.lower()
    details["Skills"] = [kw.capitalize() for kw in skill_keywords if kw in text_lower]
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(kw in chunk_text for kw in ["skill", "proficient", "experience"]) and chunk_text not in text_lower:
            details["Skills"].append(chunk.text)

    # Education
    education_keywords = ["university", "college", "bachelor", "master", "phd", "b.sc", "m.sc"]
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(kw in sent_text for kw in education_keywords):
            details["Education"].append(sent.text.strip())

    # Work Experience
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(kw in sent_text for kw in ["experience", "worked", "employed", "position"]):
            details["Work Experience"].append(sent.text.strip())

    # Projects
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(kw in sent_text for kw in ["project", "developed", "built"]):
            details["Projects"].append(sent.text.strip())

    # Certificates
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(kw in sent_text for kw in ["certificate", "certified", "certification"]):
            details["Certificates"].append(sent.text.strip())

    # Languages
    common_languages = ["english", "spanish", "french", "german", "chinese", "hindi"]
    for ent in doc.ents:
        if ent.label_ == "LANGUAGE":
            details["Languages"].append(ent.text)
    for lang in common_languages:
        if lang in text_lower and lang.capitalize() not in details["Languages"]:
            details["Languages"].append(lang.capitalize())

    # Additional Details
    categorized = " ".join([details["Name"], details["Professional Summary"]] + 
                           sum([details[k] for k in details if isinstance(details[k], list)], []))
    for sent in doc.sents:
        if sent.text.strip() and sent.text not in categorized:
            details["Additional Details"].append(sent.text.strip())

    # Clean empty sections
    for key in details:
        if not details[key] or (isinstance(details[key], list) and not details[key]):
            details[key] = ["Not identified"] if isinstance(details[key], list) else "Not identified"

    return details

# UI
st.title("🔍 AI Resume Screener")
st.write("Upload a PDF resume and select a job category to analyze.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_categories = [
    "Accountant", "Engineer", "Sales", "Teacher", "HR", "IT"
]
selected_category = st.selectbox("💼 Select Job Category", job_categories)

if uploaded_file and selected_category:
    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.error("No text extracted. Please upload a text-based PDF.")
            st.stop()

        cleaned_text = clean_text(resume_text)
        predicted_role = model.predict([cleaned_text])[0]
        confidence = model.predict_proba([cleaned_text]).max()

        match_percentage = round(confidence * 100)
        if predicted_role.lower() == selected_category.lower():
            match_percentage = max(match_percentage, 85)
        else:
            match_percentage = min(match_percentage, 70)

        extracted = extract_details(resume_text)

        st.subheader("🔍 Extracted Resume Details")
        st.json(extracted)

        st.subheader("📊 Matching Result")
        st.write(f"**Predicted Role:** {predicted_role}")
        st.progress(match_percentage / 100)
        st.success(f"✅ Match Percentage: {match_percentage}%")
else:
    st.info("Please upload a resume and select a category.")
