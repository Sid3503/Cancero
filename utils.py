# utils.py

import google.generativeai as genai
from google.generativeai import GenerativeModel, types
from dotenv import load_dotenv
import os
import markdown
from PyPDF2 import PdfReader

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def generate_insights(predicted_label, file_type, model_name="gemini-1.5-flash"):
    """
    Generate detailed medical insights based on the predicted label and imaging modality.

    Parameters:
        predicted_label (str): The label predicted by the model (e.g., 'breast_cancer', 'lung_malignant').
        file_type (str): The type of imaging used ('Ultrasound' for breast, 'CT Scan' for lung and brain).
        model_name (str): The name of the Generative AI model to use.

    Returns:
        str: The generated medical insights in HTML format.
    """

    # Define the comprehensive classification system including brain, breast, and expanded lung cancer types
    classification_system = (
        "The classification system utilized leverages Convolutional Neural Networks (CNNs) to categorize tissue "
        "into the following types: "
        "Lung benign tissue, "
        "Lung adenocarcinoma, "
        "Lung squamous cell carcinoma, "
        "Lung malignant tissue, "
        "Lung normal tissue, "
        "Colon adenocarcinoma, "
        "Colon benign tissue, "
        "Brain glioma, "
        "Brain meningioma, "
        "Brain non-tumor tissue, "
        "Brain pituitary tumors, "
        "Breast cancer, "
        "Breast non-cancerous tissue. "
        "The system supports multi-class classification with high sensitivity, enabling differentiation between various cancerous and non-cancerous tissue types."
    )

    # Validate the predicted_label
    valid_class_names = [
        'colon_adenocarcinoma', 'colon_benign_tissue',
        'lung_adenocarcinoma', 'lung_benign_tissue',
        'lung_squamous_cell_carcinoma',
        'brain_glioma', 
        'brain_meningioma', 
        'brain_notumor', 
        'brain_pituitary',
        'breast_cancer', 
        'breast_non_cancer', 
        'lung_benign', 
        'lung_malignant',
        'lung_normal'
    ]

    if predicted_label not in valid_class_names:
        raise ValueError(f"Invalid predicted label: {predicted_label}")

    # Determine the imaging modality based on file_type and predicted_label
    imaging_modality = ""
    if file_type.lower() == "ultrasound":
        imaging_modality = "ultrasound images"
    elif file_type.lower() in ["ct scan", "ct/mri", "ct"]:
        imaging_modality = "CT scans"
    else:
        imaging_modality = file_type  # Default to whatever is provided

    # Update the prompt to include the specific imaging modality
    default_prompt = (
        f"As a medical expert specializing in oncology, you have detected {predicted_label.replace('_', ' ').title()} based on the "
        f"provided {imaging_modality} (histological images). "
        f"{classification_system}\n\n"
        
        f"Please provide the following in your response:\n\n"
        
        f"1. **Detailed Explanation**: Briefly explain what {predicted_label.replace('_', ' ').title()} is, the characteristics of "
        f"the detected tissue type (e.g., Lung benign tissue, Lung adenocarcinoma, Lung squamous cell carcinoma, "
        f"Lung malignant tissue, Lung normal tissue, Colon adenocarcinoma, Colon benign tissue, Brain glioma, "
        f"Brain meningioma, Brain non-tumor tissue, Brain pituitary tumors, Breast cancer, Breast non-cancerous tissue), "
        f"and the significance of CNN-based detection in identifying cancerous and non-cancerous regions.\n"
        
        f"2. **Stage and Risk Factors**: Discuss potential stages of {predicted_label.split('_')[0].capitalize()} cancer (if applicable), "
        f"and elaborate on key risk factors. Mention how CNN architecture helps in differentiating tissue types, "
        f"including individual cell morphology and overall tissue organization.\n"
        
        f"3. **Immediate Next Steps**: Suggest further diagnostic tests to confirm the classification, "
        f"such as additional biopsies, molecular testing, or imaging techniques like CT scans, MRI, ultrasound, or colonoscopy. "
        f"Highlight how early detection through {imaging_modality.lower()} impacts diagnostic accuracy and treatment planning.\n"
        
        f"4. **Treatment Options**: Outline the treatment options available based on the classification "
        f"of the biopsy (e.g., surgery, radiation, chemotherapy, targeted therapy, immunotherapy). Discuss early intervention "
        f"for localized tumors or more aggressive treatments for advanced carcinomas if applicable.\n"
        
        f"5. **Lifestyle Adjustments and Prevention**: Recommend lifestyle changes or preventive measures "
        f"that the patient can take to improve health outcomes and reduce recurrence risk, especially for benign or early-stage cases. "
        f"For lung cancer, this might include smoking cessation, while for colon cancer, dietary adjustments could be emphasized. "
        f"For brain and breast cancers, include relevant lifestyle recommendations.\n"
        
        f"6. **Prognosis and Patient Care**: Provide an overview of the prognosis based on the type and stage "
        f"of the tissue classification, discussing how early intervention, continuous monitoring, and follow-ups "
        f"are vital for patient care and improved long-term outcomes."
    )

    # Initialize the Generative AI model
    model = GenerativeModel(model_name)
    
    # Generate the content using streaming
    response_stream = model.generate_content(
        [default_prompt],
        generation_config=types.GenerationConfig(temperature=0.7),
        stream=True
    )
    
    # Collect the generated text
    insights_output = ""
    for message in response_stream:
        insights_output += message.text
    
    # Resolve the stream
    response_stream.resolve()
    
    return insights_output



def get_pdf_text(pdf_path):
    """Extract text from the PDF."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def summarize_data(pdf_path):
    """Summarize content from PDF with structured markdown."""
    text_data = get_pdf_text(pdf_path)

    # Define prompt to generate markdown-structured insights
    default_prompt = (
        "Summarize the PDF content and provide key insights in markdown format. "
        "Include the following sections:\n\n"
        "1. **Document Summary**: A brief overview of the document's purpose and main points.\n"
        "2. **Key Findings**: Bullet points for the important findings or data.\n"
        "3. **Important Dates and Names**: Highlight any important dates, names, or locations mentioned.\n"
        "4. **Next Steps or Recommendations**: Mention any suggested actions or next steps based on the document's content.\n"
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response_stream = model.generate_content(
        [default_prompt, text_data],
        generation_config=genai.types.GenerationConfig(temperature=0.7),
        stream=True
    )

    # Collect the generated markdown content
    summary_output_md = ""
    for message in response_stream:
        summary_output_md += message.text

    response_stream.resolve()

    # Convert markdown to HTML
    summary_output_html = markdown.markdown(summary_output_md)

    return summary_output_html


def answer_question(question, pdf_text, ctmri_insights):
    """Generate an answer based on the question, combining PDF content and CT/MRI insights."""
    # Check if the question is provided
    if not question.strip():
        return "Please provide a question related to health, diseases, or nutrition."

    # Define a detailed prompt combining PDF (blood report) and CT/MRI (image and cancer insights)
    prompt = (
        f"You are a highly knowledgeable medical assistant specializing in cancer, diseases, and general health. You "
        f"have access to the following patient data and insights, which include both a blood report and a CT/MRI "
        f"analysis. Use this combined information to answer the patient's question accurately and comprehensively.\n\n"
        
        f"1. Patient Blood Report: The PDF content below contains important details such as white blood cell counts, "
        f"hemoglobin levels, platelet counts, and any abnormal readings:\n\n"
        f"{pdf_text}\n\n"

        f"2. CT/MRI Insights: The CT/MRI analysis includes the type of cancer detected from an image analyzed by a CNN "
        f"model, along with a summary of the disease, its causes, symptoms, and progression:\n\n"
        f"{ctmri_insights}\n\n"

        f"Using this information, provide a comprehensive answer to the following question:\n\n"
        f"Question: {question}\n\n"

        f"Please address any aspects related to cancer, general health issues, diseases, diagnostic interpretations, "
        f"possible treatments, diet, and nutrition."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response_stream = model.generate_content(
        [prompt],
        generation_config=genai.types.GenerationConfig(temperature=0.7),
        stream=True
    )

    # Collect the generated answer
    answer_output = ""
    for message in response_stream:
        answer_output += message.text

    response_stream.resolve()

    return answer_output.strip()


def answer_image_question(question, insights_html):
    """
    Generate an answer based on the provided question and image insights (in HTML format).
    
    Args:
    - question (str): The question to be answered.
    - insights_html (str): The image insights in HTML format.
    
    Returns:
    - str: The generated answer in markdown format.
    """
    # Incorporate insights_html into the prompt
    prompt = f"Based on the following image insights:\n{insights_html}\nAnswer the question: {question}"
    
    # Call the model or API to generate the answer (this is just a placeholder)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response_stream = model.generate_content(
        [prompt],
        generation_config=genai.types.GenerationConfig(temperature=0.7),
        stream=True
    )

    answer_output = ""
    for message in response_stream:
        answer_output += message.text

    response_stream.resolve()

    return answer_output.strip()
