from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from news import NEWS_API_KEY
import requests
import google.generativeai as genai
from werkzeug.utils import secure_filename
from utils import generate_insights, summarize_data, get_pdf_text, answer_question, answer_image_question
from models import pred_and_plot, predict_cancer_for_ctmri
import markdown  # Import markdown library
from scraper import scrape_doctor_data 
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry backend for non-GUI operations
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import logging
import pandas as pd
from datetime import datetime
import urllib.parse


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')  # Use environment variable for secret key

# Set up logging
logging.basicConfig(level=logging.INFO)

# Allowed file types
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_PDF_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'txt', 'doc', 'docx'}


# Helper function to check file types
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Ensure the upload folder exists
def ensure_upload_folder_exists():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

# Placeholder function to fetch scraped data
def fetch_doctor_data(location, specialization):
    return scrape_doctor_data(location, specialization)

# Function to create a visualization
def create_visualization(data):
    plt.figure(figsize=(8, 5))
    labels, values = zip(*data.items())
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Sample Visualization')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode the image to base64
    return base64.b64encode(buf.getvalue()).decode()


@app.route('/cancer-journey-stories')
def cancer_journey_stories():
     # Call the function to generate stories
    return render_template('cancer-journey-stories.html')

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Mission')
def mission():
    return render_template('Mission.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_pdf')
def upload_pdf():
    return render_template('pdf.html')


@app.route('/display_pdf', methods=['POST'])
def display_pdf():
    ensure_upload_folder_exists()
    
    if 'pdf' not in request.files:
        flash('No PDF file uploaded')
        return redirect(request.url)
    
    file = request.files['pdf']
    
    if file and allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            insights_md = summarize_data(file_path)
            insights_html = markdown.markdown(insights_md)

            # Render the results on display_pdf.html
            return render_template(
                'display_pdf.html',
                insights=insights_html,
                file_name=filename
            )
        except Exception as e:
            logging.error(f'Error processing PDF: {str(e)}')
            flash('An error occurred during processing the PDF. Please try again.')
            return redirect('/upload_pdf')
    
    else:
        flash('Invalid PDF file type')
        return redirect('/upload_pdf')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    ensure_upload_folder_exists()
    
    if 'image' not in request.files:
        flash('No image file uploaded')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            threshold = 0.9
            
            # Read and encode the uploaded image
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call model prediction for image
            prediction, probabilities, class_names, raw_pred = pred_and_plot(file_path, img_shape=224)

            max_prob = np.max(raw_pred)
            

            # Log the probabilities for debugging
            logging.info(f'Prediction: {prediction}')
            logging.info(f'Probabilities: {probabilities}')
            logging.info(f'Class Names: {class_names}')

            # Generate medical insights
            insights_md = generate_insights(predicted_label=prediction, file_type="Image")
            insights_html = markdown.markdown(insights_md)
            
            # Store insights_md in session
            session['insights_ctmri_md'] = insights_md

            # Generate the probability plot
            img = io.BytesIO()
            plt.figure(figsize=(6, 4))  # Smaller figure size
            bars = plt.bar(class_names, probabilities, color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel('Probability (%)')
            plt.title('Prediction Probabilities')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)

            # Add probability labels on top of each bar
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{prob:.2f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Log the plot URL length to ensure it's being generated
            logging.info(f'Plot URL length: {len(plot_url)}')

            # Delete the uploaded image after encoding
            os.remove(file_path)
            logging.info(f'Deleted image file: {file_path}')

            if max_prob < threshold:
                message = (
                        "Outlier detected: This image does not belong to any of the cancer types our model currently supports. "
                        "We are continuously scaling our model to accommodate more cancer classes in the future. As of now, "
                        "the model accurately detects the following cancer types:\n\n"
                        "- Colon Adenocarcinoma\n"
                        "- Colon Benign Tissue\n"
                        "- Lung Adenocarcinoma\n"
                        "- Lung Benign Tissue\n"
                        "- Lung Squamous Cell Carcinoma\n"
                        "- Brain Glioma\n"
                        "- Brain Meningioma\n"
                        "- Brain No Tumor\n"
                        "- Brain Pituitary\n"
                        "- Breast Cancer\n"
                        "- Breast Non-Cancer\n"
                        "- Lung Benign\n"
                        "- Lung Malignant\n"
                        "- Lung Normal\n\n"
                        "Stay tuned as we expand our model to cover additional cancer types, providing more comprehensive support for cancer detection."
                    )
                
                
                return render_template(
                    'results.html',
                    prediction=prediction,
                    file_type='Image',
                    file_name=filename,
                    insights=message,
                    plot_url=plot_url,         # Pass the plot URL to the template
                    uploaded_image=encoded_image
                )
                

            else:
                return render_template(
                'results.html',
                prediction=prediction,
                file_type='Image',
                file_name=filename,
                insights=insights_html,
                plot_url=plot_url,         # Pass the plot URL to the template
                uploaded_image=encoded_image  # Pass the uploaded image to the template
            )
        except Exception as e:
            logging.error(f'Error processing image: {str(e)}')
            flash('An error occurred during processing the image. Please try again.')
            return redirect('/')
    
    else:
        flash('Invalid image file type')
        return redirect('/')
    

@app.route('/ask_question', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        question = request.form['question']
        file_name = request.form['file_name']
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        pdf_text = get_pdf_text(pdf_path)

        ctmri_insights = session.get('insights_ctmri_md', '')

        answer_md = answer_question(question, pdf_text, ctmri_insights)
        answer_html = markdown.markdown(answer_md)

        insights_md = summarize_data(pdf_path)
        insights_html = markdown.markdown(insights_md)

        return render_template(
            'display_pdf.html',
            insights=insights_html,
            file_name=file_name,
            answer=answer_html
        )
    

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    doctor_data = []
    if request.method == 'POST':
        location = request.form['location']
        specialization = request.form['specialization']

        try:
            # Fetch doctors based on location and specialization
            doctor_data = fetch_doctor_data(location, specialization)
        except Exception as e:
            logging.error(f'Error fetching doctor data: {str(e)}')
            flash('An error occurred while fetching doctor data. Please try again.')

    return render_template('doctors.html', doctors=doctor_data)

@app.route('/news', methods=['GET', 'POST'])
def news():
    query = "cancer"
    health_sources = (
        "healthline,"
        "medical-news-today,"
        "cancer-news,"
        "news-medical.net,"
        "cancer.gov,"
        "nih.gov,"
        "webmd,"
        "mayoclinic.org,"
        "health.com,"
        "everydayhealth.com,"
        "cancerresearchuk.org,"
        "cancercare.org,"
        "harvard.edu,"
        "clevelandclinic.org,"
        "mdanderson.org,"
        "aacr.org,"
        "cancer.org,"
        "coping.com,"
        "cancer.net,"
        "oncolink.org,"
        "healthaffairs.org,"
        "ascopubs.org,"
        "jco.ascopubs.org,"
        "breastcancer.org,"
        "uspreventiveservicestaskforce.org"
    )
    
    # List of universal media sources
    universal_sources = (
        "bbc-news,"
        "cnn,"
        "the-verge,"
        "nytimes,"
        "reuters,"
        "forbes,"
        "the-washington-post,"
        "the-guardian,"
        "al-jazeera-english,"
        "bloomberg,"
        "abc-news,"
        "cbs-news,"
        "fox-news,"
        "national-geographic,"
        "usa-today,"
        "time"
    )

    sources = health_sources + "," + universal_sources

    if request.method == 'POST':
        query = request.form.get('query')  # Get the query from the user's input

    # URL encode the query to handle spaces and special characters
    encoded_query = urllib.parse.quote_plus(query)

    # Get today's date in 'YYYY-MM-DD' format
    today = datetime.today().strftime('%Y-%m-%d')

    # Construct the URL to fetch today's articles
    url_today = f"https://newsapi.org/v2/everything?q={encoded_query}&from={today}&to={today}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&sources={sources}"
    
    # Make the request to News API for today's articles
    response_today = requests.get(url_today)
    news_data_today = response_today.json()

    # If no articles found, try fetching articles without the date filter
    if news_data_today.get('totalResults', 0) == 0:
        print("No articles found for today. Fetching without date filter.")
        url_no_date = f"https://newsapi.org/v2/everything?q={encoded_query}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&sources={sources}"
        response_no_date = requests.get(url_no_date)
        news_data_no_date = response_no_date.json()
        articles = news_data_no_date.get('articles', [])
    else:
        articles = news_data_today.get('articles', [])

    return render_template('news.html', articles=articles)


@app.route('/charts', methods=['GET', 'POST'])
def charts():
    # Load the first CSV file for ASR (World) data
    asr_df = pd.read_csv('data/dataset-asr-inc-both-sexes-in-2022-world.csv')

    # Load the second CSV file for Mortality data
    mortality_df = pd.read_csv('data/dataset-asr-mort-both-sexes-in-2022-world.csv')

    # Extract 'Label' and 'ASR (World)' columns for the ASR chart and drop any NaN values
    asr_data = asr_df[['Label', 'ASR (World)']].dropna()

    # Extract 'Label' and 'Total' columns for the Mortality chart and drop any NaN values
    mortality_data = mortality_df[['Label', 'ASR (World)']].dropna()

    prevalent_all_df = pd.read_csv('data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-all-cancers.csv')
    prevalent_all_df = prevalent_all_df.drop_duplicates(subset='Label')

    pie_all_labels = prevalent_all_df['Label'].tolist()
    pie_all_data = prevalent_all_df['Total'].tolist()

    pie_all_colors = [
        'rgba(255, 99, 132, 0.6)',    # Red
        'rgba(54, 162, 235, 0.6)',    # Blue
        'rgba(255, 206, 86, 0.6)',    # Yellow
        'rgba(75, 192, 192, 0.6)',    # Green
        'rgba(153, 102, 255, 0.6)',   # Purple
        'rgba(255, 159, 64, 0.6)',    # Orange
        'rgba(199, 199, 199, 0.6)',   # Grey
        'rgba(83, 102, 255, 0.6)',    # Indigo
        'rgba(255, 99, 71, 0.6)',     # Tomato
        'rgba(60, 179, 113, 0.6)'     # Medium Sea Green
    ]

    if len(pie_all_labels) > len(pie_all_colors):
        factor = (len(pie_all_labels) // len(pie_all_colors)) + 1
        pie_all_colors = (pie_all_colors * factor)[:len(pie_all_labels)]


    prevalent_cancer_df = pd.read_csv('data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-continents.csv')

    prevalent_cancer_df = prevalent_cancer_df.drop_duplicates(subset='Label')

    pie_cancer_labels = prevalent_cancer_df['Label'].tolist()
    pie_cancer_data = prevalent_cancer_df['Total'].tolist()

    pie_cancer_colors = [
        'rgba(255, 99, 132, 0.6)',    # Red
        'rgba(54, 162, 235, 0.6)',    # Blue
        'rgba(255, 206, 86, 0.6)',    # Yellow
        'rgba(75, 192, 192, 0.6)',    # Green
        'rgba(153, 102, 255, 0.6)',   # Purple
        'rgba(255, 159, 64, 0.6)',    # Orange
        'rgba(199, 199, 199, 0.6)',   # Grey
        'rgba(83, 102, 255, 0.6)',    # Indigo
        'rgba(255, 99, 71, 0.6)',     # Tomato
        'rgba(60, 179, 113, 0.6)'     # Medium Sea Green
    ]

    # Ensure there are enough colors for all labels
    if len(pie_cancer_labels) > len(pie_cancer_colors):
        factor = (len(pie_cancer_labels) // len(pie_cancer_colors)) + 1
        pie_cancer_colors = (pie_cancer_colors * factor)[:len(pie_cancer_labels)]

    # Convert ASR and Mortality data to dictionary format (for use in JavaScript)
    asr_chart_data = asr_data.to_dict(orient='records')
    mortality_chart_data = mortality_data.to_dict(orient='records')

    scatter_df = pd.read_csv("data/dataset-mort-asr-world-vs-inc-asr-world-both-sexes-in-2022-all-cancers.csv")

    scatter_data = scatter_df[['Population', 'Incidence - ASR (World)', 'Mortality - ASR (World)']].to_dict(orient='records')

    # Creating a new column for ASR Asia
    df_full = pd.read_csv("data/dataset-asr-inc-both-sexes-in-2022-world-vs-asia.csv")
    asr_asia_values = df_full.groupby("Label")["ASR (World)"].transform(lambda x: x.iloc[1] if len(x) > 1 else None)

    # Adding the ASR Asia column to the DataFrame
    df_full["ASR Asia"] = asr_asia_values
    df_unique = df_full.drop_duplicates(subset=['Label'])

    compare_data = df_unique[["Label", "ASR (World)", "ASR Asia"]].to_dict(orient='records')

    # Pass both datasets to the template
    return render_template(
        'charts.html',
        asr_chart_data=asr_chart_data,
        mortality_data=mortality_chart_data,
        pie_all_labels=pie_all_labels,
        pie_all_data=pie_all_data,
        pie_all_colors=pie_all_colors,
        pie_cancer_labels=pie_cancer_labels,
        pie_cancer_data=pie_cancer_data,
        pie_cancer_colors=pie_cancer_colors,
        scatter_data=scatter_data,
        compare_data=compare_data
    )

@app.route('/upload_ct_mri', methods=['GET', 'POST'])
def upload_ct_mri():
    ensure_upload_folder_exists()
    
    if 'image' not in request.files:
        flash('No image file uploaded')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            threshold = 0.9
            
            # Read and encode the uploaded image
            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call model prediction for image
            prediction, probabilities, class_names, raw_pred = predict_cancer_for_ctmri(file_path, img_shape=224)

            max_prob = np.max(raw_pred)
            confidence_score = max_prob * 100  # Calculate confidence score as a percentage

            # Log the probabilities for debugging
            logging.info(f'Prediction: {prediction}')
            logging.info(f'Probabilities: {probabilities}')
            logging.info(f'Class Names: {class_names}')

            # Generate medical insights
            insights_md = generate_insights(predicted_label=prediction, file_type="Image")
            insights_html = markdown.markdown(insights_md)
            
            # Store insights_md in session
            session['insights_ctmri_md'] = insights_md

            # Generate the probability plot
            img = io.BytesIO()
            plt.figure(figsize=(6, 4))  # Smaller figure size
            bars = plt.bar(class_names, probabilities, color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel('Probability (%)')
            plt.title('Prediction Probabilities')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 100)

            # Add probability labels on top of each bar
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{prob:.2f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Log the plot URL length to ensure it's being generated
            logging.info(f'Plot URL length: {len(plot_url)}')

            # Delete the uploaded image after encoding
            os.remove(file_path)
            logging.info(f'Deleted CT/MRI image file: {file_path}')

            if max_prob < threshold:
                prediction = "Outlier"
                message = (
                        "Outlier detected: This image does not belong to any of the cancer types our model currently supports. "
                        "We are continuously scaling our model to accommodate more cancer classes in the future. As of now, "
                        "the model accurately detects the following cancer types:\n\n"
                        "- Colon Adenocarcinoma\n"
                        "- Colon Benign Tissue\n"
                        "- Lung Adenocarcinoma\n"
                        "- Lung Benign Tissue\n"
                        "- Lung Squamous Cell Carcinoma\n"
                        "- Brain Glioma\n"
                        "- Brain Meningioma\n"
                        "- Brain No Tumor\n"
                        "- Brain Pituitary\n"
                        "- Breast Cancer\n"
                        "- Breast Non-Cancer\n"
                        "- Lung Benign\n"
                        "- Lung Malignant\n"
                        "- Lung Normal\n\n"
                        "Stay tuned as we expand our model to cover additional cancer types, providing more comprehensive support for cancer detection."
                    )
                
                
                return render_template(
                    'results.html',
                    prediction=prediction,
                    file_type='Image',
                    file_name=filename,
                    insights=message,
                    plot_url=plot_url,         # Pass the plot URL to the template
                    uploaded_image=encoded_image,
                    confidence_score=confidence_score  # Pass the confidence score
                )
            

            else:
                return render_template(
                'results.html',
                prediction=prediction,
                file_type='Image',
                file_name=filename,
                insights=insights_html,
                plot_url=plot_url,         # Pass the plot URL to the template
                uploaded_image=encoded_image,  # Pass the uploaded image to the template
                confidence_score=confidence_score  # Pass the confidence score
            )
        except Exception as e:
            logging.error(f'Error processing image: {str(e)}')
            flash('An error occurred during processing the image. Please try again.')
            return redirect('/')
    
    else:
        flash('Invalid image file type')
        return redirect('/')

    

@app.route('/personalized-plans')
def fitness_blogs():
    return render_template('health_dashboard.html')

@app.route('/virtual-workouts')
def virtual_workouts():
    return render_template('virtual_workouts.html')

@app.route('/nutritional-guidance')
def nutritional_guidance():
    return render_template('nutritional_guidance.html')

@app.route('/webinars-workshops')
def webinars_workshops():
    return render_template('webinars_workshops.html')

@app.route('/virtual-reality')
def virtual_reality():
    return render_template('vr.html')

@app.route('/health-literacy')
def health_literacy():
    return render_template('health_literacy.html')


if __name__ == '__main__':
    app.run(debug=True)
