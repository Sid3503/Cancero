from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from news import NEWS_API_KEY
import requests
from werkzeug.utils import secure_filename
from utils import generate_insights
from models import predict_cancer, predict_cancer_from_csv
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')  # Use environment variable for secret key

logging.basicConfig(level=logging.INFO)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def ensure_upload_folder_exists():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def fetch_doctor_data(location, specialization):
    return scrape_doctor_data(location, specialization)

def create_visualization(data):
    plt.figure(figsize=(8, 5))
    labels, values = zip(*data.items())
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Sample Visualization')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            prediction = predict_cancer(image_path)

            insights_md = generate_insights(predicted_label=prediction, file_type="Image")
            insights_html = markdown.markdown(insights_md)
            
            sample_data = {'Category A': 10, 'Category B': 15, 'Category C': 7}
            plot_image = create_visualization(sample_data)

            return render_template(
                'results.html',
                prediction=prediction,
                file_type='Image',
                file_name=filename,
                insights=insights_html,
                plot_image=plot_image  # Pass plot image to the template
            )
        except Exception as e:
            logging.error(f'Error processing image: {str(e)}')
            flash('An error occurred during processing the image. Please try again.')
            return redirect('/')
    
    else:
        flash('Invalid image file type')
        return redirect('/')

@app.route('/doctors', methods=['GET', 'POST'])
def doctors():
    doctor_data = []
    if request.method == 'POST':
        location = request.form['location']
        specialization = request.form['specialization']
        
        try:
            doctor_data = fetch_doctor_data(location, specialization)
        except Exception as e:
            logging.error(f'Error fetching doctor data: {str(e)}')
            flash('An error occurred while fetching doctor data. Please try again.')

    return render_template('doctors.html', doctors=doctor_data)

@app.route('/news', methods=['GET', 'POST'])
def news():
    query = "CANCER" 

    if request.method == 'POST':
        query = request.form.get('query')  

    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('articles', [])
    except requests.RequestException as e:
        logging.error(f'Error fetching news articles: {str(e)}')
        articles = []  # Fallback to an empty list if an error occurs
        flash('An error occurred while fetching news articles. Please try again.')

    return render_template('news.html', articles=articles)


@app.route('/charts', methods=['GET', 'POST'])
def charts():
    asr_df = pd.read_csv('C:/Users/Siddharth/Desktop/devFinal/data/dataset-asr-inc-both-sexes-in-2022-world.csv')

    mortality_df = pd.read_csv('C:/Users/Siddharth/Desktop/devFinal/data/dataset-asr-mort-both-sexes-in-2022-world.csv')

    asr_data = asr_df[['Label', 'ASR (World)']].dropna()

    mortality_data = mortality_df[['Label', 'ASR (World)']].dropna()

    prevalent_all_df = pd.read_csv('C:/Users/Siddharth/Desktop/devFinal/data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-all-cancers.csv')
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

    
    prevalent_cancer_df = pd.read_csv('C:/Users/Siddharth/Desktop/devFinal/data/dataset-estimated-number-of-prevalent-cases-1-year-both-sexes-in-2022-continents.csv')

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

    if len(pie_cancer_labels) > len(pie_cancer_colors):
        factor = (len(pie_cancer_labels) // len(pie_cancer_colors)) + 1
        pie_cancer_colors = (pie_cancer_colors * factor)[:len(pie_cancer_labels)]

    asr_chart_data = asr_data.to_dict(orient='records')
    mortality_chart_data = mortality_data.to_dict(orient='records')

    scatter_df = pd.read_csv("C:/Users/Siddharth/Desktop/devFinal/data/dataset-mort-asr-world-vs-inc-asr-world-both-sexes-in-2022-all-cancers.csv")

    scatter_data = scatter_df[['Population', 'Incidence - ASR (World)', 'Mortality - ASR (World)']].to_dict(orient='records')

    df_full = pd.read_csv("C:/Users/Siddharth/Desktop/devFinal/data/dataset-asr-inc-both-sexes-in-2022-world-vs-asia.csv")
    asr_asia_values = df_full.groupby("Label")["ASR (World)"].transform(lambda x: x.iloc[1] if len(x) > 1 else None)

    df_full["ASR Asia"] = asr_asia_values
    df_unique = df_full.drop_duplicates(subset=['Label'])

    compare_data = df_unique[["Label", "ASR (World)", "ASR Asia"]].to_dict(orient='records')

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

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    ensure_upload_folder_exists()
    
    if 'csv' not in request.files:
        flash('No CSV file uploaded')
        return redirect(request.url)
    
    file = request.files['csv']
    
    if file and allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            prediction = predict_cancer_from_csv(csv_path)

            insights_md = generate_insights(predicted_label=prediction, file_type="CSV")
            insights_html = markdown.markdown(insights_md)
            
            doctor_data = fetch_doctor_data("Bangalore", prediction)  # Change location if necessary
            
            sample_data = {'Category A': 10, 'Category B': 15, 'Category C': 7}
            plot_image = create_visualization(sample_data)
            
            return render_template(
                'results.html',
                prediction=prediction,
                file_type='CSV',
                file_name=filename,
                insights=insights_html,
                doctors=doctor_data,
                plot_image=plot_image
            )
        except Exception as e:
            logging.error(f'Error processing CSV: {str(e)}')
            flash('An error occurred during processing the CSV. Please try again.')
            return redirect('/')
    
    else:
        flash('Invalid CSV file type')
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
