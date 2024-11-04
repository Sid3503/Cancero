# Cancero

This project is a Flask-based web application that allows users to upload medical images or PDFs, and it provides detailed insights by utilizing AI-generated summaries and predictions. The application uses a convolutional neural network (CNN) model for image classification and a generative model for creating structured insights from both images and PDF files.

## Features

- **Image Upload and Prediction**: Upload medical images (e.g., X-rays, ultrasounds, CT scans) for classification.
- **PDF Upload and Summarization**: Upload PDF files to generate markdown-formatted summaries.
- **AI-Generated Insights**: Provides structured insights based on the model’s classification and PDF content using Google's Generative AI model.
- **Image Probability Plot**: Displays a plot showing the model’s confidence for each class prediction.
- **Confidence Score**: Displays the confidence score for the image prediction.

## Requirements

- Python 3.8+
- Flask
- PyPDF2
- Google Generative AI Python Client
- Markdown
- NumPy
- dotenv
  
## App
![Screenshot 2024-10-10 205644](https://github.com/user-attachments/assets/0909b60e-b969-4c8f-a7e5-cbaf8de99e06)
