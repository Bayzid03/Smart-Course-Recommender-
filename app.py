from flask import Flask, render_template, request
from dotenv import load_dotenv; load_dotenv()
from recommendation_engine import get_recommendations
from genai_helper import generate_explanation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
