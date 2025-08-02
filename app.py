from flask import Flask, render_template, request
from dotenv import load_dotenv; load_dotenv()
from recommendation_engine import get_recommendations
from genai_helper import generate_explanation

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get student info from form
    student_background = request.form['background']
    career_goal = request.form['career_goal']
    interests = request.form['interests']
    
    # Get recommendations
    recommended_courses = get_recommendations(student_background, career_goal, interests)
    
    # Add AI explanations
    for course in recommended_courses:
        course['explanation'] = generate_explanation(course, student_background, career_goal)
    
    return render_template('results.html', courses=recommended_courses)

if __name__ == '__main__':
    app.run(debug=True)
