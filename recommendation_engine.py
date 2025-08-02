from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_processing import create_course_embeddings, process_student_profile, get_student_embedding

def get_recommendations(background, career_goal, interests, top_k=3):
    """Get top course recommendations for student using semantic embeddings"""
    
    # Load and process data
    courses, course_vectors, embedding_model = create_course_embeddings()
    student_text = process_student_profile(background, career_goal, interests)
    
    # Convert student profile to embedding (same vector space as courses)
    student_vector = get_student_embedding(student_text)
    
    # Calculate semantic similarities
    similarities = cosine_similarity(student_vector, course_vectors).flatten()
    
    # Get top recommendations
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    recommended_courses = []
    for idx in top_indices:
        course = courses.iloc[idx]
        similarity_score = similarities[idx]
        
        recommended_courses.append({
            'title': course['title'],
            'description': course['description'],
            'provider': course['provider'],
            'duration': course['duration'],
            'skills': course['skills'],
            'level': course.get('level', 'Intermediate'),  # Default level
            'rating': course.get('rating', 4.5),  # Default rating
            'similarity_score': float(similarity_score),  # Convert numpy float to Python float
            'match_percentage': int(similarity_score * 100)  # User-friendly percentage
        })
    
    return recommended_courses
