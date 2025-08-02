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

def get_recommendations_with_filters(background, career_goal, interests, 
                                   preferred_duration=None, skill_level=None, top_k=5):
    """Enhanced recommendations with filtering options"""
    
    # Get base recommendations
    all_recommendations = get_recommendations(background, career_goal, interests, top_k=20)
    
    # Apply filters
    filtered_recommendations = []
    
    for course in all_recommendations:
        # Duration filter (if specified)
        if preferred_duration:
            course_duration = course['duration'].lower()
            if preferred_duration.lower() not in course_duration:
                continue
        
        # Skill level filter (if specified)
        if skill_level:
            course_level = course.get('level', 'intermediate').lower()
            if skill_level.lower() != course_level:
                continue
        
        filtered_recommendations.append(course)
        
        # Stop when we have enough
        if len(filtered_recommendations) >= top_k:
            break
    
    return filtered_recommendations

def explain_recommendation_score(similarity_score):
    """Convert similarity score to human-readable explanation"""
    if similarity_score > 0.8:
        return "Perfect match!"
    elif similarity_score > 0.6:
        return "Great match"
    elif similarity_score > 0.4:
        return "Good match"
    elif similarity_score > 0.2:
        return "Fair match"
    else:
        return "Basic match"

# Debugging function to understand what the model is matching
def debug_similarity(student_text, course_title, course_description):
    """Debug function to see why courses are being matched"""
    from data_processing import embedding_model
    
    student_embedding = embedding_model.encode([student_text])
    course_embedding = embedding_model.encode([f"{course_title} {course_description}"])
    
    similarity = cosine_similarity(student_embedding, course_embedding)[0][0]
    
    print(f"Student: {student_text}")
    print(f"Course: {course_title}")
    print(f"Similarity: {similarity:.3f} ({similarity*100:.1f}%)")
    print(f"Match Quality: {explain_recommendation_score(similarity)}")
    
    return similarity
