import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import hashlib

# Initialize the embedding model
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def load_courses():
    """Load course data from CSV file"""
    courses = pd.read_csv('courses_data.csv')
    return courses

def get_dataset_hash():
    """Create a hash of the dataset to detect changes"""
    courses = load_courses()
    # Create a hash based on the content of the dataset
    dataset_content = courses.to_string()
    return hashlib.md5(dataset_content.encode()).hexdigest()

def process_student_profile(background, career_goal, interests):
    """Convert student info into searchable text"""
    student_text = f"{background} {career_goal} {interests}"
    return student_text

def create_course_embeddings():
    """Create semantic embeddings for courses with dataset change detection"""
    courses = load_courses()
    courses['full_text'] = courses['title'] + ' ' + courses['description'] + ' ' + courses['skills']

    embeddings_file = 'course_embeddings.pkl'
    hash_file = 'dataset_hash.txt'
    
    current_hash = get_dataset_hash()
    
    # Check if we need to regenerate embeddings
    should_regenerate = False
    
    if not os.path.exists(embeddings_file):
        print("No existing embeddings found. Creating new embeddings...")
        should_regenerate = True
    elif not os.path.exists(hash_file):
        print("No hash file found. Regenerating embeddings...")
        should_regenerate = True
    else:
        # Read the stored hash
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash != current_hash:
            print("Dataset has changed! Regenerating embeddings...")
            should_regenerate = True
        else:
            print("Loading existing embeddings (dataset unchanged)...")
    
    if should_regenerate:
        print("Creating new embeddings... (this may take a few minutes)")
        course_texts = courses['full_text'].tolist()
        course_vectors = embedding_model.encode(course_texts)

        # Save embeddings
        with open(embeddings_file, 'wb') as f:
            pickle.dump(course_vectors, f)
        
        # Save current dataset hash
        with open(hash_file, 'w') as f:
            f.write(current_hash)
        
        print(f"✅ Embeddings created and saved! ({len(courses)} courses processed)")
    else:
        # Load existing embeddings
        with open(embeddings_file, 'rb') as f:
            course_vectors = pickle.load(f)
        print(f"✅ Loaded existing embeddings ({len(courses)} courses)")

    return courses, course_vectors, embedding_model

def get_student_embedding(student_text):
    """Convert student profile to embedding"""
    return embedding_model.encode([student_text])

def clear_embeddings_cache():
    """Clear cached embeddings when course data changes"""
    embeddings_file = 'course_embeddings.pkl'
    hash_file = 'dataset_hash.txt'
    
    if os.path.exists(embeddings_file):
        os.remove(embeddings_file)
        print(f"Removed {embeddings_file}")
    
    if os.path.exists(hash_file):
        os.remove(hash_file)
        print(f"Removed {hash_file}")

def force_regenerate_embeddings():
    """Force regeneration of embeddings (useful for debugging)"""
    print("🔄 Forcing regeneration of embeddings...")
    clear_embeddings_cache()
    courses, course_vectors, embedding_model = create_course_embeddings()
    return courses, course_vectors, embedding_model

def check_dataset_status():
    """Check the current status of the dataset and embeddings"""
    courses = load_courses()
    embeddings_file = 'course_embeddings.pkl'
    hash_file = 'dataset_hash.txt'
    
    print("📊 Dataset Status:")
    print(f"   • Total courses: {len(courses)}")
    print(f"   • Embeddings file exists: {os.path.exists(embeddings_file)}")
    print(f"   • Hash file exists: {os.path.exists(hash_file)}")
    
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"   • Embeddings shape: {embeddings.shape}")
    
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        current_hash = get_dataset_hash()
        print(f"   • Dataset changed: {stored_hash != current_hash}")
    
    return courses
