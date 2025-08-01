from groq import Groq
import os
from dotenv import load_dotenv; load_dotenv()

def get_groq_client():
    """Get Groq client with API key"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

def generate_explanation(course, student_background, career_goal):
    """Generate AI explanation for why course is recommended"""
    
    prompt = f"""
    Student Background: {student_background}
    Career Goal: {career_goal}
    Recommended Course: {course['title']}
    Course Description: {course['description']}
    
    Explain in 2-3 sentences why this course is a good fit for this student:
    """
    
    try:
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            print("   Current working directory:", os.getcwd())
            print("   .env file exists:", os.path.exists('.env'))
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}")
        
        client = get_groq_client()
        print("🔄 Making API call to Groq...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0.7
        )
        
        result = chat_completion.choices[0].message.content.strip()
        print(f"AI Response received: {result[:50]}...")
        return result
    
    except Exception as e:
        print(f"❌ AI Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return f"This course matches your interests in {career_goal} and builds relevant skills."

def generate_learning_path_suggestion(recommended_courses, career_goal):
    """Bonus: Generate learning path with Groq"""
    
    course_titles = [course['title'] for course in recommended_courses[:3]]
    
    prompt = f"""
    Career Goal: {career_goal}
    Recommended Courses: {', '.join(course_titles)}
    
    Suggest the best order to take these courses and why. Keep it brief (2-3 sentences):
    """
    
    try:
        client = get_groq_client()
        print("🔄 Making learning path API call to Groq...")
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=80,
            temperature=0.5
        )
        
        result = chat_completion.choices[0].message.content.strip()
        print(f"Learning path response: {result[:50]}...")
        return result
    
    except Exception as e:
        print(f"❌ Learning path AI Error: {e}")
        return "Start with the fundamentals and gradually move to specialized topics."
