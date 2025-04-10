import spacy
import requests

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# API Ninjas key
API_KEY = "8mWnKLqAlWsv2CZPIEnYQg==hiijUled5aSgnHXK"
API_URL = "https://api.api-ninjas.com/v1/exercises"

# Updated mappings
EXPERIENCE_MAP = {
    "beginner": "beginner",
    "new": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "pro": "advanced"
}

GOAL_MAP = {
    "strength": ["chest", "biceps", "quadriceps"],
    "muscle": ["chest", "biceps", "quadriceps"],
    "weight loss": ["abdominals", "quadriceps"],
    "endurance": ["abdominals", "quadriceps", "glutes"],  # Broader muscles
    "tone": ["arms", "chest"]
}

EQUIPMENT_MAP = {
    "dumbbells": "dumbbell",
    "dumbbell": "dumbbell",
    "barbell": "barbell",
    "none": "body_only",
    "no equipment": "body_only",
    "kettlebell": "kettlebell",
    "gym": "all"
}

def process_input(user_input):
    """Extract key info from user text using NLP."""
    doc = nlp(user_input.lower())
    
    experience = "beginner"
    goal = "strength"
    equipment = "body_only"
    
    for token in doc:
        if token.text in EXPERIENCE_MAP:
            experience = EXPERIENCE_MAP[token.text]
        for g, muscles in GOAL_MAP.items():
            if g in user_input:
                goal = g
                break
        for eq, api_eq in EQUIPMENT_MAP.items():
            if eq in user_input:
                equipment = api_eq
                break
    
    return {"experience": experience, "goal": goal, "equipment": equipment}

def map_to_api_filters(extracted_info):
    """Map extracted info to API filters."""
    filters = {
        "difficulty": extracted_info["experience"],
        "muscle": GOAL_MAP.get(extracted_info["goal"], ["chest"])[0],
        "type": "strength" if extracted_info["goal"] in ["strength", "muscle", "tone"] else "cardio"
    }
    if extracted_info["equipment"] != "all":
        filters["equipment"] = extracted_info["equipment"]
    return filters

def get_workout(filters):
    """Fetch exercises from API Ninjas with fallback."""
    # Try primary query
    api_url = f"{API_URL}?muscle={filters['muscle']}&difficulty={filters['difficulty']}&type={filters['type']}"
    if 'equipment' in filters:
        api_url += f"&equipment={filters['equipment']}"
    
    response = requests.get(api_url, headers={'X-Api-Key': API_KEY})
    
    if response.status_code == requests.codes.ok:
        exercises = response.json()
        if exercises:
            routine = []
            for i, ex in enumerate(exercises[:3], 1):
                routine.append(f"{i}. Do 3 sets of 10 {ex['name']}, rest 60s")
            return "\n".join(routine)
    
    # Fallback: Try without equipment or difficulty
    fallback_url = f"{API_URL}?type={filters['type']}"
    response = requests.get(fallback_url, headers={'X-Api-Key': API_KEY})
    
    if response.status_code == requests.codes.ok:
        exercises = response.json()
        if exercises:
            routine = []
            for i, ex in enumerate(exercises[:3], 1):
                routine.append(f"{i}. Do 3 sets of 10 {ex['name']}, rest 60s")
            return "\n".join(routine)
    
    return "No exercises found even after fallback. Try a different goal or equipment!"

def main():
    print("Welcome to your Gym Workout Chatbot!")
    user_input = input("Tell me about yourself and your goals (e.g., 'I’m a beginner, want to build muscle, have dumbbells'): ")
    
    extracted_info = process_input(user_input)
    print(f"Extracted: {extracted_info}")
    
    filters = map_to_api_filters(extracted_info)
    print(f"API Filters: {filters}")
    
    workout = get_workout(filters)
    print("\nYour Workout Routine:")
    print(workout)

if __name__ == "__main__":
    main()