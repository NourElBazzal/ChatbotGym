import spacy
from spacy.training import Example
import random
import requests
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

# === Dataset Generation ===

TRAIN_DATA = [
    ("i’m new to working out and want to make my arms stronger.", {"entities": [(0, 3, "EXPERIENCE"), (30, 34, "BODY_PART"), (44, 52, "GOAL")]}),
    ("i’ve never exercised and want something for my legs at home.", {"entities": [(0, 6, "EXPERIENCE"), (33, 37, "BODY_PART"), (41, 45, "EQUIPMENT")]}),
    ("can you suggest an easy workout for my stomach to get in shape?", {"entities": [(10, 14, "EXPERIENCE"), (30, 37, "BODY_PART"), (41, 52, "GOAL")]}),
    ("i want to feel more fit and exercise with dumbbells.", {"entities": [(5, 17, "GOAL"), (35, 44, "EQUIPMENT")]}),
    ("i’m a beginner and want my back to look better at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (21, 25, "BODY_PART"), (29, 39, "GOAL"), (43, 46, "EQUIPMENT")]}),
    ("never done this before, but i want to work out my whole body.", {"entities": [(0, 5, "EXPERIENCE"), (40, 50, "BODY_PART")]}),
    ("i want something simple for my legs to not feel so tired.", {"entities": [(10, 16, "EXPERIENCE"), (26, 30, "BODY_PART"), (34, 50, "GOAL")]}),
    ("just starting out and want to tone my arms at home.", {"entities": [(0, 12, "EXPERIENCE"), (21, 25, "GOAL"), (29, 33, "BODY_PART"), (37, 41, "EQUIPMENT")]}),
    ("i don’t know how to start but want my stomach stronger.", {"entities": [(2, 17, "EXPERIENCE"), (29, 36, "BODY_PART"), (37, 45, "GOAL")]}),
    ("can i get a workout for my chest with a barbell to build muscle?", {"entities": [(22, 27, "BODY_PART"), (33, 40, "EQUIPMENT"), (44, 55, "GOAL")]}),
    ("hey, i’m new to the gym and want to get stronger.", {"entities": [(5, 8, "EXPERIENCE"), (10, 17, "EQUIPMENT"), (28, 39, "GOAL")]}),
    ("i want to work out at the gym to tone my legs.", {"entities": [(17, 24, "EQUIPMENT"), (26, 30, "GOAL"), (34, 38, "BODY_PART")]}),
    ("i’m starting out at the gym to build muscle.", {"entities": [(0, 3, "EXPERIENCE"), (15, 22, "EQUIPMENT"), (23, 34, "GOAL")]}),
    ("can you suggest a gym workout for beginners?", {"entities": [(10, 13, "EQUIPMENT"), (24, 33, "EXPERIENCE")]}),
    ("i want a gym workout to tone my arms.", {"entities": [(4, 7, "EQUIPMENT"), (11, 15, "GOAL"), (19, 23, "BODY_PART")]}),
    ("i’ve never been to the gym but want stronger arms.", {"entities": [(0, 6, "EXPERIENCE"), (10, 17, "EQUIPMENT"), (23, 31, "GOAL"), (32, 36, "BODY_PART")]}),
    ("never exercised, want to tone my legs at the gym.", {"entities": [(0, 5, "EXPERIENCE"), (17, 21, "GOAL"), (25, 29, "BODY_PART"), (33, 40, "EQUIPMENT")]}),
    ("i’m advanced and want muscle gain in my chest at the gym 3 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (17, 27, "GOAL"), (31, 36, "BODY_PART"), (40, 43, "EQUIPMENT"), (44, 57, "FREQUENCY")]}),
    ("i want to lose weight by working my whole body at home 4 days per week.", {"entities": [(5, 15, "GOAL"), (26, 36, "BODY_PART"), (40, 44, "EQUIPMENT"), (45, 59, "FREQUENCY")]}),
    ("advanced lifter here, aiming for weight loss with dumbbells 5 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (20, 30, "GOAL"), (36, 45, "EQUIPMENT"), (46, 60, "FREQUENCY")]}),
    ("i don’t know what to do but i want my arms to get stronger.", {"entities": [(26, 30, "BODY_PART"), (34, 45, "GOAL")]}),
    ("can you suggest something for my legs?", {"entities": [(19, 23, "BODY_PART")]}),
    ("i’m advanced and want muscle gain in my arms at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (17, 27, "GOAL"), (31, 35, "BODY_PART"), (39, 42, "EQUIPMENT")]}),
    ("beginner here, want weight loss with no equipment 4 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (14, 24, "GOAL"), (30, 42, "EQUIPMENT"), (43, 57, "FREQUENCY")]}),
    ("i want muscle gain for my chest.", {"entities": [(2, 12, "GOAL"), (17, 22, "BODY_PART")]}),
    ("i’m new and want weight loss at home.", {"entities": [(0, 3, "EXPERIENCE"), (11, 21, "GOAL"), (25, 29, "EQUIPMENT")]}),
    ("advanced user, aiming for chest muscle gain 3 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (20, 25, "BODY_PART"), (26, 36, "GOAL"), (37, 50, "FREQUENCY")]}),
    ("i want to work my whole body for weight loss.", {"entities": [(12, 22, "BODY_PART"), (27, 37, "GOAL")]}),
    ("i can do 3 days a week for my legs.", {"entities": [(4, 17, "FREQUENCY"), (22, 26, "BODY_PART")]}),
    ("i don’t know what to gain but want my chest stronger.", {"entities": [(30, 35, "BODY_PART"), (36, 44, "GOAL")]}),
    ("i want muscle gain in my Chest.", {"entities": [(2, 12, "GOAL"), (17, 22, "BODY_PART")]}),
    ("my goal is muscle gain at the Gym.", {"entities": [(7, 17, "GOAL"), (22, 25, "EQUIPMENT")]}),
    ("Chest workouts at gym for muscle gain.", {"entities": [(0, 5, "BODY_PART"), (10, 13, "EQUIPMENT"), (18, 28, "GOAL")]}),
    ("i’m advanced, want my chest to grow at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (14, 19, "BODY_PART"), (23, 27, "GOAL"), (31, 34, "EQUIPMENT")]}),
    ("3 days a week for chest muscle gain.", {"entities": [(0, 13, "FREQUENCY"), (18, 23, "BODY_PART"), (24, 34, "GOAL")]}),
    ("i want muscle gain for chest at gym.", {"entities": [(2, 12, "GOAL"), (17, 22, "BODY_PART"), (26, 29, "EQUIPMENT")]}),
    ("muscle gain is my goal at the gym.", {"entities": [(0, 10, "GOAL"), (23, 26, "EQUIPMENT")]}),
    ("i’m advanced, muscle gain in chest at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (10, 20, "GOAL"), (24, 29, "BODY_PART"), (33, 36, "EQUIPMENT")]}),
    ("at the gym for chest workouts.", {"entities": [(0, 6, "EQUIPMENT"), (11, 16, "BODY_PART")]}),
    ("working out at gym to gain muscle.", {"entities": [(11, 14, "EQUIPMENT"), (18, 28, "GOAL")]}),
    ("at gym, i want chest strength.", {"entities": [(0, 6, "EQUIPMENT"), (14, 19, "BODY_PART"), (20, 28, "GOAL")]}),
]

# Lowercase training data
TRAIN_DATA = [(text.lower(), annotations) for text, annotations in TRAIN_DATA]

templates = [
    "i’m {EXPERIENCE} and want to {GOAL} my {BODY_PART} with {EQUIPMENT} {FREQUENCY}.",
    "can you suggest something {EXPERIENCE} for my {BODY_PART} to {GOAL} {FREQUENCY}?",
    "i want to {GOAL} and exercise my {BODY_PART} at {EQUIPMENT} {FREQUENCY}.",
    "{EXPERIENCE} here, looking to {GOAL} with {EQUIPMENT} for my {BODY_PART}.",
    "never done this, but i want my {BODY_PART} to {GOAL} {FREQUENCY}."
]

experience_terms = ["new", "beginner", "never", "just starting", "don’t know how", "first time", "starting out", "no experience", "easy", "advanced"]
goal_terms = ["get stronger", "look better", "feel more fit", "tone", "build muscle", "get in shape", "move better", "not feel so tired", "stronger", "not feel so weak", "weight loss", "muscle gain"]
body_part_terms = ["arms", "legs", "stomach", "back", "whole body", "chest", "tummy", "core"]
equipment_terms = ["home", "dumbbells", "barbell", "gym", "no equipment", "kettlebell", "none"]
frequency_terms = ["3 days a week", "4 days per week", "5 days a week", "twice a week", "every day"]

temp_nlp = spacy.blank("en")

def generate_training_data(n_samples=110):
    data = []
    for _ in range(n_samples):
        template = random.choice(templates)
        experience = random.choice(experience_terms)
        goal = random.choice(goal_terms)
        body_part = random.choice(body_part_terms)
        equipment = random.choice(equipment_terms)
        frequency = random.choice(frequency_terms)
        
        text = template.format(EXPERIENCE=experience, GOAL=goal, BODY_PART=body_part, EQUIPMENT=equipment, FREQUENCY=frequency).lower()
        entities = []
        
        for term, label in [(experience, "EXPERIENCE"), (goal, "GOAL"), (body_part, "BODY_PART"), (equipment, "EQUIPMENT"), (frequency, "FREQUENCY")]:
            start_idx = text.find(term)
            if start_idx != -1:
                end_idx = start_idx + len(term)
                entities.append((start_idx, end_idx, label))
        
        if entities:
            data.append((text, {"entities": entities}))
    return data

TRAIN_DATA.extend(generate_training_data(110))

# === NER Model Training ===

def train_ner_model(train_data, output_path="gym_ner_model", n_iter=150):
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    for label in ["BODY_PART", "GOAL", "EQUIPMENT", "EXPERIENCE", "FREQUENCY"]:
        ner.add_label(label)
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        print("Training NER model...")
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.1, sgd=optimizer, losses=losses)
            print(f"Iteration {itn+1}, Losses: {losses}")
    
    nlp.to_disk(output_path)
    print(f"Model saved to {output_path}")
    return nlp

train_ner_model(TRAIN_DATA, output_path="gym_ner_model", n_iter=150)

# === Chatbot Implementation ===

nlp = spacy.load("gym_ner_model")

API_URL = "https://exercisedb-api.vercel.app/api/v1/exercises?offset=0&limit=100"

EXPERIENCE_MAP = {
    "beginner": "beginner", "new": "beginner", "never": "beginner", "just starting": "beginner",
    "don’t know how": "beginner", "first time": "beginner", "starting out": "beginner",
    "no experience": "beginner", "easy": "beginner", "advanced": "advanced"
}

GOAL_MAP = {
    "strength": ["upper arms", "upper legs", "waist", "chest"], "muscle": ["upper arms", "upper legs", "waist", "chest"],
    "stronger": ["upper arms", "upper legs", "waist", "chest"], "weight loss": ["waist", "upper legs", "chest"],
    "endurance": ["waist", "upper legs", "glutes"], "tone": ["upper arms", "waist", "chest"],
    "fit": ["waist", "upper legs"], "look better": ["upper arms", "waist", "chest"],
    "move better": ["glutes", "upper legs"], "not feel so tired": ["waist", "upper legs"],
    "build muscle": ["upper arms", "upper legs", "waist", "chest"], "get in shape": ["waist", "upper legs", "chest"],
    "feel more fit": ["waist", "upper legs"], "not feel so weak": ["upper arms", "waist", "chest"],
    "muscle gain": ["upper arms", "upper legs", "waist", "chest"], "weight loss": ["waist", "upper legs", "chest"]
}

EQUIPMENT_MAP = {
    "dumbbells": "dumbbell", "dumbbell": "dumbbell", "barbell": "barbell", "none": "body weight",
    "no equipment": "body weight", "kettlebell": "kettlebell", "gym": None, "home": "body weight"
}

BODY_PART_MAP = {
    "arms": "upper arms", "legs": "upper legs", "stomach": "waist", "tummy": "waist",
    "back": "back", "whole body": ["upper arms", "upper legs", "waist", "back", "chest"],
    "chest": "chest", "core": "waist"
}

FREQUENCY_MAP = {
    "3 days a week": 3, "4 days per week": 4, "5 days a week": 5, "twice a week": 2, "every day": 7
}

def process_input(user_input, extracted_info):
    user_input = user_input.lower().replace("at the gym", "gym").replace("the gym", "gym")
    doc = nlp(user_input)
    
    print(f"Raw entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
    confirmation = None
    for ent in doc.ents:
        if ent.label_ == "EXPERIENCE" and not extracted_info["experience"]:
            extracted_info["experience"] = EXPERIENCE_MAP.get(ent.text, "beginner")
        elif ent.label_ == "GOAL" and not extracted_info["goal"]:
            extracted_info["goal"] = ent.text.replace("want ", "") if ent.text.startswith("want ") else ent.text
            extracted_info["goal"] = extracted_info["goal"] if extracted_info["goal"] in GOAL_MAP else "strength"
        elif ent.label_ == "EQUIPMENT" and not extracted_info["equipment"] and ent.text != "home":
            extracted_info["equipment"] = EQUIPMENT_MAP.get(ent.text, "body weight")
        elif ent.label_ == "EQUIPMENT" and ent.text == "home" and extracted_info["equipment"] not in ["body weight", None]:
            confirmation = "You mentioned home, but earlier said gym. Confirm home (body weight)?"
        elif ent.label_ == "BODY_PART" and not extracted_info["muscle"]:
            extracted_info["muscle"] = BODY_PART_MAP.get(ent.text, "upper arms")
        elif ent.label_ == "FREQUENCY" and not extracted_info["frequency"]:
            extracted_info["frequency"] = FREQUENCY_MAP.get(ent.text, 3)
    
    return extracted_info, confirmation

def map_to_api_filters(extracted_info):
    if isinstance(extracted_info["muscle"], list):
        body_parts = extracted_info["muscle"]
    else:
        body_parts = [extracted_info["muscle"] if extracted_info["muscle"] else random.choice(GOAL_MAP.get(extracted_info["goal"], ["upper arms"]))]
    filters = {"body_part": body_parts}
    if extracted_info["equipment"]:
        filters["equipment"] = extracted_info["equipment"]
    return filters, extracted_info["experience"], extracted_info["frequency"]

def get_workout(filters, experience, used_exercises, num_exercises=5):
    try:
        body_parts = filters['body_part'] if isinstance(filters['body_part'], list) else [filters['body_part']]
        response = requests.get(API_URL, timeout=10)
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Text: {response.text[:200]}...")
        
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict) or 'data' not in data or 'exercises' not in data['data']:
                print(f"Error: Unexpected response format, got {type(data)}")
                raise ValueError("Invalid API response format")
            
            exercises = data['data']['exercises']
            if not isinstance(exercises, list):
                print(f"Error: Expected a list of exercises, got {type(exercises)}")
                raise ValueError("Exercises is not a list")
            
            filtered_exercises = []
            # Try chest, fallback to upper arms
            if 'chest' in body_parts:
                chest_exercises = [ex for ex in exercises if 'chest' in [b.lower() for b in ex.get('bodyParts', [])]]
                print(f"Chest exercises found: {len(chest_exercises)}")
                if not chest_exercises:
                    body_parts = ['upper arms' if bp == 'chest' else bp for bp in body_parts]
            for ex in exercises:
                matches = True
                if any(bp.lower() in [b.lower() for b in ex.get('bodyParts', [])] for bp in body_parts):
                    if filters.get('equipment') == 'body weight' and 'body weight' not in ex.get('equipments', []):
                        matches = False
                    elif filters.get('equipment') and filters['equipment'] != 'body weight' and filters['equipment'] not in ex.get('equipments', []):
                        matches = False
                    if ex['name'] in used_exercises:
                        matches = False
                    if matches:
                        filtered_exercises.append(ex)
            
            if filtered_exercises:
                random.shuffle(filtered_exercises)
                routine = []
                sets = 4 if experience == "advanced" else 3
                reps = "12-15" if experience == "advanced" else "10"
                equipment_note = " (use moderate weight)" if experience == "advanced" and filters.get('equipment') not in ["body weight", None] else ""
                # For whole body, select one per body part
                if isinstance(filters['body_part'], list):
                    selected_parts = random.sample(body_parts, min(num_exercises, len(body_parts)))
                    selected = []
                    for part in selected_parts:
                        for ex in filtered_exercises:
                            if part.lower() in [b.lower() for b in ex.get('bodyParts', [])] and ex['name'] not in [s['name'] for s in selected]:
                                selected.append(ex)
                                break
                else:
                    selected = filtered_exercises[:num_exercises]
                for ex in selected[:num_exercises]:
                    desc = "\n    - ".join(ex.get('instructions', ["No instructions available."])[:5])
                    routine.append(f"{ex['name'].title()} - {sets} sets of {reps} reps{equipment_note}\n    - {desc}")
                    used_exercises.add(ex['name'])
                return routine
            
            print("No exact matches, trying fallback...")
            filtered_exercises = []
            for ex in exercises:
                if any(bp.lower() in [b.lower() for b in ex.get('bodyParts', [])] for bp in body_parts) and ex['name'] not in used_exercises:
                    filtered_exercises.append(ex)
            if filtered_exercises:
                random.shuffle(filtered_exercises)
                routine = []
                sets = 4 if experience == "advanced" else 3
                reps = "12-15" if experience == "advanced" else "10"
                equipment_note = " (use moderate weight)" if experience == "advanced" and filters.get('equipment') not in ["body weight", None] else ""
                selected = filtered_exercises[:num_exercises]
                for ex in selected:
                    desc = "\n    - ".join(ex.get('instructions', ["No instructions available."])[:5])
                    routine.append(f"{ex['name'].title()} - {sets} sets of {reps} reps{equipment_note}\n    - {desc}")
                    used_exercises.add(ex['name'])
                return routine
            
            print("No exercises found in fallback.")
        
        routine = []
        sets = 4 if experience == "advanced" else 3
        reps = "12-15" if experience == "advanced" else "10"
        body_part = random.choice(body_parts) if isinstance(body_parts, list) else body_parts
        print(f"API failed: {response.status_code} - {response.text[:200]}...")
        for i in range(1, num_exercises + 1):
            routine.append(f"{body_part} Exercise {i} - {sets} sets of {reps} reps\n    - No instructions available.")
        return routine
    
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        routine = []
        sets = 4 if experience == "advanced" else 3
        reps = "12-15" if experience == "advanced" else "10"
        body_part = random.choice(body_parts) if isinstance(body_parts, list) else body_parts
        for i in range(1, num_exercises + 1):
            routine.append(f"{body_part} Exercise {i} - {sets} sets of {reps} reps\n    - No instructions available.")
        return routine

def generate_weekly_plan(extracted_info, filters, experience, frequency):
    workout_days = frequency if frequency else 3
    plan = []
    day_count = 1
    used_exercises = set()
    
    for i in range(1, 8):
        if day_count <= workout_days:
            if i % 2 == 1 or (workout_days > 3 and i in [4, 6][:workout_days-3]):
                exercises = get_workout(filters, experience, used_exercises, num_exercises=5)
                plan.append(f"Day {i}: Workout\n  - " + "\n  - ".join(exercises))
                day_count += 1
            else:
                plan.append(f"Day {i}: Rest")
        else:
            plan.append(f"Day {i}: Rest")
    
    return "\n".join(plan)

def main():
    print("Welcome to your Gym Workout Chatbot! How can I help you? Please tell us your goal.")
    
    extracted_info = {"experience": None, "goal": None, "equipment": None, "muscle": None, "frequency": None}
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        extracted_info, confirmation = process_input(user_input, extracted_info)
        if confirmation:
            print(confirmation)
            user_input = input("You: ").strip()
            if user_input.lower() in ["yes", "home"]:
                extracted_info["equipment"] = "body weight"
            else:
                print("Okay, sticking with gym or equipment.")
            continue
        
        if not extracted_info["experience"]:
            print("Okay, got it! Are you a beginner, advanced, or somewhere else in your fitness level?")
            continue
        if not extracted_info["goal"]:
            print("Okay, got it! Could you specify your goal (e.g., weight loss, muscle gain, get stronger)?")
            continue
        if not extracted_info["muscle"] or extracted_info["muscle"] not in BODY_PART_MAP.values():
            print("Okay, got it! Which muscle group do you want to focus on (e.g., arms, legs, whole body)?")
            continue
        if not extracted_info["frequency"]:
            print("Okay, got it! How many days per week can you work out (e.g., 3 days a week)?")
            continue
        if not extracted_info["equipment"]:
            print("Okay, got it! Will you be working out at home, the gym, or using equipment like dumbbells?")
            continue
        
        print("Okay, got it!")
        filters, experience, frequency = map_to_api_filters(extracted_info)
        print(f"Here's your detailed workout plan:")
        plan = generate_weekly_plan(extracted_info, filters, experience, frequency)
        print(plan)
        print("\nWant another plan? Just tell me your goal, or type 'exit' to quit.")
        extracted_info = {"experience": None, "goal": None, "equipment": None, "muscle": None, "frequency": None}
    
    print("\nTesting NER model on sample inputs...")
    test_inputs = [
        "i’m new and want to tone my arms at home.",
        "i’ve never exercised, want my legs to look better.",
        "i’m new to the gym, want to get stronger.",
        "i’m advanced, want muscle gain in my chest at the gym 5 days a week.",
        "beginner here, want weight loss, no equipment, 4 days a week."
    ]
    for text in test_inputs:
        doc = nlp(text.lower().replace("at the gym", "gym").replace("the gym", "gym"))
        print(f"Input: {text}")
        print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

if __name__ == "__main__":
    main()