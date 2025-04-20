import json
import random
import spacy

# Load spaCy model
nlp = spacy.load("./gym_ner_model_v6")

# Define mapping dictionaries
BODY_PART_MAP = {
    "legs": "upper legs",
    "arms": "upper arms",
    "back": "back",
    "chest": "chest",
    "shoulders": "shoulders",
    "glutes": "glutes",
    "waist": "waist",
    "lower legs": "lower legs",
    "upper arms": "upper arms",
    "upper legs": "upper legs",
    "lower arms": "lower arms",
    "cardio": "cardio",
    "stomach": "waist",
    "whole body": "full body",
    "tummy": "waist",
    "core": "waist",
    "biceps": "upper arms",
    "triceps": "upper arms",
    "abs": "waist",
    "abdominals": "waist",
    "full body": "full body",
    "upper body": "upper body",
    "lower body": "lower body",
    "quads": "upper legs",
    "hamstrings": "upper legs",
    "calves": "lower legs",
    "lats": "back",
    "traps": "back",
    "belly": "waist",
    "thighs": "upper legs",
    "pecs": "chest",
    "delts": "shoulders"
}

EQUIPMENT_MAP = {
    "dumbbells": "dumbbell",
    "dumbbell": "dumbbell",
    "barbell": "barbell",
    "barbels": "barbell",
    "kettlebell": "kettlebell",
    "resistance band": "resistance band",
    "resistance bands": "resistance band",
    "bands": "resistance band",
    "resistance band loop": "resistance band",
    "body weight": "body weight",
    "bodyweight": "body weight",
    "home": "body weight",
    "no equipment": "body weight",
    "none": "body weight",
    "nothing": "body weight",
    "no gear": "body weight",
    "gym": None,
    "at the gym": None,
    "at_the_gym": None,
    "machine": None,
    "smith machine": None,
    "cables": None,
    "cable machine": None,
    "free weights": None,
    "pull-up bar": "pull-up bar",
    "pull-up bar with neutral grips": "pull-up bar",
    "bench": "bench",
    "incline bench": "bench",
    "decline bench": "bench",
    "dip station": "dip station",
    "chest press machine": "chest press machine",
    "pec deck machine": "pec deck machine",
    "squat rack": "squat rack",
    "leg press machine": "leg press machine",
    "lying leg curl machine": "lying leg curl machine",
    "seated leg curl machine": "seated leg curl machine",
    "glute kickback machine": "glute kickback machine",
    "step or platform": "step or platform",
    "standing calf raise machine": "standing calf raise machine",
    "seated calf raise machine": "seated calf raise machine",
    "assisted pull-up machine": "assisted pull-up machine",
    "lat pulldown machine": "lat pulldown machine",
    "landmine attachment": "landmine attachment",
    "handle": "handle",
    "rope attachment": "rope attachment",
    "bar pad": "bar pad",
    "hyperextension bench": "hyperextension bench",
    "stability ball": "stability ball",
    "wall": "wall",
    "gymnastic rings": "gymnastic rings",
    "sissy squat machine": "sissy squat machine",
    "glute ham raise machine": "glute ham raise machine",
    "plate": "plate",
    "weight plate": "plate",
    "shoulder press machine": "shoulder press machine",
    "shoulder press": "shoulder press machine",
    "chest supported row machine": "chest supported row machine",
    "reverse fly machine": "reverse fly machine",
    "preacher bench": "preacher bench",
    "ez bar": "EZ bar",
    "power rack": "power rack",
    "box": "box",
    "donkey calf raise machine": "donkey calf raise machine",
    "weight block": "weight block",
    "partner": "partner",
    "wrist roller": "wrist roller"
}

# Load exercises database
def load_exercise_data():
    try:
        with open("exercices.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("exercices.json not found. Please ensure the file exists in the working directory.")
    except json.JSONDecodeError:
        raise ValueError("exercices.json is malformed. Please ensure it contains valid JSON.")
    return data

exercises_db = load_exercise_data()

def process_input(user_input, extracted_info):
    # Preprocess user input to match train_ner.py preprocessing
    processed_text = user_input.lower()
    processed_text = processed_text.replace("dumbbells", "dumbbell").replace("barbells", "barbell")
    processed_text = processed_text.replace("muscle gain", "muscle_gain")
    processed_text = processed_text.replace("at the gym", "at_the_gym")
    processed_text = processed_text.replace("ez bar", "EZ bar").replace("pullup bar", "pull-up bar")
    processed_text = processed_text.replace("resistance band loops", "resistance band loop")
    processed_text = processed_text.replace("weight plates", "weight plate")
    processed_text = processed_text.replace("i’m", "i'm")
    doc = nlp(processed_text)

    # Extract entities
    entities = [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]

    confirmation = None
    equipment_extracted = False
    goal_extracted = False

    # Corrected entities list to fix potential misclassifications
    corrected_entities = []
    for ent_text, ent_label, span in entities:
        ent_text = ent_text.replace("at_the_gym", "at the gym")
        if ent_label == "EXPERIENCE" and ent_text in EQUIPMENT_MAP:
            ent_label = "EQUIPMENT"
        elif ent_label == "EQUIPMENT" and ent_text in ["beginner", "intermediate", "advanced", "new"]:
            ent_label = "EXPERIENCE"
        corrected_entities.append((ent_text, ent_label, span))

    # Extract experience level
    for ent_text, ent_label, _ in corrected_entities:
        if ent_label == "EXPERIENCE":
            extracted_info["experience"] = "beginner" if ent_text == "new" else ent_text
            break
    if "experience" not in extracted_info:
        user_input_lower = user_input.lower()
        for exp in ["beginner", "intermediate", "advanced", "new"]:
            if exp in user_input_lower:
                extracted_info["experience"] = "beginner" if exp == "new" else exp
                break

    # Extract goal (with fallback if spaCy misses it)
    goal_mapping = {
        "tone": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
        "build muscle": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
        "get in shape": ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"],
        "muscle gain": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
        "muscle_gain": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
        "work": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
        "train": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"]
    }
    for ent_text, ent_label, _ in corrected_entities:
        if ent_label == "GOAL":
            ent_text_cleaned = ent_text.replace("muscle_gain", "muscle gain")
            if ent_text_cleaned in goal_mapping:
                extracted_info["goal_targets"] = goal_mapping[ent_text_cleaned]
                extracted_info["goal_raw"] = ent_text_cleaned
                goal_extracted = True
                break
            for goal in goal_mapping.keys():
                if goal in ent_text_cleaned:
                    extracted_info["goal_targets"] = goal_mapping[goal]
                    extracted_info["goal_raw"] = goal
                    goal_extracted = True
                    break
        if goal_extracted:
            break
    if not goal_extracted:
        user_input_lower = user_input.lower()
        for goal in goal_mapping.keys():
            if goal in user_input_lower:
                extracted_info["goal_targets"] = goal_mapping[goal]
                extracted_info["goal_raw"] = goal
                goal_extracted = True
                break

    # Extract body part (muscle focus)
    muscle = None
    for ent_text, ent_label, _ in corrected_entities:
        if ent_label == "BODY_PART":
            muscle = BODY_PART_MAP.get(ent_text, ent_text)
            extracted_info["muscle"] = muscle
            break
    if not muscle:
        user_input_lower = user_input.lower()
        for bp, mapped_bp in BODY_PART_MAP.items():
            if bp in user_input_lower:
                extracted_info["muscle"] = mapped_bp
                break

    if "muscle" not in extracted_info and extracted_info.get("goal_raw") in ["get in shape"]:
        extracted_info["muscle"] = ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"]

    if "muscle" not in extracted_info:
        extracted_info["muscle"] = "waist"

    # Extract equipment (with fallback)
    for ent_text, ent_label, _ in corrected_entities:
        if ent_label == "EQUIPMENT":
            ent_text = ent_text.replace("at the gym", "gym")
            mapped_equip = EQUIPMENT_MAP.get(ent_text)
            equipment_extracted = True
            if ent_text in ["home", "no equipment", "none", "nothing", "no gear", "bodyweight"] and mapped_equip == "body weight":
                if extracted_info.get("equipment") not in [None, "body weight"]:
                    confirmation = f"You mentioned '{ent_text}' (body weight) but previously specified other equipment ('{extracted_info['equipment']}'). Use body weight only? 🤔"
                else:
                    confirmation = f"I see you mentioned '{ent_text}'. Do you want a workout plan using only body weight (no equipment)? 🤔"
            elif ent_text in ["gym", "at_the_gym"] or mapped_equip is None:
                extracted_info["equipment"] = None
            else:
                extracted_info["equipment"] = mapped_equip
            break
    if not equipment_extracted and not confirmation:
        user_input_lower = user_input.lower()
        for eq, mapped_eq in EQUIPMENT_MAP.items():
            if eq in user_input_lower:
                if eq in ["home", "no equipment", "none", "nothing", "no gear", "bodyweight"] and mapped_eq == "body weight":
                    if extracted_info.get("equipment") not in [None, "body weight"]:
                        confirmation = f"You mentioned '{eq}' (body weight) but previously specified other equipment ('{extracted_info['equipment']}'). Use body weight only? 🤔"
                    else:
                        confirmation = f"I see you mentioned '{eq}'. Do you want a workout plan using only body weight (no equipment)? 🤔"
                elif eq in ["gym", "at_the_gym"] or mapped_eq is None:
                    extracted_info["equipment"] = None
                else:
                    extracted_info["equipment"] = mapped_eq
                break

    # Extract frequency
    frequency = extract_frequency(corrected_entities)
    if frequency:
        extracted_info["frequency"] = frequency

    return extracted_info, confirmation

def extract_frequency(entities):
    for ent_text, ent_label, _ in entities:
        if ent_label == "FREQUENCY":
            ent_text = ent_text.replace("at_the_gym ", "")
            if "days a week" in ent_text or "days per week" in ent_text:
                try:
                    if "days a week" in ent_text:
                        num_days = int(ent_text.split("days a week")[0].strip())
                    else:
                        num_days = int(ent_text.split("days per week")[0].strip())
                    return num_days
                except (ValueError, IndexError):
                    return None
    return None

def get_workout(filters, day, used_exercises):
    focus = filters["focus"]
    equipment = filters["equipment"]
    experience = filters["experience"]

    exercises = load_exercise_data()
    focus_list = focus if isinstance(focus, list) else [focus]

    # Step 1: Find exercises matching the specified equipment and focus
    matching_exercises = []
    for ex in exercises:
        if any(part in focus_list for part in ex["bodyParts"]):
            if equipment and equipment not in ex["equipments"]:
                continue
            if experience == "beginner" and ex.get("difficulty") == "advanced":
                continue
            if experience == "advanced" and ex.get("difficulty") == "beginner":
                continue
            matching_exercises.append(ex)

    # Step 2: Select up to 5 exercises with the specified equipment
    selected_exercises = []
    day_used_exercises = set()
    matching_exercises.sort(key=lambda ex: used_exercises.get(ex["name"], 0))
    random.shuffle(matching_exercises)

    for ex in matching_exercises:
        ex_name = ex["name"]
        if ex_name not in day_used_exercises and len(selected_exercises) < 5:
            if used_exercises.get(ex_name, 0) < 2:
                selected_exercises.append(ex)
                day_used_exercises.add(ex_name)
                used_exercises[ex_name] = used_exercises.get(ex_name, 0) + 1

    # Step 3: If fewer than 5 exercises, supplement with body weight exercises
    if len(selected_exercises) < 5 and equipment is not None:  # Only supplement if equipment was specified
        body_weight_exercises = []
        for ex in exercises:
            if any(part in focus_list for part in ex["bodyParts"]):
                if "body weight" not in ex["equipments"]:
                    continue
                if experience == "beginner" and ex.get("difficulty") == "advanced":
                    continue
                if experience == "advanced" and ex.get("difficulty") == "beginner":
                    continue
                if ex["name"] in day_used_exercises:  # Avoid duplicates
                    continue
                body_weight_exercises.append(ex)

        # Sort and shuffle body weight exercises
        body_weight_exercises.sort(key=lambda ex: used_exercises.get(ex["name"], 0))
        random.shuffle(body_weight_exercises)

        # Add body weight exercises to reach the target of 5
        for ex in body_weight_exercises:
            ex_name = ex["name"]
            if ex_name not in day_used_exercises and len(selected_exercises) < 5:
                if used_exercises.get(ex_name, 0) < 2:
                    selected_exercises.append(ex)
                    day_used_exercises.add(ex_name)
                    used_exercises[ex_name] = used_exercises.get(ex_name, 0) + 1

    return selected_exercises, day_used_exercises

def generate_workout_plan(extracted_info):
    if extracted_info.get("frequency") is None:
        raise ValueError("Frequency is not set in extracted_info")
    if extracted_info.get("goal_raw") is None:
        raise ValueError("Goal is not set in extracted_info")
    if extracted_info.get("equipment") is None and "equipment" not in extracted_info:
        raise KeyError("'equipment'")
    if extracted_info.get("experience") is None:
        raise ValueError("Experience is not set in extracted_info")

    frequency = extracted_info["frequency"]
    experience = extracted_info["experience"]
    equipment = extracted_info["equipment"]
    muscle = extracted_info["muscle"]
    goal = extracted_info["goal_raw"]

    workout_days = min(frequency, 7)
    workout_schedule = [None] * 7  # Initialize a 7-day schedule with None (rest days)

    if workout_days > 0:
        step = 2
        start_day = random.choice([1, 2])
        training_days = []
        current_day = start_day

        for _ in range(workout_days):
            while current_day in training_days or current_day > 7:
                current_day = (current_day % 7) + 1
            training_days.append(current_day)
            current_day += step

        training_days.sort()
        for day in training_days:
            workout_schedule[day - 1] = day

    plan = []
    filters = {
        "focus": muscle,
        "equipment": equipment,
        "experience": experience
    }
    used_exercises = {}

    warm_up = "Warm-Up: 5 minutes of Jumping Jacks 🏃\n  - Instructions:\n    1. Stand with feet together, arms at sides.\n    2. Jump while raising arms overhead and spreading legs.\n    3. Jump back to starting position.\n    4. Repeat at a moderate pace."
    cool_down = "Cool-Down: 5 minutes of Static Stretching 🧘\n  - Instructions:\n    1. Hamstring Stretch: Sit with one leg extended, reach towards toes, hold 30s per side.\n    2. Quad Stretch: Stand, pull one foot to glutes, hold 30s per side.\n    3. Arm Stretch: Cross one arm over body, pull with opposite hand, hold 30s per side."

    for day in range(1, 8):
        if workout_schedule[day - 1] is not None:
            day_used_exercises = set()
            exercises, updated_used_exercises = get_workout(filters, day, used_exercises)
            workout = f"Day {day}: Workout 💪\n"
            workout += f"  {warm_up}\n"

            # Track if body weight exercises were used as a fallback
            used_body_weight_fallback = False
            for ex in exercises:
                if equipment is not None and "body weight" in ex["equipments"] and equipment != "body weight":
                    used_body_weight_fallback = True
                    break

            if not exercises:
                workout += "  - No suitable exercises found for this day, even with body weight fallback. 😔\n"
            else:
                for ex in exercises:
                    workout += f"  - {ex['name'].title()}: 3 sets of 8-12 reps 🏋️\n"
                    workout += "    Instructions:\n"
                    for idx, step in enumerate(ex['instructions'], 1):
                        workout += f"      {idx}. {step}\n"

            # Add a note if body weight exercises were used as a fallback
            if used_body_weight_fallback:
                workout += f"  Note: Not enough exercises were found using '{equipment}'. Added body weight exercises to complete the workout. 🏃\n"

            workout += f"  {cool_down}\n"
            plan.append(workout)
        else:
            plan.append(f"Day {day}: Rest 😴")

    plan.append("\nProgression Tip: Increase reps by 1-2 each week as you get stronger! 📈")
    return "\n".join(plan)

def chatbot():
    print("Welcome to the Workout Chatbot! 🏋️‍♀️")
    print("Let’s get you a workout plan! Tell me about your fitness goals, experience level, and any equipment you have. 🎯")
    print("For example: 'I’m a beginner and want to tone my legs at home 3 days per week.'")
    print("Type 'exit' to quit. 🚪\n")

    extracted_info = {}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Goodbye! Stay fit! 👋")
            break

        extracted_info = {}
        extracted_info, confirmation = process_input(user_input, extracted_info)

        # Handle equipment confirmation for "at home" or similar
        if confirmation:
            print(f"Bot: {confirmation}")
            user_response = input("Please confirm (yes/no): ").strip().lower()
            if user_response == "yes":
                extracted_info["equipment"] = "body weight"
                print("Bot: Got it! Using body weight only. ✅")
            elif user_response == "no":
                print("Bot: Please specify any equipment you have (e.g., dumbbells, resistance band, kettlebell), or confirm you want a body weight workout. 🏋️")
                user_response = input("You: ").strip().lower()
                if user_response == "yes":
                    extracted_info["equipment"] = "body weight"
                    print("Bot: Got it! Using body weight only. ✅")
                else:
                    found_equipment = None
                    for eq, mapped_eq in EQUIPMENT_MAP.items():
                        if eq in user_response:
                            found_equipment = mapped_eq
                            break
                    if found_equipment:
                        extracted_info["equipment"] = found_equipment
                        print(f"Bot: Great, I’ll use {found_equipment} for your workout! ✅")
                    else:
                        print("Bot: I couldn’t recognize the equipment. Defaulting to body weight. 🏃")
                        extracted_info["equipment"] = "body weight"
            else:
                print("Bot: Please specify any equipment you have (e.g., dumbbells, resistance band, kettlebell), or confirm you want a body weight workout. 🏋️")
                user_response = input("You: ").strip().lower()
                if user_response == "yes":
                    extracted_info["equipment"] = "body weight"
                    print("Bot: Got it! Using body weight only. ✅")
                else:
                    found_equipment = None
                    for eq, mapped_eq in EQUIPMENT_MAP.items():
                        if eq in user_response:
                            found_equipment = mapped_eq
                            break
                    if found_equipment:
                        extracted_info["equipment"] = found_equipment
                        print(f"Bot: Great, I’ll use {found_equipment} for your workout! ✅")
                    else:
                        print("Bot: I couldn’t recognize the equipment. Defaulting to body weight. 🏃")
                        extracted_info["equipment"] = "body weight"

        # Prompt for missing entities
        if "experience" not in extracted_info:
            print("Bot: What’s your experience level? (e.g., beginner, intermediate, advanced) ❓")
            user_response = input("You: ").strip()
            user_input_lower = user_response.lower()
            for exp in ["beginner", "intermediate", "advanced", "new"]:
                if exp in user_input_lower:
                    extracted_info["experience"] = "beginner" if exp == "new" else exp
                    break
            if "experience" not in extracted_info:
                print("Bot: I couldn’t understand your experience level. Defaulting to beginner. 👶")
                extracted_info["experience"] = "beginner"

        if "goal_raw" not in extracted_info:
            print("Bot: What’s your fitness goal? (e.g., tone, muscle gain, get in shape) ❓")
            user_response = input("You: ").strip()
            user_input_lower = user_response.lower()
            goal_mapping = {
                "tone": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "build muscle": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "get in shape": ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"],
                "muscle gain": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "muscle_gain": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "work": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "train": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"]
            }
            for goal in goal_mapping.keys():
                if goal in user_input_lower:
                    extracted_info["goal_targets"] = goal_mapping[goal]
                    extracted_info["goal_raw"] = goal
                    break
            if "goal_raw" not in extracted_info:
                print("Bot: I couldn’t understand your goal. Defaulting to 'get in shape'. 🏃")
                extracted_info["goal_raw"] = "get in shape"
                extracted_info["goal_targets"] = goal_mapping["get in shape"]

        if "frequency" not in extracted_info:
            print("Bot: How often do you want to train per week? (e.g., 3 days per week) ❓")
            user_response = input("You: ").strip()
            doc = nlp(user_response.lower())
            entities = [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]
            frequency = extract_frequency(entities)
            if frequency:
                extracted_info["frequency"] = frequency
            else:
                print("Bot: I couldn’t understand the frequency. Defaulting to 3 days per week. 📅")
                extracted_info["frequency"] = 3

        if "equipment" not in extracted_info:
            print("Bot: What equipment do you have access to? (e.g., dumbbells, at home, gym) ❓")
            user_response = input("You: ").strip()
            user_input_lower = user_response.lower()
            for eq, mapped_eq in EQUIPMENT_MAP.items():
                if eq in user_input_lower:
                    if eq in ["home", "no equipment", "none", "nothing", "no gear", "bodyweight"]:
                        print(f"Bot: I see you mentioned '{eq}'. Do you want a workout plan using only body weight (no equipment)? 🤔")
                        confirm_response = input("Please confirm (yes/no): ").strip().lower()
                        if confirm_response == "yes":
                            extracted_info["equipment"] = "body weight"
                            print("Bot: Got it! Using body weight only. ✅")
                        elif confirm_response == "no":
                            print("Bot: Please specify any equipment you have (e.g., dumbbells, resistance band, kettlebell), or confirm you want a body weight workout. 🏋️")
                            user_response = input("You: ").strip().lower()
                            if user_response == "yes":
                                extracted_info["equipment"] = "body weight"
                                print("Bot: Got it! Using body weight only. ✅")
                            else:
                                found_equipment = None
                                for eq2, mapped_eq2 in EQUIPMENT_MAP.items():
                                    if eq2 in user_response:
                                        found_equipment = mapped_eq2
                                        break
                                if found_equipment:
                                    extracted_info["equipment"] = found_equipment
                                    print(f"Bot: Great, I’ll use {found_equipment} for your workout! ✅")
                                else:
                                    print("Bot: I couldn’t recognize the equipment. Defaulting to body weight. 🏃")
                                    extracted_info["equipment"] = "body weight"
                        else:
                            print("Bot: Please specify any equipment you have (e.g., dumbbells, resistance band, kettlebell), or confirm you want a body weight workout. 🏋️")
                            user_response = input("You: ").strip().lower()
                            if user_response == "yes":
                                extracted_info["equipment"] = "body weight"
                                print("Bot: Got it! Using body weight only. ✅")
                            else:
                                found_equipment = None
                                for eq2, mapped_eq2 in EQUIPMENT_MAP.items():
                                    if eq2 in user_response:
                                        found_equipment = mapped_eq2
                                        break
                                if found_equipment:
                                    extracted_info["equipment"] = found_equipment
                                    print(f"Bot: Great, I’ll use {found_equipment} for your workout! ✅")
                                else:
                                    print("Bot: I couldn’t recognize the equipment. Defaulting to body weight. 🏃")
                                    extracted_info["equipment"] = "body weight"
                    else:
                        extracted_info["equipment"] = mapped_eq
                        print(f"Bot: Great, I’ll use {mapped_eq} for your workout! ✅")
                    break
            if "equipment" not in extracted_info:
                print("Bot: I couldn’t understand the equipment. Defaulting to body weight. 🏃")
                extracted_info["equipment"] = "body weight"

        print("Bot: Okay, I’ve got all the info! Generating your plan... 🏋️‍♀️")
        try:
            workout_plan = generate_workout_plan(extracted_info)
            print("Bot: Here’s your workout plan! 🎉")
            print(workout_plan)
            extracted_info = {}
        except Exception as e:
            print(f"Bot: Sorry, I encountered an error: {e} 😔")
            print("Please try again with a different input.")

if __name__ == "__main__":
    chatbot()