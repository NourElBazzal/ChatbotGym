import json
import random
import spacy

# Load spaCy model
nlp = spacy.load("./gym_ner_model_v5")

# Load exercises database
def load_exercise_data():
    with open("exercices.json", "r") as f:
        data = json.load(f)
    print(f"Loaded data from exercices.json: {type(data)} with {len(data)} top-level items/keys.")
    body_parts = sorted(set(ex["bodyParts"] for ex in data))
    equipment = sorted(set(item for ex in data for item in ex["equipments"]))
    print(f"Available body parts: {body_parts}")
    print(f"Available equipment: {equipment}")
    return data

exercises_db = load_exercise_data()

def process_input(user_input, extracted_info):
    doc = nlp(user_input.lower())
    print(f"Processed text: {doc.text}")
    print(f"Tokens: {[token.text for token in doc]}")
    
    entities = [(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]
    print(f"Raw entities: {entities}")

    confirmation = None

    # Extract experience level
    for ent_text, ent_label, _ in entities:
        if ent_label == "EXPERIENCE":
            if ent_text in ["beginner", "intermediate", "advanced"]:
                extracted_info["experience"] = ent_text
            break

    # Extract goal
    for ent_text, ent_label, _ in entities:
        if ent_label == "GOAL":
            goal_mapping = {
                "tone": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "build muscle": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
                "get in shape": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"]
            }
            extracted_info["goal_targets"] = goal_mapping.get(ent_text, [])
            extracted_info["goal_raw"] = ent_text
            break

    # Extract body part (muscle focus)
    body_part_mapping = {
        "legs": "upper legs",
        "arms": "upper arms",
        "back": "back",
        "chest": "chest",
        "shoulders": "shoulders",
        "glutes": "glutes",
        "waist": "waist",
        "lower legs": "lower legs"
    }
    muscle = None
    for ent_text, ent_label, _ in entities:
        if ent_label == "BODY_PART":
            muscle = body_part_mapping.get(ent_text, ent_text)
            break

    # If no specific body part is mentioned and goal is general (e.g., "get in shape"), set to full-body
    if not muscle and extracted_info.get("goal_raw") in ["get in shape"]:
        muscle = ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"]
    elif muscle:
        extracted_info["muscle"] = muscle
    else:
        extracted_info["muscle"] = "waist"  # Fallback, but should rarely be used now

    # Extract equipment
    equipment_mapping = {
        "dumbbells": "dumbbell",
        "barbell": "barbell",
        "kettlebell": "kettlebell",
        "resistance band": "resistance band",
        "body weight": "body weight",
        "home": "body weight"
    }
    for ent_text, ent_label, _ in entities:
        if ent_label == "EQUIPMENT":
            if ent_text == "home":
                confirmation = "I see you mentioned 'at home'. Do you want a workout plan using only body weight (no equipment)?"
            else:
                equipment = equipment_mapping.get(ent_text, ent_text)
                extracted_info["equipment"] = equipment
            break

    # Extract frequency
    frequency = extract_frequency(entities)
    if frequency:
        extracted_info["frequency"] = frequency

    print(f"Extracted info after processing: {extracted_info}")
    return extracted_info, confirmation

def extract_frequency(entities):
    for ent_text, ent_label, _ in entities:
        if ent_label == "FREQUENCY":
            # Handle patterns like "X days a week"
            if "days a week" in ent_text:
                try:
                    num_days = int(ent_text.split()[0])
                    return num_days
                except (ValueError, IndexError):
                    print(f"Could not map frequency text: {ent_text}")
                    return None
            # Add more frequency patterns if needed
    return None

def get_workout(filters, day, used_exercises):
    focus = filters["focus"]
    equipment = filters["equipment"]
    experience = filters["experience"]

    # Load exercises for each day to ensure fresh data
    exercises = load_exercise_data()

    # Convert focus to list if it's a single string
    focus_list = focus if isinstance(focus, list) else [focus]

    # Filter exercises
    matching_exercises = []
    for ex in exercises:
        # Check if the exercise targets the focus body part
        if ex["bodyParts"] in focus_list:
            # Check equipment match (since equipments is a list, check if the requested equipment is in the list)
            if equipment and equipment not in ex["equipments"]:
                continue
            # Filter by difficulty based on experience level
            if experience == "beginner" and ex.get("difficulty") == "advanced":
                continue
            if experience == "advanced" and ex.get("difficulty") == "beginner":
                continue
            matching_exercises.append(ex)

    print(f"Filtering for focus: {focus}, required equipment: {equipment}")
    print(f"Filtered to {len(matching_exercises)} exercises matching criteria.")

    # Select up to 5 exercises, avoiding over-repetition within the same day
    selected_exercises = []
    day_used_exercises = set()
    random.shuffle(matching_exercises)

    for ex in matching_exercises:
        ex_name = ex["name"]
        # Allow some reuse across days, but not within the same day
        if ex_name not in day_used_exercises and len(selected_exercises) < 5:
            selected_exercises.append(ex)
            day_used_exercises.add(ex_name)

    return selected_exercises, day_used_exercises

def generate_workout_plan(extracted_info):
    if extracted_info.get("frequency") is None:
        raise ValueError("Frequency is not set in extracted_info")

    frequency = extracted_info["frequency"]
    experience = extracted_info["experience"]
    equipment = extracted_info["equipment"]
    muscle = extracted_info["muscle"]
    goal = extracted_info["goal_raw"]

    print("-" * 30)
    print(f"Experience: {experience.title()}")
    print(f"Goal: {goal}")
    print(f"Focus: {muscle.title() if isinstance(muscle, str) else 'Full Body'}")
    print(f"Equipment: {equipment if equipment else 'various (gym)'}")
    print(f"Frequency: {frequency} days/week")
    print("-" * 30)

    workout_days = min(frequency, 7)
    workout_schedule = []

    for day in range(1, 8):
        if len(workout_schedule) < workout_days:
            workout_schedule.append(day)
        else:
            workout_schedule.append(None)

    random.shuffle(workout_schedule)

    plan = []
    filters = {
        "focus": muscle,
        "equipment": equipment,
        "experience": experience
    }
    used_exercises = set()  # Track exercises across all days, but allow reuse with modification

    # Add warm-up and cool-down templates
    warm_up = "Warm-Up: 5 minutes of Jumping Jacks\n  - Instructions:\n    1. Stand with feet together, arms at sides.\n    2. Jump while raising arms overhead and spreading legs.\n    3. Jump back to starting position.\n    4. Repeat at a moderate pace."
    cool_down = "Cool-Down: 5 minutes of Static Stretching\n  - Instructions:\n    1. Hamstring Stretch: Sit with one leg extended, reach towards toes, hold 30s per side.\n    2. Quad Stretch: Stand, pull one foot to glutes, hold 30s per side.\n    3. Arm Stretch: Cross one arm over body, pull with opposite hand, hold 30s per side."

    for day in range(1, 8):
        print(f"Generating workout for Day {day}...")
        if day in workout_schedule:
            # Reset used_exercises each day to allow reuse, or limit tracking to avoid overuse
            day_used_exercises = set()  # Track only within the day
            exercises, updated_used_exercises = get_workout(filters, day, used_exercises)
            workout = f"Day {day}: Workout\n"
            workout += f"  {warm_up}\n"
            if not exercises:
                workout += "  - No suitable exercises found for this day.\n"
            else:
                for ex in exercises:
                    workout += f"  - {ex['name'].title()}: 3 sets of 8-12 reps\n"  # Hardcoding sets/reps since not in JSON
                    workout += "    Instructions:\n"
                    for idx, step in enumerate(ex['instructions'], 1):
                        workout += f"      {idx}. {step}\n"
            workout += f"  {cool_down}\n"
            plan.append(workout)
            used_exercises.update(day_used_exercises)  # Update global tracking, but less restrictive
        else:
            plan.append(f"Day {day}: Rest")

    # Add progression tip
    plan.append("\nProgression Tip: Increase reps by 1-2 each week as you get stronger.")
    return "\n".join(plan)

def chatbot():
    print("Welcome to the Workout Chatbot!")
    print("Tell me about your fitness goals, experience level, and any equipment you have.")
    print("For example: 'I’m a beginner and want to tone my legs at home 3 days a week.'")
    print("Type 'exit' to quit.\n")

    extracted_info = {}
    confirmation_pending = None
    user_response = None

    while True:
        if confirmation_pending:
            print(confirmation_pending)
            user_response = input("Please confirm (yes/no): ").strip().lower()
            if user_response == "yes":
                extracted_info["equipment"] = "body weight"
                confirmation_pending = None
            elif user_response == "no":
                confirmation_pending = None
            else:
                print("Please respond with 'yes' or 'no'.")
                continue

        if not confirmation_pending:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            extracted_info, confirmation = process_input(user_input, extracted_info)

            if confirmation:
                confirmation_pending = confirmation
                continue

            print("Bot: Okay, got all the info! Generating your plan...")
            try:
                workout_plan = generate_workout_plan(extracted_info)
                print(workout_plan)
            except Exception as e:
                print(f"Bot: Sorry, I encountered an error: {e}")
                print("Please try again with a different input.")

if __name__ == "__main__":
    chatbot()