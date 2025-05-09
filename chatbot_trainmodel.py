import spacy
from spacy.training import Example
from spacy.scorer import Scorer
import random
import requests
import warnings
import json
import os
import urllib.parse
import re

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
    ("i’m advanced and want muscle gain in my chest at the gym 3 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (17, 28, "GOAL"), (31, 36, "BODY_PART"), (40, 43, "EQUIPMENT"), (44, 57, "FREQUENCY")]}),
    ("i want to lose weight by working my whole body at home 4 days per week.", {"entities": [(5, 15, "GOAL"), (26, 36, "BODY_PART"), (40, 44, "EQUIPMENT"), (45, 59, "FREQUENCY")]}),
    ("advanced lifter here, aiming for weight loss with dumbbells 5 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (20, 30, "GOAL"), (36, 45, "EQUIPMENT"), (46, 60, "FREQUENCY")]}),
    ("i don’t know what to do but i want my arms to get stronger.", {"entities": [(26, 30, "BODY_PART"), (34, 45, "GOAL")]}),
    ("can you suggest something for my legs?", {"entities": [(19, 23, "BODY_PART")]}),
    ("i’m advanced and want muscle gain in my arms at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (17, 28, "GOAL"), (31, 35, "BODY_PART"), (39, 42, "EQUIPMENT")]}),
    ("beginner here, want weight loss with no equipment 4 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (14, 24, "GOAL"), (30, 42, "EQUIPMENT"), (43, 57, "FREQUENCY")]}),
    ("i want muscle gain for my chest.", {"entities": [(7, 18, "GOAL"), (22, 27, "BODY_PART")]}),
    ("i’m new and want weight loss at home.", {"entities": [(0, 3, "EXPERIENCE"), (11, 21, "GOAL"), (25, 29, "EQUIPMENT")]}),
    ("advanced user, aiming for chest muscle gain 3 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (20, 25, "BODY_PART"), (26, 37, "GOAL"), (38, 51, "FREQUENCY")]}),
    ("i want to work my whole body for weight loss.", {"entities": [(12, 22, "BODY_PART"), (27, 37, "GOAL")]}),
    ("i can do 3 days a week for my legs.", {"entities": [(9, 22, "FREQUENCY"), (27, 31, "BODY_PART")]}),
    ("i don’t know what to gain but want my chest stronger.", {"entities": [(30, 35, "BODY_PART"), (36, 44, "GOAL")]}),
    ("i want muscle gain in my Chest.", {"entities": [(7, 18, "GOAL"), (22, 27, "BODY_PART")]}),
    ("my goal is muscle gain at the Gym.", {"entities": [(11, 22, "GOAL"), (26, 29, "EQUIPMENT")]}),
    ("Chest workouts at gym for muscle gain.", {"entities": [(0, 5, "BODY_PART"), (15, 18, "EQUIPMENT"), (23, 34, "GOAL")]}),
    ("i’m advanced, want my chest to grow at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (19, 24, "BODY_PART"), (28, 32, "GOAL"), (36, 39, "EQUIPMENT")]}),
    ("3 days a week for chest muscle gain.", {"entities": [(0, 13, "FREQUENCY"), (18, 23, "BODY_PART"), (24, 35, "GOAL")]}),
    ("i want muscle gain for chest at gym.", {"entities": [(7, 18, "GOAL"), (23, 28, "BODY_PART"), (32, 35, "EQUIPMENT")]}),
    ("muscle gain is my goal at the gym.", {"entities": [(0, 11, "GOAL"), (24, 27, "EQUIPMENT")]}),
    ("i’m advanced, muscle gain in chest at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (14, 25, "GOAL"), (29, 34, "BODY_PART"), (38, 41, "EQUIPMENT")]}),
    ("at the gym for chest workouts.", {"entities": [(0, 6, "EQUIPMENT"), (15, 20, "BODY_PART")]}),
    ("working out at gym to gain muscle.", {"entities": [(15, 18, "EQUIPMENT"), (22, 33, "GOAL")]}),
    ("at gym, i want chest strength.", {"entities": [(0, 6, "EQUIPMENT"), (15, 20, "BODY_PART"), (21, 29, "GOAL")]}),
    ("i have some experience, want to focus on lower body using free weights", {"entities": [(2, 17, "EXPERIENCE"), (34, 44, "BODY_PART"), (51, 63, "EQUIPMENT")]}),
    ("as an absolute beginner, suggest cardio fitness exercises without any gear", {"entities": [(6, 23, "EXPERIENCE"), (33, 47, "GOAL"), (60, 71, "EQUIPMENT")]}),
    ("intermediate level here, need a plan for hypertrophy, mainly shoulders and back", {"entities": [(0, 18, "EXPERIENCE"), (36, 47, "GOAL"), (56, 65, "BODY_PART"), (70, 74, "BODY_PART")]}),
    ("can i get lean working out my full body with resistance bands twice a week?", {"entities": [(7, 15, "GOAL"), (29, 38, "BODY_PART"), (44, 60, "EQUIPMENT"), (61, 73, "FREQUENCY")]}),
    ("pro lifter needs a 5 day routine for functional strength, emphasize glutes", {"entities": [(0, 10, "EXPERIENCE"), (18, 31, "FREQUENCY"), (36, 54, "GOAL"), (66, 72, "BODY_PART")]}),
    ("help me burn calories using machines at the gym, i'm starting out", {"entities": [(8, 21, "GOAL"), (28, 36, "EQUIPMENT"), (40, 43, "EQUIPMENT"), (48, 61, "EXPERIENCE")]}),
    ("i want definition for my abs and chest, using cables mainly", {"entities": [(7, 17, "GOAL"), (25, 28, "BODY_PART"), (33, 38, "BODY_PART"), (46, 52, "EQUIPMENT")]}),
    ("just need a simple workout for upper body, i have dumbbells at home", {"entities": [(20, 30, "BODY_PART"), (40, 49, "EQUIPMENT"), (53, 57, "EQUIPMENT")]}),
    ("not sure where to start, maybe bodyweight exercises for legs?", {"entities": [(0, 22, "EXPERIENCE"), (31, 41, "EQUIPMENT"), (51, 55, "BODY_PART")]}),
    ("increase stamina with exercises for my core and lower legs", {"entities": [(0, 16, "GOAL"), (36, 40, "BODY_PART"), (45, 55, "BODY_PART")]}),
    ("i am seasoned and want to maintain muscle using barbels 3 days a week", {"entities": [(5, 13, "EXPERIENCE"), (28, 43, "GOAL"), (50, 57, "EQUIPMENT"), (58, 71, "FREQUENCY")]}),
    ("show me body weight exercises for the full body, please", {"entities": [(8, 20, "EQUIPMENT"), (36, 45, "BODY_PART")]}),
    ("what can i do for my thighs and glutes with kettlebells?", {"entities": [(21, 27, "BODY_PART"), (32, 38, "BODY_PART"), (44, 54, "EQUIPMENT")]}),
    ("regular lifter looking for a shoulder workout", {"entities": [(0, 14, "EXPERIENCE"), (28, 36, "BODY_PART")]}),
    ("need to lose fat around my belly area, i exercise every day", {"entities": [(5, 13, "GOAL"), (24, 29, "BODY_PART"), (43, 53, "FREQUENCY")]}),
    ("advanced workout for biceps and triceps using the smith machine", {"entities": [(0, 8, "EXPERIENCE"), (20, 26, "BODY_PART"), (31, 38, "BODY_PART"), (45, 58, "EQUIPMENT")]}),
    ("any exercises for calves using no gear?", {"entities": [(18, 24, "BODY_PART"), (31, 38, "EQUIPMENT")]}),
    ("i want to sculpt my back and traps", {"entities": [(10, 16, "GOAL"), (20, 24, "BODY_PART"), (29, 34, "BODY_PART")]}),
    ("is it possible to build muscle with resistance bands?", {"entities": [(19, 31, "GOAL"), (37, 53, "EQUIPMENT")]}),
    ("give me a plan for abdominals, 4 days per week at the gym", {"entities": [(19, 29, "BODY_PART"), (31, 45, "FREQUENCY"), (49, 52, "EQUIPMENT")]}),
    ("My primary goal is cardio fitness, I'm intermediate level.", {"entities": [(19, 33, "GOAL"), (38, 56, "EXPERIENCE")]}),
    ("Using only a pull-up bar, what back exercises can a beginner do?", {"entities": [(12, 23, "EQUIPMENT"), (29, 33, "BODY_PART"), (48, 56, "EXPERIENCE")]}),
    ("Need to work my upper body with free weights.", {"entities": [(15, 25, "BODY_PART"), (31, 43, "EQUIPMENT")]}),
    ("Can you help me bulk up? I workout 5 days a week.", {"entities": [(16, 23, "GOAL"), (35, 49, "FREQUENCY")]}),
    ("Focus on lats and shoulders, I'm advanced.", {"entities": [(9, 13, "BODY_PART"), (18, 27, "BODY_PART"), (32, 40, "EXPERIENCE")]}),
    ("i want muscle gain using dumbbells for chest.", {"entities": [(7, 18, "GOAL"), (24, 33, "EQUIPMENT"), (38, 43, "BODY_PART")]}),
    ("advanced lifter, chest muscle gain at gym 5 days.", {"entities": [(0, 8, "EXPERIENCE"), (16, 21, "BODY_PART"), (22, 33, "GOAL"), (37, 40, "EQUIPMENT"), (41, 49, "FREQUENCY")]}),
    ("gym workout for muscle gain on my pecs.", {"entities": [(0, 3, "EQUIPMENT"), (15, 26, "GOAL"), (33, 37, "BODY_PART")]}),
    ("muscle gain in my chest at the gym.", {"entities": [(0, 11, "GOAL"), (15, 20, "BODY_PART"), (24, 27, "EQUIPMENT")]}),
    ("i want muscle gain in my chest at gym 5 days a week.", {"entities": [(7, 18, "GOAL"), (22, 27, "BODY_PART"), (31, 34, "EQUIPMENT"), (35, 48, "FREQUENCY")]}),
    ("i’m advanced, want muscle gain in chest at gym 5 days a week.", {"entities": [(0, 8, "EXPERIENCE"), (14, 25, "GOAL"), (29, 34, "BODY_PART"), (38, 41, "EQUIPMENT"), (42, 55, "FREQUENCY")]}),
    ("at the gym, i want to tone my legs 5 days a week.", {"entities": [(0, 6, "EQUIPMENT"), (18, 22, "GOAL"), (26, 30, "BODY_PART"), (31, 44, "FREQUENCY")]}),
    ("5 days a week at gym for muscle gain in chest.", {"entities": [(0, 13, "FREQUENCY"), (17, 20, "EQUIPMENT"), (25, 36, "GOAL"), (40, 45, "BODY_PART")]}),
    ("i’m advanced and want muscle gain in my chest at the gym.", {"entities": [(0, 8, "EXPERIENCE"), (17, 28, "GOAL"), (32, 37, "BODY_PART"), (41, 44, "EQUIPMENT")]}),
    ("muscle gain in my chest is my goal at gym 5 days a week.", {"entities": [(0, 11, "GOAL"), (15, 20, "BODY_PART"), (34, 37, "EQUIPMENT"), (38, 51, "FREQUENCY")]}),
    ("i want muscle gain in my chest using dumbbells.", {"entities": [(7, 18, "GOAL"), (22, 27, "BODY_PART"), (33, 42, "EQUIPMENT")]}),
]

# Preprocess training data to match runtime preprocessing
TRAIN_DATA = [(text.lower().replace("at the gym", "at_the_gym").replace("dumbbells", "dumbbell").replace("barbells", "barbell").replace("muscle gain", "muscle_gain"), annotations) for text, annotations in TRAIN_DATA]

templates = [
    "i’m {EXPERIENCE} and want to {GOAL} my {BODY_PART} with {EQUIPMENT} {FREQUENCY}.",
    "can you suggest a {EXPERIENCE} workout for my {BODY_PART} to {GOAL} using {EQUIPMENT} {FREQUENCY}?",
    "i want to {GOAL} my {BODY_PART} at {EQUIPMENT} {FREQUENCY}, i’m {EXPERIENCE}.",
    "{EXPERIENCE} here, aiming to {GOAL} my {BODY_PART} with {EQUIPMENT} {FREQUENCY}.",
    "as a {EXPERIENCE} lifter, i want to {GOAL} my {BODY_PART} using {EQUIPMENT} {FREQUENCY}.",
    "i’m {EXPERIENCE}, looking to {GOAL} my {BODY_PART} with {EQUIPMENT}.",
    "i want to {GOAL} my {BODY_PART} {FREQUENCY} at {EQUIPMENT}.",
    "can you suggest something for my {BODY_PART} to {GOAL} {FREQUENCY} for a {EXPERIENCE} user?",
    "{EXPERIENCE} user here, i want my {BODY_PART} to {GOAL} {FREQUENCY}.",
    "i want to {GOAL} {FREQUENCY} with {EQUIPMENT}, i’m {EXPERIENCE}.",
    "i’m {EXPERIENCE} and want to {GOAL} my {BODY_PART}.",
    "can you suggest a workout for my {BODY_PART} to {GOAL}?",
    "i want to {GOAL} {FREQUENCY} at {EQUIPMENT}.",
    "my goal is to {GOAL} my {BODY_PART} {FREQUENCY}.",
    "{EXPERIENCE} lifter, looking for {BODY_PART} exercises with {EQUIPMENT}.",
    "what’s a good {EXPERIENCE} workout for {BODY_PART} to {GOAL}?",
    "i need a plan to {GOAL} my {BODY_PART} using {EQUIPMENT}.",
    "{FREQUENCY} {BODY_PART} workout for {EXPERIENCE} users to {GOAL}.",
    "help me {GOAL} my {BODY_PART} {FREQUENCY}, i’m {EXPERIENCE}.",
    "{FREQUENCY} at {EQUIPMENT}, i’m {EXPERIENCE}, give me a {BODY_PART} workout.",
]

experience_terms = ["new", "beginner", "never", "just starting", "don’t know how", "first time", "starting out", "no experience", "easy", "advanced", "absolute beginner", "not sure where to start", "some experience", "intermediate", "expert", "pro", "seasoned", "regular lifter"]
goal_terms = ["get stronger", "look better", "feel more fit", "tone", "build muscle", "get in shape", "move better", "not feel so tired", "stronger", "not feel so weak", "weight loss", "muscle gain", "lose fat", "burn calories", "increase stamina", "improve endurance", "get lean", "bulk up", "definition", "sculpt", "functional strength", "cardio fitness", "hypertrophy"]
body_part_terms = ["arms", "legs", "stomach", "back", "whole body", "chest", "tummy", "core", "glutes", "shoulders", "biceps", "triceps", "abs", "abdominals", "full body", "upper body", "lower body", "quads", "hamstrings", "calves", "lats", "traps", "belly", "thighs", "pecs", "delts"]
equipment_terms = ["home", "dumbbell", "barbell", "gym", "no equipment", "kettlebell", "none", "bodyweight", "resistance bands", "machine", "smith machine", "cables", "free weights", "nothing", "no gear", "pull-up bar", "bench", "bands", "barbels", "cable machine", "at_the_gym"]
frequency_terms = ["3 days a week", "4 days per week", "5 days a week", "twice a week", "every day", "daily", "3 times a week", "4 times weekly", "5 days weekly", "once a week", "6 days a week"]

def generate_training_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        template = random.choice(templates)
        experience = random.choice(experience_terms)
        goal = random.choice(goal_terms).replace("muscle gain", "muscle_gain")
        body_part = random.choice(body_part_terms)
        equipment = random.choice(equipment_terms)
        frequency = random.choice(frequency_terms)

        if equipment in ["gym", "free weights", "at_the_gym"] and "at {EQUIPMENT}" in template:
            text_segment = "at_the_gym"
            text = template.replace("at {EQUIPMENT}", text_segment).format(
                EXPERIENCE=experience, GOAL=goal, BODY_PART=body_part,
                EQUIPMENT=equipment, FREQUENCY=frequency
            ).lower()
        elif equipment in ["home", "bodyweight", "nothing", "no gear", "no equipment", "none"] and "with {EQUIPMENT}" in template:
            text_segment = random.choice([f"using {equipment}", "at home"])
            text = template.replace("with {EQUIPMENT}", text_segment).format(
                EXPERIENCE=experience, GOAL=goal, BODY_PART=body_part,
                EQUIPMENT=equipment, FREQUENCY=frequency
            ).lower()
        else:
            try:
                text = template.format(
                    EXPERIENCE=experience, GOAL=goal, BODY_PART=body_part,
                    EQUIPMENT=equipment, FREQUENCY=frequency
                ).lower()
            except KeyError:
                print(f"Skipping sample due to template mismatch: {template}")
                continue

        text = text.replace("dumbbells", "dumbbell").replace("barbells", "barbell").replace("muscle gain", "muscle_gain").replace("at the gym", "at_the_gym")

        entities = []
        term_map = [
            (experience, "EXPERIENCE"), (goal, "GOAL"), (body_part, "BODY_PART"),
            (equipment, "EQUIPMENT"), (frequency, "FREQUENCY")
        ]

        for term, label in term_map:
            if term in ["gym", "free weights", "at the gym"] and label == "EQUIPMENT":
                term = "at_the_gym"
            start_idx = text.find(term)
            if start_idx != -1:
                end_idx = start_idx + len(term)
                is_overlapping = False
                for s, e, _ in entities:
                    if (start_idx >= s and end_idx <= e) or \
                       (s >= start_idx and e <= end_idx) or \
                       (max(start_idx, s) < min(end_idx, e)):
                        is_overlapping = True
                        break
                if not is_overlapping:
                    entities.append((start_idx, end_idx, label))

        entities.sort(key=lambda x: x[0])
        valid_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                valid_entities.append((start, end, label))
                last_end = end
            else:
                print(f"Skipping overlapping entity: {(start, end, label)} in text: {text}")

        if valid_entities:
            data.append((text, {"entities": valid_entities}))
        else:
            print(f"Skipping sample with no valid entities: {text}")

        if len(data) >= n_samples:
            break

    print(f"Generated {len(data)} training samples")
    return data if data else []

generated_data = generate_training_data(500)
ALL_TRAIN_DATA = TRAIN_DATA + generated_data
print(f"Total training examples: {len(ALL_TRAIN_DATA)}")

# === NER Model Training ===

def train_ner_model(training_data, output_path="gym_ner_model", n_iter=100, eval_split=0.2, early_stopping_patience=15, dropout=0.3):
    nlp = spacy.blank("en")
    print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    labels = set()
    for _, annotations in training_data:
        for _, _, label in annotations.get("entities", []):
            labels.add(label)

    for label in labels:
        ner.add_label(label)
    print(f"Added labels to NER: {list(labels)}")

    random.shuffle(training_data)
    split_point = int(len(training_data) * (1 - eval_split))
    train_subset = training_data[:split_point]
    eval_subset = training_data[split_point:]

    print(f"Training data size: {len(train_subset)}")
    print(f"Evaluation data size: {len(eval_subset)}")

    train_examples = []
    for text, annots in train_subset:
        try:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            train_examples.append(example)
        except Exception as e:
            print(f"Error creating Example for training: {e}\nText: {text}\nAnnotations: {annots}\nSkipping this example.")

    eval_examples = []
    for text, annots in eval_subset:
        try:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            eval_examples.append(example)
        except Exception as e:
            print(f"Error creating Example for evaluation: {e}\nText: {text}\nAnnotations: {annots}\nSkipping this example.")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    best_score = -1.0
    patience_counter = 0
    best_model_path = None

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        print("Training NER model...")
        print("Iteration | Losses   | Eval P   | Eval R   | Eval F1")
        print("----------|----------|----------|----------|----------")

        for itn in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            valid_train_examples = [ex for ex in train_examples if ex is not None]
            batches = spacy.util.minibatch(valid_train_examples, size=spacy.util.compounding(4.0, 32.0, 1.001))

            for batch in batches:
                if any(ex is None for ex in batch):
                    print(f"Warning: Skipping batch with None Example object at iteration {itn+1}")
                    continue
                try:
                    nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)
                except ValueError as e:
                    print(f"ValueError during update at iteration {itn+1}: {e}")
                    print(f"Problematic batch (first text): {batch[0].text if batch else 'N/A'}")
                    continue
                except Exception as e:
                    print(f"Unexpected error during update at iteration {itn+1}: {e}")
                    continue

            valid_eval_examples = [ex for ex in eval_examples if ex is not None]
            if valid_eval_examples:
                scores = nlp.evaluate(valid_eval_examples)
                f_score = scores.get('ents_f', 0.0)
                precision = scores.get('ents_p', 0.0)
                recall = scores.get('ents_r', 0.0)

                print(
                    f"{itn+1:<9} | {losses.get('ner', 0.0):<8.2f} | {precision:<8.3f} | {recall:<8.3f} | {f_score:<8.3f}"
                )

                if f_score > best_score:
                    best_score = f_score
                    patience_counter = 0
                    best_model_path_temp = os.path.join(output_path, "best_model_temp")
                    if not os.path.exists(best_model_path_temp):
                        os.makedirs(best_model_path_temp)
                    nlp.to_disk(best_model_path_temp)
                    best_model_path = best_model_path_temp
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {itn + 1} iterations.")
                    print(f"Best F1 score achieved: {best_score:.3f}")
                    break
            else:
                print(f"{itn+1:<9} | {losses.get('ner', 0.0):<8.2f} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")

    if best_model_path and os.path.exists(best_model_path):
        print(f"\nLoading best model from iteration with F1: {best_score:.3f}")
        nlp = spacy.load(best_model_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nlp.to_disk(output_path)
    print(f"\nFinal model saved to {output_path}")
    if best_score >= 0:
        print(f"Best evaluated F1 score during training: {best_score:.3f}")

    return nlp

nlp = train_ner_model(ALL_TRAIN_DATA, output_path="gym_ner_model_v5", n_iter=100, early_stopping_patience=15)
print("Model training complete.")

# === Chatbot Implementation ===

API_URL = "https://exercisedb-api.vercel.app/api/v1/exercises?offset={offset}&limit={limit}"

EXPERIENCE_MAP = {
    "new": "beginner", "beginner": "beginner", "never": "beginner",
    "just starting": "beginner", "don’t know how": "beginner", "first time": "beginner",
    "starting out": "beginner", "no experience": "beginner", "easy": "beginner",
    "absolute beginner": "beginner", "not sure where to start": "beginner",
    "intermediate": "intermediate", "some experience": "intermediate",
    "regular lifter": "intermediate",
    "advanced": "advanced", "expert": "advanced", "pro": "advanced",
    "seasoned": "advanced", "pro lifter": "advanced"
}

GOAL_MAP = {
    "get stronger": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "build muscle": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "stronger": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "muscle gain": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "muscle_gain": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "bulk up": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "hypertrophy": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders"],
    "functional strength": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders", "glutes"],
    "maintain muscle": ["upper arms", "upper legs", "waist", "chest", "back", "shoulders", "glutes"],
    "tone": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
    "look better": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
    "get in shape": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs"],
    "feel more fit": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs", "back"],
    "get lean": ["waist", "upper legs", "chest", "back", "shoulders"],
    "definition": ["upper arms", "waist", "chest", "shoulders", "upper legs", "back"],
    "sculpt": ["upper arms", "waist", "chest", "glutes", "shoulders", "upper legs", "back"],
    "weight loss": ["waist", "upper legs", "chest", "back", "glutes"],
    "lose fat": ["waist", "upper legs", "chest", "back", "glutes"],
    "burn calories": ["waist", "upper legs", "chest", "back", "glutes"],
    "cardio fitness": ["waist", "upper legs", "lower legs", "back"],
    "endurance": ["waist", "upper legs", "glutes", "lower legs", "back"],
    "increase stamina": ["waist", "upper legs", "glutes", "lower legs", "back"],
    "improve endurance": ["waist", "upper legs", "glutes", "lower legs", "back"],
    "not feel so tired": ["waist", "upper legs", "glutes", "lower legs", "back"],
    "move better": ["glutes", "upper legs", "back", "core"],
    "not feel so weak": ["upper arms", "waist", "chest", "back", "shoulders"]
}

DEFAULT_GOAL_TARGETS = ["upper arms", "upper legs", "waist", "chest", "back"]

EQUIPMENT_MAP = {
    "home": "body weight", "body weight": "body weight", "bodyweight": "body weight",
    "nothing": "body weight", "no gear": "body weight", "no equipment": "body weight",
    "none": "body weight",
    "dumbbell": "dumbbell", "dumbell": "dumbbell",
    "barbell": "barbell", "barbels": "barbell",
    "kettlebell": "kettlebell", "kettlebells": "kettlebell",
    "resistance bands": "resistance band", "bands": "resistance band",
    "machine": "machine", "machines": "machine",
    "smith machine": "smith machine",
    "cable": "cable", "cables": "cable", "cable machine": "cable",
    "pull-up bar": "pull up bar", "pull up bar": "pull up bar",
    "bench": "bench",
    "gym": None, "free weights": None, "at_the_gym": None,
}

DEFAULT_EQUIPMENT = None

BODY_PART_MAP = {
    "arms": "upper arms", "biceps": "upper arms", "triceps": "upper arms",
    "legs": "upper legs", "quads": "upper legs", "quadriceps": "upper legs", "thighs": "upper legs",
    "hamstrings": "upper legs",
    "calves": "lower legs", "lower legs": "lower legs",
    "stomach": "waist", "tummy": "waist", "abs": "waist", "abdominals": "waist",
    "core": "waist", "belly": "waist", "waist": "waist",
    "back": "back", "lats": "back", "traps": "back",
    "chest": "chest", "pecs": "chest",
    "shoulders": "shoulders", "delts": "shoulders",
    "glutes": "glutes", "butt": "glutes",
    "whole body": ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"],
    "full body": ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"],
    "upper body": ["upper arms", "back", "chest", "shoulders"],
    "lower body": ["upper legs", "lower legs", "glutes"],
}

API_BODY_PART_MAP = {
    "upper arms": "upper arms",
    "upper legs": "upper legs",
    "lower legs": "lower legs",
    "waist": "waist",
    "back": "back",
    "chest": "chest",
    "shoulders": "shoulders",
    "glutes": "glutes",
}

API_TARGET_MUSCLE_MAP = {
    "biceps": "upper arms",
    "triceps": "upper arms",
    "quadriceps": "upper legs",
    "hamstrings": "upper legs",
    "glutes": "glutes",
    "pectorals": "chest",
    "deltoids": "shoulders",
    "abs": "waist",
    "upper back": "back",
    "lats": "back",
    "traps": "back",
    "calves": "lower legs",
}

DEFAULT_BODY_PART = "waist"

FREQUENCY_MAP = {
    "3 days a week": 3, "4 days per week": 4, "5 days a week": 5,
    "twice a week": 2, "every day": 7, "daily": 7,
    "3 times a week": 3, "4 times weekly": 4, "5 times a week": 5,
    "once a week": 1, "6 days a week": 6,
    "5 days weekly": 5
}

DEFAULT_FREQUENCY = 3

def process_input(user_input, extracted_info):
    processed_text = user_input.lower()
    processed_text = processed_text.replace("dumbbell", "dumbbell").replace("barbell", "barbell")
    processed_text = processed_text.replace("muscle gain", "muscle_gain")
    processed_text = processed_text.replace("at the gym", "at_the_gym")
    doc = nlp(processed_text)

    print(f"Processed text: {processed_text}")
    print(f"Tokens: {[token.text for token in doc]}")
    print(f"Raw entities: {[(ent.text, ent.label_, (ent.start_char, ent.end_char)) for ent in doc.ents]}")
    confirmation = None

    equipment_extracted = False
    goal_extracted = False
    for ent in doc.ents:
        ent_text = ent.text
        ent_label = ent.label_

        ent_text = ent_text.replace("at_the_gym", "at the gym")

        if ent_label == "EXPERIENCE":
            extracted_info["experience"] = EXPERIENCE_MAP.get(ent_text, extracted_info.get("experience"))
        elif ent_label == "GOAL":
            ent_text_cleaned = ent_text
            for goal in GOAL_MAP.keys():
                if goal in ent_text:
                    ent_text_cleaned = goal
                    break
            extracted_info["goal_targets"] = GOAL_MAP.get(ent_text_cleaned, extracted_info.get("goal_targets"))
            extracted_info["goal_raw"] = ent_text_cleaned.replace("muscle_gain", "muscle gain")
            goal_extracted = True
        elif ent_label == "EQUIPMENT":
            ent_text = ent_text.replace("at the gym", "gym")
            mapped_equip = EQUIPMENT_MAP.get(ent_text)
            equipment_extracted = True
            if mapped_equip == "body weight" and extracted_info.get("equipment") not in [None, "body weight"]:
                confirmation = f"You mentioned '{ent_text}' (body weight) but previously specified other equipment ('{extracted_info['equipment']}'). Use body weight only?"
            else:
                extracted_info["equipment"] = mapped_equip  # Simply set the equipment
        elif ent_label == "BODY_PART":
            extracted_info["muscle"] = BODY_PART_MAP.get(ent_text, extracted_info.get("muscle"))
        elif ent_label == "FREQUENCY":
            freq_text = ent_text.replace("at_the_gym ", "")
            if freq_text in FREQUENCY_MAP:
                extracted_info["frequency"] = FREQUENCY_MAP[freq_text]
            else:
                print(f"Could not map frequency text: {freq_text}")

    if not equipment_extracted and "at_the_gym" in processed_text:
        extracted_info["equipment"] = EQUIPMENT_MAP.get("gym")

    # Fallback for goal extraction if NER fails
    if not goal_extracted:
        for goal in GOAL_MAP.keys():
            goal_processed = goal.replace("muscle gain", "muscle_gain")
            if goal_processed in processed_text:
                extracted_info["goal_targets"] = GOAL_MAP.get(goal)
                extracted_info["goal_raw"] = goal
                break

    if "goal_targets" not in extracted_info:
        extracted_info["goal_targets"] = DEFAULT_GOAL_TARGETS
        if "goal_raw" not in extracted_info:
            extracted_info["goal_raw"] = "general fitness"

    if "muscle" not in extracted_info or extracted_info["muscle"] is None:
        # If the goal implies a general fitness focus, use full-body
        general_goals = ["get in shape", "feel more fit", "weight loss", "burn calories"]
        if extracted_info["goal_raw"] in general_goals:
            extracted_info["muscle"] = ["upper arms", "upper legs", "lower legs", "waist", "back", "chest", "shoulders", "glutes"]  # Full-body
        elif extracted_info["goal_targets"] != DEFAULT_GOAL_TARGETS and extracted_info["goal_targets"]:
            extracted_info["muscle"] = random.choice(extracted_info["goal_targets"])
        else:
            extracted_info["muscle"] = DEFAULT_BODY_PART

    if "experience" not in extracted_info:
        extracted_info["experience"] = "beginner"
    if "equipment" not in extracted_info:
        extracted_info["equipment"] = DEFAULT_EQUIPMENT
    if "frequency" not in extracted_info or extracted_info["frequency"] is None:
        extracted_info["frequency"] = DEFAULT_FREQUENCY

    print(f"Extracted info after processing: {extracted_info}")
    return extracted_info, confirmation

def validate_equipment(exercise, required_equipment):
    listed_equipments = exercise.get('equipments', [])
    instructions = " ".join(exercise.get('instructions', [])).lower()

    equipment_aliases = {
        "dumbbell": ["dumbbell", "dumbbells"],
        "barbell": ["barbell", "barbells"],
        "kettlebell": ["kettlebell", "kettlebells"],
        "resistance band": ["resistance band", "resistance bands", "bands"],
        "pull up bar": ["pull up bar", "pull-up bar"],
        "body weight": ["body weight", "bodyweight"],
        "machine": ["machine", "machines"],
        "smith machine": "smith machine",
        "cable": ["cable", "cables", "cable machine"],
        "bench": ["bench"]
    }

    matches_listed = False
    if required_equipment is None:
        matches_listed = True
    elif required_equipment == "body weight":
        matches_listed = "body weight" in listed_equipments or "bodyweight" in listed_equipments
    else:
        for equip in equipment_aliases.get(required_equipment, [required_equipment]):
            if equip in listed_equipments:
                matches_listed = True
                break

    matches_instructions = False
    if required_equipment is None:
        matches_instructions = True
    elif required_equipment == "body weight":
        other_equipment_mentioned = False
        for equip, aliases in equipment_aliases.items():
            if equip == "body weight":
                continue
            for alias in aliases:
                if alias in instructions:
                    other_equipment_mentioned = True
                    break
            if other_equipment_mentioned:
                break
        matches_instructions = not other_equipment_mentioned
    else:
        for alias in equipment_aliases.get(required_equipment, [required_equipment]):
            if alias in instructions:
                matches_instructions = True
                break

    return matches_listed or matches_instructions

def get_workout(extracted_info, day, used_exercises):
    focus = extracted_info.get("focus", "chest")
    required_equipment = extracted_info.get("equipment")
    experience_level = extracted_info.get("experience", "beginner")

    # Load exercises from exercices.json
    try:
        with open("exercices.json", "r") as f:
            loaded_data = json.load(f)
        print(f"Loaded data from exercices.json: {type(loaded_data)} with {len(loaded_data)} top-level items/keys.")
    except FileNotFoundError:
        print("Error: exercices.json file not found.")
        return [], used_exercises
    except json.JSONDecodeError:
        print("Error: exercices.json file is not a valid JSON.")
        return [], used_exercises

    # Handle different data formats
    all_exercises = []
    if isinstance(loaded_data, dict):
        possible_keys = ["exercises", "data", "workouts"]
        for key in possible_keys:
            if key in loaded_data and isinstance(loaded_data[key], list):
                all_exercises = loaded_data[key]
                break
    elif isinstance(loaded_data, list):
        all_exercises = loaded_data
    else:
        print(f"Unexpected data format in exercices.json: {type(loaded_data)}")
        return [], used_exercises

    # Process exercises
    processed_exercises = []
    for item in all_exercises:
        if isinstance(item, str):
            try:
                parsed_item = json.loads(item)
                processed_exercises.append(parsed_item)
            except json.JSONDecodeError:
                continue
        else:
            processed_exercises.append(item)
    all_exercises = processed_exercises

    # Validate exercises
    required_fields = ["name", "bodyParts", "equipments", "instructions"]
    validated_exercises = []
    for ex in all_exercises:
        if not isinstance(ex, dict):
            continue
        has_all_fields = all(field in ex for field in required_fields)
        instructions_is_list = isinstance(ex.get("instructions", []), list)
        if has_all_fields and instructions_is_list:
            ex["bodyPart"] = ex["bodyParts"][0] if ex["bodyParts"] else "unknown"
            validated_exercises.append(ex)
    all_exercises = validated_exercises

    # Debug available body parts and equipment
    body_parts = sorted(set(ex["bodyPart"] for ex in all_exercises))
    equipments = sorted(set(equip for ex in all_exercises for equip in ex["equipments"]))
    print(f"Available body parts: {body_parts}")
    print(f"Available equipment: {equipments}")

    # Debug the focus and equipment being used for filtering
    print(f"Filtering for focus: {focus}, required equipment: {required_equipment}")

    # Handle full-body focus (if focus is a list)
    if isinstance(focus, list):
        filtered_exercises = [
            ex for ex in all_exercises
            if ex["bodyPart"] in focus and validate_equipment(ex, required_equipment)
        ]
        # Distribute exercises across body parts for variety
        exercises_by_body_part = {bp: [] for bp in focus}
        for ex in filtered_exercises:
            exercises_by_body_part[ex["bodyPart"]].append(ex)
        # Select one exercise per body part for variety, if possible
        selected_exercises = []
        day_used_exercises = set()
        for bp in exercises_by_body_part:
            available_ex = [ex for ex in exercises_by_body_part[bp] if ex["name"] not in used_exercises and ex["name"] not in day_used_exercises]
            if available_ex:
                ex = random.choice(available_ex)
                selected_exercises.append(ex)
                day_used_exercises.add(ex["name"])
    else:
        # Original single-focus filtering
        filtered_exercises = [
            ex for ex in all_exercises
            if ex["bodyPart"] == focus and validate_equipment(ex, required_equipment)
        ]
        print(f"Filtered to {len(filtered_exercises)} exercises matching criteria.")
        if not filtered_exercises and all_exercises:
            print("Sample exercise for debugging:", all_exercises[0])

        # Select exercises for the day
        day_used_exercises = set()
        selected_exercises = []
        num_exercises = 5
        for ex in filtered_exercises:
            if ex["name"] not in used_exercises and ex["name"] not in day_used_exercises:
                selected_exercises.append(ex)
                day_used_exercises.add(ex["name"])
                if len(selected_exercises) >= num_exercises:
                    break

    # Generate workout
    if not selected_exercises:
        print(f"No suitable exercises found for day {day}.")
        return [], used_exercises

    sets = 4 if experience_level == "advanced" else 3
    reps = "6-10" if experience_level == "advanced" else "8-12"
    workout = []
    for ex in selected_exercises:
        workout.append({
            "name": ex["name"],
            "sets": sets,
            "reps": reps,
            "instructions": ex["instructions"]
        })
        used_exercises.add(ex["name"])

    return workout, used_exercises

# (Previous sections remain as provided in the last artifact)

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

    for day in range(1, 8):
        print(f"Generating workout for Day {day}...")
        if day in workout_schedule:
            # Reset used_exercises each day to allow reuse, or limit tracking to avoid overuse
            day_used_exercises = set()  # Track only within the day
            exercises, updated_used_exercises = get_workout(filters, day, used_exercises)
            workout = f"Day {day}: Workout\n"
            if not exercises:
                workout += "  - No suitable exercises found for this day."
            else:
                for ex in exercises:
                    workout += f"  - {ex['name'].title()}: {ex['sets']} sets of {ex['reps']} reps\n"
                    workout += "    Instructions:\n"
                    for idx, step in enumerate(ex['instructions'], 1):
                        workout += f"      {idx}. {step}\n"
            plan.append(workout)
            used_exercises.update(day_used_exercises)  # Update global tracking, but less restrictive
        else:
            plan.append(f"Day {day}: Rest")

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