import spacy
from spacy.training import Example
import random
import os

# Define the new body parts and equipment lists
body_part_terms = [
    "chest", "upper arms", "upper legs", "lower legs", "back",
    "shoulders", "waist", "lower arms", "cardio", "arms", "legs",
    "stomach", "whole body", "tummy", "core", "glutes", "biceps",
    "triceps", "abs", "abdominals", "full body", "upper body",
    "lower body", "quads", "hamstrings", "calves", "lats", "traps",
    "belly", "thighs", "pecs", "delts"
]

equipment_terms = [
    "home", "dumbbell", "barbell", "gym", "no equipment", "kettlebell",
    "none", "bodyweight", "resistance bands", "machine", "smith machine",
    "cables", "free weights", "nothing", "no gear", "pull-up bar",
    "bench", "bands", "barbels", "cable machine", "at_the_gym",
    "incline bench", "decline bench", "dip station", "chest press machine",
    "pec deck machine", "squat rack", "leg press machine",
    "lying leg curl machine", "seated leg curl machine",
    "glute kickback machine", "step or platform", "standing calf raise machine",
    "seated calf raise machine", "assisted pull-up machine",
    "pull-up bar with neutral grips", "lat pulldown machine",
    "landmine attachment", "handle", "rope attachment", "bar pad",
    "hyperextension bench", "stability ball", "resistance band loop",
    "wall", "gymnastic rings", "sissy squat machine", "glute ham raise machine",
    "plate", "shoulder press machine", "chest supported row machine",
    "reverse fly machine", "preacher bench", "EZ bar", "power rack",
    "box", "donkey calf raise machine", "weight block", "partner",
    "wrist roller", "weight plate"
]

# Add templates and other terms (same as in your script)
experience_terms = ["new", "beginner", "never", "just starting", "don’t know how", "first time", "starting out", "no experience", "easy", "advanced", "absolute beginner", "not sure where to start", "some experience", "intermediate", "expert", "pro", "seasoned", "regular lifter"]
goal_terms = ["get stronger", "look better", "feel more fit", "tone", "build muscle", "get in shape", "move better", "not feel so tired", "stronger", "not feel so weak", "weight loss", "muscle gain", "lose fat", "burn calories", "increase stamina", "improve endurance", "get lean", "bulk up", "definition", "sculpt", "functional strength", "cardio fitness", "hypertrophy"]
frequency_terms = ["3 days a week", "4 days per week", "5 days a week", "twice a week", "every day", "daily", "3 times a week", "4 times weekly", "5 days weekly", "once a week", "6 days a week"]

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

# Data generation function (same as in your script)
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
        text = text.replace("ez bar", "EZ bar").replace("pullup bar", "pull-up bar")

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

# Training function (same as modified version above)
def train_ner_model(training_data, output_path="gym_ner_model", n_iter=100, eval_split=0.2, early_stopping_patience=15, dropout=0.3, load_existing_model=None):
    if load_existing_model and os.path.exists(load_existing_model):
        nlp = spacy.load(load_existing_model)
        print(f"Loaded existing model from {load_existing_model}")
    else:
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
        optimizer = nlp.begin_training() if not load_existing_model else nlp.resume_training()
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

# Generate training data
generated_data = generate_training_data(500)
TRAIN_DATA = [
    ("i want to train my lower arms with a wrist roller at the gym.", {
        "entities": [(15, 25, "BODY_PART"), (30, 42, "EQUIPMENT"), (46, 49, "EQUIPMENT")]
    }),
    ("can you suggest a workout for my shoulders using a shoulder press machine?", {
        "entities": [(24, 33, "BODY_PART"), (40, 62, "EQUIPMENT")]
    }),
    ("i’m advanced and want to work on my cardio with a step or platform 3 days a week.", {
        "entities": [(0, 8, "EXPERIENCE"), (27, 33, "BODY_PART"), (39, 55, "EQUIPMENT"), (56, 69, "FREQUENCY")]
    }),
    ("i want to tone my upper legs using a glute kickback machine at home.", {
        "entities": [(14, 23, "BODY_PART"), (29, 50, "EQUIPMENT"), (54, 58, "EQUIPMENT")]
    }),
]
ALL_TRAIN_DATA = TRAIN_DATA + generated_data

# Fine-tune the model
nlp = train_ner_model(
    ALL_TRAIN_DATA,
    output_path="gym_ner_model_v6",
    n_iter=50,
    load_existing_model="gym_ner_model_v5"
)
print("Model fine-tuning complete.")

# Test the model
test_inputs = [
    "I want to train my lower arms with a wrist roller at the gym.",
    "Suggest a workout for my shoulders using a shoulder press machine.",
    "I’m advanced and want to work on my cardio with a step or platform 3 days a week.",
    "I want to tone my upper legs using a glute kickback machine at home."
]

for test_input in test_inputs:
    doc = nlp(test_input.lower())
    print(f"Input: {test_input}")
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}\n")