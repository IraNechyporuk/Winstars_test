import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Loading the NER model
ner_model_path = r'C:\Users\Irenchik\OneDrive\Робочий стіл\Winstars_test\task_2\ner_model\ner_model'
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_path)
tokenizer = AutoTokenizer.from_pretrained(ner_model_path)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

# Loading the image classification model
image_model = load_model(r'C:\Users\Irenchik\OneDrive\Робочий стіл\Winstars_test\task_2\img_model\img_classification_model.h5')

# List of possible classes
class_labels = ["butterfly", "cat", "cow", "dog", "elephant", "hen", "horse", "monkey", "panda", "sheep", "spider", "squirrel"]

# Function for text processing (NER)
def extract_animal_from_text(text):
    entities = ner_pipeline(text)  # Process the text using the NER pipeline
    detected_animals = {entity["word"].lower() for entity in entities if entity["word"].lower() in class_labels}  # Find animals in the text
    if not detected_animals:
        return "No animals detected in text."  # Return message if no animals are detected
    return detected_animals  # Return detected animals

# Function for image processing (classification)
def classify_image(image_path):
    image = cv2.imread(image_path)  # Read the image
    image = cv2.resize(image, (256, 256))  # Resize the image to the input size required by the model
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image
    predictions = image_model.predict(image)  # Make predictions using the image model
    predicted_class = class_labels[np.argmax(predictions)]  # Get the predicted class label
    return predicted_class

# Main pipeline function
def process_input(text, image_path):
    text_animals = extract_animal_from_text(text)  # Get the list of animals from the text
    image_animal = classify_image(image_path)  # Classify the image

    if text_animals == "No animals detected in text.":
        print("No animals detected in text.") # If no animals detected in text, display this message
    else:
        print(f"🔹 Animal from text: {text_animals}")

    print(f"🔹 Animal from image: {image_animal}")  # Display the animal detected from the image

    # Return True/False if there's a match between text and image
    match = image_animal in text_animals if isinstance(text_animals, set) else False

    # Write the result to a file
    with open(r'C:\Users\Irenchik\OneDrive\Робочий стіл\Winstars_test\task_2\output_answer\answer.txt', 'w') as output_file:
        output_file.write("Answer: {}\n".format(match))
    return match

# Reading text from a file
with open(r'C:\Users\Irenchik\OneDrive\Робочий стіл\Winstars_test\task_2\input_data\my_text.txt', 'r') as file:
    text_input = file.read().strip()

# Example of calling the pipeline
image_path = "C:\\Users\\Irenchik\\OneDrive\\Робочий стіл\\Winstars_test\\task_2\\input_data\\my_photo.jpg"  # Path to the actual image file
output = process_input(text_input, image_path)
print(output)  # True or False