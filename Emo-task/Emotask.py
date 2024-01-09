import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

# Load dataset and Sentence Transformer model
data = pd.read_csv("/content/drive/MyDrive/Emotask/training_3.csv")  # Replace with your dataset file
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Preprocess data and create embeddings
task_emotions = []
for idx, row in data.iterrows():
    task = row['Task']
    emotions = [row[f'Emotion {i}'] for i in range(1, 4)]
    task_emotions.append((task, emotions))

task_texts, emotion_texts = zip(*task_emotions)
task_embeddings = model.encode(task_texts)
emotion_embeddings = model.encode(emotion_texts)

# Store embeddings in a Faiss index (using IndexFlatL2)
concatenated_embeddings = []
for task_emb, emotion_emb in zip(task_embeddings, emotion_embeddings):
    concatenated_emb = np.concatenate((task_emb, emotion_emb))
    concatenated_embeddings.append(concatenated_emb)

emb_np = np.array(concatenated_embeddings, dtype=np.float32)
index = faiss.IndexFlatL2(emb_np.shape[1])  # Using IndexFlatL2 for distance calculations
index.add(emb_np)
# User Input
print("Enter your task(s) (type 'done' when finished):")
user_tasks = []
while True:
    task_input = input("Enter task: ")
    if task_input.lower() == "done":
        break
    user_tasks.append(task_input)
print(user_tasks)
print("Choose up to 6 emotions from the following list by entering the corresponding numbers separated by spaces:")
emotion_options = [
    'happy', 'calm', 'motivated', 'stress', 'tiredness',
    'sad', 'excited', 'anxious', 'bored', 'frustrated', 'anger'
]
print("Emotions:")
for i, emotion in enumerate(emotion_options, start=1):
    print(f"{i}. {emotion}")

user_emotions = input("Enter emotions (e.g., '1 3 5') or 'done' if finished: ").split()
user_emotions = [emotion_options[int(e) - 1] for e in user_emotions]
print(user_emotions)
# Preprocess user input
user_task_embeddings = model.encode(user_tasks)
user_emotion_embeddings = model.encode(user_emotions)
user_combined_emotion_embedding = np.mean(user_emotion_embeddings, axis=0)

# Search nearest tasks for user input using cosine similarity
nearest_tasks = []
for user_task_emb in user_task_embeddings:
    D, I = index.search(np.expand_dims(np.concatenate((user_task_emb, user_combined_emotion_embedding)), axis=0), 1)
    nearest_tasks.append(task_texts[I[0][0]])

# Match user emotions with nearest tasks' emotions and calculate similarity scores using cosine similarity
similarity_scores = []

for task_text, task_emb in zip(nearest_tasks, user_task_embeddings):
    D, I = index.search(np.expand_dims(np.concatenate((task_emb, user_combined_emotion_embedding)), axis=0), 1)
    nearest_task_emotion = emotion_texts[I[0][0]]  # Emotion inherited from the nearest task

    # Count the number of matching emotions between user input and task emotions
    matching_emotions = sum(emotion in nearest_task_emotion for emotion in user_emotions)

    # Append the task, similarity score, and matching emotion count to a list
    similarity_scores.append((task_text, matching_emotions))

# Sort tasks based on the number of matching emotions in descending order
similarity_scores.sort(key=lambda x: x[1], reverse=True)

# Print recommendations based on the sorted similarity scores
print("Tasks matched on semantic search:")

similar_tasks_emotions = []
for task_text in nearest_tasks:
    task_index = task_texts.index(task_text)  # Finding the index of the similar task in the dataset
    task_emotions = emotion_texts[task_index]  # Getting the emotions for the similar task
    similar_tasks_emotions.append((task_text, task_emotions))

# Print emotions of similar tasks
for task, emotions in similar_tasks_emotions:
    print(f"Task: {task}, Emotions: {emotions}")
# Preprocess user emotions to lowercase and remove spaces
user_emotions_processed = [emotion.lower().replace(" ", "") for emotion in user_emotions]

matching_emotions_count = []
for user_task, user_task_emb in zip(user_tasks, user_task_embeddings):
    # Search nearest task for user input using cosine similarity
    D, I = index.search(np.expand_dims(np.concatenate((user_task_emb, user_combined_emotion_embedding)), axis=0), 1)
    nearest_task = task_texts[I[0][0]]

    task_index = task_texts.index(nearest_task)  # Finding the index of the similar task in the dataset
    task_emotions = emotion_texts[task_index]  # Getting the emotions for the similar task

    # Preprocess task emotions to lowercase and remove spaces
    task_emotions_processed = [emotion.lower().replace(" ", "") for emotion in task_emotions]

    # Count the number of matching emotions between user input and task emotions
    matching_count = sum(emotion in task_emotions_processed for emotion in user_emotions_processed)
    matching_emotions_count.append((user_task, matching_count))

# Sort tasks based on the number of matching emotions in descending order
matching_emotions_count.sort(key=lambda x: x[1], reverse=True)

# Print user tasks and their matching emotion count
# Print user tasks and their matching emotion count in a tabular format
table_data = [[user_task, f"Aligns with {count} emotions"] for user_task, count in matching_emotions_count]
table_headers = ["User Task", "Matching Emotion Count"]

print(tabulate(table_data, headers=table_headers, tablefmt="pretty"))