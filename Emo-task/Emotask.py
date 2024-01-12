import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from ipywidgets import DatePicker, VBox

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

import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def select_time(hours, minutes):
    print(f"Time selected: {hours:02d}:{minutes:02d}")

# Assuming user_tasks already contains the tasks entered by the user
deadlines = {}

for task in user_tasks:
    print(f"Enter deadline for task '{task}':")

    date_picker = widgets.DatePicker(description=f'Select Date for {task}')
    hour_selector = widgets.IntSlider(min=0, max=23, description='Hour:')
    minute_selector = widgets.IntSlider(min=0, max=55, step=5, description='Minute:', value=0)

    display(widgets.VBox([date_picker, widgets.HBox([hour_selector, minute_selector])]))
    deadlines[task] = None

    def on_date_change(change, task_name=task):
        deadlines[task_name] = change['new']

    def on_time_change(hours, minutes, task_name=task):
        deadline_time = pd.Timestamp(deadlines[task_name]) if deadlines[task_name] else pd.Timestamp.now()
        new_deadline = pd.Timestamp(
            year=deadline_time.year,
            month=deadline_time.month,
            day=deadline_time.day,
            hour=hours,
            minute=minutes,
        )
        deadlines[task_name] = new_deadline

    date_picker.observe(on_date_change, names='value')
    widgets.interactive(on_time_change, hours=hour_selector, minutes=minute_selector, task_name=task)

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

print("Tasks matched on semantic search:")

similar_tasks_emotions = []
for task_text in nearest_tasks:
    task_index = task_texts.index(task_text)  # Finding the index of the similar task in the dataset
    task_emotions = emotion_texts[task_index]  # Getting the emotions for the similar task
    similar_tasks_emotions.append((task_text, task_emotions))

# Create DataFrame from the list of tuples
similar_tasks_df = pd.DataFrame(similar_tasks_emotions, columns=['Task', 'Emotions'])

# Display DataFrame
display(similar_tasks_df)
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

# Create a DataFrame from matching_emotions_count for tabular representation
df_matching_emotions = pd.DataFrame(matching_emotions_count, columns=['||User Task||', '||Matching Emotion Count||'])

# Display the DataFrame
display(df_matching_emotions)
print("TASK RECOMMENDATION BASED ON DEADLINE:")

# Convert the deadlines dictionary to a DataFrame for tabular representation
tasks = list(deadlines.keys())
deadline_dates = [pd.to_datetime(deadlines[task]).date() if deadlines[task] else None for task in tasks]
deadline_times = [pd.to_datetime(deadlines[task]).time() if deadlines[task] else None for task in tasks]

deadlines_df = pd.DataFrame({
    'Task': tasks,
    'Deadline Date': deadline_dates,
    'Deadline Time': deadline_times
})

# Sort DataFrame by 'Deadline Date' and then 'Deadline Time' columns in ascending order
deadlines_df = deadlines_df.sort_values(by=['Deadline Date', 'Deadline Time']).reset_index(drop=True)

display(deadlines_df)
def assign_priority(matching_count, deadline_date):
    if matching_count == 3:
        today = pd.to_datetime('today').date()
        if deadline_date == today:
            return 1
        elif deadline_date == today + pd.DateOffset(days=1):
            return 2
        elif deadline_date == today + pd.DateOffset(days=2):
            return 4
    elif matching_count == 2:
        today = pd.to_datetime('today').date()
        if deadline_date == today:
            return 3
        elif deadline_date == today + pd.DateOffset(days=1):
            return 7
        elif deadline_date == today + pd.DateOffset(days=2):
            return 8
    elif matching_count == 1:
        today = pd.to_datetime('today').date()
        if deadline_date == today:
            return 5
        elif deadline_date == today + pd.DateOffset(days=1):
            return 7
        elif deadline_date == today + pd.DateOffset(days=2):
            return 8
    else:
        return 6

# Merge and prioritize tasks based on emotions and deadlines
merged_df = pd.merge(df_matching_emotions, deadlines_df, left_on='||User Task||', right_on='Task', how='inner')
merged_df['Priority'] = merged_df.apply(lambda x: assign_priority(x['||Matching Emotion Count||'], x['Deadline Date']), axis=1)
sorted_df = merged_df.sort_values(by=['Priority', 'Deadline Date', 'Deadline Time']).reset_index(drop=True)

# Display the merged and sorted DataFrame
display(sorted_df)
