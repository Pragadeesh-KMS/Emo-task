
# Emo-task
Emo-task is a system that matches user-entered tasks and emotions with a database of tasks and their associated emotions, providing recommendations based on matching emotions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contact](#contact)

---

## Overview

Task Matcher is designed to assist users in finding relevant tasks based on their input tasks and associated emotions. Leveraging advanced techniques such as cosine similarity and sentence embeddings, it recommends tasks from a dataset that match the emotional context provided by the user.

---

## Features

- **Emotion-based Task Matching**: Matches user-entered tasks and emotions with similar tasks in the database.
- **Cosine Similarity**: Utilizes cosine similarity for comparing task and emotion embeddings.
- **Task Prioritization**: Ranks tasks based on the count of matching emotions with user-entered emotions.
- **User-friendly Interaction**: Offers an intuitive interface for entering tasks and emotions.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pragadeesh-KMS/Emo-task.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Emo-task
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Prepare the dataset: Replace `/path/to/dataset.csv` in `config.ini` with the path to your dataset containing tasks and emotions.
2. Run the main script:

   ```bash
   python Emotask-recommender.py
   ```

3. Follow the prompts to enter tasks and emotions.
4. Receive recommendations based on matching emotions.

---

## Configuration

- `training_3.csv`: Dataset containing tasks and their associated emotions.

---

## Contact

For inquiries or feedback, contact [Pragadeesh(mailto:kmspragadeesh6000@gmail.com).

---

Feel free to customize this README with additional information, specific instructions, or any other details relevant to your project.
