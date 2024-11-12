
![enter image description here](https://i.sstatic.net/o1puaUA4.jpg)

https://youtu.be/x2Daglt4gVM?si=TfmTIDeyxsxp_Ets

![enter image description here](https://i.sstatic.net/9QxFn77K.jpg)
![enter image description here](https://i.sstatic.net/j81G0mFd.jpg)
https://chatgpt.com/share/67319a0d-c18c-8010-af00-743cae9240b7
### Title: Creating Insightful Love Triad and Types of Love Charts Using Clubhouse Chats with ChatGPT and Hugging Face Models
#### Introduction:
This guide explains how to analyze Clubhouse chat data to create meaningful visualizations based on **Sternberg’s Triangular Theory of Love**—intimacy, passion, and commitment. By processing text data and using AI models available on Hugging Face, we can detect these elements and categorize the types of love present. This tutorial will cover steps from chat data extraction to analysis, using Hugging Face’s sentiment analysis models and ChatGPT for guidance.
---
### Workflow Summary
1. **Extract Clubhouse Chat Data**: Capture screenshots of Clubhouse chats, transcribe them using Google Lens or another OCR tool, and organize the text.
2. **Analyze Text for Love Triad Elements**: Use Hugging Face models to analyze each chat message for levels of intimacy, passion, and commitment.
3. **Categorize Types of Love**: Based on the love triad scores, calculate the percentages for different types of love (e.g., romantic love, companionate love).
4. **Visualize the Data**: Use Python code to create tables and charts visualizing the love triad elements and types of love for each user.
---
### Step-by-Step Guide
#### Step 1: Extracting Clubhouse Chat Data
Begin by capturing screenshots of your Clubhouse room chats. Use **Google Lens** or a similar OCR tool to convert these images to text. Once the text is ready, you can organize it by each user and timestamp.
#### Step 2: Detecting Love Triad Elements with Hugging Face Models
We can analyze the text for **intimacy, passion, and commitment** using top sentiment and emotion analysis models on Hugging Face. Here are some recommended models:
1. **`j-hartmann/emotion-english-distilroberta-base`**: Specialized for emotions like love, joy, and desire—perfect for detecting intimacy and passion.
   - [Model Link](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
2. **`nlptown/bert-base-multilingual-uncased-sentiment`**: Multilingual and capable of assigning scores from 1 to 5, useful for nuanced positivity analysis relevant to commitment.
   - [Model Link](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
3. **`bhadresh-savani/bert-base-uncased-emotion`**: Detects various emotions that can indicate passion and commitment elements.
   - [Model Link](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion)
#### Python Code Example
The following code uses the `j-hartmann/emotion-english-distilroberta-base` model to analyze each user’s message for signs of Sternberg’s love elements.
1. **Install Required Libraries**:
   ```python
   !pip install transformers
   !pip install torch
   ```
2. **Load and Configure the Model**:
   This code takes a text message as input and returns estimated percentages for intimacy, passion, and commitment.
   
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import torch
   import numpy as np
   model_name = "j-hartmann/emotion-english-distilroberta-base"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name)
   def detect_love_elements(text):
       inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
       with torch.no_grad():
           outputs = model(**inputs)
       scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
       # Extract scores for love, desire, and trust
       emotion_labels = ["admiration", "joy", "love", "desire", "trust"]
       intimacy_score = scores[emotion_labels.index("love")] * 100
       passion_score = scores[emotion_labels.index("desire")] * 100
       commitment_score = scores[emotion_labels.index("trust")] * 100
       return {"Intimacy": intimacy_score, "Passion": passion_score, "Commitment": commitment_score}
   # Test the function with a sample message
   text = "I feel deeply connected to you and cherish every moment we spend together."
   love_elements = detect_love_elements(text)
   print("Love Triad Elements:", love_elements)
   ```
#### Step 3: Categorizing Types of Love
Using Sternberg’s theory, the love triad scores can be combined to identify specific types of love:
   - **Romantic Love**: High intimacy and passion, low commitment.
   - **Companionate Love**: High intimacy and commitment, low passion.
   - **Consummate Love**: High scores in all three elements.
For example, based on the scores from `detect_love_elements`, you can categorize each user’s messages into types of love and assign percentages.
#### Step 4: Visualizing the Love Triad and Types of Love
The results can be visualized in a table or chart. Here’s a simple bar chart example for visualizing the intimacy, passion, and commitment levels for each user.
1. **Plotting the Love Triad Elements**:
   ```python
   import matplotlib.pyplot as plt
   def plot_love_elements(love_elements):
       plt.bar(love_elements.keys(), love_elements.values(), color=['blue', 'red', 'green'])
       plt.title("Sternberg's Triangular Theory of Love Elements")
       plt.xlabel("Love Elements")
       plt.ylabel("Percentage (%)")
       plt.show()
   plot_love_elements(love_elements)
   ```
2. **Creating a Table for Types of Love**:
   The table below provides a sample layout:
   | User      | Romantic Love (%) | Companionate Love (%) | Consummate Love (%) |
   |-----------|-------------------|-----------------------|----------------------|
   | User1     | 45%               | 30%                  | 25%                 |
   | User2     | 30%               | 40%                  | 30%                 |
   | ...       | ...               | ...                  | ...                 |
3. **Generate a Chart for Types of Love**:
   Use similar bar chart methods to show how each type of love is represented across users.
---
### Conclusion
This guide provides a full workflow—from data extraction to visualization—for analyzing love elements and types from Clubhouse chat data. By leveraging models from Hugging Face, such as `j-hartmann/emotion-english-distilroberta-base` for emotion detection, you can gain insights into relational dynamics and visualize Sternberg’s love triad and types of love for each user.
Using Hugging Face models provides flexibility in handling conversational data and detecting nuanced emotions, making it ideal for applications like analyzing Clubhouse chats.
More information like the proposal, business plan and the PowerPoint for them are available at below 👇:
![enter image description here](https://i.sstatic.net/mLfYscgD.jpg)
 
https://1drv.ms/f/s!Ahbmuw9pTocwrwYHY9YHkxlbSeee?e=vGL4LM
The reports are like this::
![enter image description here](https://i.sstatic.net/p2XYg9fg.jpg)
https://1drv.ms/w/s!Ahbmuw9pTocwrxmNXgzn9UfgMgyO?e=DsvfH2
And it's PowerPoint 👇:
![enter image description here](https://i.sstatic.net/bm7c7cjU.jpg)
https://1drv.ms/p/s!Ahbmuw9pTocwrxqlwCd62dngmSGW?e=6UvWJa
And it's business plan 👇:
![enter image description here](https://i.sstatic.net/TPHwIyJj.jpg)
https://1drv.ms/w/s!Ahbmuw9pTocwrw1FiqEPWZewuJi5?e=Azv3fu
And it's PowerPoint 👇:
![enter image description here](https://i.sstatic.net/LhzbhAId.jpg)
https://1drv.ms/p/s!Ahbmuw9pTocwrwtJkTs6lwg5vhBo?e=HrTqC1
  
![enter image description here](https://i.sstatic.net/65dPjyvB.jpg)
 
https://1drv.ms/p/s!Ahbmuw9pTocwrwo9Dtdg-TTB8u8p?e=llHInI

