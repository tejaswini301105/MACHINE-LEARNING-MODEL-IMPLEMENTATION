# 📧 Spam Email Detection using Scikit-learn

This project is a demonstration of building a **spam detection model** using Python and machine learning with the help of **Scikit-learn**. The goal is to classify messages as either **spam** or **ham** (non-spam). We generate a synthetic dataset with 30,000 messages and train a model using the **Naive Bayes algorithm**.

---

#🛠️ Tools & Libraries Used

- **Python 3.10+**
- **pandas** – for data manipulation
- **scikit-learn (sklearn)** – for ML preprocessing, model building, and evaluation
- **matplotlib** – for plotting graphs
- **seaborn** – for heatmap (confusion matrix) visualization
- **Jupyter Notebook** – for interactive development

---

#📂 Project Structure

├── spam_email_classifier.ipynb     # Jupyter Notebook with the complete workflow

├── README.md                       # Project overview and documentation


🔍 Step-by-Step Code Explanation

📄 Step 1: Import Libraries

    import pandas as pd
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
We import the required libraries for data generation, ML modeling, and visualization.

🧪 Step 2: Generate Synthetic Dataset

spam_templates = [ ... ]

ham_templates = [ ... ]

 #Function to fill placeholders in templates
    def generate_message(template):
    
        ...
    
 #Generate 15,000 spam and 15,000 ham messages
 
    spam_messages = [{"label": "spam", "message": generate_message(random.choice(spam_templates))} for _ in range(15000)]
    
    ham_messages = [{"label": "ham", "message": generate_message(random.choice(ham_templates))} for _ in range(15000)]

#Combine and shuffle

    all_messages = spam_messages + ham_messages
    
    random.shuffle(all_messages)
    
    df = pd.DataFrame(all_messages)
    
This creates a balanced dataset with 30,000 labeled messages.

🏷 Step 3: Encode Labels

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    Convert textual labels into numeric ones:

    Ham → 0

    Spam → 1

🔀 Step 4: Split the Dataset

    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42)
Split data into 80% training and 20% testing.

🧠 Step 5: Vectorize Text with TF-IDF

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)
TF-IDF converts text into a numerical format suitable for ML.

🤖 Step 6: Train the Naive Bayes Model

    model = MultinomialNB()
    model.fit(X_train_tf, y_train)
Train a simple yet powerful Multinomial Naive Bayes classifier.

🧾 Step 7: Evaluate the Model

    y_pred = model.predict(X_test_tf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
Outputs model performance metrics like precision, recall, F1-score, and accuracy.

📉 Step 8: Confusion Matrix

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
The confusion matrix visually shows correct vs incorrect predictions.

📄SUMMARY:

*The idea is to simulate and solve a real-world problem — detecting spam messages — using a well-known approach:

*Generate a dataset with realistic spam and ham messages

*Vectorize the text data

*Train a Naive Bayes classifier

*Evaluate its performance

*The model achieves high accuracy (~98%), showing its effectiveness even with a synthetic dataset.

🧠Sample output:



