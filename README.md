# SMS Spam Detector

A machine learning project that classifies SMS messages as **spam** or **ham** (normal) using Python, scikit-learn and Naive Bayes.

## Files
- `spam_detector.py`: main program
- `spam_data_big.csv`: sample dataset (label, text)

## How to Run in VS Code
1. Download or clone this repository and keep both files in one folder.
2. Open the folder in VS Code (File → Open Folder).
3. Open the terminal (Terminal → New Terminal).
4. Install the libraries:
   ```
   python -m pip install pandas scikit-learn
   ```
5. Run the program:
   ```
   python spam_detector.py
   ```
6. When `SMS:` appears, type a message and press Enter. The result shows SPAM or HAM.

## How It Works
1. Load the dataset with pandas
2. Split into training and testing data (75% / 25%)
3. Convert text to numbers with CountVectorizer
4. Train a Multinomial Naive Bayes model
5. Print accuracy and predict a new message

## Tech Stack
Python, pandas, scikit-learn

## Future Improvements
- Train on a larger real dataset
- Add confusion matrix and classification report
- Build a Streamlit web app
