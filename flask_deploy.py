from flask import Flask,render_template, request
import joblib
from nltk.corpus import stopwords


vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('mymodel.joblib')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',  methods=['POST'])
def predict_review():
    review = str(request.form.get('review')) 
#------------------------------------------Prediction-------------------------#
    stop_words = stopwords.words('english')
    user_input = ' '.join([word for word in review.split() if word not in stop_words])
   
    # Transform the input text using the vectorizer
    user_input_vec = vectorizer.transform([user_input])
    
    # Use the pre-trained model to predict sentiment
    sentiment = model.predict(user_input_vec)[0]
    result  = ''
    # Display the output
    if sentiment == 1:
        result = 'Postive'
    else:
        result = 'Negative'
    return render_template('index.html', result=result)

if __name__=='__main__':
    app.run(debug=True)
