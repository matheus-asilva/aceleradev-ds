[`Sentiment Classifier Model as a REST API with Flask`]
==============================

> Here you can find a simple project to deploy a Machine Learning model as a Sentiment Classifier using Flask.

| Role                 | Responsibility         | Full name                | e-mail       |
| -----                | ----------------       | -----------              | ---------    |
| Data Scientist       | Author                 | [`Matheus de Almeida Silva`]            | [`ms.asilvas1@gmail.com`] |

File Structure
------------
```
├── README.md          <- Overview this project.
├── data
│   ├── train.tsv      <- Data used to train the model
│   ├── test.tsv       <- Data to be predicted
│
├── models             <- Trained and serialized models and vectorizers.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── app.py             <- Flask API application.
├── model.py           <- Class object for the classifier.
├── build_model.py     <- Imports the classifier's class object from `model.py` and instanciates a new model, trains it and the pickle.
├── utils.py           <- Helper functions for `model.py`.
```

Testing the API
------------
1. Run the Flask API. Go to the directory where you can find `app.py`.
```bash
python app.py
```
2. In another terminal window, use HTTPie to make a GET request at the API's URL.
```bash
http http://127.0.0.1:5000/ query=="That was pretty cool"
```
3. Example of successful output.
```bash
HTTP/1.0 200 OK
Content-Length: 58
Content-Type: application/json
Date: Sat, 08 Jun 2019 02:53:56 GMT
Server: Werkzeug/0.14.1 Python/3.6.5

{
    "confidence": 0.998,
    "prediction": "Positive"
}
```

Next Steps
------------
1. Improve the model's accuracy. Maybe changing the model or using some stacking/blending techniques.
2. Deploy it online on an EC2 instance.
