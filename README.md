# CS6460_Final_Project

## Please visit this [link](http://gatechmlengine.com:8000/api/v1/) to try out the application. 

For pure NLP
http://gatechmlengine.com:8000/api/v1/movie_classifier/predict?ml_algorithm=nlp

For Hybrid NLP + SVD 
http://gatechmlengine.com:8000/api/v1/movie_classifier/predict?ml_algorithm=nlp_svd&user_id=5 

For Hybrid NLP + NCF 
http://gatechmlengine.com:8000/api/v1/movie_classifier/predict?ml_algorithm=nlp_ncf&user_id=5

# Some request example put in "Content:"
{"title":"Ip Man"}

{"title":"Harry Potter and the Philosopher's Stone"}

{"title":"Titanic"}

{"title":"Toy Story"}

For nlp+svd and nlp+ncf algorithms, try to set different user_id on the http link to different values such as 500, 600, 567 etc. It is to represent personalization different people will have different recommendations due to their preferences.

# One Example Screenshot

![Screenshot 2023-07-23 at 5 05 53 PM](https://github.com/warrenkwchan/ML_Configuration_Engine/assets/26699800/fc6b8d29-c0a9-48fc-b282-25150cb1c94e)

