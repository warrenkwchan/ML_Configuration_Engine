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

For nlp+svd and nlp+ncf algorithms, try to set different user_id on the HTTP link to different values such as 500, 600, 567, etc. It is to represent personalization that different people will have different recommendations due to their preferences.

**_NOTE:_** The model is trained with a subset of the MovieLens dataset with only 9000 samples so many titles are not there. The docker VM is not able to handle large preprocessed objects and runs out of memory. Please use the titles above or try some more titles.

# One Example Screenshot

![Screenshot 2023-07-23 at 5 05 53 PM](https://github.com/warrenkwchan/ML_Configuration_Engine/assets/26699800/fc6b8d29-c0a9-48fc-b282-25150cb1c94e)

# Run the app on your local machine

There are two ways 

### Using Docker
1. [Install dockers](https://docs.docker.com/engine/install/) on your machine. I work with Mac OS and Linux so I am not sure how different it is in Windows.
2. sudo docker-compose build
3. sudo docker-compose up
4. Then it should launch on the local host and can open in the browser with localhost:8000/api/v1/

### Simply run with Django 
1. You will need to install [Django](https://docs.djangoproject.com/en/4.2/topics/install/)
2. Make sure you have python or python3 install and install all the packages under "requirements.txt" file
3. Go to ./backend/server
4. run "python manage.py runserver" 

