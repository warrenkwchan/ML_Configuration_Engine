import joblib

class NLPMovieClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.cosine_sim =  joblib.load(path_to_artifacts + "nlp_recommendation_cosine_sim.joblib")
        self.titles = joblib.load(path_to_artifacts + "titles.joblib")
        self.indices = joblib.load(path_to_artifacts + "indices.joblib")
   
    def get_recommendations(self, title):
        try:
            idx = self.indices[title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:20]
            movie_indices = [i[0] for i in sim_scores]
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return {"movie_recommendations": self.titles.iloc[movie_indices], "status": "OK"}
    
        
        