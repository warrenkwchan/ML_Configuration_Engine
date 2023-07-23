import numpy as np
import copy

class NLPMovieClassifier:
    def __init__(self, recommendersVariablesBuilder):
        self.cosine_sim =  copy.copy(recommendersVariablesBuilder.cosine_sim)
        self.titles = copy.copy(recommendersVariablesBuilder.titles)
        self.indices = copy.copy(recommendersVariablesBuilder.indices)
   
    def get_recommendations(self, title):
        try:
            idx = self.indices[title]
            if not isinstance(idx, np.integer):
                idx = list(idx.values)[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:20]
            movie_indices = [i[0] for i in sim_scores]
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return {"movie_recommendations": self.titles.iloc[movie_indices], "status": "OK"}
    
        
        