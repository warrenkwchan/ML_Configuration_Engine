import numpy as np

class HybridNLPSVDRecommender:
    def __init__(self, recommendersVariablesBuilder):
        self.cosine_sim =  recommendersVariablesBuilder.cosine_sim
        self.titles = recommendersVariablesBuilder.titles 
        self.indices = recommendersVariablesBuilder.indices
        self.id_map = recommendersVariablesBuilder.id_map
        self.pmm = recommendersVariablesBuilder.pmm 
        self.svd = recommendersVariablesBuilder.svd
        self.indices_map = recommendersVariablesBuilder.indices_map

   
    def get_recommendations(self, userId, title):
        try:
            idx = self.indices[title]
            if not isinstance(idx, np.integer):
                idx = list(idx.values)[0]
            sim_scores = list(enumerate(self.cosine_sim[int(idx)]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:26]
            movie_indices = [i[0] for i in sim_scores]
            
            movies = self.pmm.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
            movies['est'] = movies['id'].apply(lambda x: self.svd.predict(userId, self.indices_map.loc[x]['movieId']).est)
            movies = movies.sort_values('est', ascending=False)
            print(userId)
            print(movies)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return {"movie_recommendations": list(movies["title"])[:10], "status": "OK"}