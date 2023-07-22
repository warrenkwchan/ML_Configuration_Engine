import numpy as np
import torch, torch.nn 
import copy 

class HybridNLPNCFRecommender:
    def __init__(self, recommendersVariablesBuilder):
        self.cosine_sim =  copy.copy(recommendersVariablesBuilder.cosine_sim)
        self.titles = copy.copy(recommendersVariablesBuilder.titles)
        self.indices = copy.copy(recommendersVariablesBuilder.indices)
        self.id_map = copy.copy(recommendersVariablesBuilder.id_map)
        self.pmm = copy.copy(recommendersVariablesBuilder.pmm)
        self.indices_map = copy.copy(recommendersVariablesBuilder.indices_map)
        self.ncf_model = copy.copy(recommendersVariablesBuilder.ncf_model)


    def get_recommendations(self, userId, title):
        try:
            userId = int(userId)
            idx = self.indices[title]
            if not isinstance(idx, np.integer):
                idx = list(idx.values)[0]
            sim_scores = list(enumerate(self.cosine_sim[int(idx)]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:26]
            movie_indices = [i[0] for i in sim_scores]
            print(movie_indices)
            predicted_labels = np.squeeze(self.ncf_model(torch.tensor([userId]*len(movie_indices)), 
                                            torch.tensor(list(movie_indices))).detach().numpy())
            top10_items = [movie_indices[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()] 
            movies = self.pmm.iloc[top10_items][['title', 'vote_count', 'vote_average', 'id']]
            print(userId)
            print(movies)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return {"movie_recommendations": list(movies["title"])[:10], "status": "OK"}