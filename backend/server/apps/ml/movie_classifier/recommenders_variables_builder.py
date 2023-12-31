import joblib
from apps.ml.movie_classifier.ncf_utility import NCF
import pandas as pd
import numpy as np
import lightning as L

class RecommendersVariablesBuilder:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.cosine_sim =  joblib.load(path_to_artifacts + "nlp_recommendation_cosine_sim.joblib")
        self.titles = joblib.load(path_to_artifacts + "titles.joblib")
        self.indices = joblib.load(path_to_artifacts + "indices.joblib")
        self.id_map = joblib.load(path_to_artifacts + "id_map.joblib")
        self.pmm = joblib.load(path_to_artifacts + "pmm.joblib")
        self.svd = joblib.load(path_to_artifacts + "svd.joblib")
        self.indices_map = joblib.load(path_to_artifacts + "indices_map.joblib")
        self.ncf_model = self.getNCFModel()

    
    def getNCFModel(self):
        # ratings = pd.read_csv("../../research/ratings_small.csv")
        # rand_userIds = np.random.choice(ratings['userId'].unique(), 
        #                         size=int(len(ratings['userId'].unique())*0.3), 
        #                         replace=False)

        # ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

        # ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'] \
        #                         .rank(method='first', ascending=False)

        # train_ratings = ratings[ratings['rank_latest'] != 1]
        # train_ratings = train_ratings[['userId', 'movieId', 'rating']]
        # num_users = ratings['userId'].max()+1
        # num_items = ratings['movieId'].max()+1
        # all_movieIds = ratings['movieId'].unique()

        # ncf_model= NCF(num_users, num_items, train_ratings, all_movieIds)
        # trainer = L.Trainer(max_epochs=5, accelerator='auto', reload_dataloaders_every_n_epochs=1)
        # trainer.fit(ncf_model)
        ncf_model = joblib.load("./ncf_model.joblib")
        return ncf_model


