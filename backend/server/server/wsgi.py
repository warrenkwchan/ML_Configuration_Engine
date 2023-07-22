# the `backend/server/server/wsgi.py file
import os
from django.core.wsgi import get_wsgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
application = get_wsgi_application()

# ML registry
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.movie_classifier.random_forest import RandomForestClassifier
from apps.ml.movie_classifier.extra_trees import ExtraTreesClassifier # import ExtraTrees ML algorithm
from apps.ml.movie_classifier.nlp_recommender import NLPMovieClassifier
from apps.ml.movie_classifier.hybrid_nlp_svd_recommender import HybridNLPSVDRecommender
from apps.ml.movie_classifier.hybrid_nlp_ncf_recommender import HybridNLPNCFRecommender
from apps.ml.movie_classifier.recommenders_variables_builder import RecommendersVariablesBuilder

try:
    registry = MLRegistry() # create ML registry
    # Random Forest classifier
    rf = RandomForestClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Warren",
                            algorithm_description="Random Forest with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(RandomForestClassifier))

    # Extra Trees classifier
    et = ExtraTreesClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=et,
                            algorithm_name="extra trees",
                            algorithm_status="testing",
                            algorithm_version="0.0.1",
                            owner="Warren",
                            algorithm_description="Extra Trees with simple pre- and post-processing",
                            algorithm_code=inspect.getsource(ExtraTreesClassifier))
    
    # get ML engine variables
    rvb = RecommendersVariablesBuilder()
    
    nlp = NLPMovieClassifier(rvb)
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_classifier",
                            algorithm_object=nlp,
                            algorithm_name="nlp",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Warren",
                            algorithm_description="NLP that takes a movie title and compute top recommendations",
                            algorithm_code=inspect.getsource(NLPMovieClassifier))
    
    hybrid_nlp_svd = HybridNLPSVDRecommender(rvb)
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_classifier",
                            algorithm_object=hybrid_nlp_svd,
                            algorithm_name="hybrid_nlp_svd",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Warren",
                            algorithm_description="Hybrid ML algorithm (NLP+SVD) that takes a movie title and compute top recommendations",
                            algorithm_code=inspect.getsource(HybridNLPSVDRecommender))
    
    hybrid_nlp_ncf = HybridNLPNCFRecommender(rvb)
    # add to ML registry
    registry.add_algorithm(endpoint_name="movie_classifier",
                            algorithm_object=hybrid_nlp_ncf,
                            algorithm_name="hybrid_nlp_ncf",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Warren",
                            algorithm_description="Hybrid ML algorithm (NLP+NCF) that takes a movie title and compute top recommendations",
                            algorithm_code=inspect.getsource(HybridNLPNCFRecommender))
    
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))