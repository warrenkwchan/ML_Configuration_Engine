import inspect
from apps.ml.registry import MLRegistry
from django.test import TestCase
from apps.ml.movie_classifier.random_forest import RandomForestClassifier
from apps.ml.movie_classifier.extra_trees import ExtraTreesClassifier
from apps.ml.movie_classifier.nlp_recommender import NLPMovieClassifier
from apps.ml.movie_classifier.hybrid_nlp_svd_recommender import HybridNLPSVDRecommender
from apps.ml.movie_classifier.hybrid_nlp_ncf_recommender import HybridNLPNCFRecommender
from apps.ml.movie_classifier.recommenders_variables_builder import RecommendersVariablesBuilder

class MLTests(TestCase):
    def __init__(self, *args, **kwargs):
        super(MLTests, self).__init__(*args, **kwargs)
        self.rvb = RecommendersVariablesBuilder()

    def test_rf_algorithm(self):
        input_data = {
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 34146,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 68,
            "native-country": "United-States"
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])

    # add below method to MLTests class:
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "movie_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
        
    def test_et_algorithm(self):
        input_data = {
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 34146,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 68,
            "native-country": "United-States"
        }
        my_alg = ExtraTreesClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])

    def test_nlp_algorithm(self):
        my_alg = NLPMovieClassifier(self.rvb)
        response = my_alg.get_recommendations('Ip Man')
        self.assertEqual('OK', response['status'])
        self.assertTrue('movie_recommendations' in response)
        self.assertTrue('Wing Chun' in response['movie_recommendations'].values)

    def test_hybrid_nlp_svd_algorithm(self):
        my_alg = HybridNLPSVDRecommender(self.rvb)
        response = my_alg.get_recommendations(5, 'Toy Story')
        self.assertEqual('OK', response['status'])
        self.assertTrue('movie_recommendations' in response)
        self.assertTrue('Toy Story 2' in response['movie_recommendations'])

    def test_hybrid_nlp_svd_algorithm(self):
        my_alg = HybridNLPNCFRecommender(self.rvb)
        response = my_alg.get_recommendations(5, 'Toy Story')
        self.assertEqual('OK', response['status'])
        self.assertTrue('movie_recommendations' in response)
        self.assertTrue('Toy Story 2' in response['movie_recommendations'])