import requests
import spacy
import keras_nlp
from google.colab import userdata
from time import sleep

class ModelInterface:
    def __init__(self):
        self.api = userdata.get('googleapi')
        self.nlp = spacy.load('en_core_web_sm')
        self.template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        self.instruction = """
            You are an AI agent tasked with answering nutrition questions in a short and simple way.
        """
        self.path_to_model = keras_nlp.models.CausalLM.from_preset("kaggle://favouryahdii/gemma-nutritionx/keras/gemma-nutritionx-2b")
        self.max_new_tokens = 128
        self.initialize_model()

    def initialize_model(self):
        self.tokenizer = tokenizers.GemmaTokenizer.from_preset(self.path_to_model)
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.path_to_model)

    def process_user_input(self, user_input, api_data="", max_length=256):
        processed_input = self.template.format(instruction=user_input, response=api_data)
        prediction = self.path_to_model.generate(processed_input, max_length)  # Adjust based on your model's predict method
        return prediction

    def extract_place_spacy(self, query):
        doc = self.nlp(query)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                return ent.text
        return None

    def get_location(self, place):
        api_key = self.api
        base_url = "https://maps.googleapis.com/maps/api/geocode/json?address="
        place_name = self.extract_place_spacy(place)
        if not place_name:
            return {"error": "No place found in the query"}
        url = f"{base_url}{place_name}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if len(data['results']) > 0:
            location = data['results'][0]['geometry']['location']
            return location
        else:
            return {"error": "Could not find the location"}

    def get_nearby_places(self, location, radius=1000, place_type='convenience_store'):
        api_key = self.api
        base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        if "error" in location:
            return location
        params = {
            'location': f"{location['lat']},{location['lng']}",
            'radius': radius,
            'type': place_type,
            'key': api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            places = response.json()['results']
            place_names = [place['name'] for place in places]
            return ", ".join(place_names)
        else:
            return {"error": "Failed to retrieve places information"}

    def extract_place_type(self, query):
        place_types = ['convenience_store', 'restaurant', 'gym', 'drugstore', 'hospital', 'department_store', 'pharmacy', 'physiotherapist']
        query_lower = query.lower()
        for place_type in place_types:
            if place_type in query_lower:
                return place_type
        return "convenience_store"

    def handle_function_call(self, user_input):
        model_response = self.process_user_input(user_input)
        if 'nearby' in user_input.lower() or 'location' in user_input.lower():
            location = self.get_location(user_input)
            place_type = self.extract_place_type(user_input)
            radius = 1000
            api_response = self.get_nearby_places(location, radius, place_type)
            if "error" not in api_response:
                return "Here are some nearby places I found: " + model_response + api_response
            else:
                return "I couldn't find any nearby places due to:" + api_response['error']
        return model_response

# Example usage
# model_interface = ModelInterface()
# user_query = "Can you find nearby grocery stores to University of Leeds?"
# response = model_interface.handle_function_call(user_query)
# print(response)
