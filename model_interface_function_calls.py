import requests
import spacy
import keras
import keras_nlp
from google.colab import userdata
api = userdata.get('googleapi')

# Load spaCy's pre-trained English model
nlp = spacy.load('en_core_web_sm')

# Define template for instruction and response
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

base = keras_nlp.models.CausalLM.from_preset("kaggle://favouryahdii/gemma-nutritionx/keras/gemma-nutritionx-2b")

# Step 2: Function to process user input and make a prediction
def process_user_input(user_input, api_data="", max_length=256):
    # Preprocess the input
    processed_input = template.format(instruction=user_input, response=api_data)
    
    # Assuming the model (base) is already defined and loaded
    prediction = base.generate(processed_input, max_length)  # Adjust based on your model's predict method
    return prediction

# Function to extract the place using spaCy's NER
def extract_place_spacy(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'GPE', 'PERSON']:  # GPE: Geopolitical Entity (like cities, countries)
            return ent.text
    return None

# Step 3: Function to convert a place to latitude and longitude using Google Geocoding API
def get_location(place):
    api_key = api
    base_url = "https://maps.googleapis.com/maps/api/geocode/json?address="

    # Extract the place name using the NER model
    place_name = extract_place_spacy(place)

    if not place_name:
        return {"error": "No place found in the query"}

    # Properly format the URL
    url = f"{base_url}{place_name}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    # Check if the response contains results
    if len(data['results']) > 0:
        location = data['results'][0]['geometry']['location']
        return location  # This will return latitude and longitude
    else:
        return {"error": "Could not find the location"}

# Step 4: Function to call an external API (Google Places API example)
def get_nearby_places(location, radius=1000, place_type='convenience_store'):
    api_key = api
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Ensure the location (lat, lng) is passed correctly from get_location
    if "error" in location:
        return location  # Return the error if location is not found

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

# Function to extract the place type from the user query
def extract_place_type(query):
    # Define common place types (you can extend this list as needed)
    place_types = ['convenience_store', 'restaurant', 'gym', 'drugstore', 'hospital', 'department_store', 'pharmacy', 'physiotherapist']
    
    # Lowercase the user query for easier matching
    query_lower = query.lower()

    # Check if any place type is mentioned in the query
    for place_type in place_types:
        if place_type in query_lower:
            return place_type

    return "convenience_store"  # Default place type if none is found


# Step 5: Handle user input, model response, and function calling
def handle_function_call(user_input):
    # Always generate a model response first
    model_response = process_user_input(user_input)

    # Check if the query is related to location or nearby search
    if 'nearby' in user_input.lower() or 'location' in user_input.lower():
        # Extract the place name
        location = get_location(user_input)  # Extract the place and get its coordinates (latitude, longitude)

        # Extract the place type from the user query dynamically
        place_type = extract_place_type(user_input)

        # Set default values for radius
        radius = 1000

        # Call the API to get nearby places using the latitude and longitude from get_location
        api_response = get_nearby_places(location, radius, place_type)

        # If there is valid API data, pass it back as input to the model
        if "error" not in api_response:
            return "Here are some nearby places I found: " + model_response + api_response
        else:
            return "I couldn't find any nearby places due to:" + api_response['error'] # model_response + "\n\nHowever, I couldn't find any nearby places due to: " + api_response['error']

    # Return the regular model response if not location-related
    return model_response


# Step 6: Example user query
user_query = "Can you find nearby grocery stores to University of Leeds?"
response = handle_function_call(user_query)
print(response)
