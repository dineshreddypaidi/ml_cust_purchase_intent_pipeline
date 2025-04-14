import requests

sample_input = {
    "data": [
        {
            "Administrative": 0,
            "Administrative_Duration": 0.0,
            "Informational": 0,
            "Informational_Duration": 0.0,
            "ProductRelated": 1,
            "ProductRelated_Duration": 0.0,
            "BounceRates": 0.2,
            "ExitRates": 0.2,
            "PageValues": 0.0,
            "SpecialDay": 0.0,
            "Month": "Feb",
            "OperatingSystems": 1,
            "Browser": 1,
            "Region": 1,
            "TrafficType": 1,
            "VisitorType": "Returning_Visitor",
            "Weekend": False
        }
    ]
}

response = requests.post("http://localhost:5000/predict", json=sample_input)
print(response.json())
