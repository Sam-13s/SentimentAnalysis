import requests

# Test home page
response = requests.get('http://127.0.0.1:5000/')
print(f"Home page status: {response.status_code}")

# Test analyze
data = {'text': 'This is a great day!'}
response = requests.post('http://127.0.0.1:5000/analyze', data=data)
print(f"Analyze status: {response.status_code}")
print("Analyze response contains sentiment:", 'sentiment' in response.text.lower())

# Test visualize without data
response = requests.get('http://127.0.0.1:5000/visualize')
print(f"Visualize status: {response.status_code}")
print("Visualize response contains no data message:", 'no sentiment data' in response.text.lower())

print("Basic tests completed.")
