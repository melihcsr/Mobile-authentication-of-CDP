# CDP Authentication API Server

This API server provides a simple interface for authenticating Copy Detection Pattern (CDP) images using the trained machine learning models.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements-api.txt
```

2. Make sure you have the trained model checkpoint in the correct location:
```
checkpoints/rgb_supervised_classifier_n_2_one_by_one/
```

3. Start the API server:
```bash
python server.py
```

The server will run on port 5000 by default.

## API Endpoints

### 1. Status Check
```
GET /api/status
```
Returns information about the API server status and loaded model.

Example response:
```json
{
  "status": "running",
  "model_loaded": true,
  "checkpoint_path": "checkpoints/rgb_supervised_classifier_n_2_one_by_one/latest.weights.h5",
  "model_classes": 2,
  "image_type": "rgb"
}
```

### 2. Image Verification
```
POST /api/verify
```
Verifies a CDP image and returns authentication results.

**Parameters:**
- `image`: The image file to verify (multipart/form-data)

**Example response:**
```json
{
  "is_authentic": true,
  "confidence": 0.98,
  "predictions": [0.98, 0.02]
}
```

## Client Example

A sample client implementation is provided in `client_example.py`. You can use it as follows:

```bash
python client_example.py --image path/to/image.jpg
```

## Mobile Integration

To integrate with a mobile application:

1. Host this API on a server accessible to your mobile app.
2. From the mobile app, capture the CDP image using the device camera.
3. Send the image to the `/api/verify` endpoint as a multipart form-data request.
4. Process the response to show authentication results to the user.

Example mobile app integration can use standard HTTP libraries available in your mobile platform. For example, in Swift for iOS:

```swift
// Example using URLSession in Swift
let url = URL(string: "http://your-api-server:5000/api/verify")!
var request = URLRequest(url: url)
request.httpMethod = "POST"

let boundary = UUID().uuidString
request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

var data = Data()
data.append("--\(boundary)\r\n".data(using: .utf8)!)
data.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
data.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
data.append(imageData) // Your captured image data
data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

let task = URLSession.shared.uploadTask(with: request, from: data) { data, response, error in
    if let error = error {
        print("Error: \(error)")
        return
    }
    
    guard let data = data else {
        print("No data received")
        return
    }
    
    do {
        let result = try JSONDecoder().decode(VerificationResult.self, from: data)
        // Use the result in your app
        print("Is Authentic: \(result.is_authentic)")
    } catch {
        print("Failed to decode response: \(error)")
    }
}

task.resume()
```

## Security Considerations

- This API server is designed for development and testing. For production use, implement proper authentication, HTTPS, and rate limiting.
- Consider implementing token-based authentication for the API endpoints.
- Use HTTPS in production to encrypt the data being transmitted. 