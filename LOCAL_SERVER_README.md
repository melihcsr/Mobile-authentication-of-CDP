# Local QR Verification Server Setup

This README provides instructions on how to set up and run the local server for QR code verification with the SwiftQR mobile app.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Tensorflow and other required packages

## Setup

1. Install the required Python packages:

```bash
pip install -r requirements-api.txt
```

2. Make sure your model checkpoint is available in the `checkpoints` directory. The server will look for the latest checkpoint in the configured directory.

## Running the Server

1. Start the local server:

```bash
python server.py
```

This will run the server on `http://localhost:5000`.

2. Verify the server is running by accessing `http://localhost:5000` in your browser. You should see a JSON response indicating the API is running.

3. You can check the status of the server and model by accessing `http://localhost:5000/api/status`.

## Using with SwiftQR App

The SwiftQR app has been configured to connect to the local server at `http://localhost:5000/verify-qr`.

1. Make sure your iOS device is on the same network as the computer running the server.

2. If testing on a real device (not a simulator), you'll need to replace `localhost` with the actual IP address of your computer in the `APIClient.swift` file:

```swift
// Change from
private let apiEndpoint = "http://localhost:5000/verify-qr"

// To 
private let apiEndpoint = "http://YOUR_COMPUTER_IP:5000/verify-qr"
```

3. For iOS simulators, "localhost" should work without modification.

## API Endpoints

- `GET /` - Basic status check
- `GET /api/status` - Get server and model status
- `POST /api/verify` - Verify image (general endpoint)
- `POST /verify-qr` - Verify QR code image (specific endpoint for SwiftQR app)

### Using the `/verify-qr` Endpoint

To test the endpoint manually:

```bash
curl -X POST -F "qrImage=@/path/to/your/qrcode.jpg" http://localhost:5000/verify-qr
```

The response will be a JSON object with the verification result:

```json
{
  "isValid": true,
  "confidence": 0.95,
  "predictions": [0.95, 0.05]
}
```

## Troubleshooting

1. **Network Issues**: Make sure the iOS device and the server are on the same network.
   
2. **ATS Settings**: If using a real device, you may need to configure App Transport Security (ATS) in your Info.plist file to allow connections to a non-HTTPS server.

3. **Model Loading Errors**: Check the server console output for any errors related to loading the model checkpoint.

4. **CORS Issues**: The server has CORS enabled, but if you're experiencing issues, check that the appropriate headers are being set. 