import socket
import json
import requests
import io
import os
from dotenv import load_dotenv
import copy
# from pydub import AudioSegment
# from pydub.playback import play

# Load environment variables from .env file
load_dotenv()

# PlayDialog API Configuration
PLAYDIALOG_URL = "https://api.play.ai/api/v1/tts/stream"

# API Credentials from environment variables
PLAYDIALOG_API_KEY = os.getenv("PLAYDIALOG_API_KEY")
PLAYDIALOG_USER_ID = os.getenv("PLAYDIALOG_USER_ID")

# Socket Configuration from environment variables
socket_port = int(os.getenv("SOCKET_PORT", 5015))

def process_response_and_play(response_json):
    """Extracts and combines LLM and socket responses into a single call to TTS."""
    
    text_parts = []

    # Include the LLM response if available (stored in response_json)
    if "llm_response" in response_json:
        text_parts.append(response_json["llm_response"])

    # Include numerical data, ensuring negative values are spoken correctly
    if "data" in response_json:
        formatted_data = []
        for key, value in response_json["data"].items():
            if isinstance(value, (int, float)):  # Check if it's a number
                formatted_data.append(f"{key} is negative {abs(value)}" if value < 0 else f"{key} is {value}")
            else:
                formatted_data.append(f"{key} is {value}")

        if formatted_data:
            text_parts.append("Received data: " + ", ".join(formatted_data))

    # Include status if available
    if "status" in response_json:
        text_parts.append(f"Response: {response_json['status']}")

    # Create the final text output and call TTS only **once**
    final_text = ". ".join(text_parts)

    # if final_text:
    #     text_to_speech_and_play(final_text)  # Only ONE call to TTS
    # else:
    #     print("Invalid response format or empty response.")

# def text_to_speech_and_play(text):
#     """Convert text to speech using PlayDialog and play it immediately."""
#     if not PLAYDIALOG_API_KEY or not PLAYDIALOG_USER_ID:
#         print("Error: Missing API Key or User ID. Set them as environment variables.")
#         return

#     headers = {
#         'Authorization': f'Bearer {PLAYDIALOG_API_KEY}',
#         'X-USER-ID': PLAYDIALOG_USER_ID,
#         'Content-Type': 'application/json'
#     }

#     json_data = {
#         'model': 'PlayDialog',
#         'text': text,
#         'language': 'english',
#         'voice': 's3://voice-cloning-zero-shot/801a663f-efd0-4254-98d0-5c175514c3e8/jennifer/manifest.json',  # Jennifer's voice
#         "prompt": "You are a helpful satellite assistant who delivers status reports in a concise manner.",
#         'outputFormat': 'mp3',
#         'seed': 34,
#         'speed': 1.05,
#         'pitch': 1.0,
#         'temperature': 1.5
#     }

#     try:
#         response = requests.post(PLAYDIALOG_URL, headers=headers, json=json_data, stream=True)

#         if response.status_code == 200:
#             audio_data = response.raw.read()
#             if not audio_data or len(audio_data) < 100:
#                 print("Error: Received empty or corrupted audio data from PlayDialog.")
#                 return
#             audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
#             play(audio)
#         else:
#             print(f"Failed to generate audio: {response.status_code}, {response.text}")
#     except Exception as e:
#         print(f"An error occurred: {e}")


# General function to handle all socket-based requests and responses
def send_request_and_play(llm_text, message, host='127.0.0.1', port=socket_port):
    """
    Send a JSON request via socket, receive response, and play it as TTS.
    
    Ensures LLM-generated pre-response message is included.
    """
    # Extract LLM-generated response message


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"Sent: {message}")
            s.sendall(message.encode('utf-8'))

            # Receive response from socket
            response = s.recv(1024).decode('utf-8')
            response_json = json.loads(response)
            # print(f"Received: {response_json}")

            # Store LLM response inside the received response to avoid separate TTS calls
            original_response = copy.deepcopy(response_json) 
            response_json["llm_response"] = llm_text

            # Pass everything together to TTS function
            process_response_and_play(response_json)
            # print(original_response)

            return original_response, response_json
        
        except Exception as e:
            print(f"Error: {e}")
            return {"type": "error", "status": str(e)}

# Socket functions using the common send_request_and_play function
def point_numbers(llm_response, numbers, type, host='127.0.0.1', port=socket_port):
    """Send XYZ rotation numbers and play response as TTS."""
    message = json.dumps({"type": type, "data": {"x": numbers[0], "y": numbers[1], "z": numbers[2]}})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def view_point(llm_response, point, type, host='127.0.0.1', port=socket_port):
    """Send orientation point request and play response as TTS."""
    message = json.dumps({"type": type, "data": {"orientation": point}})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def change_numbers(llm_response, numbers, type, host='127.0.0.1', port=socket_port):
    """Change orientation points and angles through a socket connection"""
    message = json.dumps({"type": type, "data": numbers})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def add_numbers(llm_response, numbers, type, host='127.0.0.1', port=socket_port):
    """Send add request and play response as TTS."""
    message = json.dumps({"type": type, "data": numbers})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def delete_numbers(llm_response, numbers, type, host='127.0.0.1', port=socket_port):
    """Send delete request and play response as TTS."""
    message = json.dumps({"type": type, "data": numbers})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def get_report_request(llm_response, report_type, host='127.0.0.1', port=socket_port):
    """Send report request and play response as TTS."""
    import json

    message = json.dumps({"type": report_type})  
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    llm_nlp = ""

    if report_type == "get_orientation":
        llm_nlp = "\n".join([
            "Current satellite orientation:",
            f"  X-axis: {original_response['data']['x']:.2f} degrees",
            f"  Y-axis: {original_response['data']['y']:.2f} degrees",
            f"  Z-axis: {original_response['data']['z']:.2f} degrees"
        ])

    elif report_type == "get_velocity":
        llm_nlp = "\n".join([
            "Current satellite velocity:",
            f"  X-axis: {original_response['data']['x']:.2f} degrees/second",
            f"  Y-axis: {original_response['data']['y']:.2f} degrees/second",
            f"  Z-axis: {original_response['data']['z']:.2f} degrees/second"
        ])

    elif report_type == "get_pointing_info":
        status = "is" if original_response['data']['is_pointed'] else "is not"
        llm_nlp = "\n".join([
            f"Satellite {status} currently pointed at target.",
            f"  Time taken to point: {original_response['data']['timePointing']:.2f} seconds",
            f"  Time since pointed: {original_response['data']['timePointed']:.2f} seconds",
            f"  Angle offset: {original_response['data']['angleOff']:.2f} degrees"
        ])
        
    elif report_type == "get_target":
        llm_nlp = "\n".join([
            "Current target information:",
            f"  Target: {original_response['data']['current_target']}",
            "  Target position:",
            f"    X: {original_response['data']['x']:.2f}",
            f"    Y: {original_response['data']['y']:.2f}",
            f"    Z: {original_response['data']['z']:.2f}"
        ])

    else:
        llm_nlp = "I'm sorry, I don't understand that report type."

    print(llm_nlp)
    return llm_nlp


def get_reset_sim(llm_response, host='127.0.0.1', port=socket_port):
    """Send reset simulation request and play response as TTS."""
    message = json.dumps({"type": "reset_simulation"})
    original_response, _ = send_request_and_play(llm_response, message, host, port)
    return original_response

def fail_state(response):
    """Handle failure in LLM command."""
    print(f"Failure: {response}")
