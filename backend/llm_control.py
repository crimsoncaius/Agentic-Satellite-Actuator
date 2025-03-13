from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from typing import List, Dict, Optional
from datetime import datetime
import re
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
from audio_utils import record_audio, transcribe_audio
from llm_socket import fail_state, point_numbers, add_numbers, delete_numbers, view_point, get_report_request, get_reset_sim, change_numbers

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
galamad_api = os.getenv("GALAMAD_API_KEY")
alvin_api = os.getenv("ALVIN_API_KEY")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = galamad_api
OPENAI_API_KEY = galamad_api  # Initialize shared queue

# Current satellite state
satellite_status = {
    "position": "earth orbit",
    "battery_level": "85%",
    "last_command": "",
    "desired_angle": [100, 80, 0],
    "time_to_destination": 849,
    "time_in_current": 4502,
    "target": "e2",
    "target_angles": None,
    "rotation": {"X": 0, "Y": 0, "Z": 0},
    "rotation_history": []
}

orientation_dict = {
    "e1": {"x": 100, "y": 80, "z": 0},
    "e2": {"x": 90, "y": 20, "z": 30},
    "e3": {"x": 80, "y": 10, "z": 75},
    "sun": {"x": -90, "y": -90, "z": 35}
    }

system_prompt_template = """
You are an advanced assistant responsible for managing and reasoning about the state of a satellite orbiting Earth. You are also responsible for 
issuing commands to actuate reaction wheels and thrusters to rotate the satellite using Euler's Angles (XYZ rotations). 

The orientation names and angles are stored in the form of a dictionary, which is dynamically updated as new commands are issued.

### Orientation Dictionary (ALWAYS USE THIS WHEN RESPONDING)
The **ONLY** valid orientation list is:
{orientation_dict}

If the user asks for the current orientation list, you **must** use the values inside `orientation_dict`.  
- Do **NOT** use cached values.  
- Do **NOT** return any orientations that are not explicitly listed here.  
- If a point was deleted or reset, it **must not** appear in the response.  

### Your Role
1. **Retrieve Information**:
   - When the user asks a question, identify the relevant key(s) in the satellite state dictionary and retrieve their values.
   - If multiple keys are needed, extract all relevant values and combine them into a natural language response.
2. **Update State**:
   - If the user issues a command to update the satellite's state, provide a confirmation of the update.
   - Include the updated state in your response.
3. **Recall Past Instructions**:
   - Refer to the `last_command` key or other stored instructions to justify or explain the satellite's current behavior.
4. **Rotation Commands**:
   - For any rotation commands, validate that the input specifies an axis (X, Y, Z) and an angle in degrees.
   - Chained rotation commands are allowed, rotation history log should be updated with all axes that have a rotation with an angle.
   - If the command is valid, update the `rotation` field with the current rotation and append the action to the `rotation_history`.
   - Clear the `rotation` field after the rotation has been successfully completed.
5. **Handle Errors Gracefully**:
   - If the user queries a key that does not exist in the dictionary, respond with: "The requested information is not available in the current state."
   - If the input for a rotation command is invalid, specify the issue and ask the user to restate the instruction.
6. **Maintain Clarity**:
   - Your responses should be concise and precise, using natural language.
7. **When User Asks About `time_to_destination` or `time_in_current`**:
   - If the user specifically asks, for example, "How long until the satellite reaches the desired target?" or "How long have we been in the current orientation?":
     - Look up `time_to_destination` or `time_in_current` from the satellite_status dictionary.
8. **Always refer to the orientation_dict when points are mentioned in user input.


### Response Format:
You must respond with a JSON object in one of these formats:

1. For general responses:
   "response": "your natural language response here"

2. When a command has been issued that wants the satellite to point to a specific location, extract the specific location name
from the orientation_dict[name] and the XYZ angles from it (3 angles in total):
   "response": "Pointing to [x], changing orientation to X, Y, Z",
   "command": "point_sat",
   "target": [x],
   "target_angles": [X, Y, Z]

3. When a command has been issued that wants the satellite to point to a target x (eg. "Point to E1") without specifying angles, DO NOT include target angles in the response. 
Only return the target name:
   "response": "Pointing to [x]",
   "command": "point_x",
   "target": e1

4. When a command has been issued that instructs the satellite to point at specific XYZ angles (3 angles in total), regardless of whether the word "target" appears in the input, check the below first:
   - Always set `"target"` to `"manual input"` unless a named entry from `orientation_dict` is explicitly referenced.
   - The response format should always include `"target_angles"` when angles are provided, even if the word "target" appears in the input, and respond in the format below:

   "response": "Manual input, changing orientation to X, Y, Z",
   "command": "point_sat",
   "target": "manual input",
   "target_angles": [X, Y, Z]

5.  For time_to_destination:
    "response": "We have x seconds left to reach the desired target."
  
6.  For time_in_current:
    "response": "We have been in our current orientation for x seconds."

7. When a command has been issued that wants to add an orientation point x and its related XYZ angles to the system:

    - Check if the orientation point x already exists in the orientation dictionary orientation_dict:
        - If the orientation point does not already exist, extract the name of the orientation point and its related angles, and respond as follows:
          "response": "Adding new orientation point x with angles X Y Z to the system.",
          "command": "add_o",
          "orientation": x,
          "angles": [X, Y, Z]

        - If the orientation point already exists, and respond as follows:
          "response": "Point x already exists in the system, no action taken.",

    - Use the provided orientation_dict to check for the existence of the orientation point. If the point exists, include its angles from orientation_dict in the response.

8. When a command has been issued that wants to change or modify an orientation point x and its related XYZ angles to the system:

    - **FIRST** Check if the orientation point x already exists in the orientation dictionary orientation_dict:

        - If the orientation point already exists, extract the name of the orientation point and its related angles, and respond as follows:
            "response": "Changing orientation point x with angles X Y Z to the system.",
            "command": "change_o",
            "orientation": x,
            "angles": [X, Y, Z]

        - If the orientation point does not exist in `orientation_dict`, you MUST return:
            "response": "Orientation point doesn't exist, no action taken."


        - DO NOT attempt to modify the orientation if it does not exist.
        - DO NOT guess values or create a response that is not explicitly defined above.

9. When a command has been issued that wants to delete an orientation point x and its related XYZ angles in the system:

    - **FIRST**, check if the orientation point x already exists in orientation_dict:

        - If the orientation point doesn't exist in orientation_dict, respond as follows:
          "response": "Point x doesn't exist in the system. No action taken.",

        - If the orientation point already exists in orientation_dict, respond as follows:
            "response": "Point x deleted from system."
            "command": "delete_o",
            "orientation": x
          
    - Always use the provided orientation_dict to check for the existence of the orientation point.

10. When a command has been issued to reset simulation, use the following response:

    "response": "Resetting simulation now"
    "command": "reset_simulation"

11. When a command has been issued that is related to getting specific information (e.g., querying orientation, velocity, target info), use the following rules:

    - For commands that involve an action, such as moving, pointing the satellite (`"point_sat"`, `"add_o"`, `"change_o"`, `"delete_o"`, etc.), DO NOT include `"report_type"`.

    - If the command is to get the current orientation :
        "response": "Getting current orientation"
        "command": "get_orientation"
        "report_type": "get_orientation"

    - If the command is to get current target info:
        "response": "Getting current target info"
        "command": "get_target"
        "report_type": "get_target"

    - If the command is to get pointing info:
        "response": "Getting pointing info now"
        "command": "get_pointing_info"
        "report_type": "get_pointing_info"

    - If the command is to get velocty:
        "response": "Getting current velocity now"
        "command": "get_velocity"
        "report_type": "get_velocity"

12. When a query is made to get the current orientation list, retrieve the latest orientation_dict and state all the orientation points in orientation_diot. Do not issue any commands, such as get_orientation, just display the response as
follows:
    "response": "list out all the orientation points in the orientation_dict"

### Conversation History:
{chat_history}

### User Input:
{user_input}

Assistant Response:
"""

# Initialize LangChain components
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_input",
    return_messages=True
)

prompt = PromptTemplate(
    template=system_prompt_template,
    input_variables=["chat_history", "user_input", "satellite_status"]
)

llm = ChatOpenAI(model="gpt-4", temperature=0)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

class GraphState(BaseModel):
    messages: List[str]
    current_command: Optional[Dict] = None
    next_step: str = "determine_command_type"

    class Config:
        arbitrary_types_allowed = True

def reset():
    global orientation_dict, satellite_status

    orientation_dict = {
    "e1": {"x": 100, "y": 80, "z": 0},
    "e2": {"x": 90, "y": 20, "z": 30},
    "e3": {"x": 80, "y": 10, "z": 75},
    "sun": {"x": -90, "y": -90, "z": 35}
    }

    get_reset_sim("")



def determine_command_type(state: GraphState) -> Dict:
    """Determines if the incoming request is a command or a report based on the LLM response."""
    messages = state.messages
    global orientation_dict, satellite_status

    # Ensure messages exist before accessing last_message
    if not messages:
        print("Warning: No messages found in state.")
        return {"error": "No user input detected"}

    last_message = messages[-1]

    # Ensure current_command exists before accessing it
    parsed_response = getattr(state, "current_command", None) or {}

    # Extract parsed fields safely
    report_type = parsed_response.get("report_type") if parsed_response else None
    command_type = parsed_response.get("command") if parsed_response else None

    # Define known report and command types
    reporting_keywords = ["status", "report", "info", "query", "get", "fetch"]
    report_map = {"get_orientation", "get_target", "get_pointing_info", "get_velocity"}
    command_map = {"point_sat", "add_o", "change_o", "delete_o", "reset_simulation"}

    # If report_type is explicitly recognized, treat as a report
    if report_type in report_map:
        return {
            "current_command": {
                "request_type": "reporting",
                "message": last_message
            },
            "next_step": "parse_report"
        }

    # If command_type is explicitly recognized, treat as a command
    if command_type in command_map:
        return {
            "current_command": {
                "request_type": "command",
                "message": last_message
            },
            "next_step": "parse_command"
        }

    # 3️If report_type is missing but a command is detected, assume it's a command action
    if not report_type and command_type:
        print("⚠️ Warning: `report_type` missing, but `command` detected. Redirecting to `parse_command`.")
        return {
            "current_command": {
                "request_type": "command",
                "message": last_message
            },
            "next_step": "parse_command"
        }

    # 4️⃣ If no valid command is found, check message keywords
    is_reporting = any(keyword in last_message.lower() for keyword in reporting_keywords)

    # Default case: route based on message content
    return {
        "current_command": {
            "request_type": "reporting" if is_reporting else "command",
            "message": last_message
        },
        "next_step": "parse_report" if is_reporting else "parse_command"
    }

def parse_report(state: GraphState) -> Dict:

    print("parse_report")

    """Node to determine specific type of report requested"""
    global orientation_dict, satellite_status
    messages = state.messages
    last_message = messages[-1]

    # Format satellite status for prompt
    sat_status_str = ' '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                              for k, v in satellite_status.items()])
    orient_dict_str = ' '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                      for k, v in orientation_dict.items()])

    try:
        response = chain.run(
            chat_history=messages[:-1],
            user_input=last_message,
            satellite_status=sat_status_str,
            orientation_dict=json.dumps(orientation_dict)
        )
        print(f"Raw LLM response: {response}")
        
        parsed_response = json.loads(response)
        print(f"Parsed Response: {parsed_response}")

        # Ensure `report_type` is only added if the command is a report
        report_commands = {"get_orientation", "get_target", "get_pointing_info", "get_velocity"}
        
        if "report_type" not in parsed_response and parsed_response.get("command") in report_commands:
            print("Warning: report_type missing, setting it to command")
            parsed_response["report_type"] = parsed_response.get("command")
        
        next_step = "handle_report" if parsed_response.get("command") in report_commands else "end"
        
        return {
            "current_command": parsed_response,
            "next_step": next_step
        }

    except Exception as e:
        print(f"Error in parse_report: {str(e)}")
        return {
            "current_command": {"response": f"Error processing command: {str(e)}"},
            "next_step": "end"
        }

def parse_command(state: GraphState) -> Dict:
    """Node to parse and validate incoming commands"""
    global satellite_status, orientation_dict
    
    messages = state.messages
    last_message = messages[-1]  # Ensure last_message is defined before usage

    print("DEBUG: Latest orientation_dict:", orientation_dict)
    
    # Extract target name and angles from the user messagechain
    match = re.search(r"Add (\w+), angles (\d+), (\d+), (\d+)", last_message)
    if match:
        target_name = match.group(1)
        target_angles = [int(match.group(2)), int(match.group(3)), int(match.group(4))]

        # Check if the orientation point already exists
        if target_name in orientation_dict:
            existing_angles = orientation_dict[target_name]
            return {
                "current_command": {
                    "response": f"The orientation point '{target_name}' already exists in the system with angles {existing_angles}.",
                    "command": "exists",
                    "orientation": target_name,
                    "angles": existing_angles
                },
                "next_step": "end"
            }

    # Format satellite status for prompt
    sat_status_str = ' '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                              for k, v in satellite_status.items()])

    orient_dict_str = ' '.join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" 
                      for k, v in orientation_dict.items()])
    
    try:
        response = chain.run(
            chat_history=messages[:-1],
            user_input=last_message,
            satellite_status=sat_status_str,
            orientation_dict=json.dumps(orientation_dict)
        )

        if isinstance(response, tuple):
            response = response[0]
        
        print(f"Raw LLM response: {response}")
        
        
        parsed_response = json.loads(response)

        command_map = {
            "point_sat": "update_target",
            "point_x": "switch_point",
            "add_o": "add_point",
            "change_o": "change_point",
            "delete_o": "delete_point",
            "exists": "end",
            "reset_simulation": "reset_simulation"
        }
    
        next_step = command_map.get(parsed_response.get("command"), "end")
    
        return {
            "current_command": parsed_response,
            "next_step": next_step
        }
        
    except Exception as e:
        print(f"Error in parse_command: {str(e)}")
        return {
            "current_command": {"response": f"Error processing command: {str(e)}"},
            "next_step": "end"
        }

    
def handle_report(state: GraphState) -> Dict:
    """Handles all report-type requests."""

    socket_port = 5005
    if not state.current_command:
        return {
            "current_command": {"type": "error"},
            "next_step": "end"
        }

    report_type = state.current_command.get("report_type")
    llm_response_text = state.current_command.get("response", "")

    if not report_type:
        print("Error: Missing report_type in current_command.")
        return {
            "current_command": {"type": "error"},
            "next_step": "end"
        }

    return {
        "current_command": {"report_type": report_type, "response": llm_response_text},
        "next_step": "end"
    }

    
def update_target(state: GraphState) -> Dict:
    """Node to handle target updates"""
    global satellite_status
    command = state.current_command
    
    try:
        if command["command"] == "point_sat":
            if "target" in command:
                satellite_status["target"] = command["target"]
            if "target_angles" in command:
                satellite_status["target_angles"] = command["target_angles"]
            elif "target_angles" not in command:
                satellite_status["target"] = "Point not in system."
                satellite_status["target_angles"] = "Point not in system."
            satellite_status["last_command"] = "point_sat"
        
        return {"current_command": command}
    except Exception as e:
        print(f"Error in update_target: {str(e)}")
        return {
            "current_command": {"response": f"Error updating target: {str(e)}"},
            "next_step": "end"
        }
    
def switch_point(state: GraphState) -> Dict:
    """Node to handle point only updates"""
    global satellite_status
    command = state.current_command
    
    try:
        if command["command"] == "point_x":
            if "target" in command:
                satellite_status["target"] = command["target"]

            satellite_status["last_command"] = "point_x"
        
        return {"current_command": command}
    except Exception as e:
        print(f"Error in update_target: {str(e)}")
        return {
            "current_command": {"response": f"Error updating target: {str(e)}"},
            "next_step": "end"
        }

def add_point(state: GraphState) -> Dict:
    """Node to handle adding one point by sending request to server."""
    command = state.current_command
    try:
        if command["command"] == "add_o":
            return {"current_command": command} 
        return {"current_command": command}
            
    except Exception as e:
        print(f"Error in add_point: {str(e)}")
        return {
            "current_command": {"response": f"Error processing request: {str(e)}"},
            "next_step": "end"
        }
    
def change_point(state: GraphState) -> Dict:
    """Node to handle changing one point by sending request to server."""
    command = state.current_command
    try:
        if command["command"] == "change_o":
            return {"current_command": command}  # Only pass along command, no local processing
        return {"current_command": command}
            
    except Exception as e:
        print(f"Error in add_point: {str(e)}")
        return {
            "current_command": {"response": f"Error processing request: {str(e)}"},
            "next_step": "end"
        }

def delete_point(state: GraphState) -> Dict:
    """Node to handle deletion requests for one or multiple orientation points without modifying orientation_dict."""
    global satellite_status  # No longer need orientation_dict here
    command = state.current_command
    try:
        if command["command"] == "delete_o":
            # Extract the orientation name(s)
            orientations = command.get("orientation")
            if not orientations:
                raise ValueError("No orientation specified in the command.")
            
            # Update status to reflect a deletion request, but don't modify orientation_dict.
            satellite_status["last_command"] = "delete_o"
            
            # For a single orientation, simply return a confirmation message.
            if isinstance(orientations, str):
                return {
                    "current_command": command,
                    "response": f"Deletion requested for point '{orientations}'"
                }
            # For multiple orientations, construct a response accordingly.
            elif isinstance(orientations, list):
                return {
                    "current_command": command,
                    "response": f"Deletion requested for points: {', '.join(orientations)}"
                }
        # In cases where the command isn't 'delete_o'
        return {"current_command": command}
    
    except Exception as e:
        print(f"Error in delete_point: {str(e)}")
        return {
            "current_command": {"response": f"Error deleting point(s): {str(e)}"},
            "next_step": "end"
        }
    
def reset_simulation(state: GraphState) -> Dict:
    """Node to handle reset simulation commands"""
    global satellite_status, orientation_dict
    command = state.current_command
    
    try:
        if command.get("command") == "reset_simulation":

            return {
                "current_command": {
                    "response": "Resetting simulation now",  # Ensure response is included
                    "command": "reset_simulation"
                }
            }
    
    except Exception as e:
        print(f"Error in reset_simulation: {str(e)}")
        return {
            "current_command": {"response": f"Error resetting simulation: {str(e)}"},
            "next_step": "end"
        }

def end_node(state: GraphState) -> Dict:
    """Final node that returns the response and handles socket communication"""
    try:        
        if state.current_command and "response" in state.current_command:
            return {"response": state.current_command["response"]}
        return {"response": "Command processed"}
    except Exception as e:
        print(f"Error in end_node: {str(e)}")
        return {"response": f"Error: {str(e)}"}

def get_next_step(state: GraphState) -> str:
    return state.next_step

def create_satellite_graph():
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("determine_request", RunnableLambda(determine_command_type))
    workflow.add_node("parse_command", RunnableLambda(parse_command))
    workflow.add_node("update_target", RunnableLambda(update_target))
    workflow.add_node("switch_point", RunnableLambda(switch_point))
    workflow.add_node("change_point", RunnableLambda(change_point))
    workflow.add_node("add_point", RunnableLambda(add_point))
    workflow.add_node("delete_point", RunnableLambda(delete_point))
    workflow.add_node("reset_simulation", RunnableLambda(reset_simulation))
    workflow.add_node("parse_report", RunnableLambda(parse_report))
    workflow.add_node("handle_report", RunnableLambda(handle_report))
    workflow.add_node("end", RunnableLambda(end_node))
    
    # Add edges with direct conditional functions
    workflow.add_conditional_edges(
        "determine_request",
        lambda x: x.next_step,
        {
            "parse_command": "parse_command",
            "parse_report": "parse_report"
        }
    )
    
    workflow.add_conditional_edges(
        "parse_command",
        lambda x: x.next_step,
        {
            "update_target": "update_target",
            "switch_point": "switch_point",
            "change_point": "change_point",
            "add_point": "add_point",
            "delete_point": "delete_point",
            "reset_simulation": "reset_simulation",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "parse_report",
        lambda x: x.next_step,
        {
            "handle_report": "handle_report",
            "end": "end"
        }
    )
    
    workflow.add_edge("handle_report", "end")
    
    workflow.set_entry_point("determine_request")
    workflow.set_finish_point("end")
    
    return workflow.compile()

def run_satellite_command(message: str, prev_messages: List[str] = None):
    if prev_messages is None:
        prev_messages = []
        
    graph = create_satellite_graph()
    
    try:
        initial_state = GraphState(
            messages=prev_messages + [message],
            current_command=None,
            next_step="parse_command"
        )
        
        result_tuple = graph.invoke(initial_state.model_dump())
        if isinstance(result_tuple, tuple):
            result = result_tuple[0]
        else:
            result = result_tuple

        return result
    except Exception as e:
        print(f"Error executing graph: {str(e)}")
        return {"response": f"Error: {str(e)}"}
    
def parse_result(input):
    global orientation_dict, satellite_status
    conversation_history = []

    result = run_satellite_command(input)
        
    try:
        if isinstance(result, dict):
            if 'response' in result and 'current_command' not in result:
                print("current_command not in result")
                print(result['response'])
            
            elif 'current_command' in result:
                # Handle report types
                if 'report_type' in result['current_command']:
                    print("report_type")
                    llm_response = result['current_command'].get("response", "")
                    report_type = result['current_command']['report_type']
                    return get_report_request(llm_response, report_type)

                # Handle command types
                elif result['current_command'].get('command') == "point_sat":
                    llm_response = result['current_command'].get("response", "")
                    if 'target_angles' in result['current_command']:
                        angles = result['current_command']['target_angles']
                        type = "manual_pointing"
                        if angles and len(angles) == 3:
                            point_numbers(llm_response, angles, type)
                            success_status = "Manual pointing initiated!"
                            print(success_status)
                            return success_status


                elif result['current_command'].get('command') == "point_x":
                    points = ""
                    llm_response = result['current_command'].get("response", "")
                    if isinstance(result['current_command']['target'], str):
                        point = result['current_command']['target'].lower()
                        points = point
                        type = "orientation_point"
                        view_point(llm_response, points, type)
                        success_status = "Target pointing successful!"
                        print(success_status)
                        return success_status

                # Handle adding an orientation point (with dictionary update)
                elif result['current_command'].get('command') == "add_o":
                    add_points = {}
                    llm_response = result['current_command'].get("response", "")
                    type = "add_orientation"

                    if isinstance(result['current_command']['orientation'], str):
                        point = result['current_command']['orientation'].lower()
                        angles = result['current_command']['angles']
                        add_points["id"] = point
                        add_points["orientationStruct"] = {"x": angles[0], "y": angles[1], "z": angles[2]}                        
                    else:
                        for point, angles in zip(result['current_command']['orientation'], result['current_command']['angles']):
                            point = point.lower()
                            add_points["id"] = point
                            add_points["orientationStruct"] = {"x": angles[0], "y": angles[1], "z": angles[2]}

                    # Send the request and wait for response
                    response = add_numbers(llm_response, add_points, type)

                    # Process and update orientation_dict if successful
                    if response.get("status") == "success":
                        if isinstance(result['current_command']['orientation'], str):
                            orientation_dict[result['current_command']['orientation']] = result['current_command']['angles']
                        else:
                            for point, angles in zip(result['current_command']['orientation'], result['current_command']['angles']):
                                orientation_dict[point] = angles

                        success_status = "Orientation point added successfully and updated in system!"
                        print(success_status)
                        return success_status

                    else:
                        fail_status = f"Failed to add orientation: {response.get('status')}"
                        print(fail_status)
                        return fail_status

                elif result['current_command'].get('command') == "change_o":
                    change_points = {}
                    llm_response = result['current_command'].get("response", "")
                    type = "modify_orientation"

                    if isinstance(result['current_command']['orientation'], str):
                        point = result['current_command']['orientation'].lower()
                        angles = result['current_command']['angles']
                        change_points["id"] = point
                        change_points["orientationStruct"] = {"x": angles[0], "y": angles[1], "z": angles[2]}
                    else:
                        for point, angles in zip(result['current_command']['orientation'], result['current_command']['angles']):
                            point = point.lower()
                            change_points["id"] = point
                            change_points["orientationStruct"] = {"x": angles[0], "y": angles[1], "z": angles[2]}

                    response = change_numbers(llm_response, change_points, type)

                    # Process and update orientation_dict if successful
                    if response.get("status") == "success":
                        if isinstance(result['current_command']['orientation'], str):
                            orientation_dict[result['current_command']['orientation']] = result['current_command']['angles']
                        else:
                            for point, angles in zip(result['current_command']['orientation'], result['current_command']['angles']):
                                orientation_dict[point] = angles

                        success_status = "Orientation point modified successfully and updated in system!"
                        print(success_status)
                        return success_status

                    else:
                        fail_status = f"Failed to modify orientation: {response.get('status')}"
                        print(fail_status)
                        return fail_status

                elif result['current_command'].get('command') == "delete_o":
                    llm_response = result['current_command'].get("response", "")
                    delete_points = {"id": ""}
                    type = "delete_orientation"

                    if isinstance(result['current_command']['orientation'], str):
                        point = result['current_command']['orientation'].lower()
                        delete_points["id"] = point

                    response = delete_numbers(llm_response,delete_points, type)

                    if response.get("status") == "success":
                        if point is not None:
                            # Safely remove the orientation point using pop()
                            removed_value = orientation_dict.pop(point, None)
                            if removed_value is not None:
                                success_status = f"Orientation point '{point}' deleted successfully from the system!"
                                print(success_status)
                                return success_status
                            else:
                                fail_status = f"Orientation point '{point}' was not found in the system."
                                print(fail_status)
                                return fail_status
                        else:
                            # Optionally handle deletion for multiple points here
                            for pt in result['current_command']['orientation']:
                                pt_lower = pt.lower()
                                removed_value = orientation_dict.pop(pt_lower, None)
                                if removed_value is not None:
                                    print(f"Orientation point '{pt_lower}' deleted successfully from the system!")
                                else:
                                    print(f"Orientation point '{pt_lower}' was not found in the system!")
                    else:
                        fail_status = f"Failed to delete orientation: {response.get('status')}"
                        print(fail_status)
                        return fail_status

                elif result['current_command'].get('command') == "reset_simulation":
                    llm_response = result['current_command'].get("response", "")
                    response = get_reset_sim(llm_response)

                    if response.get("status") == "success":
                        print("DEBUG: Before reset, orientation_dict is:", orientation_dict)

                        orientation_dict.clear()
                        orientation_dict.update({
                            "e1": {"x": 100, "y": 80, "z": 0},
                            "e2": {"x": 90, "y": 20, "z": 30},
                            "e3": {"x": 80, "y": 10, "z": 75},
                            "sun": {"x": -90, "y": -90, "z": 35}
                        })

                        print("DEBUG: After reset, orientation_dict is:", orientation_dict)
                        success_status = f"Orientation dictionary has been reset to: {orientation_dict}"
                        print(success_status)
                        return success_status

                else:
                    fail_status = result['current_command']['response']
                    print(fail_status)
                    return fail_status

    except Exception as e:
        print(f"\nError processing command: {str(e)}")

# def main():
#     global orientation_dict
#     conversation_history = []
    
#     print("Voice-Controlled Satellite System")
#     print("Press SPACE to start recording, press SPACE again to stop")
#     print("Say 'exit' or 'quit' to end the session")
#     print("-" * 50)
    
#     while True:
#         print("\nPress SPACE to record a command...")
        
#         # Record audio
#         output_file = "command.wav"
#         if not record_audio(output_file):
#             print("Failed to record audio. Please check your microphone settings.")
#             continue
            
#         try:
#             # Transcribe audio
#             transcription = transcribe_audio(output_file, OPENAI_API_KEY)
#             print(f"Transcribed text: {transcription}")
            
#             # Check for exit command
#             if transcription.lower().strip() in ['quit', 'exit']:
#                 print("Terminating satellite control session...")
#                 break
                    
#              # Process the command
#             parse_result(transcription)

#         except Exception as e:
#             print(f"\nError processing command: {str(e)}")


# if __name__ == "__main__":
#     main()

# Use below for text input testing without audio
def main():
    global orientation_dict, satellite_status
    conversation_history = []
    
    print("Text-Controlled Satellite System")
    print("Type your commands and press ENTER to execute")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    while True:
        try:
            # Get text input from user
            command = input("\nEnter command: ").strip()
            
            # Check for exit command
            if command.lower() in ['quit', 'exit']:
                print("Terminating satellite control session...")
                break
            # Process the command
            parse_result(command)

        except Exception as e:
            print(f"\nError processing command: {str(e)}")

if __name__ == "__main__":
    main()






