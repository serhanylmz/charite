import outlines
import torch
from pydantic import BaseModel, Field
from typing import List, Literal
from enum import IntEnum

# Check for GPU availability and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the model
model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct", device=device)

# Define the score range
class Score(IntEnum):
    MIN = 0
    MAX = 6

# Define the section model
class SectionScore(BaseModel):
    section: int
    score: Score

# Define the assessment model
class Assessment(BaseModel):
    sections: List[SectionScore] = Field(default_factory=list)

# Define the function calling model
class FunctionCall(BaseModel):
    function: Literal["assess_user", "continue_conversation"]
    reason: str

# Create the generators
dialogue_generator = outlines.generate.text(model)
function_generator = outlines.generate.json(model, FunctionCall)
score_generator = outlines.generate.json(model, SectionScore)

def get_questionnaire():
    return [
        "Mood",
        "Interest",
        "Sleep",
        "Energy",
        "Appetite",
        "Self-esteem",
        "Concentration",
        "Psychomotor changes",
        "Suicidal thoughts",
        "Functional impairment"
    ]

@outlines.prompt
def system_prompt(questionnaire):
    """You are a depression assessment chatbot. Your task is to assess the user according to this questionnaire:

    {% for topic in questionnaire %}
    Section {{ loop.index }}: {{ topic }}
    {% endfor %}

    Guidelines:
    1. Maintain a natural, conversational tone.
    2. Ask one question at a time, allowing the user to respond.
    3. Follow up on the user's responses with relevant questions.
    4. Do not mention section numbers or names to the user.
    5. Transition smoothly between topics without explicitly stating the change.
    6. Do not make function calls or list remaining sections in your responses.
    7. Keep your responses concise and focused on the current topic.
    8. Do not assume or hallucinate information the user hasn't provided.
    """

@outlines.prompt
def generate_response(system_prompt, chat_history, current_section, questionnaire):
    """{{ system_prompt }}

    Current section: {{ questionnaire[current_section - 1] }}

    Chat history:
    {% for message in chat_history %}
    {{ message }}
    {% endfor %}

    Chatbot: """

@outlines.prompt
def generate_function_call(system_prompt, chat_history, current_section, questionnaire):
    """{{ system_prompt }}

    Current section: {{ questionnaire[current_section - 1] }}

    Chat history:
    {% for message in chat_history %}
    {{ message }}
    {% endfor %}

    Based on the dialogue for the current section, should we assess the user (move to the next section) or continue the conversation?
    Return JSON: {"function": "assess_user" or "continue_conversation", "reason": "brief explanation"}
    """

@outlines.prompt
def generate_score(system_prompt, chat_history, current_section, questionnaire):
    """{{ system_prompt }}

    Current section: {{ questionnaire[current_section - 1] }}

    Chat history:
    {% for message in chat_history %}
    {{ message }}
    {% endfor %}

    Based on the dialogue for the current section, assign a score between {{ Score.MIN }} and {{ Score.MAX }}.
    Return JSON: {"section": current_section_number, "score": assigned_score}
    """

def conduct_assessment():
    assessment = Assessment()
    questionnaire = get_questionnaire()
    system = system_prompt(questionnaire)
    current_section = 1
    
    print("Chatbot: Hello! I'm here to chat with you about how you've been feeling lately. How has your mood been recently?")
    
    chat_history = []
    
    while current_section <= len(questionnaire):
        user_input = input("User: ")
        chat_history.append(f"User: {user_input}")
        
        response = dialogue_generator(generate_response(system, chat_history, current_section, questionnaire))
        print(f"Chatbot: {response}")
        chat_history.append(f"Chatbot: {response}")
        
        function_call = function_generator(generate_function_call(system, chat_history, current_section, questionnaire))
        
        if function_call.function == "assess_user":
            section_score = score_generator(generate_score(system, chat_history, current_section, questionnaire))
            assessment.sections.append(section_score)
            current_section += 1
    
    print("Chatbot: Thank you for sharing your thoughts and feelings with me. I hope our conversation has been helpful. Is there anything else you'd like to discuss?")
    
    # Print scores to command line (not visible to user)
    print("\nAssessment Scores:")
    for section in assessment.sections:
        print(f"Section {section.section}: {section.score}")

    return assessment

# Run the assessment
if __name__ == "__main__":
    final_assessment = conduct_assessment()