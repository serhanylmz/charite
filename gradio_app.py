import gradio as gr
import outlines
import torch

MAX_TOKENS = 100
MAX_EXCHANGES = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct", device=device)

MADRS_ITEMS = {
    1: {
        "name": "Apparent Sadness",
        "description": "Representing despondency, gloom, and despair (more than just ordinary transient low spirits) reflected in speech, facial expression, and posture. Rate by depth and inability to brighten up.",
        "ratings": {
            0: "No sadness.",
            2: "Looks dispirited but does brighten up without difficulty.",
            4: "Appears sad and unhappy most of the time.",
            6: "Looks miserable all the time. Extremely despondent."
        }
    },
    2: {
        "name": "Reported Sadness",
        "description": "Representing reports of depressed mood, regardless of whether it is reflected in appearance or not. Includes low spirits, despondency, or the feeling of being beyond help and without hope. Rate according to intensity, duration, and the extent to which the mood is reported to be influenced by events.",
        "ratings": {
            0: "Occasional sadness in keeping with the circumstances.",
            2: "Sad or low but brightens up without difficulty.",
            4: "Pervasive feelings of sadness or gloominess. The mood is still influenced by external circumstances.",
            6: "Continuous or unvarying sadness, misery, or despondency."
        }
    }
    # Add the remaining 8 items here
}

def chatbot_decision(conversation_history, current_item, exchange_count):
    prompt = f"""You are an AI assistant conducting a subtle assessment using the Montgomery and Asberg Rating Scale (MADRS). Based on the following conversation history and the current assessment item, decide whether to continue the conversation or assess the user on the current item. Look at the User's responses so far, and see whether you have enough information.

Conversation history:
{conversation_history}

Current assessment item: {MADRS_ITEMS[current_item]['name']}
Item description: {MADRS_ITEMS[current_item]['description']}

Consider:
1. Has enough information been gathered to assess this item?
2. Would continuing the conversation naturally lead to more relevant information?
3. This is exchange number {exchange_count} out of a maximum of {MAX_EXCHANGES}.

If you decide to continue the conversation, choose -1.
If you decide to assess, choose a score from 0 to 6 based on the following ratings:
{MADRS_ITEMS[current_item]['ratings']}

Choose one option: -1, 0, 1, 2, 3, 4, 5, or 6"""

    generator = outlines.generate.choice(model, ["-1", "0", "1", "2", "3", "4", "5", "6"])
    decision = int(generator(prompt))
    
    if exchange_count >= MAX_EXCHANGES and decision == -1:
        decision = int(outlines.generate.choice(model, ["0", "1", "2", "3", "4", "5", "6"])(prompt))
    
    return decision

def get_chatbot_response(conversation_history, current_item):
    prompt = f"""You are a compassionate AI assistant conducting a subtle assessment for Ukrainian refugees using the Montgomery and Asberg Rating Scale (MADRS). Based on the conversation history and the current assessment item, provide a response to the user that naturally continues the conversation while gathering information relevant to the assessment. Your response should be empathetic, conversational, and not directly mention the assessment.

Conversation history:
{conversation_history}

Current assessment item: {MADRS_ITEMS[current_item]['name']}
Item description: {MADRS_ITEMS[current_item]['description']}

Guidelines:
1. Be empathetic and supportive.
2. Ask open-ended questions related to the current assessment item.
3. Do not generate user responses or continue the dialogue on your own.
4. Keep your response concise (2-3 sentences).

Provide a response:
Chatbot:"""

    generator = outlines.generate.text(model)
    response = generator(prompt, max_tokens=MAX_TOKENS, stop_at=["User:", "\n", "Chatbot:"])
    return response.strip()

class ChatbotState:
    def __init__(self):
        self.conversation_history = ""
        self.current_item = 1
        self.scores = {}
        self.exchange_count = 0
        self.assessment_complete = False

def chatbot(message, state):
    if state is None:
        state = ChatbotState()
        return "Hello! I'm here to chat with you about how you're feeling. How has your day been so far?", state

    if state.assessment_complete:
        return "The assessment is complete. Thank you for your participation.", state

    state.conversation_history += f"User: {message}\n"
    state.exchange_count += 1

    decision = chatbot_decision(state.conversation_history, state.current_item, state.exchange_count)

    if decision != -1 or state.exchange_count >= MAX_EXCHANGES:
        state.scores[state.current_item] = decision
        state.current_item += 1
        state.exchange_count = 0

        if state.current_item > 10:
            state.assessment_complete = True
            scores_message = "\nMADRS Assessment Scores:\n"
            for item, score in state.scores.items():
                scores_message += f"{MADRS_ITEMS[item]['name']}: {score}\n"
            return f"Thank you for sharing your thoughts with me today. Here are your assessment scores:\n{scores_message}", state

    chatbot_response = get_chatbot_response(state.conversation_history, state.current_item)
    state.conversation_history += f"Chatbot: {chatbot_response}\n"

    return chatbot_response, state

iface = gr.Interface(
    fn=chatbot,
    inputs=["text", "state"],
    outputs=["text", "state"],
    title="MADRS Assessment Chatbot",
    description="Chat with an AI assistant to complete a Montgomery and Asberg Depression Rating Scale (MADRS) assessment.",
    allow_flagging="never",
    examples=[
        ["I'm feeling a bit down today."],
        ["I've been having trouble sleeping lately."],
        ["I don't have much of an appetite these days."],
    ],
)

iface.launch()