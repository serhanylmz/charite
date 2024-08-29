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
    },
    3: {
        "name": "Inner Tension",
        "description": "Representing feelings of ill-defined discomfort, edginess, inner turmoil, mental tension mounting to either panic, dread, or anguish. Rate according to intensity, frequency, duration, and the extent of reassurance called for.",
        "ratings": {
            0: "Placid. Only fleeting inner tension.",
            2: "Occasional feelings of edginess and ill-defined discomfort.",
            4: "Continuous feelings of inner tension or intermittent panic, which the patient can only master with some difficulty.",
            6: "Unrelenting dread or anguish. Overwhelming panic."
        }
    },
    4: {
        "name": "Reduced Sleep",
        "description": "Representing the experience of reduced duration or depth of sleep compared to the subject’s own normal pattern when well.",
        "ratings": {
            0: "Sleeps as usual.",
            2: "Slight difficulty dropping off to sleep or slightly reduced, light, or fitful sleep.",
            4: "Sleep reduced or broken by at least two hours.",
            6: "Less than two or three hours of sleep."
        }
    },
    5: {
        "name": "Reduced Appetite",
        "description": "Representing the feeling of a loss of appetite compared with when well. Rate by loss of desire for food or the need to force oneself to eat.",
        "ratings": {
            0: "Normal or increased appetite.",
            2: "Slightly reduced appetite.",
            4: "No appetite. Food is tasteless.",
            6: "Needs persuasion to eat at all."
        }
    },
    6: {
        "name": "Concentration Difficulties",
        "description": "Representing difficulties in collecting one’s thoughts, amounting to incapacitating lack of concentration. Rate according to intensity, frequency, and degree of incapacity produced.",
        "ratings": {
            0: "No difficulties in concentrating.",
            2: "Occasional difficulties in collecting one’s thoughts.",
            4: "Difficulties in concentrating and sustaining thought, which reduces ability to read or hold a conversation.",
            6: "Unable to read or converse without great difficulty."
        }
    },
    7: {
        "name": "Lassitude",
        "description": "Representing a difficulty in getting started or slowness in initiating and performing everyday activities.",
        "ratings": {
            0: "Hardly any difficulty in getting started. No sluggishness.",
            2: "Difficulties in starting simple routine activities, which are carried out with effort.",
            4: "Difficulties in starting simple routine activities which are carried out with effort.",
            6: "Complete lassitude. Unable to do anything without help."
        }
    },
    8: {
        "name": "Inability to Feel",
        "description": "Representing the subjective experience of reduced interest in the surroundings, or activities that normally give pleasure. The ability to react with adequate emotion to circumstances or people is reduced.",
        "ratings": {
            0: "Normal interest in the surroundings and in other people.",
            2: "Reduced ability to enjoy usual interests.",
            4: "Loss of interest in the surroundings. Loss of feelings for friends and acquaintances.",
            6: "The experience of being emotionally paralyzed, inability to feel anger, grief, or pleasure, and a complete or even painful failure to feel for close relatives and friends."
        }
    },
    9: {
        "name": "Pessimistic Thoughts",
        "description": "Representing thoughts of guilt, inferiority, self-reproach, sinfulness, remorse, and ruin.",
        "ratings": {
            0: "No pessimistic thoughts.",
            2: "Fluctuating ideas of failure, self-reproach, or self-deprecation.",
            4: "Persistent self-accusations, or definite but still rational ideas of guilt or sin. Increasingly pessimistic about the future.",
            6: "Delusions of ruin, remorse, or unredeemable sin. Self-accusations which are absurd and unshakeable."
        }
    },
    10: {
        "name": "Suicidal Thoughts",
        "description": "Representing the feeling that life is not worth living, that a natural death would be welcome, suicidal thoughts, and preparations for suicide. Suicidal attempts should not in themselves influence the rating.",
        "ratings": {
            0: "Enjoys life or takes it as it comes.",
            2: "Weary of life. Only fleeting suicidal thoughts.",
            4: "Probably better off dead. Suicidal thoughts are common, and suicide is considered a possible solution, but without specific plans or intention.",
            6: "Explicit plans for suicide when there is an opportunity. Active preparations for suicide."
        }
    }
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