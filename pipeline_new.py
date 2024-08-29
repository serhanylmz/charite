import outlines
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set the maximum number of tokens for chatbot responses
MAX_TOKENS = 1000
# Set the maximum number of exchanges per assessment item
MAX_EXCHANGES = 3

# Initialize the model
# model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct", device=device)

model = outlines.models.transformers("meta-llama/Meta-Llama-3-8B-Instruct", device=device)

# MADRS items and their descriptions
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

def chatbot_decision(conversation_history, current_item):
    prompt = f"""You are an AI assistant conducting a subtle assessment using the Montgomery and Asberg Rating Scale (MADRS). Based on the following conversation history and the current assessment item, decide whether to continue the conversation or assess the user on the current item. Analyze the User's responses carefully to determine if you have enough information.

Conversation history:
{conversation_history}

Current assessment item: {MADRS_ITEMS[current_item]['name']}
Item description: {MADRS_ITEMS[current_item]['description']}

Consider:
1. Has enough information been gathered to assess this item accurately?
2. Are there any inconsistencies or ambiguities in the user's responses that need clarification?
3. Would continuing the conversation naturally lead to more relevant information?
4. Have you gathered information on the frequency, intensity, and duration of the symptoms related to this item?

If you decide to continue the conversation, choose -1.
If you decide to assess, choose a score from 0 to 6 based on the following ratings:
{MADRS_ITEMS[current_item]['ratings']}

Provide a brief explanation for your decision, then choose one option: -1, 0, 1, 2, 3, 4, 5, or 6"""

    generator = outlines.generate.choice(model, ["-1", "0", "1", "2", "3", "4", "5", "6"])
    decision = int(generator(prompt))

    return decision

def get_chatbot_response(conversation_history, current_item):
    prompt = f"""You are a compassionate AI assistant conducting a subtle assessment for Ukrainian refugees using the Montgomery and Asberg Rating Scale (MADRS). Based on the conversation history and the current assessment item, provide a response to the user that naturally continues the conversation while gathering information relevant to the assessment. Your response should be empathetic, conversational, and not directly mention the assessment.

Conversation history:
{conversation_history}

Current assessment item: {MADRS_ITEMS[current_item]['name']}
Item description: {MADRS_ITEMS[current_item]['description']}

Guidelines:
1. Be empathetic and supportive, considering the user's refugee status and potential trauma.
2. Ask open-ended questions related to the current assessment item, focusing on frequency, intensity, and duration of symptoms.
3. Adapt your language and tone to be culturally sensitive and appropriate for Ukrainian refugees.
4. If needed, gently probe for more information or clarification on previous responses.
5. Do not generate user responses or continue the dialogue on your own.
6. Keep your response concise (2-3 sentences).

Provide a response:
Chatbot:"""

    generator = outlines.generate.text(model)
    response = generator(prompt, max_tokens=MAX_TOKENS, stop_at=["User:", "\n", "Chatbot:"])
    return response.strip()

def run_madrs_assessment():
    scores = {}
    conversation_history = ""
    current_item = 1

    print("Chatbot: Hello! I'm here to chat with you about how you're feeling. How has your day been so far?")

    while current_item <= 10:
        user_input = input("User: ")
        conversation_history += f"User: {user_input}\n"

        decision = chatbot_decision(conversation_history, current_item)

        if decision != -1:
            print(f"Item {current_item} score is: {decision}")
            scores[current_item] = decision
            current_item += 1

        if current_item <= 10:
            chatbot_response = get_chatbot_response(conversation_history, current_item)
            print(f"Chatbot: {chatbot_response}")
            conversation_history += f"Chatbot: {chatbot_response}\n"
        else:
            print("Chatbot: Thank you for sharing your thoughts with me today. I hope you found our conversation helpful. Take care!")

    print("\nMADRS Assessment Scores:")
    for item, score in scores.items():
        print(f"{MADRS_ITEMS[item]['name']}: {score}")

if __name__ == "__main__":
    run_madrs_assessment()