import outlines
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set the maximum number of tokens for chatbot responses
MAX_TOKENS = 1000

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
    prompt = f"""As an AI conducting a MADRS assessment, decide whether to assess the current item or continue the conversation. Be decisive and efficient.

Conversation history:
{conversation_history}

Current item: {MADRS_ITEMS[current_item]['name']}
Description: {MADRS_ITEMS[current_item]['description']}

Guidelines:
1. Analyze user responses for clear indicators related to the current item.
2. If sufficient information is present, make an assessment immediately.
3. Only continue the conversation if crucial information is missing.
4. Err on the side of making an assessment rather than prolonging the conversation.

Choose -1 to continue or 0-6 to assess based on:
{MADRS_ITEMS[current_item]['ratings']}

Output one number: -1, 0, 1, 2, 3, 4, 5, or 6"""

    generator = outlines.generate.choice(model, ["-1", "0", "1", "2", "3", "4", "5", "6"])
    decision = int(generator(prompt))
    
    return decision

def get_chatbot_response(conversation_history, current_item):
    prompt = f"""You are conducting a MADRS assessment. Focus strictly on gathering information for the current item. Be concise and direct.

Conversation history:
{conversation_history}

Current item: {MADRS_ITEMS[current_item]['name']}
Description: {MADRS_ITEMS[current_item]['description']}

Guidelines:
1. Ask focused questions directly related to the current item.
2. Do not deviate from the current assessment topic.
3. Be concise. Avoid repetition or unnecessary elaboration.
4. Do not use sympathy gestures or actions in text.
5. If the user goes off-topic, gently redirect to the current item.

Provide a short, focused response or question:"""

    generator = outlines.generate.text(model)
    response = generator(prompt, max_tokens=MAX_TOKENS, stop_at=["User:", "\n"])
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
            print(f"Item {current_item} score is: {decision}, now moving on to item {current_item + 1}")
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