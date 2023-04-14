import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model from a file
model = load_model('model.h5')

# Load the tokenizer used during training
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define the maximum sequence length used during training
max_sequence_length = 50

# Prompt the user for input and generate responses
while True:
    input_text = input("You: ")
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length)
    output_seq = model.predict(input_seq)[0]
    output_text = tokenizer.sequences_to_texts([np.argmax(output_seq)])[0]
    print("ChatGPT: " + output_text)
