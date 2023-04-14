import wikipedia
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import pickle

# Define the Wikipedia page title and section to retrieve
page_title = "Artificial intelligence"
section_title = "Applications"

# Retrieve the section text from the Wikipedia page
page = wikipedia.page(page_title)
section = page.section(section_title)

# Preprocess the section text
section = re.sub(r'\n+', ' ', section)  # Replace newlines with spaces
section = re.sub(r'\[[^()]*\]', '', section)  # Remove footnotes

# Tokenize and pad the section text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts([section])
sequences = tokenizer.texts_to_sequences([section])
padded_sequences = pad_sequences(sequences, padding='post', maxlen=50)

# Define the model architecture
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=50),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the padded sequences
labels = [1] * len(padded_sequences)
model.fit(padded_sequences, labels, epochs=10, batch_size=1)

# Save the trained model to a file
model.save("model.h5")

# Save the tokenizer used during training to a pickled file
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
