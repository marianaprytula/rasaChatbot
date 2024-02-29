from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
import numpy as np

# def tokenize_sentences(sentences):
#     tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
#     return tokenized_sentences
#
#
input_sentences = ["What are the store hours?", "What is your return policy?", "How long does shipping take?",
                   "Do you ship internationally?", "How can I track my order?", "What happens if my package is lost?",
                   "What payment methods do you accept?", "How can I use my discount code?",
                   "In what formats are your eBooks available?", "How can I contact customer support?"]


target_sentences = ["Our online store is open 24/7. For physical stores, the hours are 9 AM to 8 PM from Monday to Saturday.",
                    "We accept returns within 30 days of purchase. The items must be in their original condition.",
                    "Shipping usually takes between 3 to 5 business days for domestic orders. For international orders, it can take up to 14 business days.",
                    "Yes, we offer international shipping. Extra charges may apply depending on the destination country.",
                    "Once your order has been dispatched, we will send you a confirmation email with a tracking number. You can use this number to track your order on our website.",
                    "In the unfortunate event of a lost package, please contact our customer service. They will assist you with the necessary steps for a refund or reshipment.",
                    "We accept Visa, Mastercard, American Express, and PayPal.",
                    "You can apply your discount code at checkout. Just enter the code in the box labelled 'Discount Code', and the discount will be automatically applied to your order.",
                    "Our eBooks are available in EPUB and PDF formats.",
                    "You can reach our customer support team by sending an email to support@onlinebookstore.com or by calling 123-456-7890 between 9 AM and 5 PM from Monday to Friday."]


# # Tokenize the text
# input_texts = [text.lower().split() for text in input_sentences]
# target_texts = [text.lower().split() for text in target_sentences]
# # Create vocabulary
# input_vocab = sorted(set(word for sentence in input_texts for word in sentence))
# target_vocab = sorted(set(word for sentence in target_texts for word in sentence))
#
# # Create word-to-index and index-to-word mappings
# input_word2idx = {word: idx for idx, word in enumerate(input_vocab)}
# input_idx2word = {idx: word for idx, word in enumerate(input_vocab)}
# target_word2idx = {word: idx for idx, word in enumerate(target_vocab)}
# target_idx2word = {idx: word for idx, word in enumerate(target_vocab)}
#
# # Convert text data to indices
# encoder_input_data = [[input_word2idx[word] for word in sentence] for sentence in input_texts]
# decoder_input_data = [[target_word2idx[word] for word in sentence] for sentence in target_texts]
#
# # Pad sequences to make them of equal length
# encoder_input_data = pad_sequences(encoder_input_data, padding='post')
# decoder_input_data = pad_sequences(decoder_input_data, padding='post')
#
# # Prepare decoder target data (shifted by one timestep)
# decoder_target_data = np.zeros_like(decoder_input_data)
# decoder_target_data[:, 0:-1] = decoder_input_data[:, 1:]
# decoder_target_data[:, -1] = 0  # Padding token
#
# # Define model
# vocab_size = len(target_vocab)
# embedding_dim = 50  # Adjust as needed
# hidden_units = 100  # Adjust as needed
#
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=encoder_input_data.shape[1]))
# model.add(LSTM(hidden_units))
# model.add(RepeatVector(decoder_input_data.shape[1]))
# model.add(LSTM(hidden_units, return_sequences=True))
# model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
#
# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(encoder_input_data, decoder_target_data, epochs=5, batch_size=1, validation_split=0.2)
#
# # Save the trained model
#
# model.save('my_model.keras')
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
def preprocess_input_data(sentences, tokenizer, max_sequence_length):
    # Tokenize input sentences
    input_sequences = tokenizer.texts_to_sequences(sentences)

    # Pad sequences to a fixed length
    padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

    return padded_sequences
model = load_model('my_model.keras')

input_data = preprocess_input_data(input_sentences, tokenizer, max_sequence_length)

# Get the model's predictions
predictions = model.predict(input_data)

responses = decode_predictions(predictions)

# Print or use the generated responses
for response in responses:
    print(response)