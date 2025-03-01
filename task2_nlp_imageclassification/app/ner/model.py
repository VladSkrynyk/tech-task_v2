# model.py
from transformers import BertTokenizer, TFBertForTokenClassification
from transformers import pipeline
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)

labels = ["[PAD]", "O", "animal", "B-animal", "I-animal"]

nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer)

def predict(text):
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(inputs).logits
    predictions = tf.argmax(logits, axis=-1)
    tokens = tokenizer.tokenize(text)
    predicted_labels = [labels[prediction] for prediction in predictions[0].numpy()]
    return list(zip(tokens, predicted_labels))

