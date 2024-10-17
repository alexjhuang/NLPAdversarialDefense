# Import necessary libraries
import torch
from transformers import BertForSequenceClassification, ElectraForSequenceClassification, BertTokenizer, ElectraTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import nltk
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack import Attacker
from sklearn.metrics import accuracy_score, f1_score

dataset = load_dataset("imdb")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

bert_encoded = dataset.map(lambda x: tokenize_function(x, bert_tokenizer), batched=True)
electra_encoded = dataset.map(lambda x: tokenize_function(x, electra_tokenizer), batched=True)

# Load pre-trained models for classification
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
electra_model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=2)

# Define adversarial attack (TextFooler)
attack = TextFoolerJin2019.build(model=bert_model)

# Wrap dataset in TextAttack format
wrapped_model = HuggingFaceModelWrapper(bert_model, bert_tokenizer)

# Create the attack
attack = TextFoolerJin2019.build(wrapped_model)

# Use Hugging Face dataset (IMDB)
huggingface_dataset = HuggingFaceDataset("imdb", split='test')

# Create the attacker
attacker = Attacker(attack, huggingface_dataset)

# Run the attack and generate adversarial examples
results = attacker.attack_dataset()

# Display a few adversarial examples
for i in range(5):
    print("Original:", results[i].original_text)
    print("Adversarial:", results[i].perturbed_text)
    print("-" * 50)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

trainer_bert = Trainer(
    model=bert_model,  # Fine-tune BERT model
    args=training_args,
    train_dataset=bert_encoded['train'],
    eval_dataset=bert_encoded['test'],
)

trainer_bert.train()

trainer_electra = Trainer(
    model=electra_model,  # Fine-tune Electra model
    args=training_args,
    train_dataset=electra_encoded['train'],
    eval_dataset=electra_encoded['test'],
)

# Fine-tune Electra model
trainer_electra.train()


nltk.download('wordnet')

def synonym_replacement(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = nltk.corpus.wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()  # Get the first synonym
            new_sentence.append(synonym)
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

# Example of input transformation
sentence = "The quick brown fox jumps over the lazy dog."
transformed_sentence = synonym_replacement(sentence)
print("Original sentence:", sentence)
print("Transformed sentence:", transformed_sentence)

# Majority voting for ensemble methods
def majority_voting(predictions_bert, predictions_electra):
    votes = torch.stack([predictions_bert, predictions_electra])
    majority_vote = torch.mode(votes, dim=0)[0]
    return majority_vote

preds_bert = trainer.predict(bert_encoded['test']).predictions
preds_electra = trainer.predict(electra_encoded['test']).predictions

# Get majority vote predictions
final_predictions = majority_voting(preds_bert, preds_electra)

# Evaluate using majority vote predictions
true_labels = bert_encoded['test']['label']
acc = accuracy_score(true_labels, final_predictions)
f1 = f1_score(true_labels, final_predictions, average='weighted')

print(f"Accuracy: {acc}")
print(f"F1 Score: {f1}")

bert_model.save_pretrained('./models/bert_adversarial_trained')
bert_tokenizer.save_pretrained('./models/bert_adversarial_trained')