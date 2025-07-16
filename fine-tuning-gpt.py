#import statements
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
#automodel is a parent class for a basic model that can be used for specific LMS, causal LMs are chatbot models like GPT-2
#autotokenizer is used to convert text into tokens that the model can understand
from datasets import load_dataset
from trl import SFTTrainer  # SFTTrainer is a class for training models using supervised fine-tuning
from trl.trainer import ConstantLengthDataset  # ConstantLengthDataset is a class for creating datasets with constant length sequences  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

#loading the dataset
#this dataset is a collection of 1,000 examples of question-answer pairs
DATASET_NAME = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(DATASET_NAME) 

#printing the dataset
print(dataset)

training_dataset = dataset["train"]
print(training_dataset) 

#checking some examples within the training dataset
 # print(training_dataset[0])  # First example
 # print(training_dataset[1])  # Second example
print(training_dataset[11]) 

#loading the model and tokenizer
MODEL_NAME = "distilgpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto") #returns an instance of the pretrained gpt2 device_map="auto" tells the model to use all available GPUs or CPUs
model.config.use_cache = True #speeds up training by not storing intermediate results
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True) #we can use autotokenizer to instantiate the original tokenizer used in the pre trained gpt2

#IMPLEMENTING PADDING - padding is a technique used to ensure that all input sequences are of the same length, which is necessary for batch processing in neural networks.
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token, eos_token is the end of sequence token, which is used to indicate the end of a sequence in the model
#the eos token is unique to the models tokenizer, so we can use it as the pad token
tokenizer.padding_side = "right"  # Set padding side to right
#example of right padding
# hello how are you pad (pad is added to the right as indicated by 'tokenizer.padding_side = 'right'' to ensure 
# all sequences are of the same length)
# whats up my old friend
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token id to eos token id

generation_configuration = model.generation_config
generation_configuration.pad_token_id = tokenizer.eos_token_id
generation_configuration.eos_token_id = tokenizer.eos_token_id  # Set eos token id to eos token id
generation_configuration.max_new_tokens = 1024 #this is the maximum number of new tokens that GPT-2 can generate
# this is set to 1024, which is the maximum number of tokens that GPT-2 can generate in a single pass
generation_configuration.temperature = 0.7  # Set temperature to 0.7, which controls the randomness of the model's output
generation_configuration.top_p = 0.90  # Set top_p to 0.90, which controls the diversity of the model's output
generation_configuration.top_k = 20  # Set top_k to 50, which controls the diversity of the model's output

#TRAINING AND TESTING THE MODEL

#funciton to generate the answer based on a prompt
#this function takes a prompt as input, encodes it, generates a response using the model

def generate(prompt): 
    encoded = tokenizer.encode(prompt,add_special_tokens=True, return_tensors="pt").to(device)  # Encode the prompt and convert it to a tensor
    out = model.generate(input_ids=encoded, repetition_penalty = 2.0, do_sample=True) 
    string_decoded = tokenizer.decode(out[0].tolist(), clean_up_tokenization_spaces=True)  # Decode the output tensor to a string
    print(string_decoded)  # Print the generated string

generate('this is')

print("---------------------------") 

generate('how are you') 

#training loop