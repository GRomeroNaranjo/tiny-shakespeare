# tiny-shakespeare
This repository aims to implement the decoder transformer architecture for generating infinite Shakespearean text. Particularly, employing the use of PyTorch and Andrej Karpathy's minbpe library as the primary modules. The trained transformer consists of an embed size of 384, 8 heads, 12 layers, and a block size of 256, where this in turn corresponds to approximatey 22 million parameters, all contributing towards the success of the model. This ReadME file will provide information regarding the technical details of the model, as well as featuring some model generations, and loss training curve. Moreover, this ReadME will outline the key steps in installing this tiny-shakespeare configuration and also training it.


# install
```
pip install torch numpy dataclasses tiktoken os statistics math
```
Dependencies:
- `torch` For the model implementation, and dataset loading.
- `dataclasses` For the config class.
- `tiktoken` For Andrej Karpath's minbpe.
- `statistics` For some stats tools, mean calculations.
- `math` For some stats tools, mean calculations.

Once these key features are installed, open the run.py folder, and proceed with this code configuration, where if implemented correctly this should generate the following text:

```
import torch
from torch import nn
from minbpe.basic import BasicTokenizer

model = GPT_Model(Config())
tokenizer = BasicTokenizer()

tokenizer.load("tokenizer_file")
model.load_state_dict(torch.load("model_file")

input_tokens = "input tokens"
tokens = tokenizer.encode(input_tokens)
tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
output = generator.generate(tensor, max_new_tokens, temperature)
print(tokenizer.decode(output[0].tolist()))
```

# eval
This section will now provide information regarding the models-accuracy. Particularly, it ended up achieving a train loss of 2.4327 and a validation loss of 3.5917, where below you will find an example generation:

```
generator = GPT_Model(Config())

generator.load_state_dict(torch.load("/content/drive/MyDrive/Tiny-Shakespeare/model_parameters.pth"))
input_tokens = "O gentle night, why dost thou weep so soft upon the silent earth?"
tokens = tokenizer.encode(input_tokens)
tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
output = generator.generate(tensor, 100, 1.5)
print(tokenizer.decode(output[0].tolist()))
```
O gentle night, why dost thou weep so soft upon the silent earth?
ANTIPHOLUS OF SYRACUSE. Is not that were, but not till?
DROMIO OF SYRACUSE. No, sir, then I gave it you since you had
  bid me in my gaoler I tell.
ANTIPHOLUS OF SYRACUSE. Why, but I gave it you gave me h

```
generator = GPT_Model(Config())

generator.load_state_dict(torch.load("/content/drive/MyDrive/Tiny-Shakespeare/model_parameters.pth"))
input_tokens = "When moonlight strikes the ivy'd stone, what secrets wake beneath the throne?"
tokens = tokenizer.encode(input_tokens)
tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
output = generator.generate(tensor, 100, 1.5)
print(tokenizer.decode(output[0].tolist()))
```
When moonlight strikes the ivy'd stone, what secrets wake beneath the throne?  
    O, let me seek the curtains of the axe-day;
    and then I see the sun of purposed skulls by
    the sun of a double baser and a double black of purd'ring
    the sun of a dragon, a dragon, a dragon, a drawer

Below you will find the training-curve for both val, and train losses, where both simultaneously decrease progressively serving at an indicator on how this model was not prone to extensive overfitting.
![training-curve](training-curve)
Overall, it is left evident that the model is not without limitations, where it only effectively captures the outer scope of how Shakespeare used to write. However, considering how little this model is, it has demonstrated solid performance altogether.

# attributions
Below you will find some attributions for this project:
- The minbpe library was developed my Andrej Karpathy, an extensively recognized deep learning engineer. For more information, attached will be the minbpe repository [minbpe repository](https://github.com/karpathy/minbpe)
- Furthermore, this project has used The Complete Works of William Shakespeare as the dataset, where the link will be attached [dataset link](https://www.gutenberg.org/ebooks/100)



