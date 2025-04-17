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

Once these key features are installed, open the run.py folder, and proceed with this code configuration:

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
output = generator.generate(tensor, max_new_tokens)
print(tokenizer.decode(output[0].tolist()))
```
