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
