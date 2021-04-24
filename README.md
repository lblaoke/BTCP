# balanced-TCP
A solution to label imbalance in uncertainty estimation using TCP (True Class Probability). This project is an experimental implementation of the paper "Balanced Approach to Making Uncertainty More Distinguishable".

# dataset extension
Override new subclass of torch.utils.data.Dataset  
Implement init(), __getitem__() and __len__()  
Save your code into balanced-TCP/helpers/MyDataset.py

# network extension
Setup a new .py file into balanced-TCP/models/  
Inherit torch.nn.Module  
Implement init() and forward()  
Follow the examples given in balanced-TCP/models/

# experiment extension
Setup a new .yaml file into balanced-TCP/conf/  
Follow the examples given in balanced-TCP/conf/  
Revise balanced-TCP/RUNME.sh for detailed settings
