#set a part of network into determined status
def set_trainable(layer,status):
	for child in layer.children():
		for parameter in child.parameters():
			parameter.requires_grad = status
