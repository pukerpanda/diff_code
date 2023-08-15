
from torchvision import datasets, transforms, models
from cnns.model import Linear


from resnet.model import ResNet34
from resnet.utils import copy_weights

def get_resnet_for_feature_extraction(n_classes: int) -> ResNet34:
	'''
	Creates a ResNet34 instance, replaces its final linear layer with a classifier
	for `n_classes` classes, and freezes all weights except the ones in this layer.

	Returns the ResNet model.
	'''
	# Create a ResNet34 with the default number of classes
	my_resnet = ResNet34()

	# Load the pretrained weights
	pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

	# Copy the weights over
	my_resnet = copy_weights(my_resnet, pretrained_resnet)

	# Freeze gradients for all layers (note that when we redefine the last layer, it will be unfrozen)
	my_resnet.requires_grad_(False)

	# Redefine last layer
	my_resnet.out_layers[-1] = Linear(
		my_resnet.out_features_per_group[-1],
		n_classes
	)

	return my_resnet
