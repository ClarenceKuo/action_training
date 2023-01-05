import torch
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('wide_resnet101_2', checkpoint_path='./savemodel/action_3G_T_best.pth', num_classes=3).to(device)
# checkpoint_path="./savemodel/action_3G_T_best.pth"
# weight = torch.load(checkpoint_path)
# model.load_state_dict(weight)

if __name__ == "__main__":
	import torch
	import time
	tstart = time.time()
	input = torch.rand(1,3,224,224).to(device)
	model = model.to(device)
	output = model(input)

	tend = time.time()
	print(tend-tstart)

