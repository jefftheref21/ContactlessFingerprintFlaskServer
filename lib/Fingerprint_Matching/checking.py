from graimatr_model import SwinModel as Model
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda")
model = Model().to(device)
ckpt = torch.load("/home/bhavinja/GradioDemoFingerprint/updated_demo/lib/Fingerprint_Matching/Models/ridgebase_train_vanilla_49_0.9111709286675639.pt", map_location=torch.device('cpu'))
model.load_state_dict(ckpt,strict=False)
print("Number of Trainable Parameters: - ", count_parameters(model))