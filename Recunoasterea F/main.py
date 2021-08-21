# importând toate bibliotecile necesare
import cv2
import torch
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms

# Definirea arhitecturii rețelei neuronale
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=1)
        self.conv1_dropuot = nn.Dropout2d(0.5)
        self.comv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_dropuot = nn.Dropout2d(0.5)
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_dropout = nn.Dropout2d(0.5)
        self.maxpool3 = nn.MaxPool2d(3, stride=1)
        self.linear1 = nn.Linear(3625216, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x):
        out = self.conv1_dropuot(self.maxpool1(self.conv1(x)))
        out = self.conv2_dropuot(self.maxpool2(self.comv2(out)))
        out = self.conv3_dropout(self.maxpool3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

# Încărcarea modelului și setarea acestuia în modul eval
model = torch.load(r'D:\Olesea\2021\IA Curs integrat\IA Practica\IA\Proiectul meu\SNN-main\Olesea.pt')
model.eval()

# Crearea obiectului VideoCapture
video_capture = cv2.VideoCapture(0)

# Configurarea fontului
font = cv2.FONT_HERSHEY_SIMPLEX

# Setarea unei imagini adevărate și a unei imagini de referință. Schimbă cu a ta
ref_true_img_path = r'D:\Olesea\2021\IA Curs integrat\IA Practica\IA\Proiectul meu\SNN-main\photos\ME\ME_1.jpg'
ref_false_img_path = r'D:\Olesea\2021\IA Curs integrat\IA Practica\IA\Proiectul meu\SNN-main\photos\NOT_ME\NOT_ME_1.jpg'

# Citind imagini ca PIL
ref_true_img_pil = Image.open(ref_true_img_path).convert('RGB')
ref_false_img_pil = Image.open(ref_false_img_path).convert('RGB')

# Redimensionarea imaginii
ref_false_img_pil = transforms.Resize((244, 244))(ref_false_img_pil)
ref_true_img_pil = transforms.Resize((244, 244))(ref_true_img_pil)

# Transformarea imaginilor în tensori
ref_false_img_pil = transforms.ToTensor()(ref_false_img_pil)
ref_true_img_pil = transforms.ToTensor()(ref_true_img_pil)

def loss(tested_out, known_out, non_obj_out, alpha):
    norm1 = torch.norm(tested_out - known_out, p=2)
    norm2 = torch.norm(tested_out - non_obj_out, p=2)
    return max(norm1 - norm2 + alpha, torch.zeros(1, requires_grad=True))

# The main loop
while True:
    # Citirea datelor din vide
    ret, frame = video_capture.read()

    # Convertirea frames in tensors
    im_pil = Image.fromarray(frame).convert("RGB")
    im_pil = transforms.Resize((244, 244))(im_pil)
    img_tensor = transforms.ToTensor()(im_pil)

    # Pregătirea tensoarelor pentru rețea
    out_frame = model(img_tensor.unsqueeze(1).permute(1, 0, 2, 3))
    out_true = model(ref_true_img_pil.unsqueeze(1).permute(1, 0, 2, 3))
    out_false = model(ref_false_img_pil.unsqueeze(1).permute(1, 0, 2, 3))

    # Calculul pierderii loss
    loss_param = loss(out_frame, out_true, out_false, alpha=-0.6)
    print(loss_param)

    # În funcție de pierderea tipăririi, se afiseaza dacă sunt eu sau nu
    def function(loss_param):
        if loss_param.data == 0:
            return 'OLESEA'
        else:
            return 'NOT OLESEA'

    # Plasarea textului
    cv2.putText(frame,
                function(loss_param),
                (50, 50),
                font, 1,
                (255, 255, 255),
                2)

    # Se afișează videoclipul de pe cameră
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
