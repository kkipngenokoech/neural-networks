from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

img = Image.open('../data/cat.jpg')

# I want to display the image
plt.imshow(img)
plt.savefig('cat.jpg')

# I want to convert the image to a tensor
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# I want to display the tensor
img_tensor = transform(img).unsqueeze(0)
print(img_tensor)