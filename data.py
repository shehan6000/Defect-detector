import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os

# Function to generate synthetic image data
def generate_image_data(num_samples, img_size=(150, 150)):
    data = []
    labels = []
    for i in range(num_samples):
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Randomly decide if the image is defective or not
        if np.random.rand() > 0.5:
            label = 1  # defective
            # Add random shapes to create defects
            for _ in range(np.random.randint(1, 5)):
                shape = np.random.choice(['circle', 'rectangle'])
                if shape == 'circle':
                    x, y, r = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1]), np.random.randint(5, 20)
                    draw.ellipse((x-r, y-r, x+r, y+r), fill='black')
                elif shape == 'rectangle':
                    x1, y1, x2, y2 = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1]), np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
                    draw.rectangle((x1, y1, x2, y2), fill='black')
        else:
            label = 0  # non-defective

        data.append(np.array(img).flatten())
        labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Generate synthetic data
num_samples = 1000
img_data, img_labels = generate_image_data(num_samples)

# Save as a CSV file
df = pd.DataFrame({
    'image': [list(img) for img in img_data],
    'label': img_labels
})
csv_path = '/mnt/data/synthetic_defect_data.csv'
df.to_csv(csv_path, index=False)

# Save a few sample images for verification
img_dir = '/mnt/data/synthetic_images/'
os.makedirs(img_dir, exist_ok=True)
for i in range(10):  # Save only first 10 images for inspection
    img = Image.fromarray(img_data[i].reshape(150, 150, 3))
    img.save(f'{img_dir}img_{i}.png')

csv_path, img_dir
