from PIL import Image, ImageDraw
import os
import random

def create_image(path, color, shape='rectangle'):
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a shape to differentiate "texture" slightly if ResNet cares, 
    # but mostly color is the dominant feature here.
    if shape == 'rectangle':
        draw.rectangle([50, 50, 174, 174], fill=color)
    elif shape == 'circle':
        draw.ellipse([50, 50, 174, 174], fill=color)
    elif shape == 'triangle':
        draw.polygon([(112, 50), (50, 174), (174, 174)], fill=color)
        
    img.save(path)

dirs = ['data/tops', 'data/bottoms', 'data/shoes']
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Colors
colors = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'black': (0, 0, 0),
    'white': (200, 200, 200),
    'grey': (128, 128, 128),
    'brown': (165, 42, 42),
    'purple': (128, 0, 128),
    'orange': (255, 165, 0)
}

# Generate Tops (Circle shapes)
tops_colors = ['red', 'blue', 'green', 'yellow', 'white', 'purple', 'orange']
for c in tops_colors:
    create_image(f'data/tops/{c}_top.jpg', colors[c], shape='circle')

# Generate Bottoms (Rectangle shapes)
bottoms_colors = ['black', 'grey', 'blue', 'brown', 'white', 'khaki', 'navy']
# reusing some colors map keys. khaki/navy approximated
colors['khaki'] = (240, 230, 140)
colors['navy'] = (0, 0, 128)
for c in bottoms_colors:
    val = colors.get(c, (50, 50, 50))
    create_image(f'data/bottoms/{c}_pants.jpg', val, shape='rectangle')

# Generate Shoes (Triangle shapes)
shoes_colors = ['white', 'black', 'brown', 'red', 'blue', 'grey', 'orange']
for c in shoes_colors:
    create_image(f'data/shoes/{c}_shoes.jpg', colors[c], shape='triangle')

print("Created expanded dummy dataset with 21 images.")
