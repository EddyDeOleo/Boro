import os
from PIL import Image

# Paths
out_dir = r'C:\Users\User\source\repos\Boro\demo-video\out'
contact_path = os.path.join(out_dir, 'brand_review_contact_sheet.png')

# Expected scene order
scene_files = [f'scene{i}.png' for i in range(1,9)]
images = []
for fn in scene_files:
    path = os.path.join(out_dir, fn)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing {path}')
    img = Image.open(path)
    images.append(img)

# Assume all images same size
w, h = images[0].size
# Create canvas 4 columns x 2 rows
canvas = Image.new('RGB', (w*4, h*2))
for idx, img in enumerate(images):
    row = idx // 4
    col = idx % 4
    canvas.paste(img, (col*w, row*h))

canvas.save(contact_path)
print('Contact sheet saved to', contact_path)
