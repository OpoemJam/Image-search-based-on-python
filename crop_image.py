from PIL import Image

def crop_image(path, cropped_path, p1, p2, s1, s2):
 image = Image.open(path)
 cropped = image.crop((p1, p2, s1, s2))
 cropped.save(cropped_path)