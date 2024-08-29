from PIL import Image, ImageDraw, ImageFont
import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter, convolve

# ASCII characters used to build the output text
ASCII_CHARS = [".", ",", ":", ";", "+", "*", "!", "?", "%", "#", "@"]
EDGE_CHARS = {'horizontal': '-', 'vertical': '|', 'diagonal1': '/', 'diagonal2': '\\'}

def resize_image(image, new_width=100):
    width, height = image.size
    ratio = height / width / 1.65  # Adjust ratio for better aspect
    new_height = int(new_width * ratio)
    resized_image = image.resize((new_width, new_height))
    return resized_image

def grayscale_image(image):
    return image.convert("L")

def difference_of_gaussians(image, sigma1=1, sigma2=2):
    """Apply Difference of Gaussians (DoG)"""
    img_array = np.array(image)
    blurred1 = gaussian_filter(img_array, sigma1)
    blurred2 = gaussian_filter(img_array, sigma2)
    dog = blurred2 - blurred1
    dog = (dog - dog.min()) / (dog.max() - dog.min()) * 255
    return Image.fromarray(dog.astype(np.uint8))

def sobel_filter(image):
    """Apply Sobel filter to detect edges"""
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    img_array = np.array(image)
    img_array = img_array.astype(np.float32)
    
    gradient_x = convolve(img_array, sobel_x)
    gradient_y = convolve(img_array, sobel_y)
    
    edges = np.hypot(gradient_x, gradient_y)
    edges = (edges / edges.max() * 255).astype(np.uint8)
    
    return Image.fromarray(edges)

def map_pixels_to_ascii(image, range_width=25):
    pixels = image.getdata()
    ascii_str = "".join([ASCII_CHARS[pixel // range_width] for pixel in pixels])
    return ascii_str

def convert_image_to_ascii(image, new_width=100):
    original_width, original_height = image.size
    
    image = resize_image(image, new_width)
    ascii_str = map_pixels_to_ascii(grayscale_image(image))
    img_width = image.width
    ascii_str_len = len(ascii_str)
    
    ascii_img = "\n".join([ascii_str[i:i + img_width] for i in range(0, ascii_str_len, img_width)])
    
    # Return the ASCII art and its dimensions
    return ascii_img, original_width, original_height

def save_ascii_image(ascii_image, output_path):
    with open(output_path, "w") as f:
        f.write(ascii_image)

def ascii_to_image(txt_file_path, font_path, output_image_path, original_width, original_height):
    with open(txt_file_path, "r") as f:
        ascii_text = f.read()

    ascii_lines = ascii_text.splitlines()
    max_width = max(len(line) for line in ascii_lines)
    max_height = len(ascii_lines)

    try:
        font = ImageFont.truetype(font_path, size=8)
    except IOError:
        print(f"Could not load font from {font_path}")
        return

    # Get the width and height of a single character
    char_width, char_height = font.getbbox("A")[2:4]

    # Calculate the image size based on the original image dimensions
    image_width = max_width * char_width
    image_height = max_height * char_height

    # Create a new blank image with a black background
    image = Image.new("RGB", (image_width, image_height), color="black")
    draw = ImageDraw.Draw(image)

    # Draw each line of ASCII text onto the image
    for i, line in enumerate(ascii_lines):
        draw.text((0, i * char_height), line, font=font, fill="white")

    # Resize the ASCII image to match the original image's dimensions
    image = image.resize((original_width, original_height), Image.NEAREST)

    # Save the generated image
    image.save(output_image_path)
    print(f"ASCII art saved as an image at {output_image_path}")

def create_edge_ascii_image(image, txt_file_path, edge_file_path):
    gray_image = grayscale_image(image)
    dog_image = difference_of_gaussians(gray_image)
    
    edges = sobel_filter(dog_image)
    edges = edges.convert("L")
    edges = edges.point(lambda p: p > 128 and 255)  # Convert to binary image

    edge_pixels = edges.load()

    # Map edge data to ASCII
    width, height = image.size
    edge_ascii_lines = []
    
    for y in range(height):
        line = ""
        for x in range(width):
            if edge_pixels[x, y] == 255:
                # Determine if the edge is horizontal, vertical or diagonal
                if x > 0 and edge_pixels[x-1, y] == 255:
                    line += EDGE_CHARS['horizontal']
                elif y > 0 and edge_pixels[x, y-1] == 255:
                    line += EDGE_CHARS['vertical']
                elif x > 0 and y > 0 and edge_pixels[x-1, y-1] == 255:
                    line += EDGE_CHARS['diagonal1']
                elif x > 0 and y < height - 1 and edge_pixels[x-1, y+1] == 255:
                    line += EDGE_CHARS['diagonal2']
                else:
                    line += " "
            else:
                line += " "
        edge_ascii_lines.append(line.rstrip())
    
    # Save edge ASCII art to file
    with open(edge_file_path, "w") as f:
        f.write("\n".join(edge_ascii_lines))
    print(f"Edge ASCII art saved to {edge_file_path}")

def main():
    # Check if image file is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: python ascii_art.py <image_path>")
        return
    
    # Get the image file path from the arguments
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"No file found at {image_path}")
        return
    
    # Open the image
    image = Image.open(image_path)

    # Convert the image to ASCII
    ascii_image, original_width, original_height = convert_image_to_ascii(image)
    
    # Determine the output file paths
    base_name, ext = os.path.splitext(image_path)
    ascii_txt_path = f"{base_name}_ascii.txt"
    ascii_image_path = f"{base_name}_ascii.png"
    edge_txt_path = f"{base_name}_edges.txt"

    # Save the ASCII image to a file
    save_ascii_image(ascii_image, ascii_txt_path)
    print(f"ASCII art saved to {ascii_txt_path}")

    # Create edge ASCII art
    create_edge_ascii_image(resize_image(image, 100), ascii_txt_path, edge_txt_path)

    # Convert the ASCII text file to an image
    font_path = "IBM_8x8.ttf"  # Ensure this font file exists in the same directory
    ascii_to_image(ascii_txt_path, font_path, ascii_image_path, original_width, original_height)

if __name__ == "__main__":
    main()
