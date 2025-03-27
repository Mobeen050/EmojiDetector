import pandas as pd
from PIL import Image
import io
import base64

def base64_to_image(base64_string):
    base64_string = base64_string[22:]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

# Function to process the dataframe
def process_images(dataframe, number_of_entries):
    # Assuming the 5th column has index 4 (0-based indexing)
    for base64_string in dataframe.iloc[:number_of_entries, 4]:
        image = base64_to_image(base64_string)
        image.show() # Display the image
        # image.save("filename.png")

# Read the CSV file (or any other data source)
data = pd.read_csv("./emojis.csv")

# Process the first X entries of the 5th column
X = 5  
process_images(data, X)
