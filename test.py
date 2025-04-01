import base64

# Open the image file and encode it to Base64
with open("/Users/saijeevan/Desktop/Minor Project/WhatsApp Image 2025-03-30 at 18.57.55.jpeg", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode("utf-8")

print(base64_string)
