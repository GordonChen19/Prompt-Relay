from pdf2image import convert_from_path
import os

path = "."

for file in os.listdir(path):
    if file.endswith(".pdf"):
        full_path = os.path.join(path, file)
        # Convert PDF to PNG
        pages = convert_from_path(full_path, dpi=300)
        for i, page in enumerate(pages):
            output_file = os.path.join(path, f"{os.path.splitext(file)[0]}.png")
            page.save(output_file, "PNG")

#ffmpeg -i kling3.mp4 -vf "fps=30" kling3.gif