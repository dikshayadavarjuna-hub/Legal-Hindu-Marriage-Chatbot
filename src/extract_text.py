from pypdf import PdfReader

pdf_path = "data/Hindu_Marriage_Act_1955.pdf"
output_path = "data/hindu_marriage_act.txt"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

with open(output_path, "w", encoding="utf-8") as file:
    file.write(text)

print("PDF text extracted successfully!")
print("Saved to:", output_path)