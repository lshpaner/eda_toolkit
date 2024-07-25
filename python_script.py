import codecs

input_file = "README_min.md"
output_file = "README_min_utf8.md"

# Read the content of the file with cp1252 encoding
with codecs.open(input_file, "r", "cp1252") as file:
    content = file.read()

# Write the content to a new file with utf-8 encoding
with codecs.open(output_file, "w", "utf-8") as file:
    file.write(content)
