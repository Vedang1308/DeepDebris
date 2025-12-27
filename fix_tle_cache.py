
import re
import os

file_path = "ml-service/main.py"

with open(file_path, "r") as f:
    content = f.read()

# Pattern to match TLE_CACHE block
# Starts with TLE_CACHE = {
# Ends with } on a new line (start of line)
pattern = r"TLE_CACHE = \{.*?\n\}"

# We need re.DOTALL to match across lines
# But be careful not to match too much.
# The TLE_CACHE block ends with a } at indentation 0.

# Let's find the start index
start_marker = "TLE_CACHE = {"
start_idx = content.find(start_marker)

if start_idx == -1:
    print("TLE_CACHE not found!")
    exit(1)

# Find the matching closing brace at indentation 0
# We assume the file is well formatted.
# We'll look for "\n}" after start_idx
end_marker = "\n}"
end_idx = content.find(end_marker, start_idx)

if end_idx == -1:
    print("Closing brace not found!")
    exit(1)

# The content to remove includes the closing brace
# Reconstruct file
new_content = content[:start_idx] + "TLE_CACHE = {}" + content[end_idx+2:]

with open(file_path, "w") as f:
    f.write(new_content)

print(f"Successfully emptied TLE_CACHE in {file_path}")
