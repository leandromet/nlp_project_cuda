import re

# Read the file
with open('forestry_indigenous_sort.md', 'r') as f:
    content = f.read()

# Define the absolute path to the screenshots directory
absolute_path = '/media/bndt/db_lin/gitstore/leandromet/nlp_project_cuda/ajuda_doc/screenshots'

# Replace all relative paths with absolute paths
content = re.sub(r'!\[([^\]]+)\]\(screenshots/([^)]+)\)', 
                 f'![\\1]({absolute_path}/\\2)', 
                 content)

# Write the file back
with open('forestry_indigenous_sort.md', 'w') as f:
    f.write(content)

print("All image paths updated to absolute paths!")
