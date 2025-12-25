import sys
import os
import re

def replace_markdown(file_path):
    # Read the content of the markdown file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Backup the original file
    backup_path = file_path + '.backup'
    try:
        os.rename(file_path, backup_path)
    except Exception as e:
        print(f"Error creating backup: {e}")
        return

    # Replace \(content\) with $content$
    content = re.sub(r'\\\((\s*?.*?[^\\]\s*?)\\\)', r'$\1$', content, flags=re.DOTALL)

    # Replace \[content\] with $$content$$
    content = re.sub(r'\\\[(.*?[^\\])\\\]', r'$$\1$$', content, flags=re.DOTALL)

    # Write the modified content back to the original file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully processed {file_path} and created backup at {backup_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        # Restore backup if writing fails
        try:
            os.rename(backup_path, file_path)
            print(f"Restored original file from backup due to write error.")
        except Exception as restore_error:
            print(f"Error restoring backup: {restore_error}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replace_markdown.py file.md")
    else:
        replace_markdown(sys.argv[1])