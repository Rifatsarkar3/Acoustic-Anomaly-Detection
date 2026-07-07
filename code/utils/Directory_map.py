import os
from pathlib import Path

def generate_directory_map(root_dir=".", output_filename="directory_map.txt"):
    root_path = Path(root_dir)
    
    # Ignore these folders to save AI tokens and keep the map clean
    ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.vscode', '.idea', 'node_modules'}
    
    # Ignore the output file itself so it doesn't map itself recursively
    ignore_files = {output_filename, '.DS_Store', 'generate_map.py'}

    # Using 'w' mode ensures the file is completely overwritten every time it runs
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Project Directory Map\n")
        f.write("="*40 + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root_path):
            # Modifying dirnames in-place tells os.walk to skip our ignored directories entirely
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            
            # Calculate the depth of the current folder to create the visual tree
            path = Path(dirpath)
            depth = len(path.relative_to(root_path).parts)
            
            indent = "    " * depth
            
            if depth == 0:
                f.write(f"📁 {path.absolute().name}/ (Root)\n")
            else:
                f.write(f"{indent}📁 {path.name}/\n")
            
            # Write the files within the current folder
            sub_indent = "    " * (depth + 1)
            for filename in sorted(filenames):
                if filename not in ignore_files:
                    f.write(f"{sub_indent}📄 {filename}\n")

if __name__ == "__main__":
    # Executes the function
    generate_directory_map()
    print("Map updated! Check 'directory_map.txt'")