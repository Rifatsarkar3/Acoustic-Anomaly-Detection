import os
import tkinter as tk
from tkinter import filedialog

def combine_text_files():
    # Hide the main, empty tkinter window
    root = tk.Tk()
    root.withdraw()

    # 1. Ask the user to select the input files
    print("Opening file dialog... Please select the text files you want to merge.")
    input_files = filedialog.askopenfilenames(
        title="Select Text Files to Merge",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    if not input_files:
        print("No files selected. Exiting script.")
        return

    # 2. Ask the user where to save the merged file
    print(f"Selected {len(input_files)} files. Now, choose where to save the merged document.")
    output_file = filedialog.asksaveasfilename(
        title="Save Merged File As",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    if not output_file:
        print("No save location selected. Exiting script.")
        return

    # A list of encodings to try, from most common to most forgiving
    encodings_to_try = ['utf-8', 'cp1252', 'latin-1']

    # 3. Combine the files
    try:
        # We always want the final output file to be clean UTF-8
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_name in input_files:
                if os.path.exists(file_name):
                    content = None
                    
                    # Try to read the file using our list of encodings
                    for enc in encodings_to_try:
                        try:
                            with open(file_name, 'r', encoding=enc) as infile:
                                content = infile.read()
                            # If it succeeds without error, break out of the encoding loop
                            break 
                        except UnicodeDecodeError:
                            # If it fails, silently continue to the next encoding in the list
                            continue
                    
                    # If all encodings fail (very rare), force read and replace bad characters with '?'
                    if content is None:
                        with open(file_name, 'r', encoding='utf-8', errors='replace') as infile:
                            content = infile.read()
                            print(f"Note: Some unrecognized characters in '{os.path.basename(file_name)}' were replaced.")

                    # Write the successfully extracted content to the master file
                    outfile.write(content)
                    outfile.write("\n\n") 
                    
                    print(f"Successfully added: '{os.path.basename(file_name)}'")
                else:
                    print(f"Warning: '{file_name}' was not found and has been skipped.")
                    
        print(f"\nSuccess! Your merged file has been saved to:\n{output_file}")

    except Exception as e:
        print(f"An error occurred while merging the files: {e}")

# --- Execute the Script ---
if __name__ == "__main__":
    combine_text_files()