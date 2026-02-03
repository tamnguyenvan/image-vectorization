import os
import shutil
import subprocess
from pathlib import Path

OUTPUT_DIR = "gemini_files"

def should_ignore(path, ignore_list):
    """Check if the given path should be ignored based on ignore list."""
    path = str(path).lower()
    return any(ignore.lower() in path for ignore in ignore_list)

def convert_files_to_txt(directory, extensions, ignore_list=None):
    """
    Convert all files with specified extensions to .txt files and save to gemini_files/ directory
    while maintaining the original directory structure.
    
    Args:
        directory (str): Root directory to start the conversion
        extensions (list): List of file extensions to convert (without leading dot)
        ignore_list (list, optional): List of paths to ignore (can be file or folder names)
    """
    if ignore_list is None:
        ignore_list = []
        
    # Convert extensions to lowercase for case-insensitive comparison
    extensions = [ext.lower() for ext in extensions]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(directory, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through the directory
    for root, dirs, files in os.walk(directory, topdown=True):
        # Calculate relative path from source directory
        rel_path = os.path.relpath(root, directory)
        if rel_path == '.':
            rel_path = ''
            
        # Create corresponding directory in output
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Skip ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_list)]
        
        for file in files:
            # Skip ignored files
            file_path = os.path.join(root, file)
            if should_ignore(file_path, ignore_list):
                print(f"Ignoring: {file_path}")
                continue
                
            # Get file extension (without dot)
            file_ext = Path(file).suffix[1:].lower()
            
            # Check if file has one of the target extensions
            if file_ext in extensions:
                # Get full path of source file
                src_path = os.path.join(root, file)
                
                # Create new path with .txt extension in the output directory
                new_file = os.path.splitext(file)[0] + '.txt'
                dst_path = os.path.join(current_output_dir, new_file)
                
                try:
                    # Copy the file with new extension to the output directory
                    shutil.copy2(src_path, dst_path)
                    print(f"Converted: {src_path} -> {dst_path}")
                except Exception as e:
                    print(f"Error converting {src_path}: {str(e)}")

def generate_project_structure(directory, output_file=f'{OUTPUT_DIR}/project_structure.md'):
    """
    Generate a markdown file containing the project structure using the 'tree' command.
    
    Args:
        directory (str): Root directory to generate structure for
        output_file (str): Name of the output markdown file
    """
    try:
        # Run the tree command
        result = subprocess.run(
            ['tree', '--noreport', '--dirsfirst', '-I', '__pycache__|*.pyc|*.pyo|*.pyd|*.egg-info|.git|.idea|venv|node_modules'],
            cwd=directory,
            capture_output=True,
            text=True
        )
        
        # Format as markdown
        markdown_content = f"# Project Structure\n\n```\n{result.stdout}\n```"
        
        # Write to file
        with open(os.path.join(directory, output_file), 'w') as f:
            f.write(markdown_content)
            
        print(f"Project structure saved to: {os.path.join(directory, output_file)}")
        return True
        
    except FileNotFoundError:
        print("Error: 'tree' command not found. Please install it first.")
        print("On Ubuntu/Debian: sudo apt install tree")
        print("On macOS: brew install tree")
        return False
    except Exception as e:
        print(f"Error generating project structure: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    target_directory = os.getcwd()  # Current directory, change as needed
    
    # List of extensions to convert (without leading dot)
    extensions_to_convert = ['py', 'md', 'txt']  # Add or remove extensions as needed
    
    # List of files/folders to ignore (case-insensitive partial match)
    ignore_list = [
        'venv',
        '__pycache__',
        '.git',
        '.idea',
        'node_modules',
        "clone.py",
        "gemini_files"
        # Add more ignore patterns as needed
    ]
    
    print(f"Starting conversion of files with extensions: {', '.join(extensions_to_convert)}")
    print(f"Output directory: {os.path.join(target_directory, 'gemini_files')}")
    print(f"Ignoring paths containing: {', '.join(ignore_list)}\n")
    
    # Generate project structure
    print("Generating project structure...")
    if generate_project_structure(target_directory, "ge"):
        # Only proceed with conversion if project structure was generated successfully
        convert_files_to_txt(target_directory, extensions_to_convert, ignore_list)
        print(f"\nConversion completed. Files saved to: {os.path.join(target_directory, 'gemini_files')}")
    else:
        print("Skipping file conversion due to previous errors.")