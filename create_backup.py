import os
import zipfile
import datetime

def backup_project():
    # Name the backup with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"Fashion_Project_Backup_{timestamp}.zip"
    
    # Files/Folders to include
    to_include = [
        'src', 
        'data', # The dataset
        'artifacts', # The trained model/embeddings
        'main.py', 
        'app.py', 
        'requirements.txt', 
        'README.md',
        'users_db.json'
    ]
    
    print(f"Creating backup: {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in to_include:
            if os.path.exists(item):
                if os.path.isdir(item):
                    for root, dirs, files in os.walk(item):
                        # Skip cache
                        if '__pycache__' in root:
                            continue
                            
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.getcwd())
                            try:
                                print(f"Adding {arcname}")
                            except UnicodeEncodeError:
                                print(f"Adding {arcname.encode('utf-8', 'replace')}")
                            zipf.write(file_path, arcname)
                else:
                    print(f"Adding {item}")
                    zipf.write(item, item)
            else:
                print(f"Warning: {item} not found, skipping.")
                
    print(f"\n[SUCCESS] Backup created successfully: {os.path.abspath(zip_filename)}")

if __name__ == "__main__":
    backup_project()
