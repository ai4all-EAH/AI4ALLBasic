import nbformat
import sys

def lock_cells(notebook_path):
   with open(notebook_path, 'r', encoding='utf-8') as f:
       nb = nbformat.read(f, as_version=4)
   
   locked_count = 0
   
   for cell in nb.cells:
       if 'tags' in cell.metadata and 'locked' in cell.metadata.tags:
           # Lösche vorhandene metadata
           cell.metadata.clear()
           # Setze nur die erforderlichen
           cell.metadata.tags = ['locked']
           cell.metadata.editable = False
           cell.metadata.deletable = False
           locked_count += 1
   
   with open(notebook_path, 'w', encoding='utf-8') as f:
       nbformat.write(nb, f)
       
   print(f"Erfolgreich abgeschlossen!")
   print(f"- {locked_count} Zellen wurden gesperrt")
   print(f"- Gespeichert als: {notebook_path}")

if __name__ == "__main__":
   if len(sys.argv) > 1:
       lock_cells(sys.argv[1])
