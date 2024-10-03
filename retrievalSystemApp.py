import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import asyncio
import os
from src.config import Config
from src.retrieval_system import EmbeddingRetrievalSystem

class RetrievalSystemApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Document Retrieval System")
        self.master.geometry("600x400")

        self.config = Config("config/config.yaml")
        self.retrieval_system = EmbeddingRetrievalSystem(self.config)

        self.create_widgets()

    def create_widgets(self):
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill="both")

        # Query tab
        self.query_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.query_frame, text="Query")
        self.create_query_widgets()

        # Add Documents tab
        self.add_docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.add_docs_frame, text="Add Documents")
        self.create_add_docs_widgets()

        # View Documents tab
        self.view_docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.view_docs_frame, text="View Documents")
        self.create_view_docs_widgets()

        ttk.Button(self.master, text="Rebuild Index", command=self.rebuild_index).pack(pady=10)

    def create_query_widgets(self):
        ttk.Label(self.query_frame, text="Enter your query:").pack(pady=10)
        self.query_entry = ttk.Entry(self.query_frame, width=50)
        self.query_entry.pack(pady=10)
        ttk.Button(self.query_frame, text="Search", command=self.run_query).pack(pady=10)
        self.query_result = tk.Text(self.query_frame, height=10, width=70)
        self.query_result.pack(pady=10)

    def create_add_docs_widgets(self):
        ttk.Button(self.add_docs_frame, text="Select Directory", command=self.add_documents).pack(pady=20)
        self.add_docs_label = ttk.Label(self.add_docs_frame, text="")
        self.add_docs_label.pack(pady=10)

    def create_view_docs_widgets(self):
        self.docs_listbox = tk.Listbox(self.view_docs_frame, height=15, width=70)
        self.docs_listbox.pack(pady=20)
        ttk.Button(self.view_docs_frame, text="Refresh", command=self.refresh_documents).pack()

    def run_query(self):
        query = self.query_entry.get()
        if query:
            response = asyncio.run(self.retrieval_system.generate_response(query))
            self.query_result.delete(1.0, tk.END)
            self.query_result.insert(tk.END, response)
        else:
            messagebox.showwarning("Empty Query", "Please enter a query.")

    def add_documents(self):
        directory = filedialog.askdirectory()
        if directory:
            asyncio.run(self.retrieval_system.add_documents(directory))
            self.add_docs_label.config(text=f"Documents added from {directory}")
            self.refresh_documents()
        else:
            self.add_docs_label.config(text="No directory selected")

    def refresh_documents(self):
        self.docs_listbox.delete(0, tk.END)
        documents = self.retrieval_system.db.get_all_documents()
        for doc in documents:
            self.docs_listbox.insert(tk.END, doc['filename'])

    def rebuild_index(self):
        asyncio.run(self.retrieval_system.rebuild_index())
        messagebox.showinfo("Index Rebuilt", "The index has been successfully rebuilt from the database.")

    def on_closing(self):
        self.retrieval_system.close()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = RetrievalSystemApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()