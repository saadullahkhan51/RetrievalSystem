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
        self.master.geometry("800x600")
        self.master.minsize(600, 400)

        self.config = Config("config/config.yaml")
        self.retrieval_system = EmbeddingRetrievalSystem(self.config)

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        self.create_widgets()

    def configure_styles(self):
        self.style.configure('TButton', padding=5, font=('Helvetica', 10))
        self.style.configure('TLabel', font=('Helvetica', 10))
        self.style.configure('TEntry', padding=5)
        self.style.configure('Treeview', rowheight=25)
        self.style.configure('TNotebook.Tab', padding=(10, 5))

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(expand=True, fill="both")

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

        self.create_query_tab()
        self.create_add_docs_tab()
        self.create_view_docs_tab()

        self.status_bar = ttk.Label(self.main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(self.main_frame, text="Rebuild Index", command=self.rebuild_index).pack(pady=10)

    def create_query_tab(self):
        query_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(query_frame, text="Query")

        query_frame.columnconfigure(1, weight=1)

        ttk.Label(query_frame, text="Query:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.query_entry = ttk.Entry(query_frame, width=50)
        self.query_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(query_frame, text="Top K:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.k_entry = ttk.Entry(query_frame, width=5)
        self.k_entry.grid(row=0, column=3, sticky="w", padx=5, pady=5)

        ttk.Button(query_frame, text="Search", command=self.run_query).grid(row=1, column=0, columnspan=4, pady=10)

        self.query_result = tk.Text(query_frame, height=15, width=80, wrap=tk.WORD)
        self.query_result.grid(row=2, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(query_frame, orient="vertical", command=self.query_result.yview)
        scrollbar.grid(row=2, column=4, sticky="ns")
        self.query_result.configure(yscrollcommand=scrollbar.set)

        query_frame.rowconfigure(2, weight=1)

    def create_add_docs_tab(self):
        add_docs_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(add_docs_frame, text="Add Documents")

        ttk.Button(add_docs_frame, text="Select Directory", command=self.add_documents).pack(pady=20)
        self.add_docs_label = ttk.Label(add_docs_frame, text="")
        self.add_docs_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(add_docs_frame, orient="horizontal", length=300, mode="indeterminate")
        self.progress_bar.pack(pady=10)

    def create_view_docs_tab(self):
        view_docs_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(view_docs_frame, text="View Documents")

        self.docs_tree = ttk.Treeview(view_docs_frame, columns=("Filename", "Path"), show="headings")
        self.docs_tree.heading("Filename", text="Filename")
        self.docs_tree.heading("Path", text="Path")
        self.docs_tree.column("Filename", width=200)
        self.docs_tree.column("Path", width=400)
        self.docs_tree.pack(expand=True, fill="both", pady=10)

        scrollbar = ttk.Scrollbar(view_docs_frame, orient="vertical", command=self.docs_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.docs_tree.configure(yscrollcommand=scrollbar.set)

        ttk.Button(view_docs_frame, text="Refresh", command=self.refresh_documents).pack(pady=10)

    def run_query(self):
        query = self.query_entry.get()
        k = self.k_entry.get()
        if query:
            self.status_bar.config(text="Searching...")
            self.master.update_idletasks()
            response = asyncio.run(self.retrieval_system.generate_response(query, k))
            self.query_result.delete(1.0, tk.END)
            self.query_result.insert(tk.END, response)
            self.status_bar.config(text="Search completed")
        else:
            messagebox.showwarning("Empty Query", "Please enter a query.")

    def add_documents(self):
        directory = filedialog.askdirectory()
        if directory:
            self.progress_bar.start()
            self.add_docs_label.config(text="Adding documents...")
            self.master.update_idletasks()
            asyncio.run(self.retrieval_system.add_documents(directory))
            self.progress_bar.stop()
            self.add_docs_label.config(text=f"Documents added from {directory}")
            self.refresh_documents()
            self.status_bar.config(text="Documents added successfully")
        else:
            self.add_docs_label.config(text="No directory selected")

    def refresh_documents(self):
        self.docs_tree.delete(*self.docs_tree.get_children())
        documents = self.retrieval_system.db.get_all_documents()
        for doc in documents:
            self.docs_tree.insert("", "end", values=(doc['filename'], doc.get('filepath', 'N/A')))
        self.status_bar.config(text="Document list refreshed")

    def rebuild_index(self):
        self.status_bar.config(text="Rebuilding index...")
        self.master.update_idletasks()
        asyncio.run(self.retrieval_system.rebuild_index())
        messagebox.showinfo("Index Rebuilt", "The index has been successfully rebuilt from the database.")
        self.status_bar.config(text="Index rebuilt successfully")

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