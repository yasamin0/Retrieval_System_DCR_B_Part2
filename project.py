import os
import nltk
import string
import pickle
from collections import defaultdict
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import messagebox, ttk

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Define stop words and custom stop list
stop_words = set(stopwords.words('english'))
custom_stop_list = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

# Step 1: Read Documents
def read_documents(directory):
    documents = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".html"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        html_content = file.read()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text = soup.get_text()
                        documents[file_path] = text  # Using file path as the key to handle files with the same name
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    print(f"Documents read: {len(documents)}")
    return documents

# Step 2: Preprocess Text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation and word not in stop_words and word not in custom_stop_list]
    return tokens

def get_synonyms(term):
    synonyms = set()
    for syn in wordnet.synsets(term):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Step 3: Create Inverted Index and BiWord Index
def create_inverted_and_biword_index(documents):
    inverted_index = defaultdict(list)
    biword_index = defaultdict(list)
    
    for doc_id, text in documents.items():
        tokens = preprocess(text)
        for token in set(tokens):
            inverted_index[token].append(doc_id)
        for i in range(len(tokens) - 1):
            biword = tokens[i] + " " + tokens[i + 1]
            biword_index[biword].append(doc_id)
    
    print(f"Inverted index created with {len(inverted_index)} terms.")
    print(f"BiWord index created with {len(biword_index)} terms.")
    return inverted_index, biword_index

# Step 4: Save and Load Indexes
def save_index(index, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(index, file)
        print(f"Index saved to {filename}")
    except Exception as e:
        print(f"Error saving index to {filename}: {e}")

def load_index(filename):
    try:
        with open(filename, 'rb') as file:
            index = pickle.load(file)
            print(f"Index loaded with {len(index)} terms.")
            return index
    except Exception as e:
        print(f"Error loading index from {filename}: {e}")
        return None

# Step 5: Search for Terms with AND/OR Logic and Synonyms, including BiWord and Skip List
def search_terms_with_synonyms(terms, index, biword_index, query_type='AND'):
    expanded_terms = {term: get_synonyms(term) for term in terms}
    all_terms = set(terms)
    for term in terms:
        all_terms.update(expanded_terms[term])
    
    print(f"Expanded terms for {terms}: {all_terms}")

    # For single term, always use OR logic
    if len(terms) == 1:
        query_type = 'OR'
    
    if query_type == 'AND':
        if not all_terms:
            return []
        initial_terms = [term for term in all_terms if term in index]
        if not initial_terms:
            return []
        result = set(index[initial_terms[0]])
        for term in initial_terms[1:]:
            result.intersection_update(set(index.get(term, [])))
            print(f"Result after intersecting with {term}: {result}")
    elif query_type == 'OR':
        result = set()
        for term in all_terms:
            result.update(set(index.get(term, [])))
            print(f"Result after adding {term}: {result}")
    
    # Prepare results with the found term or synonym
    results_with_terms = []
    original_term_str = ' '.join(terms)
    for doc_id in result:
        for term in terms:
            if term in index and doc_id in index[term]:
                results_with_terms.append((doc_id, term, "No", "No", original_term_str))
                break
            for synonym in expanded_terms[term]:
                if synonym in index and doc_id in index[synonym]:
                    results_with_terms.append((doc_id, synonym, "No", "No", original_term_str))
                    break

    # Add code for biword index processing
    biword_results = []
    biwords = [terms[i] + " " + terms[i + 1] for i in range(len(terms) - 1)]
    for biword in biwords:
        if biword in biword_index:
            for doc_id in biword_index[biword]:
                biword_results.append((doc_id, biword, "Yes", "No", original_term_str))
    results_with_terms.extend(biword_results)
    
    # Add code for skip list processing
    skiplist_results = []
    for term in all_terms:
        if term in index:
            postings = index[term]
            skiplist_postings = postings[::2]  # Simple skip every second entry
            for doc_id in skiplist_postings:
                skiplist_results.append((doc_id, term, "No", "Yes", original_term_str))
    results_with_terms.extend(skiplist_results)
    
    return results_with_terms

# GUI for Searching
def search_gui():
    loaded_index = load_index('inverted_index.pkl')  # Load the index from the file
    biword_index = load_index('biword_index.pkl')  # Load the BiWord index from the file
    if not loaded_index or not biword_index:
        return

    def search():
        terms = entry_terms.get().strip().split()
        query_type = query_type_var.get().strip().upper()
        search_results = search_terms_with_synonyms(terms, loaded_index, biword_index, query_type)  # Use the new function
        for i in tree.get_children():
            tree.delete(i)
        if search_results:
            for result, found_term, used_biword, used_skiplist, original_term in search_results:
                tree.insert('', 'end', values=(result, found_term, used_biword, used_skiplist, original_term))
        else:
            messagebox.showinfo("Search Results", f"No documents found containing the terms '{' '.join(terms)}' with {query_type} query.")

    root = tk.Tk()
    root.title("Document Retrieval System")

    tk.Label(root, text="Enter terms to search (separated by spaces):").pack(pady=5)
    entry_terms = tk.Entry(root, width=50)
    entry_terms.pack(pady=5)

    tk.Label(root, text="Enter query type (AND/OR):").pack(pady=5)
    query_type_var = tk.StringVar(value="AND")
    query_type_menu = ttk.Combobox(root, textvariable=query_type_var, values=["AND", "OR"], state="readonly")
    query_type_menu.pack(pady=5)

    tk.Button(root, text="Search", command=search).pack(pady=20)

    # Create a treeview for displaying the search results
    columns = ('Location', 'Found Term', 'Used BiWord', 'Used SkipList', 'Original Term')
    tree = ttk.Treeview(root, columns=columns, show='headings')
    tree.heading('Location', text='Location')
    tree.heading('Found Term', text='Found Term')
    tree.heading('Used BiWord', text='Used BiWord')
    tree.heading('Used SkipList', text='Used SkipList')
    tree.heading('Original Term', text='Original Term')
    tree.pack(pady=20, fill='both', expand=True)

    root.mainloop()

# Main Function
def main():
    documents = read_documents(r'C:\Level1')  # Read documents from 'C:\Level1' folder
    if documents:
        inverted_index, biword_index = create_inverted_and_biword_index(documents)  # Create the inverted and biword indexes
        save_index(inverted_index, 'inverted_index.pkl')  # Save the inverted index to a file
        save_index(biword_index, 'biword_index.pkl')  # Save the biword index to a file

    # Run the GUI search interface
    search_gui()

if __name__ == "__main__":
    main()
