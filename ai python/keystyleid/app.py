"""
Main application with tkinter GUI for keystroke dynamics.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import sys

# Import our modules
from features import validate_typed_text
from train import train_model
from predict import predict_user


PROMPT_TEXT = "the quick brown fox jumps over the lazy dog"
SESSIONS_FILE = 'data/sessions.json'


class KeystrokeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Dynamics - MVP")
        self.root.geometry("800x600")
        
        # Current session data
        self.current_events: List[Dict[str, Any]] = []
        self.session_start_time = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.collect_frame = ttk.Frame(self.notebook)
        self.train_frame = ttk.Frame(self.notebook)
        self.predict_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.collect_frame, text="Collect")
        self.notebook.add(self.train_frame, text="Train")
        self.notebook.add(self.predict_frame, text="Predict")
        
        self.setup_collect_tab()
        self.setup_train_tab()
        self.setup_predict_tab()
    
    def setup_collect_tab(self):
        """Setup the data collection tab."""
        # User selection
        user_frame = ttk.Frame(self.collect_frame)
        user_frame.pack(pady=10)
        
        ttk.Label(user_frame, text="User Label:").pack(side=tk.LEFT, padx=5)
        self.user_var = tk.StringVar(value="A")
        user_radio_frame = ttk.Frame(user_frame)
        user_radio_frame.pack(side=tk.LEFT)
        for label in ["A", "B", "C"]:
            ttk.Radiobutton(user_radio_frame, text=label, variable=self.user_var, 
                          value=label).pack(side=tk.LEFT, padx=5)
        
        # Prompt display
        prompt_frame = ttk.LabelFrame(self.collect_frame, text="Type this text:")
        prompt_frame.pack(fill=tk.X, padx=20, pady=10)
        
        prompt_label = ttk.Label(prompt_frame, text=PROMPT_TEXT, 
                                font=("Arial", 12), wraplength=700)
        prompt_label.pack(pady=10)
        
        # Text input area
        input_frame = ttk.LabelFrame(self.collect_frame, text="Your typing:")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_input = tk.Text(input_frame, height=5, font=("Arial", 11))
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind key events
        self.text_input.bind('<KeyPress>', self.on_key_down)
        self.text_input.bind('<KeyRelease>', self.on_key_up)
        self.text_input.bind('<FocusIn>', self.on_focus_in)
        
        # Buttons
        button_frame = ttk.Frame(self.collect_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.collect_status = ttk.Label(self.collect_frame, text="Ready to collect data")
        self.collect_status.pack(pady=5)
    
    def setup_train_tab(self):
        """Setup the training tab."""
        # Instructions
        info_frame = ttk.LabelFrame(self.train_frame, text="Instructions")
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_text = ("1. Collect at least 10 sessions per user in the 'Collect' tab\n"
                    "2. Click 'Train Model' to train the classifier\n"
                    "3. The model will be saved to artifacts/model.joblib")
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Train button
        button_frame = ttk.Frame(self.train_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Train Model", command=self.train_model_action,
                  style="Accent.TButton").pack(padx=10)
        
        # Results area
        results_frame = ttk.LabelFrame(self.train_frame, text="Training Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.train_results = scrolledtext.ScrolledText(results_frame, height=10, 
                                                       state=tk.DISABLED)
        self.train_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_predict_tab(self):
        """Setup the prediction tab."""
        # Prompt display
        prompt_frame = ttk.LabelFrame(self.predict_frame, text="Type this text:")
        prompt_frame.pack(fill=tk.X, padx=20, pady=10)
        
        prompt_label = ttk.Label(prompt_frame, text=PROMPT_TEXT, 
                                font=("Arial", 12), wraplength=700)
        prompt_label.pack(pady=10)
        
        # Text input area
        input_frame = ttk.LabelFrame(self.predict_frame, text="Your typing:")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.predict_text_input = tk.Text(input_frame, height=5, font=("Arial", 11))
        self.predict_text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind key events
        self.predict_text_input.bind('<KeyPress>', self.on_predict_key_down)
        self.predict_text_input.bind('<KeyRelease>', self.on_predict_key_up)
        self.predict_text_input.bind('<FocusIn>', self.on_predict_focus_in)
        
        # Buttons
        button_frame = ttk.Frame(self.predict_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_predict_input).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Predict", command=self.predict_action,
                  style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        # Results area
        results_frame = ttk.LabelFrame(self.predict_frame, text="Prediction Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.predict_results = scrolledtext.ScrolledText(results_frame, height=8, 
                                                         state=tk.DISABLED)
        self.predict_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Prediction events
        self.predict_events: List[Dict[str, Any]] = []
        self.predict_start_time = None
    
    def on_focus_in(self, event):
        """Reset session when text input gets focus."""
        self.current_events = []
        self.session_start_time = datetime.now().timestamp() * 1000  # milliseconds
    
    def on_key_down(self, event):
        """Record keydown event."""
        if self.session_start_time is None:
            self.session_start_time = datetime.now().timestamp() * 1000
        
        current_time = datetime.now().timestamp() * 1000
        relative_time = current_time - self.session_start_time
        
        # Get the key character
        key = event.char if event.char else event.keysym
        
        self.current_events.append({
            'key': key,
            'type': 'down',
            't': relative_time
        })
    
    def on_key_up(self, event):
        """Record keyup event."""
        if self.session_start_time is None:
            return
        
        current_time = datetime.now().timestamp() * 1000
        relative_time = current_time - self.session_start_time
        
        # Get the key character
        key = event.char if event.char else event.keysym
        
        self.current_events.append({
            'key': key,
            'type': 'up',
            't': relative_time
        })
    
    def on_predict_focus_in(self, event):
        """Reset prediction session when text input gets focus."""
        self.predict_events = []
        self.predict_start_time = datetime.now().timestamp() * 1000
    
    def on_predict_key_down(self, event):
        """Record keydown event for prediction."""
        if self.predict_start_time is None:
            self.predict_start_time = datetime.now().timestamp() * 1000
        
        current_time = datetime.now().timestamp() * 1000
        relative_time = current_time - self.predict_start_time
        
        key = event.char if event.char else event.keysym
        
        self.predict_events.append({
            'key': key,
            'type': 'down',
            't': relative_time
        })
    
    def on_predict_key_up(self, event):
        """Record keyup event for prediction."""
        if self.predict_start_time is None:
            return
        
        current_time = datetime.now().timestamp() * 1000
        relative_time = current_time - self.predict_start_time
        
        key = event.char if event.char else event.keysym
        
        self.predict_events.append({
            'key': key,
            'type': 'up',
            't': relative_time
        })
    
    def clear_input(self):
        """Clear the input text and reset events."""
        self.text_input.delete('1.0', tk.END)
        self.current_events = []
        self.session_start_time = None
        self.collect_status.config(text="Ready to collect data")
    
    def clear_predict_input(self):
        """Clear the prediction input and reset events."""
        self.predict_text_input.delete('1.0', tk.END)
        self.predict_events = []
        self.predict_start_time = None
        self.predict_results.config(state=tk.NORMAL)
        self.predict_results.delete('1.0', tk.END)
        self.predict_results.config(state=tk.DISABLED)
    
    def save_session(self):
        """Save the current typing session."""
        typed_text = self.text_input.get('1.0', tk.END).strip()
        user = self.user_var.get()
        
        if not typed_text:
            messagebox.showwarning("Warning", "Please type something first.")
            return
        
        # Validate typed text
        if not validate_typed_text(typed_text, PROMPT_TEXT):
            messagebox.showwarning("Warning", 
                                 "Typed text doesn't match the prompt well enough.\n"
                                 "Please type the exact prompt text.")
            return
        
        if len(self.current_events) < 4:
            messagebox.showwarning("Warning", "Not enough key events captured.")
            return
        
        # Load existing sessions
        sessions = []
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, 'r') as f:
                sessions = json.load(f)
        
        # Create new session
        new_session = {
            'user': user,
            'events': self.current_events.copy(),
            'typed_text': typed_text,
            'created_at': datetime.now().isoformat()
        }
        
        sessions.append(new_session)
        
        # Save to file
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        self.collect_status.config(
            text=f"Session saved! Total sessions for user {user}: "
            f"{sum(1 for s in sessions if s['user'] == user)}"
        )
        
        # Clear for next session
        self.text_input.delete('1.0', tk.END)
        self.current_events = []
        self.session_start_time = None
        
        messagebox.showinfo("Success", f"Session saved for user {user}!")
    
    def train_model_action(self):
        """Train the model and display results."""
        self.train_results.config(state=tk.NORMAL)
        self.train_results.delete('1.0', tk.END)
        self.train_results.insert(tk.END, "Training model...\n")
        self.train_results.update()
        
        try:
            # Change to the script directory to ensure relative paths work
            original_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            result = train_model()
            
            if result['success']:
                self.train_results.insert(tk.END, "Training successful!\n\n")
                self.train_results.insert(tk.END, f"Accuracy: {result['accuracy']:.2%}\n")
                self.train_results.insert(tk.END, f"Total sessions: {result['total_sessions']}\n")
                self.train_results.insert(tk.END, f"Train size: {result['train_size']}\n")
                self.train_results.insert(tk.END, f"Test size: {result['test_size']}\n\n")
                self.train_results.insert(tk.END, "User counts:\n")
                for user, count in result['user_counts'].items():
                    self.train_results.insert(tk.END, f"  User {user}: {count} sessions\n")
                messagebox.showinfo("Success", "Model trained successfully!")
            else:
                self.train_results.insert(tk.END, f"Training failed: {result['error']}\n")
                if 'user_counts' in result:
                    self.train_results.insert(tk.END, "\nCurrent user counts:\n")
                    for user, count in result['user_counts'].items():
                        self.train_results.insert(tk.END, f"  User {user}: {count} sessions\n")
                messagebox.showerror("Error", result['error'])
            
            os.chdir(original_dir)
        except Exception as e:
            self.train_results.insert(tk.END, f"Error: {str(e)}\n")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
        
        self.train_results.config(state=tk.DISABLED)
    
    def predict_action(self):
        """Predict the user based on current typing."""
        typed_text = self.predict_text_input.get('1.0', tk.END).strip()
        
        if not typed_text:
            messagebox.showwarning("Warning", "Please type something first.")
            return
        
        # Validate typed text
        if not validate_typed_text(typed_text, PROMPT_TEXT):
            messagebox.showwarning("Warning", 
                                 "Typed text doesn't match the prompt well enough.\n"
                                 "Please type the exact prompt text.")
            return
        
        if len(self.predict_events) < 4:
            messagebox.showwarning("Warning", "Not enough key events captured.")
            return
        
        self.predict_results.config(state=tk.NORMAL)
        self.predict_results.delete('1.0', tk.END)
        self.predict_results.insert(tk.END, "Predicting...\n")
        self.predict_results.update()
        
        try:
            # Change to the script directory
            original_dir = os.getcwd()
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            result = predict_user(self.predict_events)
            
            if result['success']:
                self.predict_results.delete('1.0', tk.END)
                self.predict_results.insert(tk.END, "Prediction Results:\n\n")
                self.predict_results.insert(tk.END, f"Predicted User: {result['predicted_user']}\n")
                self.predict_results.insert(tk.END, f"Confidence: {result['confidence']:.2%}\n\n")
                self.predict_results.insert(tk.END, "Probabilities:\n")
                for user, prob in sorted(result['probabilities'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    self.predict_results.insert(tk.END, f"  User {user}: {prob:.2%}\n")
            else:
                self.predict_results.delete('1.0', tk.END)
                self.predict_results.insert(tk.END, f"Error: {result['error']}\n")
                messagebox.showerror("Error", result['error'])
            
            os.chdir(original_dir)
        except Exception as e:
            self.predict_results.delete('1.0', tk.END)
            self.predict_results.insert(tk.END, f"Error: {str(e)}\n")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
        
        self.predict_results.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = KeystrokeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()

