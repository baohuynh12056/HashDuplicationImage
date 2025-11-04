import io
import os
import subprocess  # <-- ADDED for opening folder
import sys
import threading  # Multithread for running GUI
import time
import traceback  # For error reporting
from collections import defaultdict
from tkinter import filedialog

import customtkinter as ctk

# --- Your Project Imports ---
import MyHash
from Application.cluster import build_cluster_faiss, build_clusters
from Application.feature_extract import mean_extract_image_features_batch_1


# This class captures print() statements and redirects them to the GUI
# We use it mainly for printing error messages
class StdoutRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, s):
        # Ensure GUI updates happen on the main thread
        self.text_widget.after(0, self.text_widget.insert, ctk.END, s)
        self.text_widget.after(0, self.text_widget.see, ctk.END)

    def flush(self):
        pass


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Deduplication Tool")
        self.geometry("700x600")  # Made slightly taller
        ctk.set_appearance_mode("System")  # "Dark", "Light"

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Main output area takes space

        # --- Top Controls Frame ---
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(
            row=0, column=0, padx=10, pady=10, sticky="nsew"
        )
        self.frame_controls.grid_columnconfigure(1, weight=1)

        # --- Main Output Frame ---
        self.frame_output = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_output.grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="nsew"
        )
        self.frame_output.grid_columnconfigure(0, weight=1)
        self.frame_output.grid_columnconfigure(1, weight=1)  # Two columns
        self.frame_output.grid_rowconfigure(0, weight=1)

        # --- Control Widgets ---
        self.label_algo = ctk.CTkLabel(
            self.frame_controls, text="Choose Algorithm:"
        )
        self.label_algo.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.algo_var = ctk.StringVar(value="FAISS")
        self.algo_menu = ctk.CTkOptionMenu(
            self.frame_controls,
            variable=self.algo_var,
            values=["FAISS", "HashTable", "BloomFilter", "SimHash", "MinHash"],
        )
        self.algo_menu.grid(
            row=0, column=1, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.label_dir = ctk.CTkLabel(self.frame_controls, text="Image Folder:")
        self.label_dir.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.entry_dir = ctk.CTkEntry(
            self.frame_controls,
            placeholder_text="Enter your dataset's path",
        )
        self.entry_dir.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="ew")

        self.browse_button = ctk.CTkButton(
            self.frame_controls,
            text="Browse...",
            command=self.browse_folder,
            width=80,  # Fixed width for browse button
        )
        self.browse_button.grid(row=1, column=2, padx=10, pady=10)

        # --- Processing Feedback ---
        self.run_button = ctk.CTkButton(
            self.frame_controls,
            text="Start Clustering",
            command=self.start_processing_thread,
        )
        self.run_button.grid(
            row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew"
        )

        # Determinate Progress Bar (fills from 0% to 100%)
        self.progress_determinate = ctk.CTkProgressBar(
            self.frame_controls, mode="determinate"
        )
        self.progress_determinate.set(0)  # Start at 0
        self.progress_determinate.grid_remove()  # Hide it initially

        # Progress status label
        self.label_progress_status = ctk.CTkLabel(
            self.frame_controls, text="Starting...", anchor="w"
        )
        self.label_progress_status.grid_remove()  # Hide it initially

        # --- Output Area Widgets ---

        # Left Side: Final Report
        self.frame_results = ctk.CTkFrame(self.frame_output)
        self.frame_results.grid(
            row=0, column=0, padx=(0, 5), pady=0, sticky="nsew"
        )
        self.frame_results.grid_rowconfigure(1, weight=1)
        self.frame_results.grid_columnconfigure(0, weight=1)

        self.label_results_title = ctk.CTkLabel(
            self.frame_results,
            text="Final Report",
            font=ctk.CTkFont(weight="bold"),
        )
        self.label_results_title.grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )

        self.textbox_results = ctk.CTkTextbox(self.frame_results, wrap="word")
        self.textbox_results.grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="nsew"
        )
        self.textbox_results.insert(
            "end", "Results will appear here when processing is complete."
        )
        self.textbox_results.configure(state="disabled")  # Make read-only

        # Right Side: Visual Chart
        self.frame_chart = ctk.CTkFrame(self.frame_output)
        self.frame_chart.grid(
            row=0, column=1, padx=(5, 0), pady=0, sticky="nsew"
        )
        self.frame_chart.grid_columnconfigure(0, weight=1)

        self.label_chart_title = ctk.CTkLabel(
            self.frame_chart,
            text="Visual Chart",
            font=ctk.CTkFont(weight="bold"),
        )
        self.label_chart_title.grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )

        self.label_correct = ctk.CTkLabel(
            self.frame_chart, text="Correct (0%):", anchor="w"
        )
        self.label_correct.grid(
            row=1, column=0, padx=10, pady=(5, 0), sticky="ew"
        )

        self.progress_correct = ctk.CTkProgressBar(
            self.frame_chart, progress_color="#4CAF50"
        )  # Green
        self.progress_correct.set(0)
        self.progress_correct.grid(
            row=2, column=0, padx=10, pady=(0, 10), sticky="ew"
        )

        self.label_wrong = ctk.CTkLabel(
            self.frame_chart, text="Wrong (0%):", anchor="w"
        )
        self.label_wrong.grid(
            row=3, column=0, padx=10, pady=(10, 0), sticky="ew"
        )

        self.progress_wrong = ctk.CTkProgressBar(
            self.frame_chart, progress_color="#F44336"
        )  # Red
        self.progress_wrong.set(0)
        self.progress_wrong.grid(
            row=4, column=0, padx=10, pady=(0, 10), sticky="ew"
        )

        # --- NEW: Open Folder Button ---
        self.open_folder_button = ctk.CTkButton(
            self.frame_chart,
            text="Open Results Folder",
            command=self.open_results_folder,
            state="disabled",  # Start disabled
        )
        self.open_folder_button.grid(
            row=5, column=0, padx=10, pady=(15, 10), sticky="ew"
        )
        # --- End of New Button ---

        # --- Threading & State ---
        self.processing_thread = None
        self.last_results = None  # To store results from the thread
        self.last_cluster_dir = None  # <-- ADDED: Store path to results

        # Redirect stdout (for errors) to the results box
        self.stdout_redirector = StdoutRedirector(self.textbox_results)

    # --- Class Methods ---

    def browse_folder(self):
        """Open a dialog to select a directory."""
        folder_path = filedialog.askdirectory()
        if folder_path:  # If user selected a folder (didn't cancel)
            self.entry_dir.delete(0, ctk.END)  # Clear old content
            self.entry_dir.insert(0, folder_path)  # Insert new path

    def animate_progress_to(self, target_value, status_text):
        """Safely animates the progress bar to a new target value."""
        self.label_progress_status.configure(text=status_text)
        current_value = self.progress_determinate.get()

        # Calculate steps for a ~250ms animation
        steps = 10

        # Avoid division by zero if steps is 0
        increment = (
            (target_value - current_value) / steps
            if steps > 0
            else target_value
        )

        # Start the animation loop
        self._animate_step(current_value, increment, target_value, steps)

    def _animate_step(self, current_val, increment, target_val, steps_left):
        """Internal helper for the animation loop."""
        if steps_left <= 0:
            self.progress_determinate.set(
                target_val
            )  # Ensure it lands precisely
            return

        new_val = current_val + increment
        self.progress_determinate.set(new_val)

        self.after(
            25,
            self._animate_step,
            new_val,
            increment,
            target_val,
            steps_left - 1,
        )

    def reset_chart(self):
        """Resets the visual chart to zero."""
        self.textbox_results.configure(state="normal")  # Enable writing
        self.textbox_results.delete("1.0", ctk.END)
        self.textbox_results.insert("end", "Processing... Please wait.\n")
        self.textbox_results.configure(state="disabled")  # Disable writing

        self.label_correct.configure(text="Correct (0%):")
        self.progress_correct.set(0)
        self.label_wrong.configure(text="Wrong (0%):")
        self.progress_wrong.set(0)

    def animate_results(self, correct_pct, wrong_pct, step=0):
        """Animates the progress bars from 0 to their target value."""
        total_steps = 25  # 25 steps over 250ms (10ms per step)
        if step <= total_steps:
            # Calculate current position
            current_correct = (correct_pct / total_steps) * step
            current_wrong = (wrong_pct / total_steps) * step

            # Update GUI
            self.label_correct.configure(
                text=f"Correct ({current_correct*100:.1f}%):"
            )
            self.progress_correct.set(current_correct)

            self.label_wrong.configure(
                text=f"Wrong ({current_wrong*100:.1f}%):"
            )
            self.progress_wrong.set(current_wrong)

            # Schedule next step
            self.after(
                10, self.animate_results, correct_pct, wrong_pct, step + 1
            )
        else:
            # Ensure final values are precise
            self.label_correct.configure(
                text=f"Correct ({correct_pct*100:.1f}%):"
            )
            self.progress_correct.set(correct_pct)

            self.label_wrong.configure(text=f"Wrong ({wrong_pct*100:.1f}%):")
            self.progress_wrong.set(wrong_pct)

    def start_processing_thread(self):
        # Show determinate progress bar and label
        self.run_button.grid_remove()  # Hide run button
        self.progress_determinate.grid(
            row=2, column=0, columnspan=3, padx=10, pady=(10, 0), sticky="ew"
        )
        self.progress_determinate.set(0)  # Reset to 0
        self.label_progress_status.grid(
            row=3, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew"
        )
        self.label_progress_status.configure(
            text="Starting..."
        )  # Reset progress text

        # Clear old results and disable open button
        self.reset_chart()
        self.last_results = None
        self.last_cluster_dir = None  # <-- ADDED: Reset path
        self.open_folder_button.configure(
            state="disabled"
        )  # <-- ADDED: Disable button

        # Get values from GUI
        self.selected_algo = self.algo_var.get()
        self.img_dir = self.entry_dir.get()

        if not self.img_dir or not os.path.isdir(self.img_dir):
            self.textbox_results.configure(state="normal")
            self.textbox_results.delete("1.0", ctk.END)
            self.textbox_results.insert(
                "end", "ERROR: Please select a valid image folder path.\n"
            )
            self.textbox_results.configure(state="disabled")

            # Hide progress bar, show button
            self.progress_determinate.grid_remove()
            self.label_progress_status.grid_remove()
            self.run_button.grid(  # Show run button again
                row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew"
            )
            return

        # Create and start the new thread
        # This prevents the GUI from freezing
        self.processing_thread = threading.Thread(
            target=self.run_backend_logic,
            daemon=True,  # Automatically close thread when app exits
        )
        self.processing_thread.start()

    def run_backend_logic(self):
        """
        This function runs in a separate thread.
        DO NOT UPDATE GUI DIRECTLY FROM HERE.
        Use `self.after(0, ...)` to schedule GUI updates.
        """

        # Redirect print() statements (mainly for errors)
        original_stdout = sys.stdout
        sys.stdout = self.stdout_redirector

        try:
            # --- Step 1: Feature Extraction ---
            self.after(
                0,
                self.animate_progress_to,
                0.4,
                "Loading/Extracting Features...",
            )

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            FEATURE_FILE = os.path.join(BASE_DIR, "features.npy")
            NAME_FILE = os.path.join(BASE_DIR, "filenames.npy")

            features, filenames = mean_extract_image_features_batch_1(
                img_dir=self.img_dir,
                feature_file=FEATURE_FILE,
                name_file=NAME_FILE,
            )

            if features.size == 0:
                raise Exception(
                    "No features loaded/extracted. Check folder or file paths."
                )

            feature_dim = features.shape[1]

            # --- Step 2: Clustering ---
            self.after(
                0, self.animate_progress_to, 0.85, "Clustering Images..."
            )
            # time.sleep(0.5) # Removed this delay

            cluster_dir = "clusters"  # Default
            if self.selected_algo == "FAISS":
                cluster_dir = "clusters_faiss"
                build_cluster_faiss(
                    features,
                    filenames,
                    self.img_dir,
                    cluster_dir,
                    threshold=0.75,
                    K=10,
                )

            elif self.selected_algo == "HashTable":
                cluster_dir = "clusters_hash"
                ht = MyHash.HashTable(32, feature_dim)
                build_clusters(
                    ht, features, filenames, self.img_dir, 5, cluster_dir
                )

            elif self.selected_algo == "BloomFilter":
                cluster_dir = "clusters_bloom"
                ht1 = MyHash.BloomFilter(36, feature_dim, 9)
                build_clusters(
                    ht1, features, filenames, self.img_dir, 7, cluster_dir
                )

            elif self.selected_algo == "SimHash":
                cluster_dir = "clusters_simhash"
                ht2 = MyHash.SimHash(128)
                build_clusters(
                    ht2, features, filenames, self.img_dir, 13, cluster_dir
                )

            elif self.selected_algo == "MinHash":
                cluster_dir = "clusters_minhash"
                ht3 = MyHash.MinHash(128)
                build_clusters(
                    ht3, features, filenames, self.img_dir, 580, cluster_dir
                )

            # --- ADDED: Store the cluster path for the "Open Folder" button ---
            self.last_cluster_dir = cluster_dir
            # --- End of change ---

            # --- Step 3: Evaluation ---
            self.after(
                0, self.animate_progress_to, 1.0, "Evaluating Results..."
            )
            # time.sleep(0.5) # Removed this delay

            # Use the internal evaluate function
            results = self._evaluate_by_image(cluster_dir)

            # Store results to be picked up by the main thread
            self.last_results = results

        except Exception as e:
            # Store the error
            error_message = (
                f"❌ AN ERROR OCCURRED ❌\n\n{e}\n\nFull Traceback:\n"
            )
            self.last_results = {"error": error_message}
            # Print full traceback to the textbox
            traceback.print_exc()

        finally:
            # Restore stdout
            sys.stdout = original_stdout

            # Tell the main thread we are done
            self.after(0, self.processing_done)

    def processing_done(self):
        """
        This function is called by the main thread after the backend is finished.
        It's safe to update the GUI here.
        """
        # Hide progress bar, show button
        self.progress_determinate.grid_remove()
        self.label_progress_status.grid_remove()
        self.run_button.grid(
            row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew"
        )

        # Re-enable textbox for writing
        self.textbox_results.configure(state="normal")
        self.textbox_results.delete("1.0", ctk.END)  # Clear "Processing..."

        if self.last_results and "error" in self.last_results:
            # Show the error message
            self.textbox_results.insert("end", self.last_results["error"])

        elif self.last_results:
            # --- Process and Show Results ---
            total = self.last_results.get("total", 0)
            correct = self.last_results.get("correct", 0)
            wrong = self.last_results.get("wrong", 0)
            accuracy = self.last_results.get("accuracy", 0)

            # 1. Update Text Report
            report_text = (
                f"Evaluation Complete:\n\n"
                f"Accuracy:      {accuracy:.2f}%\n"
                f"Total Images:  {total}\n"
                f"Correct:       {correct}\n"
                f"Wrong:         {wrong}\n"
            )
            self.textbox_results.insert("end", report_text)

            # 2. Animate Visual Chart
            correct_pct = correct / total if total > 0 else 0
            wrong_pct = wrong / total if total > 0 else 0
            self.animate_results(correct_pct, wrong_pct)

            # 3. --- ADDED: Enable "Open Folder" button if path exists ---
            if self.last_cluster_dir:
                self.open_folder_button.configure(state="normal")

        else:
            # Should not happen, but good to check
            self.textbox_results.insert(
                "end", "Processing finished with no results."
            )

        # Disable textbox again
        self.textbox_results.configure(state="disabled")

    # --- NEW: Function to open the folder ---
    def open_results_folder(self):
        """Opens the cluster results directory in the OS file explorer."""
        if not self.last_cluster_dir:
            return  # Should not happen if button is disabled

        # Ensure the path is absolute
        path = os.path.abspath(self.last_cluster_dir)

        if not os.path.isdir(path):
            self.textbox_results.configure(state="normal")
            self.textbox_results.delete("1.0", ctk.END)
            self.textbox_results.insert(
                "end", f"Error: Cannot find directory:\n{path}"
            )
            self.textbox_results.configure(state="disabled")
            return

        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # "linux", "linux2", etc.
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.textbox_results.configure(state="normal")
            self.textbox_results.delete("1.0", ctk.END)
            self.textbox_results.insert("end", f"Error opening folder:\n{e}")
            self.textbox_results.configure(state="disabled")

    # --- End of new function ---

    def _evaluate_by_image(self, base_dir):
        """
        Internal copy of the evaluation logic.
        This function runs in the worker thread.
        It returns a dictionary with all necessary stats.
        """
        if not os.path.isdir(base_dir):
            raise Exception(
                f"Evaluation failed: Directory not found '{base_dir}'"
            )

        label_to_clusters = defaultdict(lambda: defaultdict(int))
        total_images = 0
        all_image_paths = []

        # Step 1a: Find all images and count totals
        for group_name in sorted(os.listdir(base_dir)):
            group_path = os.path.join(base_dir, group_name)
            if not os.path.isdir(group_path):
                continue

            for filename in os.listdir(group_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    total_images += 1
                    label = filename.split("_")[0]
                    label_to_clusters[label][group_name] += 1
                    all_image_paths.append((filename, label, group_name))

        if total_images == 0:
            return {"total": 0, "correct": 0, "wrong": 0, "accuracy": 0}

        # Step 2: Find the 'correct' cluster for each label
        label_best_cluster = {
            label: max(cluster_counts.items(), key=lambda x: x[1])[0]
            for label, cluster_counts in label_to_clusters.items()
        }

        # Step 3: Count correct/wrong based on the best cluster
        correct = 0
        wrong = 0

        for filename, label, group_name in all_image_paths:
            correct_cluster = label_best_cluster.get(label)
            if correct_cluster and group_name == correct_cluster:
                correct += 1
            else:
                wrong += 1

        accuracy = (correct / total_images * 100) if total_images else 0

        # Return a dictionary
        return {
            "total": total_images,
            "correct": correct,
            "wrong": wrong,
            "accuracy": accuracy,
        }


if __name__ == "__main__":
    app = App()
    app.mainloop()
