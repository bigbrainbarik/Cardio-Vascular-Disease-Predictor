import tkinter as tk
from tkinter import Toplevel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from predictor import (
    predict_risk,
    all_models,
    performance_df,
    get_confusion_matrix
)

data = pd.read_csv("healthcare_synthetic_data.csv")
TARGET_COLUMN = "Heart_Disease_Risk"

for col in data.columns:
    if "id" in col.lower() or "pid" in col.lower():
        data = data.drop(col, axis=1)

feature_names = data.drop(TARGET_COLUMN, axis=1).columns


def calculate():
    try:
        input_data = {}
        for feature, entry in zip(feature_names, entries):
            val = entry.get()
            try:
                val = float(val)
            except:
                pass
            input_data[feature] = val

        selected = model_var.get()
        prob = predict_risk(input_data, selected)
        percent = round(prob * 100, 2)

        result_label.config(
            text=f"{selected}\nRisk Probability: {percent}%"
        )

        fig, ax = plt.subplots(figsize=(5, 2))

        ax.barh(["Risk"], [percent])

        ax.set_xlim(0, 100)
        ax.set_title(f"Risk Probability: {percent}%")

        ax.text(percent + 1, 0, f"{percent}%", va="center")

        plt.tight_layout()

        canvas_plot = FigureCanvasTkAgg(fig, master=scroll_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack()


    except Exception as e:
        result_label.config(text=str(e))


def show_confusion():
    cm = get_confusion_matrix(model_var.get())
    win = Toplevel(root)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack()


root = tk.Tk()
root.geometry("600x800")
root.title("Cardiovascular Risk Predictor")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas)

scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

tk.Label(scroll_frame, text="Cardiovascular Risk Prediction",
         font=("Arial", 16, "bold")).pack(pady=10)

model_var = tk.StringVar(value="Best Model")

for name in ["Best Model", "Ensemble Model"] + list(all_models.keys()):
    roc_val = performance_df.loc[
        performance_df["Model"] == name, "ROC-AUC"
    ]
    roc_text = ""
    if not roc_val.empty:
        roc_text = f" (ROC-AUC: {round(float(roc_val.values[0]),3)})"

    tk.Radiobutton(scroll_frame,
                   text=name + roc_text,
                   variable=model_var,
                   value=name).pack(anchor="w")

entries = []
for feature in feature_names:
    frame = tk.Frame(scroll_frame)
    frame.pack(pady=3)
    tk.Label(frame, text=feature, width=22,
             anchor="w").pack(side="left")
    e = tk.Entry(frame, width=15)
    e.pack(side="right")
    entries.append(e)

tk.Button(scroll_frame, text="Predict Risk",
          command=calculate).pack(pady=10)

tk.Button(scroll_frame, text="Show Confusion Matrix",
          command=show_confusion).pack(pady=5)

result_label = tk.Label(scroll_frame, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
