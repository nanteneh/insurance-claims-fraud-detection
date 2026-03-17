# Language: Python
# Author: Nahom Anteneh
# Description: Insurance Claims Fraud Detection Application using Tkinter and Machine Learning
# Input to the program: CSV file with insurance claims data using template provided
# Output from the program: Model performance metrics and visualization of ROC curves, and list of flagged fraudulent claims.
# Things that need attention: Ensure the CSV file has the correct format and required columns for analysis, all dependencies are installed.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             roc_curve, confusion_matrix)
import xgboost as xgb
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

class FraudDetectionApp:
    """
    Insurance Claims Fraud Detection Application using Tkinter and Machine Learning.
    Author name: Nahom Anteneh
    Date : March 24, 2025
    Description of class: This class creates a GUI application for loading insurance claims data, analyzing it using machine learning models, and displaying the results.
    """
    def __init__(self, master):
        """
        Purpose: Initialize the GUI application.
        Parameters: master (tk.Tk): The main window of the application.
        Returns: None
        """
        self.master = master
        master.title("Insurance Claims Fraud Detection System")
        master.geometry("1200x900")
        self.style = ttk.Style()
        self.configure_styles()
        
        self.file_path = None
        self.create_widgets()
        self.model_metrics = {}
    
    def configure_styles(self):
        """
        Purpose: Configure the styles for the GUI components.
        Parameters: None
        Returns: None
        """
        self.style.configure('TButton', foreground='black', background='#87CEFA', font=('Arial', 10))
        self.style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Metric.TLabel', font=('Arial', 10))
        self.style.map('TButton', background=[('active', '#63B8FF')])
    
    def create_widgets(self):
        """
        Purpose: Create the GUI widgets for the application.
        Parameters: None
        Returns: None
        """
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        file_frame = ttk.LabelFrame(main_frame, text=" Upload Dataset ", style='Title.TLabel')
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.browse_btn = ttk.Button(file_frame, text="Browse CSV", command=self.load_file)
        self.browse_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze", command=self.analyze_data, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        exit_btn = ttk.Button(btn_frame, text="Exit", command=self.master.destroy)
        exit_btn.pack(side=tk.LEFT, padx=5)

        results_frame = ttk.LabelFrame(main_frame, text=" Model Results ", style='Title.TLabel')
        results_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.models = ['Logistic Regression', 'Random Forest', 'XGBoost']
        self.model_vars = []
        for i, model in enumerate(self.models):
            frame = ttk.LabelFrame(results_frame, text=model)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            ttk.Label(frame, text="Fraudulent Claims:", style='Metric.TLabel').grid(row=0, column=0, sticky='w')
            ttk.Label(frame, text="Non-Fraudulent Claims:", style='Metric.TLabel').grid(row=1, column=0, sticky='w')
            fraud_var = tk.StringVar(value='0')
            non_fraud_var = tk.StringVar(value='0')
            ttk.Label(frame, textvariable=fraud_var).grid(row=0, column=1, sticky='e')
            ttk.Label(frame, textvariable=non_fraud_var).grid(row=1, column=1, sticky='e')

            ttk.Label(frame, text="Accuracy:", style='Metric.TLabel').grid(row=2, column=0, sticky='w')
            ttk.Label(frame, text="AUC-ROC:", style='Metric.TLabel').grid(row=3, column=0, sticky='w')
            ttk.Label(frame, text="F1-Score:", style='Metric.TLabel').grid(row=4, column=0, sticky='w')
            
            acc_var = tk.StringVar(value='-')
            auc_var = tk.StringVar(value='-')
            f1_var = tk.StringVar(value='-')
            
            ttk.Label(frame, textvariable=acc_var).grid(row=2, column=1, sticky='e')
            ttk.Label(frame, textvariable=auc_var).grid(row=3, column=1, sticky='e')
            ttk.Label(frame, textvariable=f1_var).grid(row=4, column=1, sticky='e')
            
            self.model_vars.append({
                'fraud': fraud_var,
                'non_fraud': non_fraud_var,
                'acc': acc_var,
                'auc': auc_var,
                'f1': f1_var
            })

        viz_notebook = ttk.Notebook(main_frame)
        viz_notebook.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.roc_tab = ttk.Frame(viz_notebook)
        self.confusion_tab = ttk.Frame(viz_notebook)
        viz_notebook.add(self.roc_tab, text="ROC Curves")
        viz_notebook.add(self.confusion_tab, text="Confusion Matrices")
        self.viz_canvas = None
        self.confusion_canvas = None

        list_frame = ttk.LabelFrame(main_frame, text=" Top 10 Fraudulent Claims ", style='Title.TLabel')
        list_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        self.listbox = tk.Listbox(list_frame, height=6, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

    def load_file(self):
        """
        Purpose: Load a CSV file containing insurance claims data.
        Parameters: None
        Returns: None
        """
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.file_label.config(text=os.path.basename(self.file_path))
            self.analyze_btn.config(state=tk.NORMAL)

    def analyze_data(self):
        """
        Purpose: Analyze the insurance claims data using machine learning models.
        Parameters: None
        Returns: None
        """
        try:
            df = pd.read_csv(self.file_path)
            df = self.preprocess_data(df)
            
            X = df.drop(['fraud_reported', 'claim_id'], axis=1)
            y = df['fraud_reported']
            claim_ids = df['claim_id']
            
            models = {
                'Logistic Regression': CalibratedClassifierCV(LogisticRegression(max_iter=1000), method='sigmoid', cv=5),
                'Random Forest': CalibratedClassifierCV(RandomForestClassifier(n_estimators=100), method='isotonic', cv=5),
                'XGBoost': CalibratedClassifierCV(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), method='isotonic', cv=5)
            }
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            results = {}
            
            for model_name, model in models.items():
                all_y_true = []
                all_y_pred = []
                all_y_proba = []
                all_claim_ids = []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    claim_ids_val = claim_ids.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_proba = model.predict_proba(X_val)[:, 1]
                    
                    all_y_true.extend(y_val.tolist())
                    all_y_pred.extend(y_pred.tolist())
                    all_y_proba.extend(y_proba.tolist())
                    all_claim_ids.extend(claim_ids_val.tolist())
                
                accuracy = accuracy_score(all_y_true, all_y_pred)
                auc = roc_auc_score(all_y_true, all_y_proba)
                f1 = f1_score(all_y_true, all_y_pred)
                cm = confusion_matrix(all_y_true, all_y_pred)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'f1': f1,
                    'y_true': all_y_true,
                    'y_pred': all_y_pred,
                    'y_proba': all_y_proba,
                    'claim_ids': all_claim_ids,
                    'confusion_matrix': cm
                }
            
            self.update_results_display(results)
            self.save_fraud_results(results)
            self.plot_roc_curves(results)
            self.plot_confusion_matrices(results)
            
            messagebox.showinfo("Success", "Analysis completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def update_results_display(self, results):
        """
        Purpose: Update the GUI with the results of the analysis.
        Parameters: results (dict): Dictionary containing model names and their respective predictions.
        Returns: None
        """
        for idx, model_name in enumerate(self.models):
            res = results[model_name]
            fraud_count = sum(res['y_pred'])
            non_fraud_count = len(res['y_pred']) - fraud_count
            
            self.model_vars[idx]['fraud'].set(str(fraud_count))
            self.model_vars[idx]['non_fraud'].set(str(non_fraud_count))
            
            self.model_vars[idx]['acc'].set(f"{res['accuracy']:.3f}")
            self.model_vars[idx]['auc'].set(f"{res['auc']:.3f}")
            self.model_vars[idx]['f1'].set(f"{res['f1']:.3f}")
        
        self.listbox.delete(0, tk.END)
        all_flagged = set()
        for model_name in self.models:
            res = results[model_name]
            for i, pred in enumerate(res['y_pred']):
                if pred == 1:
                    all_flagged.add(res['claim_ids'][i])
        
        top_flagged = list(all_flagged)[:10]
        for claim_id in top_flagged:
            self.listbox.insert(tk.END, f"Claim ID: {claim_id}")

    def save_fraud_results(self, results):
        """
        Purpose: Save the results of the fraud detection analysis to a CSV file.
        Parameters: results (dict): Dictionary containing model names and their respective predictions.
        Returns: None
        """
        data = {'claim_id': results['Logistic Regression']['claim_ids']}
        for model_name in self.models:
            data[model_name] = results[model_name]['y_pred']
        result_df = pd.DataFrame(data)
        result_df['All_Models'] = result_df[self.models].all(axis=1)
        result_df.to_csv('fraud_results.csv', index=False)

    def plot_roc_curves(self, results):
        """
        Purpose: Plot ROC curves for each model in the results.
        Parameters: results (dict): Dictionary containing model names and their respective ROC data.
        Returns: None
        """
        if self.viz_canvas:
            self.viz_canvas.get_tk_widget().destroy()
        
        fig = plt.figure(figsize=(7, 4))
        for model_name in self.models:
            res = results[model_name]
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {res['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()

        self.viz_canvas = FigureCanvasTkAgg(fig, master=self.roc_tab)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_confusion_matrices(self, results):
        """"
        Purpose: Plot confusion matrices for each model in the results.
        Parameters: results (dict): Dictionary containing model names and their respective confusion matrices.
        Returns: None"""
        if self.confusion_canvas:
            self.confusion_canvas.get_tk_widget().destroy()
        
        #fig = plt.figure(figsize=(15, 5))
        fig = plt.figure(figsize=(8, 3))
        for idx, model_name in enumerate(self.models):
            cm = results[model_name]['confusion_matrix']
            ax = fig.add_subplot(1, 3, idx+1)
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            #plt.title(f'{model_name} Confusion Matrix')
            ax.set_title(f'{model_name} Confusion Matrix', fontsize=8, pad=10)
            fig.colorbar(cax, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Non-Fraud', 'Fraud'],fontsize=6)
            ax.set_yticklabels(['Non-Fraud', 'Fraud'],fontsize=6)
            ax.set_xlabel('Predicted',fontsize=6)
            ax.set_ylabel('Actual',fontsize=6)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.tight_layout()

        self.confusion_canvas = FigureCanvasTkAgg(fig, master=self.confusion_tab)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def preprocess_data(self, df):
        """
        Purpose: Preprocess the insurance claims data for analysis.
        Parameters: df (pd.DataFrame): The input DataFrame with insurance claims data.
        Returns: pd.DataFrame: The preprocessed DataFrame ready for analysis.
        """
    
        df['claim_id'] = np.arange(len(df)) + 1
        
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['days_policy_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
        df = df.drop(columns=['policy_bind_date', 'incident_date'])
        
        df.replace('?', np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        numeric_cols = [col for col in numeric_cols if col not in ['claim_id', 'fraud_reported']]
        cat_cols = [col for col in cat_cols if col not in ['fraud_reported', 'claim_id']]
        
        df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
        df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
        
        for col in numeric_cols:
            z = np.abs(stats.zscore(df[col]))
            df[col] = np.where(z > 3, df[col].median(), df[col])
        
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
        
        return df

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionApp(root)
    root.mainloop()
