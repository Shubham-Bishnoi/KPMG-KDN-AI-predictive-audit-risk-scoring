import os

PLOT_DIR = "../backend/plots"  # Update the directory path if needed

# List all available files in the PLOT_DIR
if os.path.exists(PLOT_DIR):
    print("✅ Available plots in the directory:")
    print(os.listdir(PLOT_DIR))
else:
    print("⚠️ The directory does not exist! Ensure the plots are being saved correctly.")
