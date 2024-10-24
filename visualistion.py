import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def open_txt_file(file):
    with open(file, 'r') as file:
        float_list = [float(line.strip()) for line in file]
    return float_list

GCN_accuracy = open_txt_file('test_accuracies_of_GCN.txt')
tem_conv_accuracy = open_txt_file('test_accuracies_of_temp_convo_model.txt')
GAT_accuracy = open_txt_file('test_accuracies_of_GAT.txt')
SAGE_accuracy = open_txt_file('test_accuracies_of_SAGE.txt')


tem_conv_loss = open_txt_file('train_loss_of_temp_convo_model.txt')
GCN_loss = open_txt_file('train_loss_of_GCN.txt')
GAT_loss = open_txt_file('train_loss_of_GAT.txt')
SAGE_loss = open_txt_file('train_loss_of_SAGE.txt')


# Accuracies DataFrame
accuracy_data = pd.DataFrame({
    'Index': range(len(GCN_accuracy)),  # Assuming all lists have the same length
    'GCN Accuracy': GCN_accuracy,
    'Temporal Convolution Accuracy': tem_conv_accuracy,
    'GAT Accuracy': GAT_accuracy,
    'SAGE Accuracy': SAGE_accuracy
})

loss_data = pd.DataFrame({
    'Index': range(len(GCN_loss)),  # Assuming all lists have the same length
    'GCN Loss': GCN_loss,
    'Temporal Convolution Loss': tem_conv_loss,
    'GAT Loss': GAT_loss,
    'SAGE Loss': SAGE_loss
})

accuracy_melted = accuracy_data.melt(id_vars=['Index'], value_vars=['GCN Accuracy', 'Temporal Convolution Accuracy', 'GAT Accuracy', 'SAGE Accuracy'],
                                     var_name='Model', value_name='Accuracy')

loss_melted = loss_data.melt(id_vars=['Index'], value_vars=['GCN Loss', 'Temporal Convolution Loss', 'GAT Loss', 'SAGE Loss'],
                             var_name='Model', value_name='Loss')

# Convert the lists to NumPy arrays for easier calculations
GCN_accuracy = np.array(GCN_accuracy)
tem_conv_accuracy = np.array(tem_conv_accuracy)
GAT_accuracy = np.array(GAT_accuracy)
SAGE_accuracy = np.array(SAGE_accuracy)

# Calculate means from the 5th epoch onward since we assume the first 5 are "adjusting" epochs
GCN_mean = GCN_accuracy[5:].mean()
tem_conv_mean = tem_conv_accuracy[5:].mean()
GAT_mean = GAT_accuracy[5:].mean()
SAGE_mean = SAGE_accuracy[5:].mean()

# Print the means for each model
print(f"GCN Accuracy Mean (from 5th epoch): {GCN_mean}")
print(f"Temporal Convolution Accuracy Mean (from 5th epoch): {tem_conv_mean}")
print(f"GAT Accuracy Mean (from 5th epoch): {GAT_mean}")
print(f"SAGE Accuracy Mean (from 5th epoch): {SAGE_mean}")

# Plotting the Accuracy Graph
sns.set(style='whitegrid')  # Set a nice background grid

# plt.figure(figsize=(10, 6))
sns.lineplot(x='Index', y='Accuracy', hue='Model', data=accuracy_melted, marker='o')

plt.title('Model Accuracies', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(title='Model')
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(x='Index', y='Loss', hue='Model', data=loss_melted, marker='o')

plt.title('Model Losses', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(title='Model')
plt.show()