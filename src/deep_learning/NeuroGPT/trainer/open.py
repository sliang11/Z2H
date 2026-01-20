import numpy as np

# pred =  np.load('/mnt/hddhelp/sliang/Cihang/results/models/upstream/GPT_lrs-4_hds-16_ChunkLen-500_NumChunks-8_ovlp-50_2025-07-02_21-0/test_predictions.npy')
# label =  np.load('/mnt/hddhelp/sliang/Cihang/results/models/upstream/GPT_lrs-4_hds-16_ChunkLen-500_NumChunks-8_ovlp-50_2025-07-02_21-0/test_label_ids.npy')
# print("Prédictions       :", pred[:10])
# print("Vraies étiquettes :", label[:10])


import pickle

pkl_path = "/mnt/hddhelp/sliang/Cihang/mice_data/230302e-b_0008/230302e-b_0008_0.03_0.06_0.5_1.0_True_0.5.pkl"

# Chargement du fichier
with open(pkl_path, 'rb') as f:
    train_samples, val_samples, test_samples, train_bi_labels, val_bi_labels, test_bi_labels = pickle.load(f)

# Vérification des labels de test
print("Shape of test_bi_labels:", test_bi_labels.shape)
print("Unique labels:", np.unique(test_bi_labels))
print("First 20 labels:", test_bi_labels[:20])

from collections import Counter

label_counts = Counter(test_bi_labels)
print("Répartition des classes dans test_bi_labels :")
for label, count in label_counts.items():
    print(f"Classe {label} : {count} exemples")