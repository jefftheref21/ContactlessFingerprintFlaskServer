import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json

def softmax_with_temperature(x, temp):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

def compute_recall_at_k(similarity_matrix, p_labels, g_labels, k):
    num_probes = len(p_labels)
    recall_at_k = 0.0
    for i in range(num_probes):
        probe_label = p_labels[i]
        sim_scores = similarity_matrix[i]
        sorted_indices = torch.argsort(sim_scores, descending=True)
        top_k_indices = sorted_indices[:k]
        correct_in_top_k = any(g_labels[idx] == probe_label for idx in top_k_indices)
        recall_at_k += correct_in_top_k
    recall_at_k /= num_probes
    return recall_at_k

contactless_base_path = "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task2/Test/Contactless/"
contactbased_base_path = "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task2/Test/Contactbased/"

# cl2cl
with open("contactless_task2_files.txt") as file:
    task2_contactless_filenames = sorted([contactless_base_path + f.strip() for f in file.readlines()])

filename_mat = []

for i,file1 in enumerate(task2_contactless_filenames):
    filename_mat.append([])
    for j, file2 in enumerate(task2_contactless_filenames):
        filename_mat[i].append((file1, file2))

with open("../misc/cl_fnames_best.txt") as file:
    task1_contactless_filenames = [f.strip().split("/")[-1] for f in file.readlines()]

task1_score_matrix = np.load("../misc/task1_cl2cl_score_matrix_best.npy")

task2_score_matrix = np.zeros((560, 560))
task2_label_matrix = np.zeros((560, 560))

task2_score_matrix_f1 = np.zeros((560, 560))
task2_score_matrix_f2 = np.zeros((560, 560))
task2_score_matrix_f3 = np.zeros((560, 560))
task2_score_matrix_f4 = np.zeros((560, 560))

probe_label_list = [-1] * 560
gallery_label_list = [-1] * 560

count_missing_scores = 0
count_missing_files = []

for index1, row in enumerate(tqdm(filename_mat)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair
        score = []
        for i in range(0,4):
            flag = False
            t1_fname1 = fname1.split("/")[-1].split(".png")[0] + "_" + str(i) + ".png"
            t1_fname2 = fname2.split("/")[-1].split(".png")[0] + "_" + str(i) + ".png"
            if (t1_fname1 in task1_contactless_filenames):
                i1 = task1_contactless_filenames.index(t1_fname1)
            else:
                flag = True
                count_missing_files.append(t1_fname1)
            if (t1_fname2 in task1_contactless_filenames): 
                i2 = task1_contactless_filenames.index(t1_fname2)
            else:
                flag = True
                count_missing_files.append(t1_fname2)  
            if flag == False:
                score.append(task1_score_matrix[i1][i2])
            else:
                score.append(0)

        id1 = fname1.split("/")[-1].split("_")[2] + fname1.split("/")[-1].split("_")[4]
        id2 = fname2.split("/")[-1].split("_")[2] + fname2.split("/")[-1].split("_")[4]
        
        task2_score_matrix_f1[index1][index2] = score[0]
        task2_score_matrix_f2[index1][index2] = score[1]
        task2_score_matrix_f3[index1][index2] = score[2]
        task2_score_matrix_f4[index1][index2] = score[3]

        task2_score_matrix[index1][index2]    = sum(score * softmax_with_temperature(score, 0.9)) #/ len(score)
        task2_label_matrix[index1][index2]    = 1 if id1 == id2 else 0

        probe_label_list[index1] = id1
        gallery_label_list[index2] = id2
        
print(len(set(count_missing_files)), " - Missing Files")

print(task2_score_matrix.max(), task2_score_matrix.min())

np.save("task2_score_matrix_cl2cl.npy", task2_score_matrix)
np.save("task2_label_matrix_cl2cl.npy", task2_label_matrix)

task2_score_matrix = np.load("task2_score_matrix_cl2cl.npy")
task2_label_matrix = np.load("task2_label_matrix_cl2cl.npy")
plt.figure()
for fingertype, scores in [("four finger", task2_score_matrix), ("Index Finger", task2_score_matrix_f1), ("Middle Finger", task2_score_matrix_f2), ("Ring Finger", task2_score_matrix_f3), ("Little Finger", task2_score_matrix_f4)]:
    scores      = np.triu(scores,k=1)
    scores      = scores[np.triu(torch.ones(scores.shape), k=1) == 1].flatten().tolist()
    # print(len(scores))
    ids         = torch.from_numpy(task2_label_matrix).float()
    # print(">", ids.shape, torch.sum(ids))
    ids_mod     = ids[torch.triu(torch.ones(ids.shape), diagonal=1) == 1]

    fpr, tpr, thresholds = roc_curve(ids_mod, scores)
    with open("../evaluate/task2_cl2cl_fpr_tpr_values.json", "w+") as file:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)

    lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
    upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
    tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    roc_auc = auc(fpr, tpr)
    print(fingertype)
    print(f"ROC for cl2cl task2:{roc_auc}")
    print(f"CL2CL task2: EER: {EER * 100}")
    print(f"CL2CL task2: TAR@FAR=10^-2 = ", tar_far_102 * 100)
    print()
    plt.plot(fpr, tpr, label= fingertype + ' : ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 0.01], [0, 1], 'k--')
plt.xlim([0, 0.01])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CL2CL task2')
plt.legend(loc="lower right")

plt.savefig("../evaluate/roc_curve_cl2cl_task2.png", dpi=300, bbox_inches='tight')


exit(0)

sim_mat = task2_score_matrix                # no need to remove diagonal for C2CL
print(sim_mat.shape, len(probe_label_list), len(gallery_label_list))
sim_mat = torch.from_numpy(sim_mat)
sim_mat = sim_mat * (1 - torch.eye(sim_mat.shape[0], sim_mat.shape[1]))
cl2clk1 = compute_recall_at_k(sim_mat, probe_label_list, gallery_label_list, 1)
print("CL2CL: R@1 = ", cl2clk1)
print("CL2CL: R@10 = ", compute_recall_at_k(sim_mat, probe_label_list, gallery_label_list, 10))
print("CL2CL: R@50 = ", compute_recall_at_k(sim_mat, probe_label_list, gallery_label_list, 50))
print("CL2CL: R@100 = ", compute_recall_at_k(sim_mat, probe_label_list, gallery_label_list, 100))

#<---------------------------------------------------------------------------------------------------------------------------------->
# cb2cl
task2_contactbase_filenames = list()
with open("contactbased_task2_files.txt") as file:
    for f in file.readlines():
        if f.strip().split("/")[-1].split("_")[-2] + "_" + f.strip().split("/")[-1].split("_")[-1].split(".bmp")[0] == "Four_Fingers":
            task2_contactbase_filenames.append(contactbased_base_path + f.strip())
    task2_contactbase_filenames = sorted(task2_contactbase_filenames)

filename_mat = list()

for i,file1 in enumerate(task2_contactbase_filenames):
    filename_mat.append([])
    for j, file2 in enumerate(task2_contactless_filenames):
        filename_mat[i].append((file1, file2))

with open("../misc/cb_fnames.txt") as file:
    task1_contactbase_filenames = [f.strip().split("/")[-1] for f in file.readlines()]

task1_score_matrix = np.load("../misc/task1_cb2cl_score_matrix.npy")
task1_score_matrix = np.transpose(task1_score_matrix)
print(task1_score_matrix.shape)

task2_score_matrix = np.zeros((51, 560))
task2_label_matrix = np.zeros((51, 560))

probe_label_list = [-1] * 560
gallery_label_list = [-1] * 51

row_labels = list()
column_labels = list()
count_missing_scores = 0
count_missing_files = []

finger_dict = {0:"Index",1:"Middle",2:"Ring",3:"Little"}
for index1, row in enumerate(tqdm(filename_mat)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair
        score = []
        for i in range(0,4):
            t1_fname1 = "_".join(fname1.split("/")[-1].split(".bmp")[0].split("_")[:3]) + "_" + finger_dict[i] + ".bmp"
            t1_fname2 = fname2.split("/")[-1].split(".png")[0] + "_" + str(i) + ".png"
            try:
                if (t1_fname1 in task1_contactbase_filenames):
                    i1 = task1_contactbase_filenames.index(t1_fname1)
                else:
                    count_missing_files.append(t1_fname1)
                if (t1_fname2 in task1_contactless_filenames): 
                    i2 = task1_contactless_filenames.index(t1_fname2)
                else:
                    count_missing_files.append(t1_fname2)  
                    
                score.append(task1_score_matrix[i1][i2])
            except:
                continue
        try:
            global_score = sum(score)
        except:
            count_missing_scores += 1
            global_score = 0

        id1 = fname1.split("/")[-1].split("_")[1] + fname1.split("/")[-1].split("_")[2].lower()
        id2 = fname2.split("/")[-1].split("_")[2] + fname2.split("/")[-1].split("_")[4].lower()
        task2_score_matrix[index1][index2] = global_score
        task2_label_matrix[index1][index2] = 1 if id1 == id2 else 0
        gallery_label_list[index1] = id1
        probe_label_list[index2] = id2


np.save("task2_score_matrix_cb2cl.npy", task2_score_matrix)
np.save("task2_label_matrix_cb2cl.npy", task2_label_matrix)

task2_score_matrix = np.load("task2_score_matrix_cb2cl.npy")
task2_label_matrix = np.load("task2_label_matrix_cb2cl.npy")

task2_score_matrix = np.transpose(task2_score_matrix)
task2_label_matrix = np.transpose(task2_label_matrix)

print(task2_score_matrix.shape, task2_label_matrix.shape)

scores      = task2_score_matrix.flatten().tolist()
# scores      = scores[np.triu(torch.ones(scores.shape), k=1) == 1].flatten().tolist()
ids         = torch.from_numpy(task2_label_matrix).float().flatten().tolist()
# ids_mod     = ids[torch.triu(torch.ones(ids.shape), diagonal=1) == 1]
fpr, tpr, thresholds = roc_curve(ids, scores)

with open("../evaluate/task2_cl2cb_fpr_tpr_values.json", "w+") as file:
    json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
roc_auc = auc(fpr, tpr)
print(f"ROC for cb2cl task2: {roc_auc}")
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CB2CL task2')
plt.legend(loc="lower right")
plt.savefig("../evaluate/roc_curve_cl2cb_task2.png", dpi=300, bbox_inches='tight')
print(f"CB2CL task2: EER: {EER * 100}")
print(f"CB2CL task2: TAR@FAR=10^-2 = ", tar_far_102 * 100)

sim_mat = task2_score_matrix                # no need to remove diagonal for C2CL
print(sim_mat.shape, len(probe_label_list), len(gallery_label_list))
cl2cbk1 = compute_recall_at_k(torch.from_numpy(sim_mat), probe_label_list, gallery_label_list, 1)
print("C2CL: R@1 = ", cl2cbk1)
print("C2CL: R@10 = ", compute_recall_at_k(torch.from_numpy(sim_mat), probe_label_list, gallery_label_list, 10))
print("C2CL: R@50 = ", compute_recall_at_k(torch.from_numpy(sim_mat), probe_label_list, gallery_label_list, 50))
print("C2CL: R@100 = ", compute_recall_at_k(torch.from_numpy(sim_mat), probe_label_list, gallery_label_list, 100))