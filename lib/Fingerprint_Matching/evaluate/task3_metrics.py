import os
import json
import numpy as np
import torch
from tqdm import tqdm
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

def find_element(matrix, element):
    for i, row in enumerate(matrix):
        try:
            j = row.index(element)
            return i, j
        except ValueError:
            continue
    return None

contactless_base_path = "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task3/DistalMatching/set_based_test_cl2cl.json"
contactbased_base_path = "/home/bhavinja/RidgeBase/Fingerprint_Train_Test_Split/Task3/DistalMatching/set_based_test_c2cl.json"

# cl2cl
with open(contactless_base_path,'r') as js:
    cl_base_file = json.load(js)

all_queries = list()
all_galery = list()
for i in cl_base_file:
    all_queries.extend(cl_base_file[i]['query'])
    all_galery.extend(cl_base_file[i]['gallery'])
print(f"number of queries: {len(all_queries)}")
print(f"number of gallery: {len(all_galery)}")
# # exit(0)
filename_mat = []

for i,file1 in enumerate(all_queries):
    filename_mat.append([])
    for j, file2 in enumerate(all_galery):
        filename_mat[i].append((file1, file2))
print(f"dimension of the matrix of query and gallery: {len(filename_mat)},{len(filename_mat[0])}")
# print(filename_mat[0][0])
# exit(0)

with open("../misc/cl_fnames.txt") as file:
    task1_contactless_filenames = [f.strip().split("/")[-1] for f in file.readlines()]
# print(task1_contactless_filenames[0])
# exit(0)

task1_score_matrix = np.load("../misc/task1_cl2cl_score_matrix_best.npy")
print(f"dimension of task1 score matrix: {task1_score_matrix.shape}")

task3_score_matrix = np.zeros((len(filename_mat), len(filename_mat[0])))
task3_label_matrix = np.zeros((len(filename_mat), len(filename_mat[0])))

count_missing_scores = 0
count_missing_files = []

for index1, row in enumerate(tqdm(filename_mat)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair
        # print(fname1)
        # print(fname2)
        # exit(0)
        try:
            if (fname1 in task1_contactless_filenames):
                i1 = task1_contactless_filenames.index(fname1)
            else:
                count_missing_files.append(fname1)
            if (fname2 in task1_contactless_filenames): 
                i2 = task1_contactless_filenames.index(fname2)
            else:
                count_missing_files.append(fname2)  
            score = task1_score_matrix[i1][i2]
            # if score == np.float32(0):
            #     exit(0)
            id1 = fname1.split("_")[2] + fname1.split("_")[4] + fname1.split("_")[7]
            id2 = fname2.split("_")[2] + fname2.split("_")[4] + fname2.split("_")[7]
            task3_score_matrix[index1][index2] = score
            task3_label_matrix[index1][index2] = 1 if id1 == id2 else 0
        except:
            continue

print(f"temporary score matrix shape: {task3_score_matrix.shape}")
print(f"temporary label matrix shape: {task3_label_matrix.shape}")
# exit(0)
np.save("task3_score_matrix_cl2cl_best.npy", task3_score_matrix)
np.save("task3_label_matrix_cl2cl_best.npy", task3_label_matrix)

all_unique_task3 = list()
for i in cl_base_file:
    all_unique_task3.append(i)

filename_mat_final = []
for i,file1 in enumerate(all_unique_task3):
    filename_mat_final.append([])
    for j, file2 in enumerate(all_unique_task3):
        filename_mat_final[i].append((file1, file2))
print(f"Shape of task 3 score matrix needed: {len(filename_mat_final)},{len(filename_mat_final[0])}")

task3_score_matrix = np.load("task3_score_matrix_cl2cl_best.npy")
task3_label_matrix = np.load("task3_label_matrix_cl2cl_best.npy")

task3_score_matrix_final = np.zeros((len(filename_mat_final), len(filename_mat_final[0])))
task3_label_matrix_final = np.zeros((len(filename_mat_final), len(filename_mat_final[0])))

count_missing_scores = 0
count_missing_files = []
probes_list_final = [-1] * len(filename_mat_final)
gallery_list_final = [-1] * len(filename_mat_final[0])
# filename_mat = np.array(filename_mat)
for index1, row in enumerate(tqdm(filename_mat_final)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair[0], filepair[1]
        query_list   = cl_base_file[fname1]['query']
        gallery_list = cl_base_file[fname2]['gallery']
        score = []
        for i in query_list:
            for j in gallery_list:
                # i1, i2 = find_element(filename_mat, (i,j))
                i1 = all_queries.index(i)
                i2 = all_galery.index(j)
                score.append(task3_score_matrix[i1][i2])
        global_score = max(score)
        task3_score_matrix_final[index1][index2] = global_score
        task3_label_matrix_final[index1][index2] = 1 if fname1 == fname2 else 0
        probes_list_final[index1] = fname1
        gallery_list_final[index2] = fname2
# print(np.array(task3_label_matrix_final))
# print(np.array(task3_score_matrix_final))
# exit(0)
np.save("task3_score_matrix_cl2cl_final_best.npy", task3_score_matrix_final)
np.save("task3_label_matrix_cl2cl_final_best.npy", task3_label_matrix_final)

# result reproduction
pg_dic = {'probes': probes_list_final, 'gallery': gallery_list_final}
with open("probes_gallery_list_recall_cl2cl_task3.json",'w') as js:
    json.dump(pg_dic, js, indent = 4)

task3_score_matrix_final = np.load("task3_score_matrix_cl2cl_final_best.npy")
task3_label_matrix_final = np.load("task3_label_matrix_cl2cl_final_best.npy")

scores      = task3_score_matrix_final.flatten().tolist()
ids         = torch.from_numpy(task3_label_matrix_final).float().flatten().tolist()
fpr, tpr, thresholds = roc_curve(ids, scores)
with open("../evaluate/task3_cl2cl_fpr_tpr_values.json", "w+") as file:
    json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
roc_auc = auc(fpr, tpr)
print(f"ROC for cl2cl task3:{roc_auc}")
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CL2CL task3')
plt.legend(loc="lower right")
plt.savefig("../evaluate/roc_curve_cl2cl_task3.png", dpi=300, bbox_inches='tight')
print(f"CL2CL task3: EER: {EER * 100}")
print(f"CL2CL task3: TAR@FAR=10^-2 = ", tar_far_102 * 100)

sim_mat = task3_score_matrix_final
print(sim_mat.shape, len(probes_list_final), len(gallery_list_final))
sim_mat = torch.from_numpy(sim_mat)
# sim_mat = sim_mat * (1 - torch.eye(sim_mat.shape[0], sim_mat.shape[1]))
cl2clk1 = compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 1)
print("CL2CL: R@1 = ", cl2clk1)
print("CL2CL: R@10 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 10))
print("CL2CL: R@50 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 50))
print("CL2CL: R@100 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 100))


# <=============================================================================================================================>
# cl2cb
with open(contactbased_base_path,'r') as js:
    cb_base_file = json.load(js)

# print(cb_base_file)

all_queries = list()
all_galery = list()
for i in cb_base_file:
    all_queries.extend(cb_base_file[i]['query'])
    all_galery.extend(cb_base_file[i]['gallery'])
print(f"number of queries: {len(all_queries)}")
print(f"number of gallery: {len(all_galery)}")
# print(all_queries[0])
# print(all_galery[0])
# exit(0)
filename_mat = list()
for i,file1 in enumerate(all_queries):
    filename_mat.append([])
    for j, file2 in enumerate(all_galery):
        filename_mat[i].append((file1, file2))
print(f"dimension of the matrix of query and gallery: {len(filename_mat)},{len(filename_mat[0])}")
# print(filename_mat[0][0])
# exit(0)

with open("../misc/cl_fnames.txt") as file:
    task1_contactless_filenames = [f.strip().split("/")[-1] for f in file.readlines()]
with open("../misc/cb_fnames.txt") as file:
    task1_contactbase_filenames = [f.strip().split("/")[-1] for f in file.readlines()]
# print(task1_contactless_filenames[0])
# print(task1_contactbase_filenames[0])
# exit(0)

task1_score_matrix = np.load("../misc/task1_cb2cl_score_matrix_best.npy")
task1_score_matrix = np.transpose(task1_score_matrix)
print(f"dimension of task1 score matrix: {task1_score_matrix.shape}")

task3_score_matrix = np.zeros((len(filename_mat), len(filename_mat[0])))
task3_label_matrix = np.zeros((len(filename_mat), len(filename_mat[0])))

print(f"Shape of the temporary score matrix needed: {task3_score_matrix.shape}")
print(f"Shape of the temporary label matrix needed: {task3_label_matrix.shape}")
# exit(0)
count_missing_scores = 0
count_missing_files = []
finger_dic = {"Index":0,"Middle":1,"Ring":2,"Little":3}
for index1, row in enumerate(tqdm(filename_mat)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair
        try:
            if (fname1 in task1_contactbase_filenames):
                i1 = task1_contactbase_filenames.index(fname1)
                # print(i1)
            else:
                count_missing_files.append(fname1)
            if (fname2 in task1_contactless_filenames): 
                i2 = task1_contactless_filenames.index(fname2)
                # print(i2)
            else:
                count_missing_files.append(fname2)  
            score = task1_score_matrix[i1][i2]

            id1 = fname1.split("_")[1] + fname1.split("_")[2].lower() + str(finger_dic[os.path.splitext(fname1.split("_")[3])[0]])
            id2 = fname2.split("_")[2] + fname2.split("_")[4].lower() + os.path.splitext(fname2.split("_")[7])[0]
            # print(id1)
            # print(id2)
            # exit(0)
            task3_score_matrix[index1][index2] = score
            task3_label_matrix[index1][index2] = 1 if id1 == id2 else 0
        except:
            continue

# print(task3_label_matrix)
print(f"temporary score matrix shape: {task3_score_matrix.shape}")
print(f"temporary label matrix shape: {task3_label_matrix.shape}")
# exit(0)
np.save("task3_score_matrix_cb2cl_best.npy", task3_score_matrix)
np.save("task3_label_matrix_cb2cl_best.npy", task3_label_matrix)

all_unique_task3 = list()
for i in cb_base_file:
    all_unique_task3.append(i)
    # print(i)
    # exit(0)

filename_mat_final = []
for i,file1 in enumerate(all_unique_task3):
    filename_mat_final.append([])
    for j, file2 in enumerate(all_unique_task3):
        filename_mat_final[i].append((file1, file2))
print(f"Shape of task 3 score matrix needed: {len(filename_mat_final)},{len(filename_mat_final[0])}")

task3_score_matrix = np.load("task3_score_matrix_cb2cl_best.npy")
task3_label_matrix = np.load("task3_label_matrix_cb2cl_best.npy")

task3_score_matrix_final = np.zeros((len(filename_mat_final), len(filename_mat_final[0])))
task3_label_matrix_final = np.zeros((len(filename_mat_final), len(filename_mat_final[0])))

count_missing_scores = 0
count_missing_files = []
probes_list_final = [-1] * len(filename_mat_final)
gallery_list_final = [-1] * len(filename_mat_final[0])
# filename_mat = np.array(filename_mat)
for index1, row in enumerate(tqdm(filename_mat_final)):
    for index2, filepair in enumerate(row):
        fname1, fname2 = filepair[0], filepair[1]
        query_list   = cb_base_file[fname1]['query']
        gallery_list = cb_base_file[fname2]['gallery']
        score = []
        for i in query_list:
            for j in gallery_list:
                # i1, i2 = find_element(filename_mat, (i,j))
                i1 = all_queries.index(i)
                i2 = all_galery.index(j)
                score.append(task3_score_matrix[i1][i2])
        # rel_weights = softmax_with_temperature(score, 0.01)
        # print(sum(score * rel_weights), max(score))
        global_score = max(score)
        task3_score_matrix_final[index1][index2] = global_score
        task3_label_matrix_final[index1][index2] = 1 if fname1 == fname2 else 0
        probes_list_final[index1] = fname1
        gallery_list_final[index2] = fname2
# print(np.array(task3_label_matrix_final))
# print(np.array(task3_score_matrix_final))
# exit(0)
np.save("task3_score_matrix_cl2cb_final_best.npy", task3_score_matrix_final)
np.save("task3_label_matrix_cl2cb_final_best.npy", task3_label_matrix_final)

# result reproduction
pg_dic = {'probes': probes_list_final, 'gallery': gallery_list_final}
with open("probes_gallery_list_recall_cl2cl_task3.json",'w') as js:
    json.dump(pg_dic, js, indent = 4)

task3_score_matrix_final = np.load("task3_score_matrix_cl2cb_final_best.npy")
task3_label_matrix_final = np.load("task3_label_matrix_cl2cb_final_best.npy")


scores      = task3_score_matrix_final.flatten().tolist()
ids         = torch.from_numpy(task3_label_matrix_final).float().flatten().tolist()

fpr, tpr, thresholds = roc_curve(ids, scores)
with open("../evaluate/task3_cl2cb_fpr_tpr_values.json", "w+") as file:
    json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, file)

lower_fpr_idx = max(i for i, val in enumerate(fpr) if val < 0.01)
upper_fpr_idx = min(i for i, val in enumerate(fpr) if val >= 0.01)
tar_far_102 = (tpr[lower_fpr_idx]+tpr[upper_fpr_idx])/2
fnr = 1 - tpr
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
roc_auc = auc(fpr, tpr)
print(f"ROC for cl2cb task3:{roc_auc}")
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CL2CB task3')
plt.legend(loc="lower right")
plt.savefig("../evaluate/roc_curve_cl2cb_task3.png", dpi=300, bbox_inches='tight')
print(f"CL2CB task3: EER: {EER * 100}")
print(f"CL2CB task3: TAR@FAR=10^-2 = ", tar_far_102 * 100)

# task3_score_matrix_final = np.transpose(task3_score_matrix_final)
# task3_label_matrix_final = np.transpose(task3_label_matrix_final)

sim_mat = task3_score_matrix_final
print(sim_mat.shape, len(probes_list_final), len(gallery_list_final))
sim_mat = torch.from_numpy(sim_mat)

# eps = 1e-5  # To avoid division by zero
# # Find minimum and maximum values in each row
# min_vals, _ = torch.min(sim_mat, dim=1, keepdim=True)
# max_vals, _ = torch.max(sim_mat, dim=1, keepdim=True)
# # Normalize each row using min and max values
# sim_mat = (sim_mat - min_vals) / (max_vals - min_vals + eps)


cl2clk1 = compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 1)
print("CL2CB: R@1 = ", cl2clk1)
print("CL2CB: R@10 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 10))
print("CL2CB: R@50 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 50))
print("CL2CB: R@100 = ", compute_recall_at_k(sim_mat, probes_list_final, gallery_list_final, 100))