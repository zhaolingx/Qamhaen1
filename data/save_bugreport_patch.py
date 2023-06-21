import experiment.config as config
import os
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
# import ML4Prediciton
import signal
import json

def save_bugreport_patch(path_patch, ):
    # dataset_text = ''
    dataset_text_with_description = ''
    file_name = 'bugreport_patch.txt'
    # if os.path.exists(file_name):
    #     return
    with open('BugReport/Bug_Report_All.json', 'rb') as f:
        bugReportText = json.load(f)

    project_ids = set()
    project_ids_noBugReport = set()
    tmp = set()

    cnt_patch, cnt_patch_with_bugreport = 0, 0
    datasets = os.listdir(path_patch)
    for dataset in datasets:
        path_dataset = os.path.join(path_patch, dataset)
        benchmarks = os.listdir(path_dataset)
        for benchmark in benchmarks:
            path_benchmark = os.path.join(path_dataset, benchmark)
            tools = os.listdir(path_benchmark)
            for tool in tools:
                path_tool = os.path.join(path_benchmark, tool)
                labels = os.listdir(path_tool)
                for label in labels:
                    path_label = os.path.join(path_tool, label)
                    projects = os.listdir(path_label)
                    for project in projects:
                        path_project = os.path.join(path_label, project)
                        ids = os.listdir(path_project)
                        for id in ids:
                            path_id = os.path.join(path_project, id)
                            patches = os.listdir(path_id)
                            for patch in patches:
                                cnt_patch += 1
                                if benchmark == 'Bugsjar' and '+' in project:
                                        project1 = project.split('+')[1]
                                        project1 = project1.lower()
                                        project_id = project1 + '-' + id
                                else:
                                    project_id = project + '-' + id
                                print('collecting {}'.format(project_id))
                                label_int = 1 if label == 'Correct' else 0
                                project_id = project_id.lower()
                                tmp.add(project_id)
                                if tool == 'Developer':
                                    project_ids.add(project_id)

                                # extract bug report
                                if project_id in bugReportText.keys():
                                    bug_report_text = bugReportText[project_id]
                                    bug_report_summary = bug_report_text[0].strip()
                                    bug_report_description = bug_report_text[1].strip()
                                else:
                                    bug_report_summary = 'None'
                                    bug_report_description = 'None'
                                    project_ids_noBugReport.add(project_id)

                                # extract patch text
                                patch_text = ''
                                path_single_patch = os.path.join(path_id, patch)
                                for root, dirs, files in os.walk(path_single_patch):
                                    for file in files:
                                        if file.endswith('.txt'):
                                            try:
                                                with open(os.path.join(root, file), 'r+') as f:
                                                    patch_text += f.readlines()[0].strip('\n')
                                            except Exception as e:
                                                print(e)
                                                continue

                                patch_id = patch + '-' + project_id + '_' + tool + '_' + dataset
                                if patch_text == '':
                                    print('problematic patch')

                                if bug_report_summary == 'None' or patch_text == '':
                                    continue
                                # integrate and save them
                                # TODO: use test suite exception
                                # dataset_text += '$$'.join([project_id, bug_report_summary, name, commit_content, str(label)]) + '\n'
                                dataset_text_with_description += '$$'.join(
                                    [project_id, bug_report_summary, bug_report_description, patch_id, patch_text,
                                     str(label_int)]) + '\n'
                                cnt_patch_with_bugreport += 1
    with open(file_name, 'w+') as f:
        f.write(dataset_text_with_description)
    print('patch number: {}, patch with available bug report: {}'.format(cnt_patch, cnt_patch_with_bugreport))
    print('All bug ids: {}, Available: {}, BugReportMissing: {}'.format(len(project_ids), len(project_ids)-len(project_ids_noBugReport), len(project_ids_noBugReport)))
    # print(tmp-project_ids)
    # print(project_ids)
    # print(project_ids_noBugReport)


if __name__ == '__main__':
    path_patch = config.Config().path_patch

    # save bug report and patch in text
    save_bugreport_patch(path_patch)
