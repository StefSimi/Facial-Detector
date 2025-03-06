from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
#from evalueaza_solutie import *

params: Parameters = Parameters()
params.dim_window = 64  #36
params.dim_hog_cell = 12  # dimensiunea celulei 2 by default
params.overlap = 0.02
params.number_positive_examples = 5813  # numarul exemplelor pozitive
# params.number_negative_examples = 10000 # numarul exemplelor negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite
if params.use_flip_images:
    params.number_positive_examples = 2 * params.number_positive_examples
params.number_negative_examples = 10000  # numarul exemplelor negative
params.threshold = -0.25  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.scaling_ratio = 0.05 #0.1
params.use_hard_mining = True  # (optional) antrenare cu exemple puternic negative

facial_detector: FacialDetector = FacialDetector(params)



#evaluate_results_task1("data/","data/","data/")


# Pasul 1. Incarcam exemplele pozitive si exemple negative daca acestea nu sunt deja salvate
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 2. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)



if params.use_hard_mining:
    params.threshold = 0.0
    hard_negative_features = facial_detector.run(True)
    negative_features = np.concatenate((negative_features,hard_negative_features), axis=0)
    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
    print('Numarul de exemple pozitive este:', positive_features.shape[0])
    print('Numarul de exemple negative este:', negative_features.shape[0])
    params.number_negative_examples = negative_features.shape[0]
    params.threshold = 0.0 #I'm gonna fucking kill myself
    facial_detector.train_classifier(training_examples, train_labels)


# Pasul 4. Ruleaza detectorul facial pe imaginile de test.



detections, scores, file_names,model, labels = facial_detector.run()
getCharacter(detections, scores, file_names, model, params)
'''
np.save("task1/detections_all_faces.npy", detections)
np.save("task1/scores_all_faces.npy", scores)
np.save("task1/file_names_all_faces.npy", file_names)

np.save("task2/detections_dad.npy", detections)
np.save("task2/scores_dad.npy", scores)
np.save("task2/file_names_dad.npy", file_names)

np.save("task2/detections_mom.npy", detections)
np.save("task2/scores_mom.npy", scores)
np.save("task2/file_names_mom.npy", file_names)

np.save("task2/detections_dexter.npy", detections)
np.save("task2/scores_dexter.npy", scores)
np.save("task2/file_names_dexter.npy", file_names)

np.save("task2/detections_deedee.npy", detections)
np.save("task2/scores_deedeee.npy", scores)
np.save("task2/file_names_deedee.npy", file_names)
'''
# Pasul 5. Evalueaza si vizualizeaza detectiile
# Pentru imagini pentru care exista adnotari (cele din setul de date  CMU+MIT) folositi functia show_detection_with_ground_truth
# pentru imagini fara adnotari (cele realizate la curs si laborator) folositi functia show_detection_without_ground_truth
#model=facial_detector.load_saved_classifier()
#dad_detections,deedee_detections,dexter_detections,mom_detections,dad_s,deedee_s,dexter_s,mom_s=getCharacter(detections, scores, file_names, model, params)



if params.has_annotations:
    #
    np.save("task1/detections_all_faces.npy", detections)
    np.save("task1/scores_all_faces.npy", scores)
    np.save("task1/file_names_all_faces.npy", file_names)
    facial_detector.eval_detections(detections, scores, file_names)
    #facial_detector.eval_detections_character(detections,scores,file_names,'validare/task2_dad_gt_testare.txt','dad')
    #facial_detector.eval_detections_character(detections, scores, file_names, 'validare/task2_deedee_gt_validare.txt','deedee')
    #facial_detector.eval_detections_character(detections, scores, file_names, 'validare/task2_dexter_gt_validare.txt', 'dexter')
    #facial_detector.eval_detections_character(detections, scores, file_names, 'validare/task2_mom_gt_validare.txt','mom')

    #Doesn't work
    #facial_detector.eval_detections_character(dad_detections, dad_s, file_names, 'validare/task2_dad_gt_validare.txt','dad')
    #facial_detector.eval_detections_character(deedee_detections, deedee_s, file_names, 'validare/task2_deedee_gt_validare.txt','deedee')
    #facial_detector.eval_detections_character(dexter_detections, dexter_s, file_names, 'validare/task2_dexter_gt_validare.txt', 'dexter')
    #facial_detector.eval_detections_character(mom_detections, mom_s, file_names, 'validare/task2_mom_gt_validare.txt','mom')


    #show_detections_with_ground_truth(detections, scores, file_names, model, params)
#else:

show_detections_without_ground_truth(detections, scores, file_names, params)


# salavarea obiectului params
params_file_name = os.path.join(params.dir_save_files, 'params')
pickle.dump(params, open(params_file_name, 'wb'))