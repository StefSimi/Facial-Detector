import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob
from Parameters import *


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune Nx4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (212, 255, 127), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)




def show_detections_with_ground_truth(detections, scores, file_names, model, params: Parameters):

    color_map = {
        'dexter': (238, 215, 189),  # Blue
        'deedee': (102, 217, 255),  # Yellow
        'mom': (160, 48, 112),  # Purple
        'dad': (80, 208, 146)  # Lime
    }


    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]


        for idx, detection in enumerate(current_detections):
            x1, y1, x2, y2 = detection[0], detection[1], detection[2], detection[3]
            cropped_region = image[y1:y2, x1:x2]

            # Resize to 64x64
            resized_region = cv.resize(cropped_region, (64, 64), interpolation=cv.INTER_AREA)

            # Flatten the resized region into a feature vector
            feature_vector = resized_region.flatten()  # This will have 64*64*3 = 12288 features

            # Feed the feature vector to your SVM model
            prediction = model.predict([feature_vector])
            predicted_label = prediction[0]  # Assuming model returns a list of predictions

            # Choose color based on prediction
            color = color_map.get(predicted_label, (0, 0, 255))  # Default to red if label not found

            # Draw the colored bounding box around the detected region
            cv.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (x1, y1),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Score text remains red

            print(f"Prediction for detection {idx}: {predicted_label} with color {color}")



        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]




        # show ground truth bboxes
        #for detection in annotations:
            #cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def getCharacter(detections, scores, file_names, model, params: Parameters):
    ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)
    character_results = {
        "dad": {"detections": [], "scores": [], "file_names": []},
        "deedee": {"detections": [], "scores": [], "file_names": []},
        "dexter": {"detections": [], "scores": [], "file_names": []},
        "mom": {"detections": [], "scores": [], "file_names": []}
    }
    for idx2, test_file in enumerate(test_files):
        print(f"Processing test file {idx2 + 1}/{len(test_files)}: {test_file}")
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            x1, y1, x2, y2 = detection
            cropped_region = image[y1:y2, x1:x2]

            resized_region = cv.resize(cropped_region, (64, 64), interpolation=cv.INTER_AREA)
            feature_vector = resized_region.flatten()
            prediction = model.predict([feature_vector])[0]
            if prediction in character_results:
                character_results[prediction]["detections"].append(detection)
                character_results[prediction]["scores"].append(current_scores[idx])
                character_results[prediction]["file_names"].append(short_file_name)

    for character in character_results:
        np.save(os.path.join(params.dir_save_files, f"detections_{character}.npy"),
                np.array(character_results[character]["detections"]))
        np.save(os.path.join(params.dir_save_files, f"scores_{character}.npy"),
                np.array(character_results[character]["scores"]))
        np.save(os.path.join(params.dir_save_files, f"file_names_{character}.npy"),
                np.array(character_results[character]["file_names"]))

    np.save(os.path.join(params.dir_save_files, "scores_all_faces.npy"), scores)


