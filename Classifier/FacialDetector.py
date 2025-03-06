from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import glob
import os
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog




class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.character_classifier = None
        self.best_model = None
        self.classifier_path = os.path.join('character_classifier.pkl')


    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NxD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            descriptor_img = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2))

            positive_descriptors.append(descriptor_img)
            print("Am extras descriptorul pentru imaginea ", i, " care are dimensiunea de ", descriptor_img.shape)

            if self.params.use_flip_images:
                img_flip = cv.flip(img, 1)  # imaginea oglindita (pe axa y)
                img_flip = img_flip * 0.50  # marim contrastul
                descriptor_img_flip = hog(img_flip,
                                          pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                          cells_per_block=(2, 2))

                positive_descriptors.append(descriptor_img_flip)

        positive_descriptors = np.array(positive_descriptors)
        print("Dupa ce am extras toti descriptorii pentru imaginile pozitive obtinem un array de dimensiuni: ",
              positive_descriptors.shape)

        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NxD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            (h, w) = img.shape  # inaltime, respectiv latimea imaginii img
            # coltul din stanga sus
            x0 = np.random.randint(0, w - self.params.dim_window, num_negative_per_image)
            y0 = np.random.randint(0, h - self.params.dim_window, num_negative_per_image)

            for idx in range(len(x0)):
                window = img[y0[idx]:y0[idx] + self.params.dim_window, x0[idx]:x0[idx] + self.params.dim_window].copy()
                descriptor_window = hog(window, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2))

                negative_descriptors.append(descriptor_window)

        negative_descriptors = np.array(negative_descriptors)
        print("Dupa ce am extras toti descriptorii pentru imaginile negative obtinem un array de dimensiuni: ",
              negative_descriptors.shape)

        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        #TODO DECOMMENT THIS


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def load_character_data(self):
        character_data = []
        labels = []

        # Loop through each annotation file and associated folder
        characters = ['dad', 'deedee', 'dexter', 'mom']
        for character in characters:
            annotation_file = os.path.join(self.params.dir_character_annotations, f"{character}_annotations.txt")
            image_folder = os.path.join(self.params.dir_character_annotations, character)

            with open(annotation_file, 'r') as f:
                for line in f:
                    file_name, x1, y1, x2, y2, label = line.strip().split()
                    image_path = os.path.join(image_folder, file_name)
                    face_image = self.crop_face(image_path, int(x1), int(y1), int(x2), int(y2))
                    character_data.append(face_image)
                    labels.append(label)

        print("Loaded images for training")
        return np.array(character_data), np.array(labels)

    def crop_face(self, image_path, x1, y1, x2, y2):
        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        cropped_face = image[y1:y2, x1:x2]
        resized_face = cv.resize(cropped_face, (self.params.dim_window, self.params.dim_window))
        return resized_face.flatten()

    def preprocess_face(self, face_image):
        # Normalize or preprocess the face image for the classifier
        return face_image / 255.0  # Example normalization

    def train_multi_class_classifier(self, training_data, labels):
        print("Creating SVC...")
        self.character_classifier = SVC(kernel='linear', decision_function_shape='ovr')  # One-vs-Rest
        print("Fitting...")
        self.character_classifier.fit(training_data, labels)
        with open(self.classifier_path, 'wb') as f:
            pickle.dump(self.character_classifier, f)
        print("Character classifier trained and saved successfully!")

    def load_saved_classifier(self):
        if os.path.exists(self.classifier_path):
            with open(self.classifier_path, 'rb') as f:
                self.character_classifier = pickle.load(f)
            print("Character classifier loaded successfully!")
        else:
            print("No saved classifier found. Please train the classifier first.")

    def classify_detected_faces(self, detected_faces):
        labels = []
        for face in detected_faces:
            preprocessed_face = self.preprocess_face(face)
            label = self.character_classifier.predict([preprocessed_face])[0]
            labels.append(label)
        return labels



    def identify_character(self, avg_color_face):
        """
        Identifies the character closest to the given average color.

        Parameters:
        avg_color_face (tuple): Average color of the face in BGR format (as returned by cv2.mean).

        Returns:
        str: Name of the closest character.
        """
        precomputed_colors = {
            "dad": [105.25380446, 122.72450817, 154.05888914],
            "deedee": [123.13081985, 136.00677795, 169.75814127],
            "dexter": [130.51003739, 127.54555591, 143.0607478],
            "mom": [107.76775685, 111.17417735, 168.68927019],
        }
        min_distance = float('inf')
        closest_character = None

        for character, avg_color in precomputed_colors.items():
            # Compute the Euclidean distance
            distance = np.linalg.norm(np.array(avg_color) - np.array(avg_color_face))

            # Update closest character if this distance is smaller
            if distance < min_distance:
                min_distance = distance
                closest_character = character

        return closest_character




    def intersection_over_union(self, bbox_a, bbox_b, epsilon=1e-8):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / (float(box_a_area + box_b_area - inter_area) + epsilon)

        return iou

    def non_maximum_suppression(self, image_detections, image_scores, image_size, descriptors_to_return=None):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune Nx4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_image_descriptors = None
        if descriptors_to_return is not None:
            sorted_image_descriptors = descriptors_to_return[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)

        #Changed IOU threshold to .4 from .3

        iou_threshold = 0.02
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        if self.params.use_hard_mining and sorted_image_descriptors is not None:
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_image_descriptors[is_maximal]

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def scaleImage(self, image, min_size):
        yield image
        interpolation = cv.INTER_AREA
        failsafe=0
        while True and failsafe<1000:
            failsafe+=1
            new_height, new_width = int(image.shape[0] - image.shape[0] * self.params.scaling_ratio), int(
                image.shape[1] - image.shape[1] * self.params.scaling_ratio)
            if new_width < min_size or new_height < min_size:
                break

            image = cv.resize(image, (new_width, new_height), interpolation=interpolation)
            yield image

    def run(self, return_descriptors=False):
        print("started run function")

        if self.character_classifier is None:
            print("Checking for saved classifier...")
            self.load_saved_classifier()

        if self.character_classifier is None:
            print("Training multi-class classifier")
            self.train_multi_class_classifier(*self.load_character_data())

        if self.character_classifier is not None:
            print("not none in run function")

        labels = []


        if self.params.use_hard_mining and return_descriptors:
            test_images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        else:
            test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')

        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        file_names = np.array([])
        detected_faces = None

        w = self.best_model.coef_
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        descriptors_to_return = None

        for i in range(num_test_images):
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            current_file_name = os.path.basename(test_files[i])
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            (L, C) = img.shape
            current_image_detections = None
            current_image_descriptors = None
            current_image_faces = None

            current_image_scores = np.array([])

            iterate_once = False
            iter=0
            for img_resised in self.scaleImage(img,self.params.dim_window):
                iter+=1
                if iterate_once:
                    break
                if self.params.use_hard_mining and return_descriptors:
                    iterate_once = True

                (H, W) = img_resised.shape
                hog_descriptor = hog(image=img_resised,
                                     pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                     cells_per_block=(2, 2), feature_vector=False)
                l, c = hog_descriptor.shape[0], hog_descriptor.shape[1]

                hog_descriptor = np.reshape(hog_descriptor, (l, c, 2 * 2 * 9))
                k = self.params.dim_window // self.params.dim_hog_cell - 1
                for y_min in range(l - k):
                    for x_min in range(c - k):
                        x_max, y_max = x_min + k, y_min + k
                        descr = hog_descriptor[y_min:y_max, x_min:x_max].copy().flatten()
                        if return_descriptors and not self.params.use_hard_mining:
                            if descriptors_to_return is not None:
                                descriptors_to_return = np.concatenate((descriptors_to_return, descr), axis=0)
                            else:
                                descriptors_to_return = np.array([descr])
                        descr = np.reshape(descr, (descr.shape[0], 1))

                        score = w @ descr + bias
                        if score[0][0] > self.params.threshold:
                            if self.params.use_hard_mining and return_descriptors:
                                if current_image_descriptors is None:
                                    current_image_descriptors = np.reshape(descr, (1, descr.shape[0]))
                                else:
                                    descr = np.reshape(descr, (1, descr.shape[0]))
                                    current_image_descriptors = np.concatenate((current_image_descriptors, descr), axis=0)

                            current_image_scores = np.concatenate((current_image_scores, score), axis=None)
                            width_ratio, height_ratio = C / W, L / H
                            x0 = int(((x_min + 1) * self.params.dim_hog_cell) * width_ratio)
                            y0 = int(((y_min + 1) * self.params.dim_hog_cell) * height_ratio)
                            x1 = int(((x_max + 1) * self.params.dim_hog_cell) * width_ratio)
                            y1 = int(((y_max + 1) * self.params.dim_hog_cell) * height_ratio)


                            if current_image_detections is None:
                                current_image_detections = np.array([[x0, y0, x1, y1]])
                                #print(current_image_detections.shape)
                            else:
                                current_image_detections = np.concatenate((current_image_detections, np.array([[x0, y0, x1, y1]])), axis=0)

            if current_image_detections is not None:
                if self.params.use_hard_mining and return_descriptors:
                    current_image_detections, current_image_scores, current_image_descriptors = self.non_maximum_suppression(
                        current_image_detections, current_image_scores, np.array([L, C]), current_image_descriptors)
                else:
                    current_image_detections, current_image_scores = self.non_maximum_suppression(
                        current_image_detections, current_image_scores, np.array([L, C]))

                for detection in current_image_detections:
                    x0, y0, x1, y1 = detection
                    face_region = img[y0:y1, x0:x1]
                    face_region_resized = cv.resize(face_region, (64, 64))
                    face_region_rgb = cv.cvtColor(face_region_resized, cv.COLOR_GRAY2RGB)

                    ###
                    avg_color = cv.mean(face_region_rgb)[:3]
                    #print(self.identify_character(avg_color))
                    ###

                    preprocessed_face = face_region_rgb.flatten().reshape(1, -1)
                    #label = self.classify_detected_faces(preprocessed_face)
                    label=self.identify_character(avg_color)

                    labels.append(label)

            if current_image_detections is not None:

                scores = np.concatenate((scores, current_image_scores), axis=None)
                if detections is not None:
                    detections = np.concatenate((detections, current_image_detections), axis=0)
                else:
                    detections = current_image_detections
                file_names = np.concatenate((file_names, np.array([current_file_name] * current_image_scores.shape[0])),
                                            axis=None)
                if descriptors_to_return is not None:
                    descriptors_to_return = np.concatenate((descriptors_to_return, current_image_descriptors), axis=0)
                else:
                    descriptors_to_return = current_image_descriptors

        if return_descriptors:
            return descriptors_to_return

        print(labels)
        print(len(labels))
        return detections, scores, file_names,self.character_classifier,labels

    def getCharacter(detections, scores, file_names, model, params: Parameters):
        ground_truth_bboxes = np.loadtxt(params.path_annotations, dtype='str')
        test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        dad_detections = None
        dexter_detections = None
        deedee_detections = None
        mom_detections = None
        dad_scores = np.array([])
        dexter_scores = np.array([])
        deedee_scores = np.array([])
        mom_scores = np.array([])

        for idx2, test_file in enumerate(test_files):
            print(idx2)
            image = cv.imread(test_file)
            short_file_name = ntpath.basename(test_file)
            indices_detections_current_image = np.where(file_names == short_file_name)
            current_detections = detections[indices_detections_current_image]
            current_scores = scores[indices_detections_current_image]

            current_dad_detections = None
            current_dexter_detections = None
            current_deedee_detections = None
            current_mom_detections = None

            current_dad_scores = np.array([])
            current_dexter_scores = np.array([])
            current_deedee_scores = np.array([])
            current_mom_scores = np.array([])

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

                if predicted_label == 'dexter':
                    dexter_scores = np.concatenate((dexter_scores, scores[idx]), axis=None)
                    if dexter_detections is None:
                        dexter_detections = np.array([[x1, y1, x2, y2]])
                    else:
                        dexter_detections = np.concatenate((dexter_detections, np.array([[x1, y1, x2, y2]])), axis=0)
                    # print('dexter')
                elif predicted_label == 'dad':
                    dad_scores = np.concatenate((dad_scores, scores[idx]), axis=None)
                    if dad_detections is None:
                        dad_detections = np.array([[x1, y1, x2, y2]])
                    else:
                        dad_detections = np.concatenate((dad_detections, np.array([[x1, y1, x2, y2]])), axis=0)
                    # print('dad')
                elif predicted_label == 'deedee':
                    deedee_scores = np.concatenate((deedee_scores, scores[idx]), axis=None)
                    if deedee_detections is None:
                        deedee_detections = np.array([[x1, y1, x2, y2]])
                    else:
                        deedee_detections = np.concatenate((deedee_detections, np.array([[x1, y1, x2, y2]])), axis=0)
                    # print('deedee')
                elif predicted_label == 'mom':
                    mom_scores = np.concatenate((mom_scores, scores[idx]), axis=None)
                    if mom_detections is None:
                        mom_detections = np.array([[x1, y1, x2, y2]])
                    else:
                        mom_detections = np.concatenate((mom_detections, np.array([[x1, y1, x2, y2]])), axis=0)
                    # print('mom')

        return dad_detections, deedee_detections, dexter_detections, mom_detections, dad_scores, deedee_scores, dexter_scores, mom_scores

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int64)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        print(average_precision)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

    def eval_detections_character(self, detections, scores, file_names,ground_truth_path,character):
        ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int64)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(character + ' faces: average precision %.3f' % average_precision)
        plt.savefig('precizie_medie_' + character + '.png')
        plt.show()

