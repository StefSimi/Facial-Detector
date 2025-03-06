import os

class Parameters:
    def __init__(self):
        self.base_dir = 'data'
        self.dir_pos_examples = os.path.join('processed_greyscale_faces/combined64')#self.base_dir, 'exemplePozitive'
        self.dir_neg_examples = os.path.join('negative_samples_4000_96')#self.base_dir, 'exempleNegative'
        self.dir_test_examples = os.path.join('validare/validare') #self.base_dir, 'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join('validare/task1_gt_validare.txt')#self.base_dir, 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt'
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 64  #36
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 64  # 36
        self.overlap = 0.02
        self.number_positive_examples = 6713  # numarul exemplelor pozitive #5813
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.threshold = 2  # toate ferestrele cu scorul > threshold si maxime locale devin detectii
        self.has_annotations = True

        self.scaling_ratio = 0.05 #0.1
        self.use_hard_mining = False  # (optional) antrenare cu exemple puternic negative
        self.use_flip_images = True  # adauga imaginile cu fete oglindite