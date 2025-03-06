import os

class Parameters:
    def __init__(self):
        self.base_dir = 'data'
        #self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        self.dir_pos_examples = os.path.join('processed_greyscale_faces/combined')
        self.dir_neg_examples = os.path.join('negative_samples')
        #self.dir_test_examples = os.path.join(self.base_dir,'exempleTest/CMU+MIT')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.dir_test_examples = os.path.join('validare/validare')
        #self.path_annotations = os.path.join(self.base_dir, 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt')
        self.path_annotations = os.path.join('validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive #6713
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
