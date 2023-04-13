import os
import random
import shutil

def test_train_val_split(origin,save_path):
    
    ## Set image location
    img_path = os.path.join(origin, "Photos")
    annot_path = os.path.join(origin, "Yolo_Boxes")
    
    # Record photo/annotation names as a list
    img_list = os.listdir(img_path)
    annot_list = os.listdir(annot_path)

    
    # Check that both folders have the same number of files
    if (len(img_list) != len(annot_list)): 
        print("Error! Directories need the same number of files")
    #    return 0 # end function if not
    
    # Generate a random sample of numbers to use for the val sets (20% of the data)
    val_nums = random.sample(range(0,len(img_list)),round(len(img_list)*.2)) 
    
    # Pull the val images
    val_img_list = [img_list[i] for i in val_nums] # Pull the photos based on val_nums
    val_annot_list = [annot_list[i] for i in val_nums]
    
    # Remove the val photos from the train set
    train_img_list = list(set(img_list).difference(val_img_list))
    
    # Generate a random sample of numbers to use for the val sets (10% of the data)
    test_nums = random.sample(range(0,len(train_img_list)),round(len(img_list)*.1))
    
    # Pull the test images
    test_img_list = [train_img_list[i] for i in test_nums]
    test_annot_list = [annot_list[i] for i in test_nums]
    
    # Remove the test photos from the train set
    train_img_list = list(set(train_img_list).difference(test_img_list))
    train_annot_list = list(set(annot_list).difference(test_annot_list+val_annot_list))
    
    
    #### Save Photos ####
    
    # Delete any photos already in paths
    [os.remove(f) for f in os.join(save_path+'train')]
    [os.remove(f) for f in os.join(save_path+'val')]
    [os.remove(f) for f in os.join(save_path+'test')]
    
    
    # Move training images
    [shutil.copyfile(os.path.join(origin, 'Photos', i), os.path.join(save_path, 'images','train', i)) for i in train_img_list]  
    
    
    
test_train_val_split(origin = 'C:\School\Project\Machine-Learning\Hemo_data\Small Grid', 
                     save_path = 'C:\School\Project\Yolo\datasets\hemo')

