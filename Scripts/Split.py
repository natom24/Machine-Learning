import os
import random
import shutil

def test_train_val_split(origin,save_path):
    
    ## Set image location
    img_path = os.path.join(origin, "Photos")
    annot_path = os.path.join(origin, "VOC_Boxes")
    
    # Record photo/annotation names as a list
    img_list = os.listdir(img_path)
    annot_list = os.listdir(annot_path)

    
    # Check that both folders have the same number of files
    if (len(img_list) != len(annot_list)): 
        print("Error! Directories need the same number of files")
    #    return 0 # end function if not
    
    # Generate a random sample of numbers to use for the val sets (20% of the data)
    total_nums = random.sample(range(0,len(img_list)-1),round(len(img_list)*.3)) 
    
    # Split the nums into correct proportions for test and train
    val_nums = total_nums[:round(len(total_nums)*(2/3))]
    test_nums = total_nums[round(len(total_nums)*(2/3)):]
    
    # Pull the val images
    val_img_list = [img_list[i] for i in val_nums] # Pull the photos based on val_nums
    val_annot_list = [annot_list[i] for i in val_nums]
    
    # Pull the test images
    test_img_list = [img_list[i] for i in test_nums]
    test_annot_list = [annot_list[i] for i in test_nums]

    # Remove the val photos from the train set
    #train_img_list = list(set(img_list).difference(val_img_list))
        
    # Remove the test photos from the train set
    train_img_list = list(set(img_list).difference(test_img_list+val_img_list))
    train_annot_list = list(set(annot_list).difference(test_annot_list + val_annot_list))
    
    #### Save Photos ####
    
    # Delete any photos already in image paths
    [os.remove(os.path.join(save_path, 'images', 'train', f)) for f in os.listdir(os.path.join(save_path, 'images', 'train'))]
    [os.remove(os.path.join(save_path, 'images', 'val', f)) for f in os.listdir(os.path.join(save_path, 'images', 'val'))]
    [os.remove(os.path.join(save_path, 'images', 'test', f)) for f in os.listdir(os.path.join(save_path, 'images', 'test'))]
  
    # Delete any photos already in label paths
    [os.remove(os.path.join(save_path, 'labels', 'train', f)) for f in os.listdir(os.path.join(save_path, 'labels', 'train'))]
    [os.remove(os.path.join(save_path, 'labels', 'val', f)) for f in os.listdir(os.path.join(save_path, 'labels', 'val'))]
    [os.remove(os.path.join(save_path, 'labels', 'test', f)) for f in os.listdir(os.path.join(save_path, 'labels', 'test'))]
    
    
    
    # Move training data
    [shutil.copyfile(os.path.join(img_path, i), os.path.join(save_path, 'images','train', i)) for i in train_img_list]  
    [shutil.copyfile(os.path.join(annot_path, i), os.path.join(save_path, 'labels','train', i)) for i in train_annot_list]
    
    # Move val Data
    [shutil.copyfile(os.path.join(img_path, i), os.path.join(save_path, 'images','val', i)) for i in val_img_list]  
    [shutil.copyfile(os.path.join(annot_path, i), os.path.join(save_path, 'labels','val', i)) for i in val_annot_list]

    # Mone test Data
    [shutil.copyfile(os.path.join(img_path, i), os.path.join(save_path, 'images','test', i)) for i in test_img_list]  
    [shutil.copyfile(os.path.join(annot_path, i), os.path.join(save_path, 'labels','test', i)) for i in test_annot_list]

    
test_train_val_split(origin = 'C:\School\Project\Machine-Learning\Hemo_data\Small_Grid', save_path = 'C:\School\Project\Machine-Learning\Hemo_data\VOC_Split')

