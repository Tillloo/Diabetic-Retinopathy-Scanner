import gdown
import os
import shutil



def main(): 
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Parent folder
    DATA_DIR = os.path.join(BASE_DIR,"data")

    model_url = "https://drive.google.com/uc?id=1BQCjRp-T8RhHXCuKY1DvkxInqTdQoQGB"
    model_path = os.path.join(BASE_DIR,"resNet_model.keras")

    if(not os.path.isdir(model_path)):
        gdown.download(model_url)

    try:
        os.mkdir(DATA_DIR)
    except:
        print("Directory Already Exists")
    os.chdir(DATA_DIR)


    test_url = "https://drive.google.com/uc?id=1Eioe9ymboPoFEcJMYskBzRLXQ1UbIult"
    train_url = "https://drive.google.com/uc?id=1RqG3xfTRYn3xC-2iGZss1cn3GnGE_Eqw"
    label_url = "https://drive.google.com/uc?id=1P_BzvW2KomHKv2OFg1czKPyX6VMCh_Kl"
    test_path = os.path.join(DATA_DIR,"test")
    train_path = os.path.join(DATA_DIR,"train")
    label_path = os.path.join(DATA_DIR,"trainLables.csv")
    format = "zip"

    if(not os.path.isdir(test_path)):
        gdown.download(test_url)
        test_file = os.path.join(DATA_DIR,"test.zip")
        shutil.unpack_archive(test_file, DATA_DIR, format)
        os.remove(test_file)  

    if(not os.path.isdir(train_path)):
        gdown.download(train_url)   
        train_file = os.path.join(DATA_DIR,"train.zip")
        shutil.unpack_archive(train_file, DATA_DIR, format)
        os.remove(train_file)  

    if(not os.path.isdir(label_path)):    
        gdown.download(label_url)    
    
    
    



if __name__ == "__main__":
    main()
