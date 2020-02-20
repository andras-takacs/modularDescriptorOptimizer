#include "Images/Database.h"



Database::Database()
{
    size = 0;
    root_folder = "/root/folder/is/not/set/";
    labels_folder_name = "labels/";
    images_folder_name = "images/";
    images_subfolder_name = "subForlder/";
    database_id = -1;

}

Database::Database(int& _database_identifier, int& _computer_identifier, int& _test_or_train,int _sub_folder_id)
{
    setDataBase(_database_identifier, _computer_identifier, _test_or_train, _sub_folder_id);
}

Database::~Database()
{

}

void Database::setDataBase(const int &database_identifier, const int &computer_identifier, const int &test_or_train, int _sub_folder_id){

    string image_list_file="not set";
    string homography_list_file="not_set";


    if(database_identifier==LABELME_FACADE_DB_ALL){
        database_id = LABELME_FACADE_DB_ALL;

        if(computer_identifier==UNI_COMPUTER){ root_folder = "/home/andras/UAQ/Tesis/DatabaseCollection/LabelMeFacade/";
        }
        else if(computer_identifier==LAPTOP){ root_folder = "/media/andras/ResearchDatabases/LabelMeFacade/";
        }

        if (test_or_train==TRAIN){
            image_list_file = "loadTrainImage.txt";
        }else if(test_or_train==TEST){
            image_list_file = "loadTestImage.txt";
        }
        labels_folder_name = "labels/";
        images_folder_name = "images/";

    }else if(database_identifier==LABELME_FACADE_DB_JENA){
        database_id = LABELME_FACADE_DB_JENA;

        if(computer_identifier==UNI_COMPUTER){ root_folder = "/home/andras/UAQ/Tesis/DatabaseCollection/LabelMeFacade/";
        }
        else if(computer_identifier==LAPTOP){ root_folder = "/media/andras/ResearchDatabases/LabelMeFacade/";
        }

        if (test_or_train==TRAIN){
            image_list_file = "train.txt";
        }else if(test_or_train==TEST){
            image_list_file = "test.txt";
        }

        labels_folder_name = "labels/";
        images_folder_name = "images/";

    }else if(database_identifier==BRIGHTON_DB){
        database_id = BRIGHTON_DB;

        if(computer_identifier==UNI_COMPUTER){ root_folder = "../datasets/brighton/";
        }
        else if(computer_identifier==LAPTOP){ root_folder = "../../datasets/brighton/";
        }
        if (test_or_train==TRAIN){
            image_list_file = "brightonTrainImages.txt";
        }else if(test_or_train==TEST){
            image_list_file = "brightonTestImages.txt";
        }
        labels_folder_name = "labels/4_label_mask/";
        images_folder_name = "images/";

    }else if(database_identifier==TILDE_DB){
        database_id = TILDE_DB;

        if(computer_identifier==UNI_COMPUTER){
            root_folder = "../datasets/tilde/";
        }
        else if(computer_identifier==LAPTOP){
            root_folder = "../../datasets/tilde/";
        }

        setSubFolders(_sub_folder_id,database_identifier);

        if (test_or_train==TRAIN){
            image_list_file = "loadTrainImage.txt";
            images_folder_name = "train/";
        }else if(test_or_train==TEST){
            image_list_file = "loadTestImage.txt";
            images_folder_name = "test/image_color/";
        }
        labels_folder_name = "labels/";


    }else if(database_identifier==OXFORD_DB){
        database_id = OXFORD_DB;

        if(computer_identifier==UNI_COMPUTER){
            root_folder = "../datasets/Oxford/";
        }
        else if(computer_identifier==LAPTOP){
            root_folder = "../../datasets/Oxford/";
        }

        setSubFolders(_sub_folder_id,database_identifier);

        if(test_or_train==TEST){
            image_list_file = "loadTestImage.txt";
            images_folder_name = "test/image_color/";
            homography_list_file="loadHomographyMatrices.txt";
            homography_folder_name = "test/homography/";
        }
        labels_folder_name = "labels/";


    }

    std::cout<<"Computer identifier: "<<computer_identifier<<std::endl;
    std::cout<<"Root folder: "<<root_folder<<std::endl;

    ifstream inputList(root_folder+image_list_file);
    string line;

    if(!inputList){
        std::cout<<"Image list at"<<root_folder+image_list_file<< " could not be opened!"<<std::endl;
    }

    image_list.clear();
    while(std::getline(inputList, line)){

        image_list.push_back(line);
    }

    size = image_list.size();

    if(database_identifier==OXFORD_DB){

        ifstream hInputList(root_folder+homography_list_file);
        string hLine;

        if(!hInputList){
            std::cout<<"Homography list file could not be opened!"<<std::endl;
        }

        homography_list.clear();
        while(std::getline(hInputList, hLine)){

            homography_list.push_back(hLine);
        }
    }

//    std::cout<<"Got all the database information!"<<std::endl;

}


void Database::setSubFolders(const int& _db_subfolder, const int& _im_dataBase){

    if(_im_dataBase==TILDE_DB){

        switch (_db_subfolder)
        {
        case CHAMONIX:
            root_folder = root_folder+"WebcamRelease/Chamonix/";
            images_subfolder_name = "Chamonix";
            break;
        case COURBEVOIE:
            root_folder = root_folder+"WebcamRelease/Courbevoie/";
            images_subfolder_name = "Courbevoie";
            break;
        case FRANKFURT:
            root_folder = root_folder+"WebcamRelease/Frankfurt/";
            images_subfolder_name = "Frankfurt";
            break;
        case MEXICO:
            root_folder = root_folder+"WebcamRelease/Mexico/";
            images_subfolder_name = "Mexico";
            break;
        case PANORAMA:
            root_folder = root_folder+"WebcamRelease/Panorama/";
            images_subfolder_name = "Panorama";
            break;
        case STLOUIS:
            root_folder = root_folder+"WebcamRelease/StLouis/";
            images_subfolder_name = "StLouis";
            break;
        default:
            images_subfolder_name = "Not Set!!";
            break;
        }

    }else if(_im_dataBase==OXFORD_DB){

        switch (_db_subfolder)
        {
        case BARK:
            root_folder = root_folder+"bark/";
            images_subfolder_name = "bark";
            break;
        case BIKES:
            root_folder = root_folder+"bikes/";
            images_subfolder_name = "bikes";
            break;
        case BOAT:
            root_folder = root_folder+"boat/";
            images_subfolder_name = "boat";
            break;
        case GRAF:
            root_folder = root_folder+"graf/";
            images_subfolder_name = "graf";
            break;
        case LEUVEN:
            root_folder = root_folder+"leuven/";
            images_subfolder_name = "leuven";
            break;
        case NOTREDAME:
            root_folder = root_folder+"notredame/";
            images_subfolder_name = "notredame";
            break;
        case OBAMA:
            root_folder = root_folder+"obama/";
            images_subfolder_name = "obama";
            break;
        case PAINTEDLADIES:
            root_folder = root_folder+"paintedladies/";
            images_subfolder_name = "paintedladies";
            break;
        case RUSHMORE:
            root_folder = root_folder+"rushmore/";
            images_subfolder_name = "rushmore";
            break;
        case TREES:
            root_folder = root_folder+"trees/";
            images_subfolder_name = "trees";
            break;
        case UBC:
            root_folder = root_folder+"ubc/";
            images_subfolder_name = "ubc";
            break;
        case WALL:
            root_folder = root_folder+"wall/";
            images_subfolder_name = "wall";
            break;
        case YOSEMITE:
            root_folder = root_folder+"yosemite/";
            images_subfolder_name = "yosemite";
            break;
        default:
            images_subfolder_name = "Not Set!!";
            break;
        }


    }
}
