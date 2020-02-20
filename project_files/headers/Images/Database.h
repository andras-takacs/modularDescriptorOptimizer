#ifndef DATABASE_H
#define DATABASE_H

#include "utils.h"


using namespace std;
using namespace cv;


class Database
{
public:
    Database();
    Database(int& _database_identifier, int& _computer_identifier, int& _test_or_train,int _sub_folder_id = 3);
    ~Database();

    int size, database_id, subfolder_id;
    string root_folder, labels_folder_name, images_folder_name,images_subfolder_name,homography_folder_name;
    vector<string> image_list,homography_list;


    void setDataBase(const int& database_identifier,const int& computer_identifier,const int& test_or_train, int _sub_folder_id = 3);
    void setSubFolders(const int& _db_subfolder, const int& _im_dataBase);

};

#endif // DATABASE_H
