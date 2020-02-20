//
// Created by andras on 06/02/20.
//

#include "Dataset/ParsingCSV.h"

ParsingCSV::ParsingCSV(){

};

ParsingCSV::~ParsingCSV() {

};

std::vector<std::vector<int> > ParsingCSV::getData()
{
    std::ifstream file(fileName);

    std::vector<std::vector<int> > dataList;

    std::string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        std::vector<std::string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        std::vector<int> int_vec;
        for (size_t i = 0; i < vec[0].size(); ++i)
        {
            // This converts the char into an int and pushes it into vec
            int_vec.push_back(vec[0][i] - '0');  // The digits will be in the same order as before
        }
        dataList.push_back(int_vec);
    }
    // Close the File
    file.close();

    return dataList;
}

void ParsingCSV::printVector(std::vector<int> const &input){

        for(int i = 0; i < input.size(); i++) {
            std::cout << input.at(i) << ',';
        }
        std::cout<<endl;
}


