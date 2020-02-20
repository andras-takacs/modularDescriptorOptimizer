//
// Created by andras on 06/02/20.
//

#ifndef PARSINGCSV_H
#define PARSINGCSV_H

#include "utils.h"
#include "boost/algorithm/string.hpp"
#include <typeinfo>


class ParsingCSV{

    std::string fileName;
    std::string delimeter;

public:

    ParsingCSV();
    ~ParsingCSV();

    ParsingCSV(std::string filename, std::string delm = ","):fileName(filename), delimeter(delm){ }

    // Function to fetch data from a CSV File
    std::vector<std::vector<int> > getData();
    void printVector(std::vector<int> const &input);




};

#endif //PARSINGCSV_H
