#ifndef TESTTILDE_H
#define TESTTILDE_H

#include "utils.h"

using namespace cv;
using namespace std;
using namespace chrono;

class TestTilde
{
public:
    TestTilde();
    ~TestTilde();


    vector<KeyPoint> testAndDump(const Mat &I,const string &pathFilter, const int &nbTest = 1, const char* ext = NULL, Mat* score = NULL);
    vector<KeyPoint> test_fast(const Mat &I,const string &pathFilter, const int &nbTest = 1, Mat* score = NULL);
    void evaluate();


};

#endif // TESTTILDE_H
