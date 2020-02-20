#ifndef BARDIALOG_H
#define BARDIALOG_H


#include "utils.h"
#include <QDialog>

using namespace cv;
using namespace std;

namespace Ui {
class BarDialog;
}

class BarDialog: public QDialog
{
    Q_OBJECT

public:
    explicit BarDialog (QWidget *parent = 0);
    ~BarDialog();

    void plotBar();

    QVector<double> time_plot_vector;

private:
    Ui::BarDialog *bui;
};


#endif // BARDIALOG_H
