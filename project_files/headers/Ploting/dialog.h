#ifndef DIALOG_H
#define DIALOG_H

#include "utils.h"
#include <QDialog>

using namespace cv;
using namespace std;

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

    void plot(int _eval_type);

    vector<QVector<double> >sift_plot_vector, surf_plot_vector, orb_plot_vector, brief_plot_vector, brisk_plot_vector, freak_plot_vector, akaze_plot_vector, project_plot_vector;

private:
    Ui::Dialog *ui;
};

#endif // DIALOG_H
