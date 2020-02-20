#ifndef POSTDIALOG_H
#define POSTDIALOG_H


#include "utils.h"
#include <QDialog>

using namespace cv;
using namespace std;

namespace Ui {
class PostDialog;
}

class PostDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PostDialog(QWidget *parent = 0);
    ~PostDialog();

    void plot(int _eval_type);

    std::vector<double> sift_plot_vector, surf_plot_vector, orb_plot_vector, brief_plot_vector, brisk_plot_vector, freak_plot_vector, latch_plot_vector, edd_plot_vector, project_plot_vector;
    QVector<double> cases_x;

private:
    Ui::PostDialog *pui;
};

#endif // POSTDIALOG_H
