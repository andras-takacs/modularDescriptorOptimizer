#include "bardialog.h"
#include "ui_barPlot.h"

BarDialog::BarDialog(QWidget *parent) :
    QDialog(parent),
    bui(new Ui::BarDialog)
{

}

BarDialog::~BarDialog()
{
    delete bui;
}


void BarDialog::plotBar(){

    bui->setupUi(this);
//    bui->bar_plot->legend->setVisible(true);
    bui->bar_plot->setFont(QFont("Helvetica",9));
    QString fileName = "calculation_time.pdf";

    QVector<double> ticks;
    QVector<QString> labels;
    //    QVector<double> time_v(time_plot_vector.size());
    ticks << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9;
    labels << "SIFT" << "SURF" << "ORB" << "BRIEF" << "BRISK" << "FREAK" << "LATCH" << "EDD" << "MODULAR";

    QVector<double> times = time_plot_vector;
    QCPBars *timings = new QCPBars(bui->bar_plot->xAxis, bui->bar_plot->yAxis);
    bui->bar_plot->addPlottable(timings);

        timings->setData(ticks,times);
        timings->setPen(QPen(QColor("#705BE8")));

        for(int x=1; x < 10; ++x){
            double y = times[x-1];

            //Creating and configuring an item
            QCPItemText *textLabel = new QCPItemText(bui->bar_plot);
            bui->bar_plot->addItem(textLabel);
            textLabel->setClipToAxisRect(false);
            textLabel->position->setAxes(bui->bar_plot->xAxis,bui->bar_plot->yAxis);
            textLabel->position->setType(QCPItemPosition::ptPlotCoords);
            //placing the item over the bar with a spacing of 0.25
            textLabel->position->setCoords(x,y+0.01);
            //Customizing the item
            textLabel->setText(QString::number(y,'f',3));
            textLabel->setFont(QFont(font().family(), 9));
            textLabel->setPen(QPen(Qt::NoPen));
        }


    bui->bar_plot->xAxis->setAutoTicks(false);
    bui->bar_plot->xAxis->setAutoTickLabels(false);
    bui->bar_plot->xAxis->setTickVector(ticks);
    bui->bar_plot->xAxis->setTickVectorLabels(labels);
    bui->bar_plot->xAxis->setLabel("Descriptor");
    bui->bar_plot->xAxis->setRange(0,10);
    bui->bar_plot->yAxis->setRange(0,0.4);
    bui->bar_plot->yAxis->setPadding(3); // a bit more space to the left border
    bui->bar_plot->yAxis->grid()->setSubGridVisible(true);
    bui->bar_plot->yAxis->setLabel("Calculation time in sec");

    bui->bar_plot->xAxis2->setVisible(true);
    bui->bar_plot->yAxis2->setVisible(true);
    bui->bar_plot->xAxis2->setTickLabels(false);
    bui->bar_plot->yAxis2->setTickLabels(false);
    bui->bar_plot->xAxis2->setTicks(false);
    bui->bar_plot->yAxis2->setTicks(false);

        QFile file("../results/"+fileName);

        if (!file.open(QIODevice::WriteOnly|QFile::WriteOnly))
        {
            QMessageBox::warning(0,"Could not create Project File",
                                 QObject::tr( "\n Could not create Project File on disk"));


        }
        bui->bar_plot->savePdf("../results/"+fileName,  0, 0, 1.0);


}
