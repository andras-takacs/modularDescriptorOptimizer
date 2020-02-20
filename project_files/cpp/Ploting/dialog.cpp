#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{

}

Dialog::~Dialog()
{
    delete ui;
}


void Dialog::plot(int _eval_type)
{
    ui->setupUi(this);
    ui->plot->legend->setVisible(true);
    ui->plot->setFont(QFont("Helvetica",9));
    QString fileName;

    QVector<double> x(sift_plot_vector.at(0).size()),
            y1(sift_plot_vector.at(1).size()),
            y2(surf_plot_vector.at(1).size()),
            y3(orb_plot_vector.at(1).size()),
            y4(brief_plot_vector.at(1).size()),
            y5(brisk_plot_vector.at(1).size()),
            y6(freak_plot_vector.at(1).size()),
            y7(akaze_plot_vector.at(1).size()),
            y8(project_plot_vector.at(1).size());



    x = sift_plot_vector.at(0);
    y1 = sift_plot_vector.at(1);
    y2 = surf_plot_vector.at(1);
    y3 = orb_plot_vector.at(1);
    y4 = brief_plot_vector.at(1);
    y5 = brisk_plot_vector.at(1);
    y6 = freak_plot_vector.at(1);
    y7 = akaze_plot_vector.at(1);
    y8 = project_plot_vector.at(1);

    ui->plot->addGraph();
    ui->plot->graph(0)->setName("SIFT");
    ui->plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, QPen(QColor(0,0,153)), QBrush(QColor(0,0,153)), 6));
    ui->plot->graph(0)->setPen(QPen(QColor(0,0,153)));
    ui->plot->graph(0)->setData(x,y1);

    ui->plot->addGraph();
    ui->plot->graph(1)->setName("SURF");
    ui->plot->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, QColor(204,0,0), QColor(204,0,0), 6));
    ui->plot->graph(1)->setPen(QPen(QColor(204,0,0)));
    ui->plot->graph(1)->setData(x,y2);

    ui->plot->addGraph();
    ui->plot->graph(2)->setName("ORB");
    ui->plot->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, QColor(255,128,0), QColor(255,128,0), 6));
    ui->plot->graph(2)->setPen(QPen(QColor(255,128,0)));
    ui->plot->graph(2)->setData(x,y3);

    ui->plot->addGraph();
    ui->plot->graph(3)->setName("BRISK");
    ui->plot->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen(QColor(0,102,0)), QBrush(QColor(0,102,0)), 6));
    ui->plot->graph(3)->setPen(QPen(QColor(0,102,0)));
    ui->plot->graph(3)->setData(x,y4);

    ui->plot->addGraph();
    ui->plot->graph(4)->setName("BRIEF");
    ui->plot->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen("#705BE8"), QBrush("#705BE8"), 6));
    ui->plot->graph(4)->setPen(QPen("#705BE8"));
    ui->plot->graph(4)->setData(x,y5);

    ui->plot->addGraph();
    ui->plot->graph(5)->setName("FREAK");
    ui->plot->graph(5)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen("#705BE8"), QBrush("#705BE8"), 6));
    ui->plot->graph(5)->setPen(QPen("#705BE8"));
    ui->plot->graph(5)->setData(x,y6);

    ui->plot->addGraph();
    ui->plot->graph(6)->setName("AKAZE");
    ui->plot->graph(6)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen("#705BE8"), QBrush("#705BE8"), 6));
    ui->plot->graph(6)->setPen(QPen("#705BE8"));
    ui->plot->graph(6)->setData(x,y7);

    ui->plot->addGraph();
    ui->plot->graph(7)->setName("PROJECT");
    ui->plot->graph(7)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen("#705BE8"), QBrush("#705BE8"), 6));
    ui->plot->graph(7)->setPen(QPen("#705BE8"));
    ui->plot->graph(7)->setData(x,y8);




    QVector<double> ticks, y_ticks;
    QVector<QString> labels, y_labels;

    ui->plot->xAxis->setRange(x.first(),x.back());
    ui->plot->yAxis->setRange(0,100);
    ui->plot->axisRect()->setMinimumMargins(QMargins(0,30,30,0));

    y_ticks << 0<<10<<20<<30<<40<<50<<60<<70<<80<<90<<100;
    y_labels << "0"<<"10"<<"20"<<"30"<<"40"<<"50"<<"60"<<"70"<<"80"<<"90"<<"100";
    ui->plot->yAxis->setAutoTicks(false);
    ui->plot->yAxis->setAutoTickLabels(false);
    ui->plot->yAxis->setTickVector(y_ticks);
    ui->plot->yAxis->setTickVectorLabels(y_labels);

    ui->plot->xAxis->setAutoTicks(false);
    ui->plot->xAxis->setAutoTickLabels(false);
    ui->plot->xAxis->setSubTickCount(0);

    if(_eval_type==INTENSITY_CHANGE){
        ticks << 0<<1<<2<<3<<4<<5<<6<<7<<8<<9<<10<<11<<12;
        labels << "0.33"<<"0.4"<<"0.5"<<"0.67"<<"0.8"<<"0.9"<<"1.0"<<"1.1"<<"1.2"<<"1.5"<<"2.0"<<"2.5"<<"3";
         ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignCenter);
        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Intensity Scaling (with scalar a)");
        fileName = "intensityChange.pdf" ;
    }else if(_eval_type==INTENSITY_SHIFT){
        ticks << -100<<-75<<-50<<-25<<0<<25<<50<<75<<100;
//        labels << "-20"<<"-15"<<"-10"<<"-5"<<"0"<<"5"<<"10"<<"15"<<"20";
         labels << "-100"<<"-75"<<"-50"<<"-25"<<"0"<<"25"<<"50"<<"75"<<"100";
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignCenter|Qt::AlignBottom);
//        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Intensity Shift (with offset o1)");
        fileName = "intensityShift.pdf" ;
    }else if(_eval_type==INTENSITY_TEMP){
        ticks << 3200<<3400<<3600<<3800<<4000<<4200<<4400<<4600<<4800<<
                 5000<<5200<<5400<<5600<<5800<<6000<<6200<<6400<<6600;
        labels <<"3200"<<"3400"<<"3600"<<"3800"<<"4000"<<"4200"<<"4400"<<"4600"<<"4800"<<
                 "5000"<<"5200"<<"5400"<<"5600"<<"5800"<<"6000"<<"6200"<<"6400"<<"6600";
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignCenter);
        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Illumination Colour temperature (K)");
        fileName = "intensityTemperature.pdf" ;
    } else if(_eval_type==BLUR){
        ticks << 1<<3<<5<<7<<9<<11<<13<<15<<17<<19;
        labels << "1"<<"3"<<"5"<<"7"<<"9"<<"11"<<"13"<<"15"<<"17"<<"19";
//        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignRight|Qt::AlignCenter);
        ui->plot->xAxis->setLabel("Gaussian blur kernel size");
        fileName = "blur.pdf" ;
    }else if(_eval_type==SIZE){
        ticks << 0<<1<<2<<3<<4<<5<<6<<7<<8;
        labels << "0.6"<<"0.7"<<"0.8"<<"0.9"<<"1"<<"1.1"<<"1.2"<<"1.3"<<"1.5";
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignBottom);
//        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Image rezise");
        fileName = "size.pdf" ;
    }else if(_eval_type==AFFINE){
        ticks <<0<<1<<2<<3<<4<<5<<6<<7<<8<<9;
        labels << "1"<<"2"<<"3"<<"4"<<"5"<<"6"<<"7"<<"8"<<"9"<<"10";
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignRight|Qt::AlignTop);
//        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Affine transformation case");
        fileName = "affine.pdf" ;
    }else if(_eval_type==ROTATE){
        ticks << 0<<15<<30<<45<<60<<75<<90<<105<<120<<135<<150<<165<<180<<195<<210<<225<<240<<255<<270<<285<<300<<315<<330<<345<<360;
        labels << "0"<<"15"<<"30"<<"45"<<"60"<<"75"<<"90"<<"105"<<"120"<<"135"<<"150"<<"165"<<"180"<<
                  "195"<<"210"<<"225"<<"240"<<"255"<<"270"<<"285"<<"300"<<"315"<<"330"<<"345"<<"360";
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignCenter|Qt::AlignBottom);
        ui->plot->xAxis->setTickLabelRotation(60);
        ui->plot->xAxis->setLabel("Image rotation (angle)");
        fileName = "rotation.pdf" ;
    }else if(_eval_type==TILDE_EVAL){

        ui->plot->xAxis->setAutoTicks(true);
        ui->plot->xAxis->setAutoTickLabels(true);
        ui->plot->xAxis->setTickStep(5);
        ui->plot->xAxis->setSubTickCount(5);
        ui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignTop);
        ui->plot->xAxis->setLabel("TILDE image library");
        fileName = "tildeEvaluation.pdf" ;
    }else{
        fileName = "graph.pdf" ;
    }

    ui->plot->xAxis->setTickVector(ticks);
    ui->plot->xAxis->setTickVectorLabels(labels);

    ui->plot->xAxis2->setVisible(true);
    ui->plot->yAxis2->setVisible(true);
    ui->plot->xAxis2->setTickLabels(false);
    ui->plot->yAxis2->setTickLabels(false);
    ui->plot->xAxis2->setTicks(false);
    ui->plot->yAxis2->setTicks(false);

    ui->plot->yAxis->setLabel("Good mateches in %");


    QFile file("../"+fileName);

    if (!file.open(QIODevice::WriteOnly|QFile::WriteOnly))
    {
        QMessageBox::warning(0,"Could not create Project File",
                             QObject::tr( "\n Could not create Project File on disk"));


    }
    ui->plot->savePdf("../"+fileName,  0, 0, 1.0);

}

