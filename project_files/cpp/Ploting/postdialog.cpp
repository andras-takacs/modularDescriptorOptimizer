#include "postdialog.h"
#include "ui_postDialog.h"

PostDialog::PostDialog(QWidget *parent) :
    QDialog(parent),
    pui(new Ui::PostDialog)
{

}

PostDialog::~PostDialog()
{

    delete pui;
}

void PostDialog::plot(int _eval_type)
{
    pui->setupUi(this);
    pui->plot->legend->setVisible(true);
    pui->plot->setFont(QFont("Helvetica",9));
    QString fileName;

    QVector<double> x(cases_x.size()),
            y1(sift_plot_vector.size()),
            y2(surf_plot_vector.size()),
            y3(orb_plot_vector.size()),
            y4(brief_plot_vector.size()),
            y5(brisk_plot_vector.size()),
            y6(freak_plot_vector.size()),
            y7(latch_plot_vector.size()),
            y8(edd_plot_vector.size()),
            y9(project_plot_vector.size());




    x = cases_x;
    for(int i=0;i<(int)sift_plot_vector.size();++i){
    y1[i] = sift_plot_vector[i]*100.0;
    y2[i] = surf_plot_vector[i]*100.0;
    y3[i] = orb_plot_vector[i]*100.0;
    y4[i] = brief_plot_vector[i]*100.0;
    y5[i] = brisk_plot_vector[i]*100.0;
    y6[i] = freak_plot_vector[i]*100.0;
    y7[i] = latch_plot_vector[i] * 100.0;
    y8[i] = edd_plot_vector[i]*100.0;
    y9[i] = project_plot_vector[i]*100.0;
    }

    pui->plot->addGraph();
    pui->plot->graph(0)->setName("SIFT");
    pui->plot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, QPen(QColor(0,0,153)), QBrush(QColor(0,0,153)), 6));
    pui->plot->graph(0)->setPen(QPen(QColor(0,0,153)));
    pui->plot->graph(0)->setData(x,y1);

    pui->plot->addGraph();
    pui->plot->graph(1)->setName("SURF");
    pui->plot->graph(1)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, QColor(204,0,0), QColor(204,0,0), 6));
    pui->plot->graph(1)->setPen(QPen(QColor(204,0,0)));
    pui->plot->graph(1)->setData(x,y2);

    pui->plot->addGraph();
    pui->plot->graph(2)->setName("ORB");
    pui->plot->graph(2)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssStar, QColor(255,128,0), QColor(255,128,0), 6));
    pui->plot->graph(2)->setPen(QPen(QColor(255,128,0)));
    pui->plot->graph(2)->setData(x,y3);

    pui->plot->addGraph();
    pui->plot->graph(3)->setName("BRIEF");
    pui->plot->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangleInverted, QColor(0,102,0), QColor(0,102,0), 6));
    pui->plot->graph(3)->setPen(QPen(QColor(0,102,0)));
    pui->plot->graph(3)->setData(x,y4);

    pui->plot->addGraph();
    pui->plot->graph(4)->setName("BRISK");
    pui->plot->graph(4)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, QColor(128,255,0), QColor(128,255,0), 6));
    pui->plot->graph(4)->setPen(QPen(QColor(128,255,0)));
    pui->plot->graph(4)->setData(x,y5);

    pui->plot->addGraph();
    pui->plot->graph(5)->setName("FREAK");
    pui->plot->graph(5)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, QColor(0,102,255), QColor(0,102,255), 6));
    pui->plot->graph(5)->setPen(QPen(QColor(0,102,255)));
    pui->plot->graph(5)->setData(x,y6);

    pui->plot->addGraph();
    pui->plot->graph(6)->setName("LATCH");
    pui->plot->graph(6)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssSquare, QColor(128,255,0), QColor(128,255,0), 6));
    pui->plot->graph(6)->setPen(QPen(QColor(128,255,0)));
    pui->plot->graph(6)->setData(x,y7);

    pui->plot->addGraph();
    pui->plot->graph(7)->setName("EDD");
    pui->plot->graph(7)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QPen("#705BE8"), QBrush("#705BE8"), 6));
    pui->plot->graph(7)->setPen(QPen("#705BE8"));
    pui->plot->graph(7)->setData(x,y8);

    pui->plot->addGraph();
    pui->plot->graph(8)->setName("MODULAR");
    pui->plot->graph(8)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDiamond, QColor(128,255,128), QColor(128,255,128), 6));
    pui->plot->graph(8)->setPen(QPen(QColor(128,255,0)));
    pui->plot->graph(8)->setData(x,y9);


    QVector<double> ticks, y_ticks;
    QVector<QString> labels, y_labels;

    pui->plot->xAxis->setRange(x.first(),x.back());
    pui->plot->yAxis->setRange(0,100);
    pui->plot->axisRect()->setMinimumMargins(QMargins(0,30,30,0));

    y_ticks << 0<<10<<20<<30<<40<<50<<60<<70<<80<<90<<100;
    y_labels << "0"<<"10"<<"20"<<"30"<<"40"<<"50"<<"60"<<"70"<<"80"<<"90"<<"100";
    pui->plot->yAxis->setAutoTicks(false);
    pui->plot->yAxis->setAutoTickLabels(false);
    pui->plot->yAxis->setTickVector(y_ticks);
    pui->plot->yAxis->setTickVectorLabels(y_labels);

    pui->plot->xAxis->setAutoTicks(false);
    pui->plot->xAxis->setAutoTickLabels(false);
    pui->plot->xAxis->setSubTickCount(0);

    if(_eval_type==LIGHT_CH_TR){
        ticks << 0<<1<<2<<3<<4;
        labels << "0"<<"1"<<"2"<<"3"<<"4";
         pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignCenter);
        pui->plot->xAxis->setLabel("Light Change case");
        fileName = "lightChange.pdf" ;
    }else if(_eval_type==LIGHT_COND_TR){
        ticks << 0<<1<<2<<3<<4;
        labels << "0"<<"1"<<"2"<<"3"<<"4";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignCenter|Qt::AlignBottom);
        pui->plot->xAxis->setLabel("Light Condition Change case");
        fileName = "lightCondChange.pdf" ;
    }else if(_eval_type==JPEG_COMPRESSION){
        ticks << 0<<1<<2<<3<<4;
        labels << "0"<<"1"<<"2"<<"3"<<"4";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignCenter);
        pui->plot->xAxis->setLabel("JPEG compression case");
        fileName = "jpegCompression.pdf" ;
    } else if(_eval_type==BLUR_TR){
        ticks << 1<<3<<5<<7<<9<<11<<13<<15<<17<<19;
        labels << "1"<<"3"<<"5"<<"7"<<"9"<<"11"<<"13"<<"15"<<"17"<<"19";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignRight|Qt::AlignCenter);
        pui->plot->xAxis->setLabel("Gaussian blur kernel size");
        fileName = "blur.pdf" ;
    }else if(_eval_type==SIZE_TR){
        ticks <<0<<1<<2<<3<<4<<5<<6<<7<<8;
        labels <<"60"<<"70"<<"80"<<"90"<<"100"<<"110"<<"120"<<"130"<<"150";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignBottom);
        pui->plot->xAxis->setLabel("Image rezise");
        fileName = "size.pdf" ;
    }else if(_eval_type==AFFINE_TR){
        ticks <<0<<1<<2<<3<<4<<5<<6<<7<<8<<9;
        labels << "1"<<"2"<<"3"<<"4"<<"5"<<"6"<<"7"<<"8"<<"9"<<"10";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignRight|Qt::AlignTop);
        pui->plot->xAxis->setLabel("Affine transformation case");
        fileName = "affine.pdf" ;
    }else if(_eval_type==ROTATION_TR){
        ticks << 0<<15<<30<<45<<60<<75<<90<<105<<120<<135<<150<<165<<180<<195<<210<<225<<240<<255<<270<<285<<300<<315<<330<<345<<360;
        labels << "0"<<"15"<<"30"<<"45"<<"60"<<"75"<<"90"<<"105"<<"120"<<"135"<<"150"<<"165"<<"180"<<
                  "195"<<"210"<<"225"<<"240"<<"255"<<"270"<<"285"<<"300"<<"315"<<"330"<<"345"<<"360";
        pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignCenter|Qt::AlignBottom);
        pui->plot->xAxis->setTickLabelRotation(60);
        pui->plot->xAxis->setLabel("Image rotation (angle)");
        fileName = "rotation.pdf" ;
    }else if(_eval_type==LIGHT_COND_TR2){

        ticks << 0<<1<<2<<3<<4<<5<<6<<7<<8;
        labels << "0"<<"1"<<"2"<<"3"<<"4"<<"5"<<"6"<<"7"<<"8"<<"9";
         pui->plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignLeft|Qt::AlignCenter);
        pui->plot->xAxis->setLabel("Light Condition Change 2 case");
        pui->plot->xAxis->setLabel("TILDE image library");
        fileName = "tildeEvaluation.pdf" ;
    }else{
        fileName = "graph.pdf" ;
    }

    pui->plot->xAxis->setTickVector(ticks);
    pui->plot->xAxis->setTickVectorLabels(labels);

    pui->plot->xAxis2->setVisible(true);
    pui->plot->yAxis2->setVisible(true);
    pui->plot->xAxis2->setTickLabels(false);
    pui->plot->yAxis2->setTickLabels(false);
    pui->plot->xAxis2->setTicks(false);
    pui->plot->yAxis2->setTicks(false);

    pui->plot->yAxis->setLabel("Good mateches in %");


    QFile file("../genetic_results/"+fileName);

    if (!file.open(QIODevice::WriteOnly|QFile::WriteOnly))
    {
        QMessageBox::warning(0,"Could not create Project File",
                             QObject::tr( "\n Could not create Project File on disk"));


    }
    pui->plot->savePdf("../genetic_results/"+fileName,  0, 0, 1.0);

}
