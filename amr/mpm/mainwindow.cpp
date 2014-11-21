#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "InitialConditions.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //bu default AMR flag is set to true
    AMR = true;
    Vec2D v1 = {0.0, 0.0};
    //Vec2D v2 = {12.0, 12.0};
    Vec2D v2 = {100.0, 100.0};
    BoundingBox domain(v1, v2);
    //initial condition combobox
    ui->ic_box->addItem("advection");
    ui->ic_box->addItem("compression");
    /*
    mpm.SetDomain(domain, 1);
    mpm.GenerateParticles(const_vel_gen);
    mpm.AdaptMesh();
    */
    //refining 4 center elements
    mpm.SetDomain(domain, 2);
    mpm.SetIC(p_gen);
    mpm.SetScene(&scene);
    mpm.Initialize();
    //mpm.AdaptMesh();
    //mpm.RefineElementByID(1);
    //mpm.ForceMeshUpdate();
    //mpm.RefineElementByID(5);
    //mpm.ForceMeshUpdate();
    //mpm.RefineElementByID(11);
    //mpm.RefineElementByID(6);
    //mpm.RefineElementByID(8);
    //mpm.RefineElementByID(7);

    scene.setBackgroundBrush(QBrush(Qt::white));

    //connecting scene to view and inverting y-axis
    ui->view->setScene(&scene);
    ui->view->scale(1.0,-1.0);
    ui->view->setRenderHints(QPainter::Antialiasing | QPainter::SmoothPixmapTransform);
    ui->view->show();

    //timestep edit field
    ui->timestep->setValidator(new QDoubleValidator(0.00001, 1000.0, 5, ui->timestep));
    ui->timestep->setText("0.1");
    //timer step edit field
    ui->timer_step->setValidator(new QIntValidator(1, 1000, ui->timer_step));
    ui->timer_step->setText("50");
    //times initialization
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(step()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::step()
{
    try
    {
        mpm.DoTimestep(ui->timestep->text().toDouble(), AMR);
    }
    catch(Error e)
    {
        if(e == Error::P_OUTSIDE_DOMAIN)
        {
            cout << "\nParticle is outside domain, stopping simulation\n";
            //stopping the timer and releasing the run button
            if(timer->isActive()) timer->stop();
            //if(ui->run->isFlat()) ui->run->click();
            if(ui->run->isChecked()) ui->run->setChecked(false);
        }
    }

    scene.update(scene.sceneRect());
}

void MainWindow::on_next_clicked()
{
    step();
    //mpm.DoTimestep(ui->timestep->text().toDouble(), AMR);
    //scene.update(scene.sceneRect());
}

void MainWindow::on_run_toggled(bool checked)
{
    if(!checked)
    {
        timer->stop();
        ui->run->setText("Run");
    }
    else
    {
        ui->run->setText("Stop");
        timer->start(ui->timer_step->text().toInt());
    }
}

void MainWindow::on_amr_toggled(bool checked)
{
    if(!checked)
    {
        AMR = false;
        ui->amr->setText("AMR: OFF");
    }
    else
    {
        AMR = true;
        ui->amr->setText("AMR: ON");
    }
}

void MainWindow::on_ic_box_currentIndexChanged(int index)
{
    switch(index)
    {
    case 0:
        p_gen = const_vel_gen;
        break;
    case 1:
        p_gen = deformation_gen;
        break;
    };
}

void MainWindow::on_reset_clicked()
{
    mpm.SetIC(p_gen);
    mpm.Initialize();
    scene.update(scene.sceneRect());
}
