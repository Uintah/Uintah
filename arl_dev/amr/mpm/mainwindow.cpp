#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "InitialConditions.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    Vec2D v1 = {0.0, 0.0};
    //Vec2D v2 = {12.0, 12.0};
    Vec2D v2 = {24.0, 24.0};
    BoundingBox domain(v1, v2);
    /*
    mpm.SetDomain(domain, 1);
    mpm.GenerateParticles(const_vel_gen);
    mpm.AdaptMesh();
    */

    //refining 4 center elements
    mpm.SetDomain(domain, 2);
    mpm.GenerateParticles(const_vel_gen);
    mpm.RefineElementByID(7);
    mpm.RefineElementByID(12);
    mpm.RefineElementByID(13);
    mpm.RefineElementByID(18);
    mpm.ForceMeshUpdate();
    mpm.RefineElementByID(21);

    scene.setBackgroundBrush(QBrush(Qt::white));
    mpm.InitQtScene(&scene);

    ui->view->setScene(&scene);
    ui->view->scale(1.0,-1.0);
    //ui->view->setAlignment(Qt::AlignLeft|Qt::AlignBottom);
    ui->view->show();

    //timestep edit field
    ui->timestep->setValidator(new QDoubleValidator(0.00001, 1000.0, 5, ui->timestep));
    ui->timestep->setText("0.1");
    //timer step edit field
    ui->timer_step->setValidator(new QIntValidator(1, 1000, ui->timer_step));
    ui->timer_step->setText("100");
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
    mpm.DoTimestep(ui->timestep->text().toDouble());
    scene.update(scene.sceneRect());
}

void MainWindow::on_next_clicked()
{
    mpm.DoTimestep(ui->timestep->text().toDouble());
    scene.update(scene.sceneRect());
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
