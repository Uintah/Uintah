#include <QtGui>
#include <iostream>
#include <fstream>

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "EntryList.h"
#include "Highlighter.h"
#include "vartable.h"
#include "exprnamedialog.h"

#include "CreateExpr.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  exprName_ = "DefaultName";
  info_ = new Info();

  tpList_  = new EntryList();
  depList_ = new VarTable();

  tpDock_  = new QDockWidget( "Template Parameters" );
  depDock_ = new QDockWidget( "Expression Dependencies" );

  tpDock_ ->setWidget( tpList_  );
  depDock_->setWidget( depList_ );

  addDockWidget( Qt::LeftDockWidgetArea, tpDock_ );
  addDockWidget( Qt::LeftDockWidgetArea, depDock_ );

  tpDock_ ->hide();
  depDock_->hide();

  setupEditor();
}

MainWindow::~MainWindow()
{
  delete ui;
  delete info_;
}

void MainWindow::on_actionGenerate_triggered()
{
  info_->clear();
  info_->set( Info::EXPR_NAME, exprName_ );
  info_->set( Info::FILE_NAME, exprName_ );
  info_->set( Info::FIELD_TYPE_NAME, tpList_->get_result_fieldt_name() );

  const std::vector<std::string> tpNames = tpList_->get_list_entries();
  for( std::vector<std::string>::const_iterator i=tpNames.begin(); i!=tpNames.end(); ++i ){
    info_->set( Info::EXTRA_TEMPLATE_PARAMS, *i );
    std::cout << *i << std::endl;
  }

  const std::vector< VarEntry > depVars = depList_->get_entries();
  for( std::vector<VarEntry>::const_iterator i=depVars.begin(); i!=depVars.end(); ++i ){
    info_->set_dep_field( i->name, i->type );
    std::cout << *i << std::endl;
  }

  info_->finalize();

  // update the window display.
  CreateExpr ce( *info_ );
  editor_->clear();
  editor_->setPlainText( QString::fromStdString(ce.get_stream().str()) );
}

void MainWindow::on_actionTemplate_Parameters_triggered()
{
  if( tpDock_->isVisible() )
    tpDock_->hide();
  else
    tpDock_->showNormal();
}

void MainWindow::on_actionDependent_Fields_triggered()
{
  if( depDock_->isVisible() )
    depDock_->hide();
  else
    depDock_->showNormal();
}

void MainWindow::setupEditor()
{
  QFont font;
  font.setFamily("Courier");
  font.setFixedPitch(true);
  font.setPointSize(10);

  editor_ = new QTextEdit();
  editor_->setFont(font);

  new Highlighter( editor_->document() );
  setCentralWidget( editor_ );
}

void MainWindow::on_actionSet_Expression_Name_triggered()
{
  ExprNameDialog dlg;
  dlg.exec();
  std::cout << dlg.get_string_value() << std::endl;
  exprName_ = dlg.get_string_value();
}

void MainWindow::on_actionSave_As_triggered()
{
  QString fileName = QFileDialog::getSaveFileName( 0,
                                                   tr("Select File Name"),
                                                   "",
                                                   tr("C++ header files (*.h)") );
  if( fileName.isNull() ) return;

  std::ofstream fout( fileName.toStdString().c_str(), std::ios_base::out );
  fout << editor_->toPlainText().toStdString();
  fout.close();

  QMessageBox box;
  box.setText( QString( "File<br><tt>&nbsp;&nbsp;&nbsp;"+fileName+"</tt><br>has been generated" ) );
  box.exec();
}
