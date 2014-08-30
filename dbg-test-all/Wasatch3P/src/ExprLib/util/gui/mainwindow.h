#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QDockWidget;
class QTextEdit;
class EntryList;
class VarTable;

class Info;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = 0);
  ~MainWindow();

private slots:
  void on_actionGenerate_triggered();

  void on_actionTemplate_Parameters_triggered();

  void on_actionDependent_Fields_triggered();

  void on_actionSet_Expression_Name_triggered();

  void on_actionSave_As_triggered();

private:
  void setupEditor();

  std::string exprName_;
  QTextEdit* editor_;
  Info* info_;
  EntryList *tpList_;
  VarTable *depList_;
  QDockWidget *tpDock_, *depDock_;
  Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
