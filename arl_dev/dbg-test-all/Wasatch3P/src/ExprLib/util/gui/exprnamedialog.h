#ifndef EXPRNAMEDIALOG_H
#define EXPRNAMEDIALOG_H

#include <QDialog>
#include <string>

namespace Ui {
class ExprNameDialog;
}

class ExprNameDialog : public QDialog
{
  Q_OBJECT

public:
  explicit ExprNameDialog(QWidget *parent = 0);
  ~ExprNameDialog();
  std::string get_string_value() const{ return value_.toStdString(); }

private slots:
  void on_buttonBox_accepted();

private:
  Ui::ExprNameDialog *ui;
  QString value_;
};

#endif // EXPRNAMEDIALOG_H
