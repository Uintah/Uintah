#include "exprnamedialog.h"
#include "ui_exprnamedialog.h"

ExprNameDialog::ExprNameDialog(QWidget *parent) :
  QDialog(parent),
  ui(new Ui::ExprNameDialog)
{
  ui->setupUi(this);
  value_ = "DefaultName";
  ui->lineEdit->setPlaceholderText(value_);
}

ExprNameDialog::~ExprNameDialog()
{
  delete ui;
}

void ExprNameDialog::on_buttonBox_accepted()
{
  value_ = ui->lineEdit->text();
}
