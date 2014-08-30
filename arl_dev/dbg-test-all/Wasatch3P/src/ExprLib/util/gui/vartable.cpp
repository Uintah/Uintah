#include <QtGui>

#include "vartable.h"
#include "ui_vartable.h"

std::ostream& operator<<( std::ostream& os, const VarEntry& v )
{
  os << "[ " << v.name << ", " << v.type << " ]" << std::endl;
  return os;
}

VarTable::VarTable(QWidget *parent) :
  QDialog(parent),
  ui(new Ui::VarTable)
{
  ui->setupUi(this);
  ui->tableWidget->verticalHeader()->setMovable(true);
}

VarTable::~VarTable()
{
  delete ui;
}

void VarTable::on_addButton_clicked()
{
  const int row = ui->tableWidget->rowCount();
  ui->tableWidget->insertRow( row );

//  QComboBox* box = new QComboBox();
//  box->addItem("STATE_NONE");
//  box->addItem("STATE_N");
//  box->addItem("STATE_NP1");
//  ui->tableWidget->setCellWidget(row,1,box);
  ui->rmButton   ->setEnabled(true);

  // set default values
  ui->tableWidget->setItem(row,0,new QTableWidgetItem("VarName"));
  ui->tableWidget->setItem(row,1,new QTableWidgetItem("FieldT"));
}


void VarTable::on_rmButton_clicked()
{
  const int row = ui->tableWidget->currentRow();
  ui->tableWidget->removeRow(row);
  if( ui->tableWidget->rowCount() == 0 ){
    ui->rmButton  ->setEnabled(false);
  }
}

std::vector< VarEntry > VarTable::get_entries() const
{
  std::vector<VarEntry> entries;
  const int nrow = ui->tableWidget->rowCount();
  for( int i=0; i<nrow; ++i ){
    VarEntry entry;
    QTableWidgetItem *item;
    item = ui->tableWidget->item(i,0);
    item = ui->tableWidget->item(i,2);
    entry.name    = ui->tableWidget->item(i,0)->text().toStdString();
//    const QWidget* cwidget = ui->tableWidget->cellWidget(i,1);
//    entry.context = dynamic_cast<const QComboBox*>(cwidget)->currentText().toStdString();
    entry.type    = ui->tableWidget->item(i,1)->text().toStdString();
    entries.push_back( entry );
  }
  return entries;
}
