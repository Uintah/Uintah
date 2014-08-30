#include "EntryList.h"
#include "ui_EntryList.h"

#include <QInputDialog>
#include <QListWidgetItem>
#include <QMessageBox>

EntryList::EntryList(QWidget *parent) :
  QDialog(parent),
  ui(new Ui::EntryList)
{
  ui->setupUi(this);
  ui->lineEdit->setPlaceholderText("FieldT");
}

EntryList::~EntryList()
{
  delete ui;
}

void EntryList::on_add_button_clicked()
{
  QInputDialog dlg;
  QString name = dlg.getText( this, "Enter the name", "", QLineEdit::Normal );
  QListWidgetItem *item = new QListWidgetItem( name, ui->listWidget );
  item->setFlags( item->flags() | Qt::ItemIsEditable );
  ui->listWidget->addItem( item );
}

void EntryList::on_rm_button_clicked()
{
  // remove
  QListWidget* lw = ui->listWidget;
  lw->takeItem( lw->row( lw->currentItem() ) );
  if( lw->count() == 0 )
    ui->rm_button->setDisabled(true);
}

void EntryList::on_listWidget_itemClicked()
{
  ui->rm_button->setDisabled(false);
}

std::vector<std::string> EntryList::get_list_entries() const
{
  std::vector<std::string> names;
  for( int i=0; i<ui->listWidget->count(); ++i ){
    names.push_back( ui->listWidget->item(i)->text().toStdString() );
  }
  return names;
}

bool EntryList::is_result_type_tparam() const
{
  return ui->checkBox->isChecked();
}

std::string EntryList::get_result_fieldt_name() const
{
  return ui->lineEdit->text().toStdString();
}

void EntryList::on_checkBox_clicked()
{
  const QString entry = ui->lineEdit->text();
  QListWidget* lw = ui->listWidget;
  const bool isInList = !lw->findItems( entry, Qt::MatchFixedString | Qt::MatchCaseSensitive ).empty();

  if( ui->checkBox->isChecked() ){
    // the box was just checked - ensure that it is in the list
    if( entry.isEmpty() ){
      QMessageBox box;
      box.setText("ERROR: You must specify a field name");
      box.exec();
      ui->checkBox->setChecked(false);
      return;
    }
    if( !isInList ){
      QListWidgetItem *item = new QListWidgetItem( ui->lineEdit->text(), lw );
      item->setFlags( Qt::NoItemFlags );
      lw->insertItem( 0, item );
    }
  }
  else{
    // the box was just unchecked - remove from list
    if( isInList ) lw->takeItem( 0 );
  }
  lw->show();
}

void EntryList::on_lineEdit_textChanged(const QString &arg1)
{
  if( ui->checkBox->isChecked() )
    ui->listWidget->item(0)->setText(arg1);
}
