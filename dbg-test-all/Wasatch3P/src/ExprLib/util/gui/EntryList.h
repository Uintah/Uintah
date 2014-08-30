#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <string>

class QListWidgetItem;
namespace Ui {
class EntryList;
}

class EntryList : public QDialog
{
  Q_OBJECT

public:
  explicit EntryList( QWidget *parent = 0 );
  ~EntryList();

  std::vector<std::string> get_list_entries() const;
  std::string get_result_fieldt_name() const;
  bool is_result_type_tparam() const;

private slots:
  void on_rm_button_clicked();
  void on_add_button_clicked();

  void on_listWidget_itemClicked();

  void on_checkBox_clicked();

  void on_lineEdit_textChanged(const QString &arg1);

private:
  Ui::EntryList *ui;
  std::string resultFieldT_;
};

#endif // DIALOG_H
