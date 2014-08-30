#ifndef VARTABLE_H
#define VARTABLE_H

#include <QDialog>
#include <vector>
#include <string>
#include <ostream>

namespace Ui {
class VarTable;
}

struct VarEntry{
  std::string name, type;
};

std::ostream& operator<<( std::ostream& os, const VarEntry& v );

class VarTable : public QDialog
{
  Q_OBJECT

public:
  explicit VarTable(QWidget *parent = 0);
  ~VarTable();
  std::vector< VarEntry > get_entries() const;

private slots:
  void on_addButton_clicked();

  void on_rmButton_clicked();

private:
  Ui::VarTable *ui;
};

#endif // VARTABLE_H
