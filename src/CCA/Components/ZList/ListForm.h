/****************************************************************************
** Form interface generated from reading ui file 'ListForm.ui'
**
** Created: Fri Apr 12 19:53:10 2002
**      by:  The User Interface Compiler (uic)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/
#ifndef LISTFORM_H
#define LISTFORM_H

#include <qvariant.h>
#include <qdialog.h>
class QVBoxLayout; 
class QHBoxLayout; 
class QGridLayout; 
class QLineEdit;
class QListBox;
class QListBoxItem;
class QPushButton;
class ZList;
class ListForm : public QDialog
{ 
    Q_OBJECT

public:
    ListForm(ZList *com, QWidget* parent = 0, const char* name = 0, bool modal = FALSE, WFlags fl = 0 );
    ~ListForm();

    QLineEdit* numLineEdit;
    QPushButton* insertPushButton;
    QListBox* numListBox;
    QPushButton* deletePushButton;
    QPushButton* refreshPushButton;
    QPushButton* closePushButton;


public slots:
    virtual void enableDelete( int i );
    virtual void enableInsert( const QString & s );
    virtual void insert();
    virtual void refresh();
    virtual void del();

protected:
    QHBoxLayout* ListFormLayout;
    QVBoxLayout* Layout8;
    QHBoxLayout* Layout2;
    QHBoxLayout* Layout7;
    QVBoxLayout* Layout6;
private:
    ZList *com;	
};

#endif // LISTFORM_H
