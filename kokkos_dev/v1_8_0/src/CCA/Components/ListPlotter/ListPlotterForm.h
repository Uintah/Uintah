/****************************************************************************
** Form interface generated from reading ui file 'ListPlotterForm.ui'
**
** Created: Tue Apr 9 18:07:16 2002
**      by:  The User Interface Compiler (uic)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/
#ifndef LISTPLOTTERFORM_H
#define LISTPLOTTERFORM_H

#include <qvariant.h>
#include <qdialog.h>
class QVBoxLayout; 
class QHBoxLayout; 
class QGridLayout; 
class QCheckBox;
class QPushButton;
class ZGraph;

class ListPlotterForm : public QDialog
{ 
    Q_OBJECT

public:
    ListPlotterForm( QWidget* parent = 0, const char* name = 0, bool modal = TRUE, WFlags fl = 0 );
    ~ListPlotterForm();

    ZGraph* listZGraph;
    QCheckBox* connectedCheckBox;
    QPushButton* closeQuitPushButton;


signals:
    void dataChanged(const double*, int);

public slots:
    virtual void setData( const double * val, int size );

protected:
    QHBoxLayout* ListPlotterFormLayout;
    QVBoxLayout* Layout8;
    QHBoxLayout* Layout4;
};

#endif // LISTPLOTTERFORM_H
