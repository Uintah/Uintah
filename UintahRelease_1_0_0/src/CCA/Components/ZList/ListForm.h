/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

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
