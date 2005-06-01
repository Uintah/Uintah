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

/*
 * ClusterDialog.h
 *
 * Written by:
 *  Ayla Khan
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  September 2004
 *
 *  Copyright (C) 2004 SCI Institute
 *
 */

#ifndef ClusterDialog_h
#define ClusterDialog_h

#include <qvariant.h>
#include <qdialog.h>

class QVBoxLayout;
class QHBoxLayout;
class QGridLayout;
class QSpacerItem;
class QLabel;
class QComboBox;
class QLineEdit;
class QSpinBox;
class QButtonGroup;
class QRadioButton;
class QPushButton;
class QString;

class ClusterDialog : public QDialog
{
    Q_OBJECT

public:
    ClusterDialog(const char* defaultLoader, const char* defaultDomain,
		    const char* defaultLogin, const char* defaultPath, QWidget* parent = 0,
		    const char* name = 0, bool modal = FALSE, WFlags fl = 0);
    ClusterDialog(QWidget* parent = 0, const char* name = 0,
		    bool modal = FALSE, WFlags fl = 0);
    ~ClusterDialog();

    QString loader() const;
    QString domain() const;
    QString login() const;
    QString path() const;
    QString password() const;
    QString copies() const;
    QString where() const;

protected:
    QVBoxLayout* layoutDialog;
    QSpacerItem* verticalSpacing1;
    QSpacerItem* verticalSpacing2;
    QHBoxLayout* layoutLoader;
    QHBoxLayout* layoutDomain;
    QHBoxLayout* layoutLogin;
    QHBoxLayout* layoutPath;
    QHBoxLayout* layoutPasswd;
    QHBoxLayout* layoutCopies;
    QGridLayout* layoutWhere;
    QHBoxLayout* layoutButton;
    QHBoxLayout* layoutButtons;
    QSpacerItem* horizontalSpacing;

protected slots:
    virtual void languageChange();

private:
    void setDefaultText(const char* defaultLoader, const char* defaultDomain, const char* defaultLogin, const char* defaultPath);
    void setWidgets(const char* name, bool modal, WFlags fl);

    QLabel* textLabelLoader;
    QComboBox* comboBoxLoader;
    QLabel* textLabelDomain;
    QComboBox* comboBoxDomain;
    QLabel* textLabelLogin;
    QComboBox* comboBoxLogin;
    QLabel* textLabelPath;
    QComboBox* comboBoxPath;
    QLabel* textLabelPasswd;
    QLineEdit* lineEditPasswd;
    QLabel* textLabelCopies;
    QSpinBox* spinBoxCopies;
    QButtonGroup* buttonGroup;
    QRadioButton* radioButtonCPU;
    QRadioButton* radioButtonNode;
    QLabel* textLabelWhere;
    QLineEdit* lineEditWhere;
    QPushButton* buttonHelp;
    QPushButton* pushButtonReset;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;
};

#endif
