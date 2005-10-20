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
 * PathDialog.h
 *
 * Written by:
 *  Ayla Khan
 *  SCI
 *  University of Utah
 *  November 2004
 *
 */


#ifndef PathDialog_h
#define PathDialog_h

#include <qvariant.h>
#include <qdialog.h>

#include <vector>

class QVBoxLayout;
class QHBoxLayout;
class QSpacerItem;
class QComboBox;
class QGridLayout;
class QLabel;
class QLineEdit;
class QPushButton;
class QString;

class PathDialog : public QDialog
{
    Q_OBJECT

public:
    PathDialog(QWidget* parent = 0, const char* name = 0, bool modal = TRUE, WFlags fl = 0);
    ~PathDialog();

    QString selectedDirectory() const;
    QString selectedComponentModel() const;
    //void insertComponentModels(std::vector<std::string> &models);
    void insertComponentModels();

protected:
    QVBoxLayout* layoutDialog;
    QSpacerItem* verticalSpacing;
    QHBoxLayout* layoutName;
    QHBoxLayout* layoutPath;
    QHBoxLayout* layoutButtons;
    QSpacerItem* horizontalSpacing;

protected slots:
    virtual void languageChange();

private:
    QLabel* textLabelComponents;
    QComboBox* comboBoxName;
    QPushButton* buttonPath;
    QLineEdit* lineEditPath;
    QPushButton* buttonHelp;
    QPushButton* buttonOk;
    QPushButton* buttonCancel;
    QString workingDir;

private slots:
    void fileDialog();
};

#endif // PathDialog_h
