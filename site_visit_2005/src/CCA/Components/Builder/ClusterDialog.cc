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
 * ClusterDialog.cc
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

#include <qvariant.h>
#include <qpushbutton.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qlineedit.h>
#include <qspinbox.h>
#include <qbuttongroup.h>
#include <qradiobutton.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>

#include <CCA/Components/Builder/ClusterDialog.h>

#include <iostream>

ClusterDialog::ClusterDialog(const char* defaultLoader, const char* defaultDomain, const char* defaultLogin, QWidget* parent, const char* name, bool modal, WFlags fl) :  QDialog(parent, name, modal, fl)
{
    setWidgets(name, modal, fl);
    setDefaultText(defaultLoader, defaultDomain, defaultLogin);
}

/*
 *  Constructs a ClusterDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ClusterDialog::ClusterDialog(QWidget* parent, const char* name, bool modal, WFlags fl)
    : QDialog(parent, name, modal, fl)
{
    setWidgets(name, modal, fl);

}

void ClusterDialog::setWidgets(const char* name, bool modal, WFlags fl)
{
    if (!name) {
        setName("Add Loader command");
    }

    setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, sizePolicy().hasHeightForWidth()));
    setMinimumSize(QSize(450, 350));
    setMaximumSize(QSize(450, 350));
    QFont f(font());
    f.setPointSize(11);
    setFont(f); 
    setSizeGripEnabled(FALSE);
    setModal(TRUE);

    QWidget* privateLayoutWidget = new QWidget(this, "layoutDialog");
    privateLayoutWidget->setGeometry(QRect(0, 0, 450, 350));
    layoutDialog = new QVBoxLayout(privateLayoutWidget, 4, 6, "layoutDialog"); 

    layoutLoader = new QHBoxLayout(0, 0, 2, "layoutLoader"); 

    textLabelLoader = new QLabel(privateLayoutWidget, "textLabelLoader");
    textLabelLoader->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelLoader->sizePolicy().hasHeightForWidth()));
    textLabelLoader->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));
    layoutLoader->addWidget(textLabelLoader);

    comboBoxLoader = new QComboBox(FALSE, privateLayoutWidget, "comboBoxLoader");
    comboBoxLoader->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, comboBoxLoader->sizePolicy().hasHeightForWidth()));
    comboBoxLoader->setEditable(TRUE);
    comboBoxLoader->setSizeLimit( 5 );
    comboBoxLoader->setMaxCount(20);
    comboBoxLoader->setInsertionPolicy(QComboBox::AtTop);
    comboBoxLoader->setDuplicatesEnabled(FALSE);
    layoutLoader->addWidget(comboBoxLoader);
    layoutDialog->addLayout(layoutLoader);

    layoutDomain = new QHBoxLayout(0, 0, 2, "layoutDomain"); 

    textLabelDomain = new QLabel(privateLayoutWidget, "textLabelDomain");
    textLabelDomain->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelDomain->sizePolicy().hasHeightForWidth()));
    textLabelDomain->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));
    layoutDomain->addWidget(textLabelDomain);

    comboBoxDomain = new QComboBox(FALSE, privateLayoutWidget, "comboBoxDomain");
    comboBoxDomain->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, comboBoxDomain->sizePolicy().hasHeightForWidth()));
    comboBoxDomain->setEditable(TRUE);
    comboBoxDomain->setSizeLimit( 5 );
    comboBoxDomain->setMaxCount(20);
    comboBoxDomain->setInsertionPolicy(QComboBox::AtTop);
    comboBoxDomain->setDuplicatesEnabled(FALSE);
    layoutDomain->addWidget(comboBoxDomain);
    layoutDialog->addLayout(layoutDomain);

    layoutLogin = new QHBoxLayout(0, 0, 2, "layoutLogin"); 

    textLabelLogin = new QLabel(privateLayoutWidget, "textLabelLogin");
    textLabelLogin->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelLogin->sizePolicy().hasHeightForWidth()));
    textLabelLogin->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));
    layoutLogin->addWidget(textLabelLogin);

    comboBoxLogin = new QComboBox(FALSE, privateLayoutWidget, "comboBoxLogin");
    comboBoxLogin->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)5, (QSizePolicy::SizeType)0, 0, 0, comboBoxLogin->sizePolicy().hasHeightForWidth()));
    comboBoxLogin->setEditable(TRUE);
    comboBoxLogin->setSizeLimit( 5 );
    comboBoxLogin->setMaxCount(20);
    comboBoxLogin->setInsertionPolicy(QComboBox::AtTop);
    comboBoxLogin->setDuplicatesEnabled(FALSE);
    layoutLogin->addWidget(comboBoxLogin);
    layoutDialog->addLayout(layoutLogin);

    layoutPasswd = new QHBoxLayout(0, 0, 2, "layoutPasswd"); 

    textLabelPasswd = new QLabel(privateLayoutWidget, "textLabelPasswd");
    textLabelPasswd->setEnabled(FALSE);
    textLabelPasswd->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelPasswd->sizePolicy().hasHeightForWidth()));
    textLabelPasswd->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));
    layoutPasswd->addWidget(textLabelPasswd);

    lineEditPasswd = new QLineEdit(privateLayoutWidget, "lineEditPasswd");
    lineEditPasswd->setEnabled(FALSE);
    lineEditPasswd->setCursor(QCursor(4));
    lineEditPasswd->setMaxLength(100);
    lineEditPasswd->setEchoMode(QLineEdit::Password);
    lineEditPasswd->setAlignment(int(QLineEdit::AlignLeft));
    layoutPasswd->addWidget(lineEditPasswd);

    layoutDialog->addLayout(layoutPasswd);
    verticalSpacing1 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layoutDialog->addItem(verticalSpacing1);

    layoutCopies = new QHBoxLayout(0, 0, 2, "layoutCopies"); 

    textLabelCopies = new QLabel(privateLayoutWidget, "textLabelCopies");
    textLabelCopies->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));
    textLabelCopies->setEnabled(FALSE);
    layoutCopies->addWidget(textLabelCopies);

    spinBoxCopies = new QSpinBox(privateLayoutWidget, "spinBoxCopies");
    spinBoxCopies->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, spinBoxCopies->sizePolicy().hasHeightForWidth()));
    spinBoxCopies->setEnabled(FALSE);
    spinBoxCopies->setMinimumSize(QSize(70, 0));
    spinBoxCopies->setMaximumSize(QSize(70, 32767));
    spinBoxCopies->setButtonSymbols(QSpinBox::UpDownArrows);
    spinBoxCopies->setMaxValue(100);
    spinBoxCopies->setMinValue(1);
    spinBoxCopies->setValue(1);
    layoutCopies->addWidget(spinBoxCopies);
    layoutDialog->addLayout(layoutCopies);

    layoutWhere = new QGridLayout(0, 1, 1, 0, 2, "layoutWhere"); 

    buttonGroup = new QButtonGroup(privateLayoutWidget, "buttonGroup");
    buttonGroup->setEnabled(FALSE);
    buttonGroup->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, buttonGroup->sizePolicy().hasHeightForWidth()));
    buttonGroup->setMinimumSize(QSize(135, 35));
    buttonGroup->setMaximumSize(QSize(135, 35));
    buttonGroup->setFrameShadow(QButtonGroup::Sunken);
    buttonGroup->setLineWidth(1);
    buttonGroup->setAlignment(int(QButtonGroup::AlignCenter));
    buttonGroup->setCheckable(FALSE);
    buttonGroup->setProperty("selectedId", -1);

    QWidget* privateLayoutWidget_2 = new QWidget(buttonGroup, "layoutButton");
    privateLayoutWidget_2->setGeometry(QRect(6, 3, 124, 27));
    layoutButton = new QHBoxLayout(privateLayoutWidget_2, 2, 2, "layoutButton"); 

    radioButtonCPU = new QRadioButton(privateLayoutWidget_2, "radioButtonCPU");
    radioButtonCPU->setEnabled(FALSE);
    radioButtonCPU->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, radioButtonCPU->sizePolicy().hasHeightForWidth()));
    radioButtonCPU->setMinimumSize(QSize(60, 25));
    radioButtonCPU->setMaximumSize(QSize(60, 25));
    radioButtonCPU->setChecked(TRUE);
    layoutButton->addWidget(radioButtonCPU);

    radioButtonNode = new QRadioButton(privateLayoutWidget_2, "radioButtonNode");
    radioButtonNode->setEnabled(FALSE);
    radioButtonNode->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, radioButtonNode->sizePolicy().hasHeightForWidth()));
    radioButtonNode->setMinimumSize(QSize(60, 25));
    radioButtonNode->setMaximumSize(QSize(60, 25));
    radioButtonNode->setChecked(FALSE);
    layoutButton->addWidget(radioButtonNode);

    layoutWhere->addWidget(buttonGroup, 1, 1);

    textLabelWhere = new QLabel(privateLayoutWidget, "textLabelWhere");
    textLabelWhere->setEnabled(FALSE);
    textLabelWhere->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelWhere->sizePolicy().hasHeightForWidth()));
    textLabelWhere->setAlignment(int(QLabel::AlignVCenter | QLabel::AlignRight));

    layoutWhere->addWidget(textLabelWhere, 0, 0);

    lineEditWhere = new QLineEdit(privateLayoutWidget, "lineEditWhere");
    lineEditWhere->setEnabled(FALSE);
    lineEditWhere->setCursor(QCursor(4));
    lineEditWhere->setAlignment(int(QLineEdit::AlignLeft));

    layoutWhere->addWidget(lineEditWhere, 0, 1);
    layoutDialog->addLayout(layoutWhere);
    verticalSpacing2 = new QSpacerItem(20, 30, QSizePolicy::Minimum, QSizePolicy::Expanding);
    layoutDialog->addItem(verticalSpacing2);

    layoutButtons = new QHBoxLayout(0, 0, 2, "layoutButtons"); 

    buttonHelp = new QPushButton(privateLayoutWidget, "buttonHelp");
    buttonHelp->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, buttonHelp->sizePolicy().hasHeightForWidth()));
    buttonHelp->setAutoDefault(TRUE);
    layoutButtons->addWidget(buttonHelp);

    pushButtonReset = new QPushButton(privateLayoutWidget, "pushButtonReset");
    pushButtonReset->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, pushButtonReset->sizePolicy().hasHeightForWidth()));
    layoutButtons->addWidget(pushButtonReset);
    horizontalSpacing = new QSpacerItem(190, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    layoutButtons->addItem(horizontalSpacing);

    buttonOk = new QPushButton(privateLayoutWidget, "buttonOk");
    buttonOk->setSizePolicy(QSizePolicy((QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, buttonOk->sizePolicy().hasHeightForWidth()));
    buttonOk->setAutoDefault(TRUE);
    buttonOk->setDefault(TRUE);
    layoutButtons->addWidget(buttonOk);

    buttonCancel = new QPushButton(privateLayoutWidget, "buttonCancel");
    buttonCancel->setAutoDefault(TRUE);
    layoutButtons->addWidget(buttonCancel);
    layoutDialog->addLayout(layoutButtons);

    // buddies
    textLabelLoader->setBuddy(comboBoxLoader);
    textLabelDomain->setBuddy(comboBoxDomain);
    textLabelLogin->setBuddy(comboBoxLogin);
    textLabelPasswd->setBuddy(lineEditPasswd);
    textLabelCopies->setBuddy(spinBoxCopies);

    languageChange();

    resize(QSize(450, 350).expandedTo(minimumSizeHint()));
    clearWState(WState_Polished);

    // signals and slots connections
    connect(buttonOk, SIGNAL(clicked()), this, SLOT(accept()));
    connect(buttonCancel, SIGNAL(clicked()), this, SLOT(reject()));
}

/*
 *  Destroys the object and frees any allocated resources
 */
ClusterDialog::~ClusterDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

/*
 *  Sets the strings of the subwidgets using the current
 *  language.
 */
void ClusterDialog::languageChange()
{
    setCaption(tr("Start a parallel component loader"));
    QToolTip::add(this, tr("set command to add loader to framework", "toolTip comment"));
    QWhatsThis::add(this, tr("dialog to enter command to add loader to framework", "toolTip comment"));
    textLabelLoader->setText(tr("Loader name"));
    textLabelDomain->setText(tr("Domain name"));
    textLabelLogin->setText(tr("Login"));
    textLabelPasswd->setText(tr("Password"));
    QToolTip::add(lineEditPasswd, QString::null);
    textLabelCopies->setText(tr("Run this many copies"));
    QToolTip::add(spinBoxCopies, tr("Run this many copies of the program.", "Run this many copies of the program on the given nodes."));
    buttonGroup->setTitle(QString::null);
    radioButtonCPU->setText(tr("CPU"));
    radioButtonNode->setText(tr("Node"));
    textLabelWhere->setText(tr("Where"));
    QToolTip::add(lineEditWhere, QString::null);
    buttonHelp->setText(tr("&Help"));
    buttonHelp->setAccel(QKeySequence(tr("F1")));
    pushButtonReset->setText(tr("&Reset"));
    pushButtonReset->setAccel(QKeySequence(tr("Alt+R")));
    buttonOk->setText(tr("&OK"));
    buttonOk->setAccel(QKeySequence(QString::null));
    buttonCancel->setText(tr("&Cancel"));
    buttonCancel->setAccel(QKeySequence(QString::null));

}

void ClusterDialog::setDefaultText(const char* defaultLoader, const char* defaultDomain, const char* defaultLogin)
{
    comboBoxLoader->clear();
    comboBoxLoader->insertItem( tr(defaultLoader) );
    comboBoxLoader->setCurrentItem(0);

    comboBoxDomain->clear();
    comboBoxDomain->insertItem( tr(defaultDomain) );
    comboBoxDomain->setCurrentItem(0);

    comboBoxLogin->clear();
    comboBoxLogin->insertItem( tr(defaultLogin) );
    comboBoxLogin->setCurrentItem(0);
}

QString ClusterDialog::loader() const
{
    return comboBoxLoader->currentText();
}

QString ClusterDialog::domain() const
{
    return comboBoxDomain->currentText();
}

QString ClusterDialog::login() const
{
    return comboBoxLogin->currentText();
}

QString ClusterDialog::password() const
{
    return lineEditPasswd->text();
}

QString ClusterDialog::where() const
{
    return lineEditWhere->text();
}

