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
 * PathDialog.cc
 *
 * Written by:
 *  Ayla Khan
 *  SCI
 *  University of Utah
 *  November 2004
 *
 */

#include <wx/sizer.h>
#include <wx/combobox.h>
#include <wx/button.h>

#include <CCA/Components/GUIBuilder/XMLPathDialog.h>
#include <Core/Util/Environment.h>
#include <include/sci_metacomponents.h>

namespace GUIBuilder {

using namespace SCIRun;

// BEGIN_EVENT_TABLE(XMLPathDialog, wxDialog)
// END_EVENT_TABLE()

XMLPathDialog::XMLPathDialog(wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style) : wxDialog(parent, id, caption, pos, size, style)
{
  // The framework should have a way to query available component models
  // from a factory (available through the builder).
  // A dialog shouldn't know this much about framework's component models.
  componentModels.Add("CCA");
  // SCIRun Dataflow is discovered using SCIRun packages.
#if HAVE_BABEL
  componentModels.Add("Babel");
#endif
#if HAVE_VTK
  componentModels.Add("VTK");
#endif
#if HAVE_TAO
  componentModels.Add("CORBA");
  componentModels.Add("Tao");
#endif

  SetLayout();
}

///////////////////////////////////////////////////////////////////////////
// protected member functions

// constructor helpers

void XMLPathDialog::SetLayout()
{
  wxBoxSizer *topSizer = new wxBoxSizer(wxVERTICAL);
  topSizer->AddSpacer(10);
  //int centerFlags = wxALIGN_CENTER|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int leftFlags = wxALIGN_LEFT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;
  const int rightFlags = wxALIGN_RIGHT|wxLEFT|wxRIGHT|wxALIGN_CENTER_VERTICAL;

  wxBoxSizer *cmSizer = new wxBoxSizer(wxHORIZONTAL);
  // assumes that the componentModels string array has been filled
  cmComboBox = new wxComboBox(this, ID_COMBOBOX_COMP_MODEL, componentModels[0], wxDefaultPosition, wxDefaultSize, componentModels, wxCB_DROPDOWN|wxCB_SORT);
  cmSizer->Add(cmComboBox, 1, wxALIGN_CENTER, 0);
  topSizer->Add(cmSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  wxBoxSizer *okCancelSizer = new wxBoxSizer(wxHORIZONTAL);
  okCancelSizer->Add( new wxButton( this, wxID_OK, wxT("&OK") ), 1, leftFlags, 4 );
  okCancelSizer->Add( new wxButton( this, wxID_CANCEL, wxT("&Cancel") ), 1, rightFlags, 4 );
  topSizer->Add(okCancelSizer, 1, wxALIGN_CENTER, 0);
  topSizer->AddSpacer(10);

  SetAutoLayout(true);
  SetSizer(topSizer);

  topSizer->Fit(this);
  topSizer->SetSizeHints(this);
}

}

#if 0

// #include <qvariant.h>
// #include <qpushbutton.h>
// #include <qlabel.h>
// #include <qlayout.h>
// #include <qtooltip.h>
// #include <qwhatsthis.h>
// #include <qfiledialog.h>
// #include <qmessagebox.h>
// #include <qcombobox.h>
// #include <qlineedit.h>


/*
 *  Constructs a PathDialog as a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
// PathDialog::PathDialog( QWidget* parent, const char* name, bool modal, WFlags fl )
//     : QDialog( parent, name, modal, fl )
// {
//     if ( !name ) {
//         setName( "PathDialog" );
//     }
//     setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, sizePolicy().hasHeightForWidth() ) );
//     setMinimumSize( QSize( 450, 200 ) );
//     setMaximumSize( QSize( 475, 190 ) );
//     QFont f( font() );
//     f.setPointSize( 11 );
//     setFont( f ); 
//     setSizeGripEnabled( FALSE );
//     setModal( TRUE );

//     QWidget* privateLayoutWidget = new QWidget( this, "layoutDialog" );
//     privateLayoutWidget->setGeometry( QRect( 0, 0, 470, 200 ) );
//     layoutDialog = new QVBoxLayout( privateLayoutWidget, 4, 6, "layoutDialog"); 

//     layoutName = new QHBoxLayout( 0, 0, 6, "layoutName"); 

//     textLabelComponents = new QLabel( privateLayoutWidget, "textLabelComponents" );
//     textLabelComponents->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, textLabelComponents->sizePolicy().hasHeightForWidth() ) );
//     textLabelComponents->setMinimumSize( QSize( 168, 25 ) );
//     textLabelComponents->setMaximumSize( QSize( 168, 25 ) );
//     QFont textLabelComponents_font(  textLabelComponents->font() );
//     textLabelComponents->setFont( textLabelComponents_font ); 
//     textLabelComponents->setAlignment( int( QLabel::AlignVCenter | QLabel::AlignRight ) );
//     textLabelComponents->setIndent( 0 );
//     layoutName->addWidget( textLabelComponents );

//     comboBoxName = new QComboBox( FALSE, privateLayoutWidget, "comboBoxName" );
//     comboBoxName->setMinimumSize( QSize( 250, 25 ) );
//     comboBoxName->setMaximumSize( QSize( 1000, 25 ) );
//     comboBoxName->setSizeLimit( 5 );
//     comboBoxName->setMaxCount( 50 );
//     comboBoxName->setDuplicatesEnabled( FALSE );
//     layoutName->addWidget( comboBoxName );
//     layoutDialog->addLayout( layoutName );

//     layoutPath = new QHBoxLayout( 0, 0, 6, "layoutPath"); 

//     buttonPath = new QPushButton( privateLayoutWidget, "buttonPath" );
//     buttonPath->setSizePolicy( QSizePolicy( (QSizePolicy::SizeType)0, (QSizePolicy::SizeType)0, 0, 0, buttonPath->sizePolicy().hasHeightForWidth() ) );
//     buttonPath->setMaximumSize( QSize( 120, 34 ) );
//     layoutPath->addWidget( buttonPath );

//     lineEditPath = new QLineEdit( privateLayoutWidget, "lineEditPath" );
//     lineEditPath->setMinimumSize( QSize( 350, 25 ) );
//     lineEditPath->setMaximumSize( QSize( 1000, 25 ) );
//     lineEditPath->setMaxLength( 1024 );
//     layoutPath->addWidget( lineEditPath );
//     layoutDialog->addLayout( layoutPath );
//     verticalSpacing = new QSpacerItem( 20, 60, QSizePolicy::Minimum, QSizePolicy::Expanding );
//     layoutDialog->addItem( verticalSpacing );

//     layoutButtons = new QHBoxLayout( 0, 0, 6, "layoutButtons"); 

//     buttonHelp = new QPushButton( privateLayoutWidget, "buttonHelp" );
//     QFont buttonHelp_font(  buttonHelp->font() );
//     buttonHelp->setFont( buttonHelp_font ); 
//     buttonHelp->setAutoDefault( TRUE );
//     layoutButtons->addWidget( buttonHelp );
//     horizontalSpacing = new QSpacerItem( 200, 20, QSizePolicy::Expanding, QSizePolicy::Minimum );
//     layoutButtons->addItem( horizontalSpacing );

//     buttonOk = new QPushButton( privateLayoutWidget, "buttonOk" );
//     QFont buttonOk_font(  buttonOk->font() );
//     buttonOk->setFont( buttonOk_font ); 
//     buttonOk->setAutoDefault( TRUE );
//     buttonOk->setDefault( TRUE );
//     layoutButtons->addWidget( buttonOk );

//     buttonCancel = new QPushButton( privateLayoutWidget, "buttonCancel" );
//     QFont buttonCancel_font(  buttonCancel->font() );
//     buttonCancel->setFont( buttonCancel_font ); 
//     buttonCancel->setAutoDefault( TRUE );
//     layoutButtons->addWidget( buttonCancel );
//     layoutDialog->addLayout( layoutButtons );
//     languageChange();
//     resize( QSize(475, 200).expandedTo(minimumSizeHint()) );
//     clearWState( WState_Polished );

//     // signals and slots connections
//     connect( buttonPath, SIGNAL( clicked() ), this, SLOT( fileDialog() ) );
//     connect( buttonOk, SIGNAL( clicked() ), this, SLOT( accept() ) );
//     connect( buttonCancel, SIGNAL( clicked() ), this, SLOT( reject() ) );
//     workingDir = QString(SCIRun::sci_getenv("SCIRUN_SRCDIR"));
// }

// /*
//  *  Destroys the object and frees any allocated resources
//  */
// PathDialog::~PathDialog()
// {
//     // no need to delete child widgets, Qt does it all for us
// }

// /*
//  *  Sets the strings of the subwidgets using the current
//  *  language.
//  */
// void PathDialog::languageChange()
// {
//     setCaption( tr( "Add directory with component XML files" ) );
//     textLabelComponents->setText( tr( "Component model name" ) );
//     // args to change...
//     insertComponentModels();

//     QToolTip::add( comboBoxName, tr( "Select component model name." ) );
//     QWhatsThis::add( comboBoxName, tr( "Select Component model type name here." ) );
//     buttonPath->setText( tr( "Get directory" ) );
//     QToolTip::add( buttonPath, tr( "Get path to XML files using the directory select dialog box." ) );
//     QWhatsThis::add( buttonPath, tr( "Shows select directory dialog box." ) );
//     QToolTip::add( lineEditPath, tr( "Enter the full path to the directory storing the XML descriptions for components belonging to the selected component model.\nDisplays directory selected using the directory select dialog box." ) );
//     QWhatsThis::add( lineEditPath, tr( "Use this to enter the full path to XML files for the selected component model or display the directory selected using the directory select dialog box." ) );
//     buttonHelp->setText( tr( "&Help" ) );
//     buttonHelp->setAccel( QKeySequence( tr( "F1" ) ) );
//     buttonOk->setText( tr( "&OK" ) );
//     buttonOk->setAccel( QKeySequence( QString::null ) );
//     buttonCancel->setText( tr( "&Cancel" ) );
//     buttonCancel->setAccel( QKeySequence( QString::null ) );
// }

// //void PathDialog::insertComponentModels(std::vector<std::string> &models)
// void PathDialog::insertComponentModels()
// {
//     comboBoxName->clear();

//     // hardcoded for now, should be set via framework
//     comboBoxName->insertItem( tr("CCA") );
//     comboBoxName->insertItem( tr("babel") );
//     comboBoxName->insertItem( tr("Vtk") );

//     comboBoxName->setCurrentItem(0);
// }

// void PathDialog::fileDialog()
// {
//     QFileDialog* fd = new QFileDialog(this, "Component XML path dialog", TRUE);
//     fd->setMode(QFileDialog::DirectoryOnly);
//     fd->setDir(workingDir);

//     if (fd->exec() == QDialog::Accepted) {
//         QString s = fd->selectedFile().stripWhiteSpace();
//         if (s.endsWith("/", FALSE)) {
//             s.truncate(s.length() - 1);
//         }
//         lineEditPath->setText(s);
//     }
// }


// QString PathDialog::selectedDirectory() const
// {
//     return lineEditPath->text().stripWhiteSpace();
// }

// QString PathDialog::selectedComponentModel() const
// {
//     return comboBoxName->currentText();
// }

#endif
