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

#include <CCA/Components/ZList/ListForm.h>

#if 0
// static QPixmap uic_load_pixmap_ListForm( const QString &name )
// {
//     const QMimeSource *m = QMimeSourceFactory::defaultFactory()->data( name );
//     if ( !m )
//  return QPixmap();
//     QPixmap pix;
//     QImageDrag::decode( m, pix );
//     return pix;
// }
#endif

/*
 *  Constructs a ListForm which is a child of 'parent', with the
 *  name 'name'.
 */
ListForm::ListForm(ZList *com, wxWindow* parent, const wxString& name)
  : wxDialog(parent, wxID_ANY, name, wxDefaultPosition, wxSize(310, 480), wxCAPTION|wxRESIZE_BORDER|wxSTAY_ON_TOP),
    com(com)
{
//   setCaption( trUtf8( "List Test" ) );
//   ListFormLayout = new QHBoxLayout( this, 11, 6, "ListFormLayout");

//   Layout8 = new QVBoxLayout( 0, 0, 6, "Layout8");

//   Layout2 = new QHBoxLayout( 0, 0, 6, "Layout2");

//   numLineEdit = new QLineEdit( this, "numLineEdit" );
//   Layout2->addWidget( numLineEdit );

//   insertPushButton = new QPushButton( this, "insertPushButton" );
//   insertPushButton->setEnabled( FALSE );
//   insertPushButton->setText( trUtf8( "Insert" ) );
//   Layout2->addWidget( insertPushButton );
//   Layout8->addLayout( Layout2 );

//   Layout7 = new QHBoxLayout( 0, 0, 6, "Layout7");

//   numListBox = new QListBox( this, "numListBox" );
//   numListBox->insertItem( trUtf8( "1.0" ) );
//   numListBox->insertItem( trUtf8( "2.0" ) );
//   numListBox->insertItem( trUtf8( "3.0" ) );
//   numListBox->insertItem( trUtf8( "4.0" ) );
//   Layout7->addWidget( numListBox );

//   Layout6 = new QVBoxLayout( 0, 0, 6, "Layout6");

//   deletePushButton = new QPushButton( this, "deletePushButton" );
//   deletePushButton->setEnabled( FALSE );
//   deletePushButton->setText( trUtf8( "Delete" ) );
//   Layout6->addWidget( deletePushButton );
//   QSpacerItem* spacer = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
//   Layout6->addItem( spacer );

//   refreshPushButton = new QPushButton( this, "refreshPushButton" );
//   refreshPushButton->setText( trUtf8( "Refresh" ) );
//   Layout6->addWidget( refreshPushButton );

//   closePushButton = new QPushButton( this, "closePushButton" );
//   closePushButton->setText( trUtf8( "Close" ) );
//   Layout6->addWidget( closePushButton );
//   Layout7->addLayout( Layout6 );
//   Layout8->addLayout( Layout7 );
//   ListFormLayout->addLayout( Layout8 );

//   // signals and slots connections
//   connect( insertPushButton, SIGNAL( clicked() ), this, SLOT( insert() ) );
//   connect( refreshPushButton, SIGNAL( clicked() ), this, SLOT( refresh() ) );
//   connect( numLineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( enableInsert(const QString&) ) );
//   connect( closePushButton, SIGNAL( clicked() ), this, SLOT( close() ) );
//   connect( numListBox, SIGNAL( highlighted(int) ), this, SLOT( enableDelete(int) ) );
//   connect( deletePushButton, SIGNAL( clicked() ), this, SLOT( del() ) );

//   if(com->datalist.size()>0){
//     numListBox->clear();
//     for(unsigned int i=0; i<com->datalist.size();i++){
//       char s[20];
//       sprintf(s,"%lf",com->datalist[i]);
//       numListBox->insertItem(s, i);
//     }
//   }
}

/*
 *  Destroys the object and frees any allocated resources
 */
ListForm::~ListForm()
{
  // no need to delete child widgets, wxWidgets does it all for us
}
