/****************************************************************************
** Form implementation generated from reading ui file 'ListForm.ui'
**
** Created: Fri Apr 12 19:53:10 2002
**      by:  The User Interface Compiler (uic)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/
#include "ListForm.h"

#include <qvariant.h>   // first for gcc 2.7.2
#include <qlineedit.h>
#include <qlistbox.h>
#include <qpushbutton.h>
#include <qmime.h>
#include <qdragobject.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include "ListForm.ui.h"
#include <qimage.h>
#include <qpixmap.h>
#include "ZList.h"
#include <string.h>
#if 0
static QPixmap uic_load_pixmap_ListForm( const QString &name )
{
    const QMimeSource *m = QMimeSourceFactory::defaultFactory()->data( name );
    if ( !m )
	return QPixmap();
    QPixmap pix;
    QImageDrag::decode( m, pix );
    return pix;
}
#endif
/* 
 *  Constructs a ListForm which is a child of 'parent', with the 
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ListForm::ListForm(ZList *com, QWidget* parent,  const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
	this->com=com;
    if ( !name )
	setName( "ListForm" );
    resize( 310, 480 ); 
    setCaption( trUtf8( "List Test" ) );
    ListFormLayout = new QHBoxLayout( this, 11, 6, "ListFormLayout"); 

    Layout8 = new QVBoxLayout( 0, 0, 6, "Layout8"); 

    Layout2 = new QHBoxLayout( 0, 0, 6, "Layout2"); 

    numLineEdit = new QLineEdit( this, "numLineEdit" );
    Layout2->addWidget( numLineEdit );

    insertPushButton = new QPushButton( this, "insertPushButton" );
    insertPushButton->setEnabled( FALSE );
    insertPushButton->setText( trUtf8( "Insert" ) );
    Layout2->addWidget( insertPushButton );
    Layout8->addLayout( Layout2 );

    Layout7 = new QHBoxLayout( 0, 0, 6, "Layout7"); 

    numListBox = new QListBox( this, "numListBox" );
    numListBox->insertItem( trUtf8( "1.0" ) );
    numListBox->insertItem( trUtf8( "2.0" ) );
    numListBox->insertItem( trUtf8( "3.0" ) );
    numListBox->insertItem( trUtf8( "4.0" ) );
    Layout7->addWidget( numListBox );

    Layout6 = new QVBoxLayout( 0, 0, 6, "Layout6"); 

    deletePushButton = new QPushButton( this, "deletePushButton" );
    deletePushButton->setEnabled( FALSE );
    deletePushButton->setText( trUtf8( "Delete" ) );
    Layout6->addWidget( deletePushButton );
    QSpacerItem* spacer = new QSpacerItem( 20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding );
    Layout6->addItem( spacer );

    refreshPushButton = new QPushButton( this, "refreshPushButton" );
    refreshPushButton->setText( trUtf8( "Refresh" ) );
    Layout6->addWidget( refreshPushButton );

    closePushButton = new QPushButton( this, "closePushButton" );
    closePushButton->setText( trUtf8( "Close" ) );
    Layout6->addWidget( closePushButton );
    Layout7->addLayout( Layout6 );
    Layout8->addLayout( Layout7 );
    ListFormLayout->addLayout( Layout8 );

    // signals and slots connections
    connect( insertPushButton, SIGNAL( clicked() ), this, SLOT( insert() ) );
    connect( refreshPushButton, SIGNAL( clicked() ), this, SLOT( refresh() ) );
    connect( numLineEdit, SIGNAL( textChanged(const QString&) ), this, SLOT( enableInsert(const QString&) ) );
    connect( closePushButton, SIGNAL( clicked() ), this, SLOT( close() ) );
    connect( numListBox, SIGNAL( highlighted(int) ), this, SLOT( enableDelete(int) ) );
    connect( deletePushButton, SIGNAL( clicked() ), this, SLOT( del() ) );

    if(com->datalist.size()>0){
        numListBox->clear();
	for(unsigned int i=0; i<com->datalist.size();i++){
	  char s[20];
	  sprintf(s,"%lf",com->datalist[i]);
	    numListBox->insertItem(s, i);
	}    
    } 
}

/*  
 *  Destroys the object and frees any allocated resources
 */
ListForm::~ListForm()
{
    // no need to delete child widgets, Qt does it all for us
}

