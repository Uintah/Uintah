/****************************************************************************
** Form implementation generated from reading ui file 'ListPlotterForm.ui'
**
** Created: Tue Apr 9 18:07:16 2002
**      by:  The User Interface Compiler (uic)
**
** WARNING! All changes made in this file will be lost!
****************************************************************************/
#include "ListPlotterForm.h"

#include <qvariant.h>   // first for gcc 2.7.2
#include "ZGraph.h"
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qmime.h>
#include <qdragobject.h>
#include <qlayout.h>
#include <qtooltip.h>
#include <qwhatsthis.h>
#include <qimage.h>
#include <qpixmap.h>

#if 0
static QPixmap uic_load_pixmap_ListPlotterForm( const QString &name )
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
 *  Constructs a ListPlotterForm which is a child of 'parent', with the 
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ListPlotterForm::ListPlotterForm( QWidget* parent,  const char* name, bool modal, WFlags fl )
    : QDialog( parent, name, modal, fl )
{
    if ( !name )
	setName( "ListPlotterForm" );
    resize( 479, 352 ); 
    setCaption( trUtf8( "ZGraph Test" ) );
    ListPlotterFormLayout = new QHBoxLayout( this, 11, 6, "ListPlotterFormLayout"); 

    Layout8 = new QVBoxLayout( 0, 0, 6, "Layout8"); 

    listZGraph = new ZGraph( this, "listZGraph" );
    Layout8->addWidget( listZGraph );

    Layout4 = new QHBoxLayout( 0, 0, 6, "Layout4"); 

    connectedCheckBox = new QCheckBox( this, "connectedCheckBox" );
    connectedCheckBox->setText( trUtf8( "Connected" ) );
    Layout4->addWidget( connectedCheckBox );

    closeQuitPushButton = new QPushButton( this, "closeQuitPushButton" );
    closeQuitPushButton->setText( trUtf8( "Close" ) );
    Layout4->addWidget( closeQuitPushButton );
    Layout8->addLayout( Layout4 );
    ListPlotterFormLayout->addLayout( Layout8 );

    // signals and slots connections
    connect( closeQuitPushButton, SIGNAL( clicked() ), this, SLOT( close() ) );
    connect( connectedCheckBox, SIGNAL( toggled(bool) ), listZGraph, SLOT( setStyle(bool) ) );
}

/*  
 *  Destroys the object and frees any allocated resources
 */
ListPlotterForm::~ListPlotterForm()
{
    // no need to delete child widgets, Qt does it all for us
}

void ListPlotterForm::setData( const double * val, int size )
{
    listZGraph->setData(val, size);
}

