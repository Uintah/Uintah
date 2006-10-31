/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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

#include <CCA/Components/ListPlotter/ListPlotterForm.h>
#include <CCA/Components/ListPlotter/ZGraph.h>

#if 0
// static QPixmap uic_load_pixmap_ListPlotterForm( const QString &name )
// {
//   const QMimeSource *m = QMimeSourceFactory::defaultFactory()->data( name );
//   if ( !m )
//     return QPixmap();
//   QPixmap pix;
//   QImageDrag::decode( m, pix );
//   return pix;
// }
#endif

/*
 *  Constructs a ListPlotterForm which is a child of 'parent', with the
 *  name 'name' and widget flags set to 'f'.
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ListPlotterForm::ListPlotterForm(wxWindow* parent, const wxString& name)
  : wxDialog( parent, wxID_ANY, name, wxDefaultPosition, wxSize(479, 352), wxCAPTION|wxSTAY_ON_TOP|wxRESIZE_BORDER)
{
//   resize( 479, 352 );
//   setCaption( trUtf8( "ZGraph Test" ) );
//   ListPlotterFormLayout = new QHBoxLayout( this, 11, 6, "ListPlotterFormLayout");

//   Layout8 = new QVBoxLayout( 0, 0, 6, "Layout8");

  listZGraph = new ZGraph(this);
//   Layout8->addWidget( listZGraph );

//   Layout4 = new QHBoxLayout( 0, 0, 6, "Layout4");

//   connectedCheckBox = new QCheckBox( this, "connectedCheckBox" );
//   connectedCheckBox->setText( trUtf8( "Connected" ) );
//   Layout4->addWidget( connectedCheckBox );

//   closeQuitPushButton = new QPushButton( this, "closeQuitPushButton" );
//   closeQuitPushButton->setText( trUtf8( "Close" ) );
//   Layout4->addWidget( closeQuitPushButton );
//   Layout8->addLayout( Layout4 );
//   ListPlotterFormLayout->addLayout( Layout8 );

//   // signals and slots connections
//   connect( closeQuitPushButton, SIGNAL( clicked() ), this, SLOT( close() ) );
//   connect( connectedCheckBox, SIGNAL( toggled(bool) ), listZGraph, SLOT( setStyle(bool) ) );
}

/*
 *  Destroys the object and frees any allocated resources
 */
ListPlotterForm::~ListPlotterForm()
{
  // no need to delete child widgets, wxWidgets does it all for us
}

void ListPlotterForm::setData( const double * val, int size )
{
  // need to refresh graph after this is called
  listZGraph->setData(val, size);
}
