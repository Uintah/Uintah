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
 *  MainWindow.cc
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */


#include <qlayout.h>
#include <qobject.h>
#include <qcombobox.h>
#include <qcheckbox.h>

#include "MainWindow.h"
#include "ViewerWindow.h"
#include "Colormap.h"

MainWindow::MainWindow( QWidget *parent, const char *name,
			SSIDL::array1<double> nodes1d, 
		        SSIDL::array1<int> triangles, 
			SSIDL::array1<double> solution)
        : QWidget( parent, name )
{
    setGeometry(QRect(200,200,500,500));
    Colormap *cmap=new Colormap(this);
    ViewerWindow *viewer=new ViewerWindow(this,cmap, nodes1d, triangles, solution);

    QCheckBox *optionMesh=new QCheckBox("Show Mesh",this);
    QCheckBox *optionCoordinates=new QCheckBox("Show Coordinates",this);

    QGridLayout *grid = new QGridLayout( this, 2, 2, 10 );
    //2x2, 10 pixel border

    QBoxLayout * hlayout = new QHBoxLayout(grid );
    hlayout->addWidget(optionMesh);
    hlayout->addWidget(optionCoordinates);
    

    QComboBox *type=new QComboBox(this);
    type->insertItem("Gray");
    type->insertItem("Color");

    grid->addWidget( viewer, 0, 0 );
    grid->addLayout( hlayout, 1, 0);
    connect(type, SIGNAL(activated(const QString&)), 
	    viewer, SLOT(refresh(const QString&) ) );  
    connect(optionMesh, SIGNAL(clicked()), 
	    viewer, SLOT(toggleMesh() ) );  
    connect(optionCoordinates, SIGNAL(clicked()), 
	    viewer, SLOT(toggleCoordinates() ) ); 
    grid->addWidget( type, 1, 1 );
    grid->addWidget( cmap, 0, 1 );
    grid->setColStretch( 0, 10 );
    grid->setColStretch( 1, 1 );
    grid->setRowStretch( 0, 10 );
    grid->setRowStretch( 1, 1 );
}


