/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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


