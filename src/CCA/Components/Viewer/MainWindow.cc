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


#include <CCA/Components/Viewer/MainWindow.h>
#include <CCA/Components/Viewer/ViewerWindow.h>
#include <CCA/Components/Viewer/Colormap.h>

#include <wx/checkbox.h>
#include <wx/event.h>
#include <wx/gbsizer.h>

namespace Viewer {


MainWindow::MainWindow(wxWindow *parent,
                       const char *name,
                       SSIDL::array1<double> nodes1d,
                       SSIDL::array1<int> triangles,
                       SSIDL::array1<double> solution)
  : wxFrame( parent, wxID_ANY, wxT(name), wxPoint(X, Y), wxSize(WIDTH, HEIGHT), wxMINIMIZE_BOX|wxCAPTION|wxCLOSE_BOX)
{

  std::cerr << "MainWindow::MainWindow(..)" << std::endl;

  //setGeometry(QRect(200,200,500,500));
  wxGridBagSizer *topSizer = new wxGridBagSizer(4, 4);
  topSizer->AddGrowableRow(1);

  std::cerr << "MainWindow::MainWindow(..): sizer" << std::endl;

//   cmap = new Colormap(this);
//   std::cerr << "MainWindow::MainWindow(..): colormap" << std::endl;

//   viewer = new ViewerWindow(this, cmap, nodes1d, triangles, solution);

//   std::cerr << "MainWindow::MainWindow(..): viewer" << std::endl;

//   meshCheckBox = new wxCheckBox(this, ID_CHECKBOX_MESH, "Show Mesh");
//   PushEventHandler(meshCheckBox);
//   meshCheckBox->Connect(ID_CHECKBOX_MESH, wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler(ViewerWindow::OnToggleMesh));
//   topSizer->Add(meshCheckBox, wxGBPosition(0, 0), wxGBSpan(1, 1), wxALL|wxALIGN_CENTER, 2);

//   std::cerr << "MainWindow::MainWindow(..): meshCheckBox" << std::endl;

//   coordsCheckBox = new wxCheckBox(this, ID_CHECKBOX_COORDS, "Show Coordinates");
//   PushEventHandler(coordsCheckBox);
//   coordsCheckBox->Connect(ID_CHECKBOX_COORDS, wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler(ViewerWindow::OnToggleCoordinates));
//   topSizer->Add(coordsCheckBox, wxGBPosition(0, 1), wxGBSpan(1, 1), wxALL|wxALIGN_CENTER, 2);

//   std::cerr << "MainWindow::MainWindow(..): coordsCheckBox" << std::endl;

//   topSizer->Add(viewer, wxGBPosition(1, 0), wxGBSpan(1, 1), wxALL|wxALIGN_CENTER, 2);
//   std::cerr << "MainWindow::MainWindow(..): add viewer" << std::endl;

//   topSizer->Add(cmap, wxGBPosition(1, 1), wxGBSpan(1, 1), wxALL|wxALIGN_CENTER, 2);
//   std::cerr << "MainWindow::MainWindow(..): add colormap" << std::endl;

//     QGridLayout *grid = new QGridLayout( this, 2, 2, 10 );
//     //2x2, 10 pixel border

//     QBoxLayout * hlayout = new QHBoxLayout(grid );
//     hlayout->addWidget(optionMesh);
//     hlayout->addWidget(optionCoordinates);


//     QComboBox *type=new QComboBox(this);
//     type->insertItem("Gray");
//     type->insertItem("Color");

//     grid->addWidget( viewer, 0, 0 );
//     grid->addLayout( hlayout, 1, 0);
//     connect(type, SIGNAL(activated(const QString&)),
//       viewer, SLOT(refresh(const QString&) ) );
//     connect(optionMesh, SIGNAL(clicked()),
//       viewer, SLOT(toggleMesh() ) );
//     connect(optionCoordinates, SIGNAL(clicked()),
//       viewer, SLOT(toggleCoordinates() ) );
//     grid->addWidget( type, 1, 1 );
//     grid->addWidget( cmap, 0, 1 );
//     grid->setColStretch( 0, 10 );
//     grid->setColStretch( 1, 1 );
//     grid->setRowStretch( 0, 10 );
//     grid->setRowStretch( 1, 1 );

  SetAutoLayout(true);
  SetSizer(topSizer);

  topSizer->Fit(this);
  topSizer->SetSizeHints(this);

  std::cerr << "MainWindow::MainWindow(..): done" << std::endl;
}

MainWindow::~MainWindow()
{
//   meshCheckBox->Disconnect();
//   coordsCheckBox->Disconnect();
  // event cleanup: pop last 2 event handlers on this window's stack
//   PopEventHandler();
//   PopEventHandler();

//   if (cmap) delete cmap;
//   if (viewer) delete viewer;
}

}
