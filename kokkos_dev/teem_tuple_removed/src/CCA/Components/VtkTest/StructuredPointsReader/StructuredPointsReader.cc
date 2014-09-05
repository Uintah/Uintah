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
 *  StructuredPointsReader.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <iostream>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkPolyData.h>

#include <CCA/Components/VtkTest/StructuredPointsReader/StructuredPointsReader.h>
#include <qfiledialog.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_StructuredPointsReader()
{
  return new StructuredPointsReader;
}

StructuredPointsReader::StructuredPointsReader(){
  //set output port name
  OutPort::setName("StructuredPointsReader::output");
  reader=vtkStructuredPointsReader::New();
  setOutput(reader->GetOutput());
  addPort(this);
  enableUI();
}

StructuredPointsReader::~StructuredPointsReader(){
  reader->Delete();
}


int
StructuredPointsReader::popupUI(){
  QString fn = QFileDialog::getOpenFileName(
	    "./","Vtk StructuredPoints Files(*.vtk)");
  if(fn.isNull())   return 1;
  reader->SetFileName(fn);
  reader->Update();

  return 0;
}
