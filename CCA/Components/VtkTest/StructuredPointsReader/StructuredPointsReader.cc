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
#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VtkTest/StructuredPointsReader/StructuredPointsReader.h>

#include "vtkStructuredPointsReader.h"
#include "vtkStructuredPoints.h"
#include "vtkPolyData.h"

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_StructuredPointsReader()
{
  return new StructuredPointsReader;
}

//Output Port

OPort::OPort(vtkStructuredPointsReader *reader){
  this->reader=reader;
  static char * filename="/home/sci/kzhang/vtk/vtkdata/head.60.vtk";
  reader->SetFileName(filename);
}

OPort::~OPort(){

}

bool
OPort::isInput(){
  return false;
}

std::string
OPort::getName(){
  return "StructuredPointsReader::output";
}

vtkObject*
OPort::getObj(){
  return reader->GetOutput();
}


StructuredPointsReader::StructuredPointsReader(){

  reader=vtkStructuredPointsReader::New();
  oports.push_back(new OPort(reader));
}

StructuredPointsReader::~StructuredPointsReader(){
  for(unsigned int i=0; i<iports.size(); i++){
    delete iports[i];
  }
  for(unsigned int i=0; i<oports.size(); i++){
    delete oports[i];
  }
  reader->Delete();
}
