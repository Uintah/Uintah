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
#include <iostream>
#include <vtkJPEGReader.h>
#include <vtkImageData.h>

#include "JPEGReader.h"
#include <qfiledialog.h>

extern "C" SCIRun::vtk::Component* make_Vtk_JPEGReader()
{
  return new SCIRun::vtk::JPEGReader;
}

SCIRun::vtk::JPEGReader::JPEGReader()
{
  //set output port name
  OutPort::setName("JPEGReader::output");
  this->reader = vtkJPEGReader::New();
  OutPort::setOutput(this->reader->GetOutput());
  addPort(this);
  enableUI();
}

SCIRun::vtk::JPEGReader::~JPEGReader()
{
  reader->Delete();
}

int
SCIRun::vtk::JPEGReader::popupUI()
{
  QString fn = QFileDialog::getOpenFileName("./","Vtk Image Files");
  if(fn.isNull())
    {
    return 1;
    }
  reader->SetFileName(fn);
  reader->Update();

  return 0;
}
