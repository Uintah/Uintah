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
#ifndef SCIRun_VTK_Components_ImageReader_h
#define SCIRun_VTK_Components_ImageReader_h

#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>
#include "vtkImageReader.h"
#include <vector>

class vtkImageReader;

namespace SCIRun {
namespace vtk {

class ImageReader: public Component, public OutPort
{
public:
  //constructor
  ImageReader();
  
  //destructor
  ~ImageReader();
  
  //Component interface
  int popupUI();
  
private:
  vtkImageReader *reader;
  ImageReader(const ImageReader&);
  ImageReader& operator=(const ImageReader&);
};

} //namespace vtk
} //namespace SCIRun


#endif
