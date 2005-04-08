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

#include "StructuredPointsReader.h"
#include <qfiledialog.h>




#include "vtkStructuredPoints.h"
#include "vtkStructuredGrid.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"

#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"



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

  vol=vtkStructuredPoints::New();
  vol->SetDimensions(1, 1, 1);
  float origin[3]={0,0,0};
  float spacing[3]={0,0,0};
  vol->SetScalarTypeToFloat();
  vtkFloatArray *scalars=vtkFloatArray::New();
  scalars->SetNumberOfValues(1);
  int offset=0;
  scalars->SetValue(0, 0);
  vol->GetPointData()->SetScalars(scalars);
  scalars->Delete();

  //reader->SetFileName("./feet50.vtk");
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
  update(Port::RESETCAMERA);
  return 0;
}

vtkObject*
StructuredPointsReader::getOutput(){
  if(reader->IsFileStructuredPoints()){
    return reader->GetOutput();
  }else{
    return vol;
  }
}
