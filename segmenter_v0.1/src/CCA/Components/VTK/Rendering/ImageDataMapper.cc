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
 *  ImageDataMapper.cc:
 *
 *  Written by:
 *   Kosta
 *   Department of Computer Science
 *   University of Utah
 *   2005
 *
 */

#include <iostream>
#include <SCIRun/Vtk/Port.h>
#include "ImageDataMapper.h"

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_ImageDataMapper()
{
  return new ImageDataMapper;
}

bool 
ImageDataMapper::accept(OutPort* port){
  return dynamic_cast<vtkImageData*>(port->getOutput())!=0;
}

void
ImageDataMapper::connect(OutPort* port){
  vtkImageData* img = dynamic_cast<vtkImageData*>(port->getOutput());

  magnitude->SetInput(img);

  geometry->SetInput(magnitude->GetOutput());
                                                                                                                                
  warp->SetInput(geometry->GetOutput());
  warp->SetScaleFactor(4.0/(2*3.14));
                                                                                                                                
  merge->SetGeometry(warp->GetOutput());
  merge->SetScalars(img);
                                                                                                                                
  mapper->SetInput(merge->GetOutput());
  mapper->SetScalarRange(0,1);
  mapper->ImmediateModeRenderingOff();
}

ImageDataMapper::ImageDataMapper(){
  //set input port name
  InPort::setName("ImageDataMapper::input");

  //set output port name
  OutPort::setName("ImageDataMapper::output");

                                                                                                                                              
  magnitude=vtkImageMagnitude::New();
  geometry=vtkImageDataGeometryFilter::New();
  warp=vtkWarpScalar::New();
  merge=vtkMergeFilter::New();
  mapper=vtkDataSetMapper::New();

  addPort(dynamic_cast<InPort*>(this));
  addPort(dynamic_cast<OutPort*>(this));
}

ImageDataMapper::~ImageDataMapper(){
  magnitude->Delete();
  geometry->Delete();
  warp->Delete();
  merge->Delete();
  mapper->Delete();
}

vtkObject*
ImageDataMapper::getOutput(){
  return mapper;
}
