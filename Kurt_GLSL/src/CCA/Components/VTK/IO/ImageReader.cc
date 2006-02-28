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

#include <sci_defs/qt_defs.h>
#include <iostream>
#include <vtkImageReader.h>
#include <vtkImageData.h>

#include "ImageReader.h"

#if HAVE_QT
 #include <qfiledialog.h>
#endif

extern "C" SCIRun::vtk::Component* make_Vtk_ImageReader()
{
  return new SCIRun::vtk::ImageReader;
}

SCIRun::vtk::ImageReader::ImageReader()
{
  //set output port name
  OutPort::setName("ImageReader::output");
  this->reader = vtkImageReader::New();
  addPort(this);
  enableUI();
}

SCIRun::vtk::ImageReader::~ImageReader()
{
  reader->Delete();
}

int
SCIRun::vtk::ImageReader::popupUI()
{
#if HAVE_QT
  QString fn = QFileDialog::getOpenFileName("./","Vtk Image Files");
  if(fn.isNull())
    {
    return 1;
    }
  reader->SetFileName(fn);
  reader->Update();
#endif

  return 0;
}

vtkObject*
SCIRun::vtk::ImageReader::getOutput(){
  return reader->GetOutput();
}
