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
 *  ContourFilter.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <iostream>
#include <vtkContourFilter.h>
#include <vtkDataSet.h>
#include <vtkStructuredPoints.h>
#include <vtkPolyData.h>

#include <SCIRun/Vtk/Port.h>
#include <CCA/Components/VTK/Graphics/ContourFilter.h>

using namespace std;
using namespace SCIRun;
using namespace vtk;

extern "C" vtk::Component* make_Vtk_ContourFilter()
{
  return new ContourFilter;
}

bool 
ContourFilter::accept(OutPort* port){
  return dynamic_cast<vtkDataSet*>(port->getOutput())!=0;
}

void
ContourFilter::connect(OutPort* port){
  filter->SetInput(dynamic_cast<vtkDataSet*>(port->getOutput()));
  //TODO: use GUI
  filter->SetValue(0,70);
}

ContourFilter::ContourFilter(){
  //set input port name
  InPort::setName("ContourFilter::input");

  //set output port name
  OutPort::setName("ContourFilter::output");

  filter=vtkContourFilter::New();


  addPort(dynamic_cast<InPort*>(this));
  addPort(dynamic_cast<OutPort*>(this));
}

vtkObject *
ContourFilter::getOutput(){
  return filter->GetOutput();
}

ContourFilter::~ContourFilter(){
  filter->Delete();
}

