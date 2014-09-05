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
 *  ImageDataMapper.h
 *
 *  Written by:
 *   Kosta 
 *   Department of Computer Science
 *   University of Utah
 *   2005
 *
 */

#ifndef SCIRun_VTK_Components_ImageDataMapper_h
#define SCIRun_VTK_Components_ImageDataMapper_h

#include <vtkImageData.h>
#include <vtkImageMagnitude.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkWarpScalar.h>
#include <vtkMergeFilter.h>
#include <vtkDataSetMapper.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>

#include <Framework/Vtk/InPort.h>
#include <Framework/Vtk/OutPort.h>
#include <Framework/Vtk/Component.h>
#include <vector>

class vtkImageDataMapper;

namespace SCIRun {
  namespace vtk{
    class ImageDataMapper: public Component, public InPort, public OutPort{
      
    public:
      //constructor
      ImageDataMapper();

      //destructor
      ~ImageDataMapper();

      //InPort interface
      bool accept(OutPort* port);

      //InPort interface
      void connect(OutPort* port);

      vtkObject* getOutput();
    private:
      vtkImageMagnitude *magnitude;
      vtkImageDataGeometryFilter *geometry;
      vtkWarpScalar *warp;
      vtkMergeFilter *merge;
      vtkDataSetMapper *mapper;

      
      ImageDataMapper(const ImageDataMapper&);
      ImageDataMapper& operator=(const ImageDataMapper&);
    };
    
  } //namespace vtk
} //namespace SCIRun


#endif
