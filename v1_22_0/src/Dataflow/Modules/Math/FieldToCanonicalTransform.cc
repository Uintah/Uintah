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

//    File   : FieldToCanonicalTransform.cc
//    Author : Martin Cole
//    Date   : Tue Jun 17 14:23:35 2003

#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Plane.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class FieldToCanonicalTransform : public Module {
  FieldIPort* ifield_;
  MatrixOPort* omatrix_;

public:
  FieldToCanonicalTransform(GuiContext* ctx);
  virtual ~FieldToCanonicalTransform();
  virtual void execute();
};

DECLARE_MAKER(FieldToCanonicalTransform)
  static string module_name("FieldToCanonicalTransform");
static string widget_name("TransformWidget");

FieldToCanonicalTransform::FieldToCanonicalTransform(GuiContext* ctx) : 
  Module("FieldToCanonicalTransform", ctx, Filter, "Math", "SCIRun"),
  ifield_(0),
  omatrix_(0)
{
}

FieldToCanonicalTransform::~FieldToCanonicalTransform()
{
}

void FieldToCanonicalTransform::execute()
{
  ifield_ = (FieldIPort *)get_iport("Field");
  omatrix_ = (MatrixOPort *)get_oport("Matrix");

  if (!ifield_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!omatrix_) {
    error("Unable to initialize oport 'Matrix'.");
    return;
  }

  // get the input matrix if there is one
  FieldHandle input_fhandle;
  Transform input_transform;
  if (ifield_->get(input_fhandle) && input_fhandle.get_rep()) {
    input_fhandle->mesh()->get_canonical_transform(input_transform);
  } else {
    warning("no input field");
    return;
  }
  
  
  DenseMatrix *dm = scinew DenseMatrix(input_transform);
  omatrix_->send(MatrixHandle(dm));
}


} // End namespace SCIRun
