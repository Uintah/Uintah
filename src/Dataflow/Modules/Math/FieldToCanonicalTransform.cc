//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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
