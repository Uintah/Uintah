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
 *  SamplePlane.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class SamplePlane : public Module
{
public:
  SamplePlane(GuiContext* ctx);
  virtual ~SamplePlane();

  virtual void execute();

private:

  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt axis_;
  GuiDouble padpercent_;
  GuiDouble position_;
  GuiString data_at_;
  GuiString update_type_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SamplePlane)
  
SamplePlane::SamplePlane(GuiContext* ctx) : 
  Module("SamplePlane", ctx, Filter, "Fields", "SCIRun"),
  size_x_(ctx->subVar("sizex")),
  size_y_(ctx->subVar("sizey")),
  axis_(ctx->subVar("axis")),
  padpercent_(ctx->subVar("padpercent")),
  position_(ctx->subVar("pos")),
  data_at_(ctx->subVar("data-at")),
  update_type_(ctx->subVar("update_type"))
{
}



SamplePlane::~SamplePlane()
{
}

void
SamplePlane::execute()
{
  const int axis = Min(2, Max(0, axis_.get()));
  Transform trans;
  trans.load_identity();

  double angle = 0;
  Vector axis_vector(0.0, 0.0, 1.0);
  switch (axis)
  {
  case 0:
    angle = M_PI * -0.5; 
    axis_vector = Vector(0.0, 1.0, 0.0);
    break;

  case 1:
    angle = M_PI * 0.5; 
    axis_vector = Vector(1.0, 0.0, 0.0);
    break;

  case 2:
    angle = 0.0;
    axis_vector = Vector(0.0, 0.0, 1.0);
    break;

  default:
    break;
  }
  trans.pre_rotate(angle, axis_vector);

  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  DataTypeEnum datatype;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    datatype = SCALAR;
  }
  else
  {
    datatype = SCALAR;
    if (ifieldhandle->query_vector_interface(this).get_rep())
    {
      datatype = TENSOR;
    }
    else if (ifieldhandle->query_tensor_interface(this).get_rep())
    {
      datatype = VECTOR;
    }
  
    // Compute Transform.
    BBox box = ifieldhandle->mesh()->get_bounding_box();

    Point loc(box.min());
    Vector diag(box.diagonal());
    position_.reset();
    double dist = position_.get()/2.0 + 0.5;
    switch (axis)
    {
    case 0:
      loc.x(loc.x() + diag.x() * dist);
      break;

    case 1:
      loc.y(loc.y() + diag.y() * dist);
      break;

    case 2:
      loc.z(loc.z() + diag.z() * dist);
      break;
      
    default:
      break;
    }

    trans.pre_scale(diag);
    trans.pre_translate(Vector(loc));
  }
  
  // Create blank mesh.
  unsigned int sizex = Max(2, size_x_.get());
  unsigned int sizey = Max(2, size_y_.get());
  Point minb(0.0, 0.0, 0.0);
  Point maxb(1.0, 1.0, 1.0);
  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent_.get()/100.0));
  minb -= diag;
  maxb += diag;


  ImageMeshHandle imagemesh = scinew ImageMesh(sizex, sizey, minb, maxb);

  Field::data_location data_at;
  if (data_at_.get() == "Nodes") data_at = Field::NODE;
  else if (data_at_.get() == "Edges") data_at = Field::EDGE;
  else if (data_at_.get() == "Faces") data_at = Field::FACE;
  else if (data_at_.get() == "None") data_at = Field::NONE;
  else {
    error("Unsupported data_at location " + data_at_.get() + ".");
    return;
  }

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == VECTOR)
  {
    ofh = scinew ImageField<Vector>(imagemesh, data_at);
  }
  else if (datatype == TENSOR)
  {
    ofh = scinew ImageField<Tensor>(imagemesh, data_at);
  }
  else
  {
    ofh = scinew ImageField<double>(imagemesh, data_at);
  }

  // Transform field.
  ofh->mesh()->transform(trans);

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp) {
    error("Unable to initialize oport 'Output Sample Field'.");
    return;
  }
  ofp->send(ofh);
}


} // End namespace SCIRun

