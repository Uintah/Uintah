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
 *  ClippingPlane.cc:  Make an ImageField that fits the source field.
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
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class ClippingPlane : public Module
{
public:
  ClippingPlane(const string& id);
  virtual ~ClippingPlane();

  virtual void execute();

private:
  
  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
  DataTypeEnum datatype_;
};


extern "C" Module* make_ClippingPlane(const string& id) {
  return new ClippingPlane(id);
}


ClippingPlane::ClippingPlane(const string& id)
  : Module("ClippingPlane", id, Filter, "Fields", "SCIRun")
{
}



ClippingPlane::~ClippingPlane()
{
}

void
ClippingPlane::execute()
{
  const int axis = 2;  // x y z planes
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
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    datatype_ = SCALAR;
  }
  else
  {
    datatype_ = SCALAR;
    if (ifieldhandle->query_vector_interface())
    {
      datatype_ = TENSOR;
    }
    else if (ifieldhandle->query_tensor_interface())
    {
      datatype_ = VECTOR;
    }
  
    // Compute Transform.
    BBox box = ifieldhandle->mesh()->get_bounding_box();

    Point loc(box.min());
    Vector diag(box.diagonal());
    switch (axis)
    {
    case 0:
      loc.x(loc.x() + diag.x() * 0.5);
      break;

    case 1:
      loc.y(loc.y() + diag.y() * 0.5);
      break;

    case 2:
      loc.z(loc.z() + diag.z() * 0.5);
      break;
      
    default:
      break;
    }

    trans.pre_scale(diag);
    trans.pre_translate(Vector(loc));
  }
  //trans.pre_rotate(angle, axis_vector);
  
  // Create blank mesh.
  unsigned int sizex = 20;
  unsigned int sizey = 20;
  const Point minb(0.0, 0.0, 0.0);
  const Point maxb(1.0, 1.0, 1.0);
  ImageMeshHandle imagemesh = scinew ImageMesh(sizex, sizey, minb, maxb);

  // Create Image Field.
  FieldHandle ofh;
  if (datatype_ == VECTOR)
  {
    ofh = scinew ImageField<Vector>(imagemesh, Field::NODE);
  }
  else if (datatype_ == TENSOR)
  {
    ofh = scinew ImageField<Tensor>(imagemesh, Field::NODE);
  }
  else
  {
    ofh = scinew ImageField<double>(imagemesh, Field::NODE);
  }

  // Transform field.
  ofh->mesh()->transform(trans);

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Clipping Plane");
  ofp->send(ofh);
}


} // End namespace SCIRun

