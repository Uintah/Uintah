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
 *  SampleLattice.cc:  Make an ImageField that fits the source field.
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
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class SampleLattice : public Module
{
public:
  SampleLattice(const string& id);
  virtual ~SampleLattice();

  virtual void execute();

private:

  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt size_z_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


extern "C" Module* make_SampleLattice(const string& id) {
  return new SampleLattice(id);
}


SampleLattice::SampleLattice(const string& id)
  : Module("SampleLattice", id, Filter, "Fields", "SCIRun"),
    size_x_("sizex", id, this),
    size_y_("sizey", id, this),
    size_z_("sizez", id, this)
{
}



SampleLattice::~SampleLattice()
{
}

void
SampleLattice::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }

  Point minb, maxb;
  DataTypeEnum datatype;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    datatype = SCALAR;
    minb = Point(0.0, 0.0, 0.0);
    maxb = Point(1.0, 1.0, 1.0);
  }
  else
  {
    datatype = SCALAR;
    if (ifieldhandle->query_vector_interface())
    {
      datatype = TENSOR;
    }
    else if (ifieldhandle->query_tensor_interface())
    {
      datatype = VECTOR;
    }
    BBox bbox = ifieldhandle->mesh()->get_bounding_box();
    minb = bbox.min();
    maxb = bbox.max();
  }

  // Create blank mesh.
  unsigned int sizex = Max(2, size_x_.get());
  unsigned int sizey = Max(2, size_y_.get());
  unsigned int sizez = Max(2, size_z_.get());
  LatVolMeshHandle mesh = scinew LatVolMesh(sizex, sizey, sizez, minb, maxb);

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == VECTOR)
  {
    ofh = scinew LatticeVol<Vector>(mesh, Field::NODE);
  }				    
  else if (datatype == TENSOR)	    
  {				    
    ofh = scinew LatticeVol<Tensor>(mesh, Field::NODE);
  }				    
  else				    
  {				    
    ofh = scinew LatticeVol<double>(mesh, Field::NODE);
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofp->send(ofh);
}


} // End namespace SCIRun

