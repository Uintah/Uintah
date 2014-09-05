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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class SampleLattice : public Module
{
public:
  SampleLattice(GuiContext* ctx);
  virtual ~SampleLattice();

  virtual void execute();

private:
  GuiInt size_x_;
  GuiInt size_y_;
  GuiInt size_z_;
  GuiDouble padpercent_;
  GuiString data_at_;

  enum DataTypeEnum { SCALAR, VECTOR, TENSOR };
};


DECLARE_MAKER(SampleLattice)

SampleLattice::SampleLattice(GuiContext* ctx)
  : Module("SampleLattice", ctx, Filter, "FieldsCreate", "SCIRun"),
    size_x_(ctx->subVar("sizex")),
    size_y_(ctx->subVar("sizey")),
    size_z_(ctx->subVar("sizez")),
    padpercent_(ctx->subVar("padpercent")),
    data_at_(ctx->subVar("data-at"))
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
    error("Unable to initialize iport 'Input Field'.");
    return;
  }

  Point minb, maxb;
  DataTypeEnum datatype;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    datatype = SCALAR;
    minb = Point(-1.0, -1.0, -1.0);
    maxb = Point(1.0, 1.0, 1.0);
  }
  else
  {
    datatype = SCALAR;
    if (ifieldhandle->query_vector_interface(this).get_rep())
    {
      datatype = VECTOR;
    }
    else if (ifieldhandle->query_tensor_interface(this).get_rep())
    {
      datatype = TENSOR;
    }
    BBox bbox = ifieldhandle->mesh()->get_bounding_box();
    minb = bbox.min();
    maxb = bbox.max();
  }

  Vector diag((maxb.asVector() - minb.asVector()) * (padpercent_.get()/100.0));
  minb -= diag;
  maxb += diag;

  // Create blank mesh.
  unsigned int sizex = Max(2, size_x_.get());
  unsigned int sizey = Max(2, size_y_.get());
  unsigned int sizez = Max(2, size_z_.get());
  LatVolMeshHandle mesh = scinew LatVolMesh(sizex, sizey, sizez, minb, maxb);

  Field::data_location data_at;
  if (data_at_.get() == "Nodes") data_at = Field::NODE;
  else if (data_at_.get() == "Edges") data_at = Field::EDGE;
  else if (data_at_.get() == "Faces") data_at = Field::FACE;
  else if (data_at_.get() == "Cells") data_at = Field::CELL;
  else if (data_at_.get() == "None") data_at = Field::NONE;
  else {
    error("Unsupported data_at location " + data_at_.get() + ".");
    return;
  }

  // Create Image Field.
  FieldHandle ofh;
  if (datatype == SCALAR)
  {
    LatVolField<double> *lvf = scinew LatVolField<double>(mesh, data_at);
    if (data_at != Field::NONE)
    {
      LatVolField<double>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = 0.0;
	++itr;
      }
    }
    ofh = lvf;
  } 
  else if (datatype == VECTOR)
  {
    LatVolField<Vector> *lvf = scinew LatVolField<Vector>(mesh, data_at);
    if (data_at != Field::NONE)
    {
      LatVolField<Vector>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Vector(0.0, 0.0, 0.0);
	++itr;
      }
    }
    ofh = lvf;
  }				    
  else // if (datatype == TENSOR)	    
  {				    
    LatVolField<Tensor> *lvf = scinew LatVolField<Tensor>(mesh, data_at);
    if (data_at != Field::NONE)
    {
      LatVolField<Tensor>::fdata_type::iterator itr = lvf->fdata().begin();
      while (itr != lvf->fdata().end())
      {
	*itr = Tensor(0.0);
	++itr;
      }
    }
    ofh = lvf;
  }				    

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp) {
    error("Unable to initialize oport 'Output Sample Field'.");
    return;
  }

  ofp->send(ofh);
}


} // End namespace SCIRun

