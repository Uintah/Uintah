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
 *  MaskLattice.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Clipper.h>
#include <Core/Datatypes/FieldInterface.h>
#include <iostream>

namespace SCIRun {

class MaskLattice : public Module
{
public:
  MaskLattice(GuiContext* ctx);
  virtual ~MaskLattice();

  virtual void execute();

private:
  GuiString maskfunction_;
};


DECLARE_MAKER(MaskLattice)


class ScalarClipper : public Clipper
{
private:
  ScalarFieldInterface *sfi_;
  string function_;
  GuiInterface *gui_;
  string id_;

public:
  ScalarClipper(ScalarFieldInterface *sfi,
		string function,
		GuiInterface *gui,
		string id) :
    sfi_(sfi),
    function_(function),
    gui_(gui),
    id_(id)
  { 
  }

  virtual bool inside_p(const Point &p)
  {
    double val;
    if (sfi_->interpolate(val, p))
    {
      string result;
      gui_->eval(id_ + " functioneval " +
		 to_string(val) + " {" + function_ + "}",
		 result);
      if (result == "1")
      {
	return true;
      }
    }
    return false;
  }

  virtual bool mesh_p() { return true; }
};


MaskLattice::MaskLattice(GuiContext* ctx)
  : Module("MaskLattice", ctx, Filter, "Fields", "SCIRun"),
    maskfunction_(ctx->subVar("maskfunction"))
{
}



MaskLattice::~MaskLattice()
{
}

void
MaskLattice::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("MaskLattice Module requires input.");
    return;
  }

  if (!ifieldhandle->query_scalar_interface(this))
  {
    error("This module only works on fields containing scalar data.");
    return;
  }
  

  LatVolField<double> *infield = 
    dynamic_cast<LatVolField<double> *>(ifieldhandle.get_rep());
  LatVolMesh *inmesh = infield->get_typed_mesh().get_rep();
  const BBox bbox = inmesh->get_bounding_box();
  MaskedLatVolMesh *mesh = scinew MaskedLatVolMesh(inmesh->get_ni(), 
						   inmesh->get_nj(), 
						   inmesh->get_nk(),
						   bbox.min(), bbox.max());

  MaskedLatVolField<double> *of = 
    scinew MaskedLatVolField<double>(mesh, infield->data_at());

  ScalarClipper clipper(ifieldhandle->query_scalar_interface(this), 
			maskfunction_.get(),gui, id);


  switch (infield->data_at())
  {
  case Field::NODE:
    {
      MaskedLatVolMesh::Node::iterator iter, iend;
      mesh->begin(iter);
      mesh->end(iend);
      while (iter != iend)
	{
	  Point p;
	  mesh->get_center(p,*iter);
	  if (!clipper.inside_p(p))
	    {
	      MaskedLatVolMesh::Cell::index_type 
		idx((*iter).mesh_, (*iter).i_, (*iter).j_, (*iter).k_);
	      mesh->mask_cell(idx);
	    }
	  ++iter;
	}
    }
    break;
  case Field::EDGE:
    {
      MaskedLatVolMesh::Edge::iterator iter, iend;
      mesh->begin(iter);
      mesh->end(iend);
      while (iter != iend)
	{
	  Point p;
	  mesh->get_center(p,*iter);
	  if (!clipper.inside_p(p))
	    {
	      MaskedLatVolMesh::Cell::index_type 
		idx((*iter).mesh_, (*iter).i_, (*iter).j_, (*iter).k_);
	      mesh->mask_cell(idx);
	    }
	  ++iter;
	}
    }
    break;
  case Field::FACE:
    {
      MaskedLatVolMesh::Face::iterator iter, iend;
      mesh->begin(iter);
      mesh->end(iend);
      while (iter != iend)
	{
	  Point p;
	  mesh->get_center(p,*iter);
	  if (!clipper.inside_p(p))
	    {
	      MaskedLatVolMesh::Cell::index_type 
		idx((*iter).mesh_, (*iter).i_, (*iter).j_, (*iter).k_);
	      mesh->mask_cell(idx);
	    }
	  ++iter;
	}
    }
    break;
  default:
  case Field::CELL:
    {
      MaskedLatVolMesh::Node::iterator iter, iend;
      mesh->begin(iter);
      mesh->end(iend);
      while (iter != iend)
	{	
	  Point p;
	  mesh->get_center(p,*iter);
	  if (!clipper.inside_p(p))
	    {
	      MaskedLatVolMesh::Cell::index_type 
		idx((*iter).mesh_, (*iter).i_, (*iter).j_, (*iter).k_);
	      mesh->mask_cell(idx);
	    }
	  ++iter;
	}
    }
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Masked Field");
  if (!ofp) 
  {
    error("Unable to initialize oport 'Output Sample Field'.");
    return;
  }
  FieldHandle ofh = of;
  ofp->send(ofh);
}


} // End namespace SCIRun

