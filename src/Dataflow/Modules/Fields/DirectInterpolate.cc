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
 *  DirectInterpolate.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/Dispatch1.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class DirectInterpolate : public Module
{
  FieldIPort *src_port_;
  FieldIPort *dst_port_;
  FieldOPort *ofp_; 
  ScalarFieldInterface *sfi_;
  GuiString   interp_op_gui_;

public:
  DirectInterpolate(const string& id);
  virtual ~DirectInterpolate();
  virtual void execute();

  template <class Fld> void callback1(Fld *fld);

};

extern "C" Module* make_DirectInterpolate(const string& id)
{
  return new DirectInterpolate(id);
}

DirectInterpolate::DirectInterpolate(const string& id) : 
  Module("DirectInterpolate", id, Filter, "Fields", "SCIRun"),
  interp_op_gui_("interp_op_gui", id, this)
{
}

DirectInterpolate::~DirectInterpolate()
{
}


template <class Fld>
void
DirectInterpolate::callback1(Fld *fld2)
{

  Fld *fld = fld2->clone();
  typename Fld::mesh_handle_type mesh = fld->get_typed_mesh();

  switch (fld->data_at())
  {
  case Field::NODE:
    {
      typedef typename Fld::mesh_type::Node::iterator Itr;
      Itr itr = mesh->node_begin();
      Itr itr_end = mesh->node_end();
      while (itr != itr_end)
      {
	Point p;
	mesh->get_center(p, *itr);

	double val;
	if (sfi_->interpolate(val, p))
	{
	  fld->set_value((typename Fld::value_type)val, *itr);
	}

	++itr;
      }
    }
    break;

  case Field::EDGE:
    {
      typedef typename Fld::mesh_type::Edge::iterator Itr;
      Itr itr = mesh->edge_begin();
      Itr itr_end = mesh->edge_end();
      while (itr != itr_end)
      {
	Point p;
	mesh->get_center(p, *itr);

	double val;
	if (sfi_->interpolate(val, p))
	{
	  fld->set_value((typename Fld::value_type)val, *itr);
	}

	++itr;
      }
    }
    break;

  case Field::FACE:
    {
      typedef typename Fld::mesh_type::Face::iterator Itr;
      Itr itr = mesh->face_begin();
      Itr itr_end = mesh->face_end();
      while (itr != itr_end)
      {
	Point p;
	mesh->get_center(p, *itr);

	double val;
	if (sfi_->interpolate(val, p))
	{
	  fld->set_value((typename Fld::value_type)val, *itr);
	}

	++itr;
      }
    }
    break;

  case Field::CELL:
    {
      typedef typename Fld::mesh_type::Cell::iterator Itr;
      Itr itr = mesh->cell_begin();
      Itr itr_end = mesh->cell_end();
      while (itr != itr_end)
      {
	Point p;
	mesh->get_center(p, *itr);

	double val;
	if (sfi_->interpolate(val, p))
	{
	  fld->set_value((typename Fld::value_type)val, *itr);
	}

	++itr;
      }
    }
    break;

  default:
    break;
  }

  FieldHandle ofh(fld);
  ofp_->send(ofh);
}



void
DirectInterpolate::execute()
{
  dst_port_ = (FieldIPort *)get_iport("Destination");
  FieldHandle dfieldhandle;
  if (!(dst_port_->get(dfieldhandle) && dfieldhandle.get_rep()))
  {
    return;
  }

  src_port_ = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  if (!(src_port_->get(sfieldhandle) && sfieldhandle.get_rep()))
  {
    return;
  }

  ofp_ = (FieldOPort *)get_oport("Interpolant");

  if (!(sfi_ = sfieldhandle->query_scalar_interface()))
  {
    warning("Source not a scalar field.");
    return;
  }

  dispatch1(dfieldhandle, callback1);
}

} // End namespace SCIRun
