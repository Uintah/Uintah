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
 *  BuildInterpolant.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   David Weinstein
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class BuildInterpolant : public Module
{
  FieldIPort *src_port;
  FieldIPort *dst_port;
  FieldOPort *ofp; 
  GuiString   interp_op_gui_;

public:
  BuildInterpolant(const string& id);
  virtual ~BuildInterpolant();
  virtual void execute();

  template <class MSRC, class MDST, class LSRC, class LDST, class FOUT>
  void callback(MSRC *src_mesh, MDST *dst_mesh, LSRC *, LDST *, FOUT *,
		Field::data_location dst_loc);

  //template <class Mesh, class Index>
  //void find_closest(Mesh *mesh, typename Index::index_type &idx, Point &p);
};

extern "C" Module* make_BuildInterpolant(const string& id)
{
  return new BuildInterpolant(id);
}

BuildInterpolant::BuildInterpolant(const string& id) : 
  Module("BuildInterpolant", id, Filter, "Fields", "SCIRun"),
  interp_op_gui_("interp_op_gui", id, this)
{
}

BuildInterpolant::~BuildInterpolant()
{
}

#if 0
template <class Mesh, class Index>
void
BuildInterpolant::find_closest(Mesh *mesh, typename Index::index_type &idx,
			       Point &p)
{
  bool first_p = true;
  double min_dist = 1.0e6;
  typename Index::iterator itr = mesh->tbegin((typename Index::iterator *)0);
  typename Index::iterator eitr = mesh->tend((typename Index::iterator *)0);
  while (itr != eitr)
  {
    Point p2;

    mesh->get_center(p2, *itr);

    const double dist = (p - p2).length2();
    if (dist < min_dist || first_p)
    {
      idx = *itr;
      min_dist = dist;
      first_p = false;
    }
   
    ++itr;
  }
}
#endif

template <class MSRC, class MDST, class LSRC, class LDST, class FOUT>
void
BuildInterpolant::callback(MSRC *src_mesh, MDST *dst_mesh,
			   LSRC *, LDST *, FOUT *,
			   Field::data_location dst_loc)
{
  FOUT *ofield = new FOUT(dst_mesh, dst_loc);

  typedef typename LDST::iterator DSTITR; 
  DSTITR itr = dst_mesh->tbegin((DSTITR *)0);
  DSTITR end_itr = dst_mesh->tend((DSTITR *)0);

  while (itr != end_itr)
  {
    typename LSRC::array_type locs;
    vector<double> weights;
    Point p;

    dst_mesh->get_center(p, *itr);

    src_mesh->get_weights(p, locs, weights);

    vector<pair<typename LSRC::index_type, double> > v;
    if (weights.size() > 0)
    {
      for (unsigned int i = 0; i < locs.size(); i++)
      {
	v.push_back(pair<typename LSRC::index_type, double>
		    (locs[i], weights[i]));
      }
    }
    else
    {
      //typename LSRC::index_type index;
      //find_closest(src_mesh, (LSRC *)0, index, p);
      //v.push_back(pair<typename LSRC::index_type, double>(index, 1.0));
    }

    ofield->set_value(v, *itr);
    ++itr;
  }

  FieldHandle fh(ofield);
  ofp->send(fh);
}




#define CALLBACK_WRAP(MSRC, FDST, fsrc, fdst)\
{\
  switch (fsrc->data_at())\
  {\
  case Field::NODE:\
    switch (fdst->data_at())\
    {\
    case Field::NODE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Node *)0,\
	       (FDST<double>::mesh_type::Node *)0,\
	       (FDST<vector<pair<MSRC::Node::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::EDGE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Node *)0,\
	       (FDST<double>::mesh_type::Edge *)0,\
	       (FDST<vector<pair<MSRC::Node::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::FACE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Node *)0,\
	       (FDST<double>::mesh_type::Face *)0,\
	       (FDST<vector<pair<MSRC::Node::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::CELL:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Node *)0,\
	       (FDST<double>::mesh_type::Cell *)0,\
	       (FDST<vector<pair<MSRC::Node::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
    \
    default:\
      break;\
    }\
    break;\
\
  case Field::EDGE:\
    switch (fdst->data_at())\
    {\
    case Field::NODE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Edge *)0,\
	       (FDST<double>::mesh_type::Node *)0,\
	       (FDST<vector<pair<MSRC::Edge::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::EDGE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Edge *)0,\
	       (FDST<double>::mesh_type::Edge *)0,\
	       (FDST<vector<pair<MSRC::Edge::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::FACE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Edge *)0,\
	       (FDST<double>::mesh_type::Face *)0,\
	       (FDST<vector<pair<MSRC::Edge::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::CELL:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Edge *)0,\
	       (FDST<double>::mesh_type::Cell *)0,\
	       (FDST<vector<pair<MSRC::Edge::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
    \
    default:\
      break;\
    }\
    break;\
\
  case Field::FACE:\
    switch (fdst->data_at())\
    {\
    case Field::NODE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Face *)0,\
	       (FDST<double>::mesh_type::Node *)0,\
	       (FDST<vector<pair<MSRC::Face::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::EDGE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Face *)0,\
	       (FDST<double>::mesh_type::Edge *)0,\
	       (FDST<vector<pair<MSRC::Face::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::FACE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Face *)0,\
	       (FDST<double>::mesh_type::Face *)0,\
	       (FDST<vector<pair<MSRC::Face::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::CELL:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Face *)0,\
	       (FDST<double>::mesh_type::Cell *)0,\
	       (FDST<vector<pair<MSRC::Face::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
    \
    default:\
      break;\
    }\
    break;\
\
  case Field::CELL:\
    switch (fdst->data_at())\
    {\
    case Field::NODE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Cell *)0,\
	       (FDST<double>::mesh_type::Node *)0,\
	       (FDST<vector<pair<MSRC::Cell::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::EDGE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Cell *)0,\
	       (FDST<double>::mesh_type::Edge *)0,\
	       (FDST<vector<pair<MSRC::Cell::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::FACE:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Cell *)0,\
	       (FDST<double>::mesh_type::Face *)0,\
	       (FDST<vector<pair<MSRC::Cell::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
\
    case Field::CELL:\
      callback((MSRC *)(fsrc->mesh().get_rep()),\
	       (FDST<double>::mesh_type *)(fdst->mesh().get_rep()),\
	       (MSRC::Cell *)0,\
	       (FDST<double>::mesh_type::Cell *)0,\
	       (FDST<vector<pair<MSRC::Cell::index_type, double> > > *)0,\
	       fdst->data_at());\
      break;\
    \
    default:\
      break;\
    }\
    break;\
\
  default:\
    break;\
  }\
}
	   


void
BuildInterpolant::execute()
{
  dst_port = (FieldIPort *)get_iport("Destination");
  FieldHandle dfieldhandle;
  Field *dst_field;
  if (!(dst_port->get(dfieldhandle) && (dst_field = dfieldhandle.get_rep())))
  {
    return;
  }

  src_port = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  Field *src_field;
  if (!(src_port->get(sfieldhandle) && (src_field = sfieldhandle.get_rep())))
  {
    return;
  }
  ofp = (FieldOPort *)get_oport("Interpolant");
  const string dst_mesh_name = dst_field->get_type_name(0);
  const string src_mesh_name = src_field->get_type_name(0);

  if (src_mesh_name == "TetVolMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(TetVolMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(TetVolMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(TetVolMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(TetVolMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(TetVolMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(TetVolMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(TetVolMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "LatVolMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(LatVolMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(LatVolMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(LatVolMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(LatVolMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(LatVolMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(LatVolMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(LatVolMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "TriSurfMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(TriSurfMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "ImageMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(ImageMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(ImageMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(ImageMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(ImageMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(ImageMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(ImageMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(ImageMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "ContourMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(ContourMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(ContourMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(ContourMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(ContourMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(ContourMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(ContourMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(ContourMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "ScanlineMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(ScanlineMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else if (src_mesh_name == "PointCloudMesh")
  {
    if (dst_mesh_name == "TetVolMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, TetVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "LatVolMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, LatticeVol, src_field, dst_field);
    }
    else if (dst_mesh_name == "ImageMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, ImageField, src_field, dst_field);
    }
    else if (dst_mesh_name == "TriSurfMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, TriSurf, src_field, dst_field);
    }
    else if (dst_mesh_name == "ScanlineMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, ScanlineField, src_field, dst_field);
    }
    else if (dst_mesh_name == "ContourMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, ContourField, src_field, dst_field);
    }
    else if (dst_mesh_name == "PointCloudMesh")
    {
      CALLBACK_WRAP(PointCloudMesh, PointCloud, src_field, dst_field);
    }
    else
    {
      error("Unrecognized destination mesh type");
    }
  }
  else
  {
    error("Unrecognized source mesh type");
  }
}

} // End namespace SCIRun
