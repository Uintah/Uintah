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
 *  ApplyInterpolant.cc:  Apply an interpolant field to project the data
 *                 from one field onto the mesh of another field.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/Dispatch2.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::cerr;
using std::vector;
using std::pair;


#if 0
template <> const string find_type_name(TetVolMesh::node_index *);
template <> const string find_type_name(TetVolMesh::edge_index *);
template <> const string find_type_name(TetVolMesh::face_index *);
template <> const string find_type_name(TetVolMesh::cell_index *);
template <> const string find_type_name(LatVolMesh::node_index *);
template <> const string find_type_name(LatVolMesh::edge_index *);
template <> const string find_type_name(LatVolMesh::face_index *);
template <> const string find_type_name(LatVolMesh::cell_index *);
#endif

template <> const string find_type_name(vector<pair<TetVolMesh::node_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::edge_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::face_index, double> > *);
template <> const string find_type_name(vector<pair<TetVolMesh::cell_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::node_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::edge_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::face_index, double> > *);
template <> const string find_type_name(vector<pair<LatVolMesh::cell_index, double> > *);

void Pio(Piostream &, TetVolMesh::node_index);
void Pio(Piostream &, TetVolMesh::edge_index);
void Pio(Piostream &, TetVolMesh::face_index);
void Pio(Piostream &, TetVolMesh::cell_index);

void Pio(Piostream &, LatVolMesh::node_index);
void Pio(Piostream &, LatVolMesh::edge_index);
void Pio(Piostream &, LatVolMesh::face_index);
void Pio(Piostream &, LatVolMesh::cell_index);

// TetVol
// TetVol

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<TetVolMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;

// LatVol

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<LatVolMesh, FData3d<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;

// TriSurf

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<TetVolMesh::cell_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::node_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::edge_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::face_index, double> > > >::query_interpolate() const;

template <>
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::interp_type *
GenericField<TriSurfMesh, vector<vector<pair<LatVolMesh::cell_index, double> > > >::query_interpolate() const;


template <> Vector TetVol<vector<pair<TetVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<TetVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::node_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::edge_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::face_index, double> > >::cell_gradient(TetVolMesh::cell_index);
template <> Vector TetVol<vector<pair<LatVolMesh::cell_index, double> > >::cell_gradient(TetVolMesh::cell_index);

template <> bool LatticeVol<vector<pair<TetVolMesh::node_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::face_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<TetVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::node_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::edge_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::face_index, double> > >::get_gradient(Vector &, Point &);
template <> bool LatticeVol<vector<pair<LatVolMesh::cell_index, double> > >::get_gradient(Vector &, Point &);


class ApplyInterpolant : public Module {
public:
  ApplyInterpolant(const clString& id);
  virtual ~ApplyInterpolant();

  virtual void execute();

  template <class FDST, class FSRC, class FITP>
  void dispatch_src_node(FDST *fdst, FSRC *fsrc, FITP *fitp);

  template <class FDST, class FSRC, class FITP>
  void dispatch_src_edge(FDST *fdst, FSRC *fsrc, FITP *fitp);

  template <class FDST, class FSRC, class FITP>
  void dispatch_src_face(FDST *fdst, FSRC *fsrc, FITP *fitp);

  template <class FDST, class FSRC, class FITP>
  void dispatch_src_cell(FDST *fdst, FSRC *fsrc, FITP *fitp);
};


extern "C" Module* make_ApplyInterpolant(const clString& id)
{
  return new ApplyInterpolant(id);
}


ApplyInterpolant::ApplyInterpolant(const clString& id)
  : Module("ApplyInterpolant", id, Filter, "Fields", "SCIRun")
{
}


ApplyInterpolant::~ApplyInterpolant()
{
}



template <class FDST, class FSRC, class FITP>
void
ApplyInterpolant::dispatch_src_node(FDST *fdst, FSRC *fsrc, FITP *fitp)
{
  FDST *fout = fdst->clone();
  FieldHandle fhout(fout);

  switch(fdst->data_at())
  {
  case Field::NODE:
    {
      typename FDST::mesh_type::node_iterator iter =
	fout->get_typed_mesh()->node_begin();
      while (iter != fout->get_typed_mesh()->node_end())
      {
	vector<pair<typename FSRC::mesh_type::node_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::node_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::EDGE:
    {
      typename FDST::mesh_type::edge_iterator iter =
	fout->get_typed_mesh()->edge_begin();
      while (iter != fout->get_typed_mesh()->edge_end())
      {
	vector<pair<typename FSRC::mesh_type::node_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::node_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::FACE:
    {
      typename FDST::mesh_type::face_iterator iter =
	fout->get_typed_mesh()->face_begin();
      while (iter != fout->get_typed_mesh()->face_end())
      {
	vector<pair<typename FSRC::mesh_type::node_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::node_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::CELL:
    {
      typename FDST::mesh_type::cell_iterator iter =
	fout->get_typed_mesh()->cell_begin();
      while (iter != fout->get_typed_mesh()->cell_end())
      {
	vector<pair<typename FSRC::mesh_type::node_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::node_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }

  default:
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  ofp->send(fhout);
}


template <class FDST, class FSRC, class FITP>
void
ApplyInterpolant::dispatch_src_edge(FDST *fdst, FSRC *fsrc, FITP *fitp)
{
  FDST *fout = fdst->clone();
  FieldHandle fhout(fout);

  switch(fdst->data_at())
  {
  case Field::NODE:
    {
      typename FDST::mesh_type::node_iterator iter =
	fout->get_typed_mesh()->node_begin();
      while (iter != fout->get_typed_mesh()->node_end())
      {
	vector<pair<typename FSRC::mesh_type::edge_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::edge_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::EDGE:
    {
      typename FDST::mesh_type::edge_iterator iter =
	fout->get_typed_mesh()->edge_begin();
      while (iter != fout->get_typed_mesh()->edge_end())
      {
	vector<pair<typename FSRC::mesh_type::edge_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::edge_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::FACE:
    {
      typename FDST::mesh_type::face_iterator iter =
	fout->get_typed_mesh()->face_begin();
      while (iter != fout->get_typed_mesh()->face_end())
      {
	vector<pair<typename FSRC::mesh_type::edge_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::edge_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::CELL:
    {
      typename FDST::mesh_type::cell_iterator iter =
	fout->get_typed_mesh()->cell_begin();
      while (iter != fout->get_typed_mesh()->cell_end())
      {
	vector<pair<typename FSRC::mesh_type::edge_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::edge_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  default:
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  ofp->send(fhout);
}

template <class FDST, class FSRC, class FITP>
void
ApplyInterpolant::dispatch_src_face(FDST *fdst, FSRC *fsrc, FITP *fitp)
{
  FDST *fout = fdst->clone();
  FieldHandle fhout(fout);

  switch(fdst->data_at())
  {
  case Field::NODE:
    {
      typename FDST::mesh_type::node_iterator iter =
	fout->get_typed_mesh()->node_begin();
      while (iter != fout->get_typed_mesh()->node_end())
      {
	vector<pair<typename FSRC::mesh_type::face_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::face_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::EDGE:
    {
      typename FDST::mesh_type::edge_iterator iter =
	fout->get_typed_mesh()->edge_begin();
      while (iter != fout->get_typed_mesh()->edge_end())
      {
	vector<pair<typename FSRC::mesh_type::face_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::face_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::FACE:
    {
      typename FDST::mesh_type::face_iterator iter =
	fout->get_typed_mesh()->face_begin();
      while (iter != fout->get_typed_mesh()->face_end())
      {
	vector<pair<typename FSRC::mesh_type::face_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::face_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::CELL:
    {
      typename FDST::mesh_type::cell_iterator iter =
	fout->get_typed_mesh()->cell_begin();
      while (iter != fout->get_typed_mesh()->cell_end())
      {
	vector<pair<typename FSRC::mesh_type::face_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::face_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  default:
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  ofp->send(fhout);
}

template <class FDST, class FSRC, class FITP>
void
ApplyInterpolant::dispatch_src_cell(FDST *fdst, FSRC *fsrc, FITP *fitp)
{
  FDST *fout = fdst->clone();
  FieldHandle fhout(fout);

  switch(fdst->data_at())
  {
  case Field::NODE:
    {
      typename FDST::mesh_type::node_iterator iter =
	fout->get_typed_mesh()->node_begin();
      while (iter != fout->get_typed_mesh()->node_end())
      {
	vector<pair<typename FSRC::mesh_type::cell_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::cell_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::EDGE:
    {
      typename FDST::mesh_type::edge_iterator iter =
	fout->get_typed_mesh()->edge_begin();
      while (iter != fout->get_typed_mesh()->edge_end())
      {
	vector<pair<typename FSRC::mesh_type::cell_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::cell_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::FACE:
    {
      typename FDST::mesh_type::face_iterator iter =
	fout->get_typed_mesh()->face_begin();
      while (iter != fout->get_typed_mesh()->face_end())
      {
	vector<pair<typename FSRC::mesh_type::cell_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::cell_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  case Field::CELL:
    {
      typename FDST::mesh_type::cell_iterator iter =
	fout->get_typed_mesh()->cell_begin();
      while (iter != fout->get_typed_mesh()->cell_end())
      {
	vector<pair<typename FSRC::mesh_type::cell_index, double> > v;
	fitp->value(v, *iter);
	typename FDST::value_type val = fsrc->value(v[0].first) * v[0].second;
	vector<pair<typename FSRC::mesh_type::cell_index, double> >::size_type j;
	for (j = 1; j < v.size(); j++)
	{
	  val += fsrc->value(v[j].first) * v[j].second;
	}
	fout->set_value(val, *iter);
      }
    }
    break;

  default:
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output");
  ofp->send(fhout);
}


#define HAIRY_MACRO(FDST, DDST, FSRC, DSRC) \
switch(src_field->data_at())\
{\
case Field::NODE:\
  dispatch_src_node((FDST<DDST> *) dst_field,\
		    (FSRC<DSRC> *) src_field,\
		    (FDST<vector<pair<FSRC<DSRC>::mesh_type::node_index, double> > > *)itp_field);\
  break;\
\
case Field::EDGE:\
  dispatch_src_edge((FDST<DDST> *) dst_field,\
		    (FSRC<DSRC> *) src_field,\
		    (FDST<vector<pair<FSRC<DSRC>::mesh_type::edge_index, double> > > *)itp_field);\
  break;\
\
case Field::FACE:\
  dispatch_src_face((FDST<DDST> *) dst_field,\
		    (FSRC<DSRC> *) src_field,\
		    (FDST<vector<pair<FSRC<DSRC>::mesh_type::face_index, double> > > *)itp_field);\
  break;\
\
case Field::CELL:\
  dispatch_src_cell((FDST<DDST> *) dst_field,\
		    (FSRC<DSRC> *) src_field,\
		    (FDST<vector<pair<FSRC<DSRC>::mesh_type::cell_index, double> > > *)itp_field);\
  break;\
\
default:\
  return;\
}


void
ApplyInterpolant::execute()
{
  FieldIPort *dst_port = (FieldIPort *)get_iport("Destination");
  FieldHandle dfieldhandle;
  Field *dst_field;
  if (!(dst_port->get(dfieldhandle) && (dst_field = dfieldhandle.get_rep())))
  {
    return;
  }

  FieldIPort *src_port = (FieldIPort *)get_iport("Source");
  FieldHandle sfieldhandle;
  Field *src_field;
  if (!(src_port->get(sfieldhandle) && (src_field = sfieldhandle.get_rep())))
  {
    return;
  }

  FieldIPort *itp_port = (FieldIPort *)get_iport("Interpolant");
  FieldHandle ifieldhandle;
  Field *itp_field;
  if (!(itp_port->get(ifieldhandle) && (itp_field = ifieldhandle.get_rep())))
  {
    return;
  }

  const string dst_geom_name = dst_field->get_type_name(0);
  const string dst_data_name = dst_field->get_type_name(1);
  const string src_geom_name = src_field->get_type_name(0);
  const string src_data_name = src_field->get_type_name(1);

  if (dst_geom_name == "TetVol" && dst_data_name == "double" &&
      src_geom_name == "TetVol" && src_data_name == "double")
  {
    HAIRY_MACRO(TetVol, double, TetVol, double)
  }
  else if (dst_geom_name == "TetVol" && dst_data_name == "double" &&
	   src_geom_name == "LatticeVol" && src_data_name == "double")
  {
    HAIRY_MACRO(TetVol, double, LatticeVol, double)
  }
  else if (dst_geom_name == "LatticeVol" && dst_data_name == "double" &&
	   src_geom_name == "TetVol" && src_data_name == "double")
  {
    HAIRY_MACRO(LatticeVol, double, TetVol, double)
  }
  else if (dst_geom_name == "LatticeVol" && dst_data_name == "double" &&
	   src_geom_name == "LatticeVol" && src_data_name == "double")
  {
    HAIRY_MACRO(LatticeVol, double, LatticeVol, double)
  }
  else if (dst_geom_name == "TriSurf" && dst_data_name == "double" &&
	   src_geom_name == "TetVol" && src_data_name == "double")
  {
    HAIRY_MACRO(TriSurf, double, TetVol, double)
  }
  else if (dst_geom_name == "TriSurf" && dst_data_name == "double" &&
	   src_geom_name == "LatticeVol" && src_data_name == "double")
  {
    HAIRY_MACRO(TriSurf, double, LatticeVol, double)
  }
}

} // End namespace SCIRun


