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
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/InterpolantTypes.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class BuildInterpolant : public Module
{
  GuiString   interp_op_gui_;

public:
  BuildInterpolant(const string& id);
  virtual ~BuildInterpolant();
  virtual void execute();

  template <class FOUT, class MDST, class MSRC>
  void dispatch_src_node(MDST *mdst, MSRC *msrc,
			 Field::data_location mdstloc,
			 FOUT *);

  template <class FOUT, class MDST, class MSRC>
  void dispatch_src_edge(MDST *mdst, MSRC *msrc,
			 Field::data_location mdstloc,
			 FOUT *);

  template <class FOUT, class MDST, class MSRC>
  void dispatch_src_face(MDST *mdst, MSRC *msrc,
			 Field::data_location mdstloc,
			 FOUT *);

  template <class FOUT, class MDST, class MSRC>
  void dispatch_src_cell(MDST *mdst, MSRC *msrc,
			 Field::data_location mdstloc,
			 FOUT *);

  double point_distance(const Point &p1, const Point &p2);
  double point_distance2(const Point &p1, const Point &p2);
  void normalize(vector<pair<int, double> > &v);

  template <class Mesh>
  void find_closest_node(Mesh *mesh, typename Mesh::node_index &idx, Point &p);

  template <class Mesh>
  void find_closest_edge(Mesh *mesh, typename Mesh::edge_index &idx, Point &p);

  template <class Mesh>
  void find_closest_face(Mesh *mesh, typename Mesh::face_index &idx, Point &p);

  template <class Mesh>
  void find_closest_cell(Mesh *mesh, typename Mesh::cell_index &idx, Point &p);

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

double
BuildInterpolant::point_distance(const Point &p1, const Point &p2)
{
  const double dx = p2.x() - p1.x();
  const double dy = p2.y() - p1.y();
  const double dz = p2.z() - p1.z();

  return sqrt(dx * dx + dy * dy + dz * dz);
}

double
BuildInterpolant::point_distance2(const Point &p1, const Point &p2)
{
  const double dx = p2.x() - p1.x();
  const double dy = p2.y() - p1.y();
  const double dz = p2.z() - p1.z();

  return dx * dx + dy * dy + dz * dz;
}



template <class Mesh>
void
BuildInterpolant::find_closest_node(Mesh *mesh, typename Mesh::node_index &idx,
				    Point &p)
{
  bool first_p = true;
  double min_dist = 1.0e6;
  typename Mesh::node_iterator itr = mesh->node_begin();
  while (itr != mesh->node_end())
  {
    Point p2;

    mesh->get_center(p2, *itr);

    const double dist = point_distance2(p, p2);
    if (dist < min_dist || first_p)
    {
      idx = *itr;
      min_dist = dist;
      first_p = false;
    }
   
    ++itr;
  }
}

template <class Mesh>
void
BuildInterpolant::find_closest_edge(Mesh *mesh, typename Mesh::edge_index &idx,
				    Point &p)
{
  bool first_p = true;
  double min_dist = 1.0e6;
  typename Mesh::edge_iterator itr = mesh->edge_begin();
  while (itr != mesh->edge_end())
  {
    Point p2;

    mesh->get_center(p2, *itr);

    const double dist = point_distance2(p, p2);
    if (dist < min_dist || first_p)
    {
      idx = *itr;
      min_dist = dist;
      first_p = false;
    }
   
    ++itr;
  }
}

template <class Mesh>
void
BuildInterpolant::find_closest_face(Mesh *mesh, typename Mesh::face_index &idx,
				    Point &p)
{
  bool first_p = true;
  double min_dist = 1.0e6;
  typename Mesh::face_iterator itr = mesh->face_begin();
  while (itr != mesh->face_end())
  {
    Point p2;

    mesh->get_center(p2, *itr);

    const double dist = point_distance2(p, p2);
    if (dist < min_dist || first_p)
    {
      idx = *itr;
      min_dist = dist;
      first_p = false;
    }
   
    ++itr;
  }
}

template <class Mesh>
void
BuildInterpolant::find_closest_cell(Mesh *mesh, typename Mesh::cell_index &idx,
				    Point &p)
{
  bool first_p = true;
  double min_dist = 1.0e6;
  typename Mesh::cell_iterator itr = mesh->cell_begin();
  while (itr != mesh->cell_end())
  {
    Point p2;

    mesh->get_center(p2, *itr);

    const double dist = point_distance2(p, p2);
    if (dist < min_dist || first_p)
    {
      idx = *itr;
      min_dist = dist;
      first_p = false;
    }
   
    ++itr;
  }
}


void
BuildInterpolant::normalize(vector<pair<int, double> > &v)
{
  if (v.empty()) { return; } 

  double sum = 0.0;
  vector<pair<int, double> >::size_type i;
  for (i = 0; i < v.size(); i++)
  {
    sum += v[i].second;
  }

  sum = 1.0 / sum;
  for (i = 0; i < v.size(); i++)
  {
    v[i].second *= sum;
  }
}



template <class FOUT, class MDST, class MSRC>
void
BuildInterpolant::dispatch_src_node(MDST *mdst, MSRC *msrc,
				    Field::data_location mdstloc,
				    FOUT *)
{
  FOUT *ofield = new FOUT(mdst, mdstloc);
  FieldHandle fh(ofield);
  
  switch(mdstloc)
  {
  case Field::NODE:
    {
      typename MDST::node_iterator itr = mdst->node_begin();
      while (itr != mdst->node_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::node_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::node_array array;
	  msrc->get_nodes(array, idx);
  
	  typename MSRC::node_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	    
	    v.push_back(pair<typename MSRC::node_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::node_index idx2;
	  find_closest_node(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::node_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::EDGE:
    {
      typename MDST::edge_iterator itr = mdst->edge_begin();
      while (itr != mdst->edge_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::node_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::node_array array;
	  msrc->get_nodes(array, idx);
  
	  typename MSRC::node_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::node_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::node_index idx2;
	  find_closest_node(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::node_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::FACE:
    {
      typename MDST::face_iterator itr = mdst->face_begin();
      while (itr != mdst->face_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::node_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::node_array array;
	  msrc->get_nodes(array, idx);
  
	  typename MSRC::node_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::node_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::node_index idx2;
	  find_closest_node(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::node_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::CELL:
    {
      typename MDST::cell_iterator itr = mdst->cell_begin();
      while (itr != mdst->cell_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::node_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::node_array array;
	  msrc->get_nodes(array, idx);
  
	  typename MSRC::node_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::node_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::node_index idx2;
	  find_closest_node(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::node_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  default:
    // no data at value for destination field
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Interpolant");
  ofp->send(fh);
}


template <class FOUT, class MDST, class MSRC>
void
BuildInterpolant::dispatch_src_edge(MDST *mdst, MSRC *msrc,
				    Field::data_location mdstloc,
				    FOUT *)
{
  FOUT *ofield = new FOUT(mdst, mdstloc);
  FieldHandle fh(ofield);
  
  switch(mdstloc)
  {
  case Field::NODE:
    {
      typename MDST::node_iterator itr = mdst->node_begin();
      while (itr != mdst->node_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::edge_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::edge_array array;
	  msrc->get_edges(array, idx);
  
	  typename MSRC::edge_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::edge_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::edge_index idx2;
	  find_closest_edge(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::edge_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::EDGE:
    {
      typename MDST::edge_iterator itr = mdst->edge_begin();
      while (itr != mdst->edge_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::edge_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::edge_array array;
	  msrc->get_edges(array, idx);
  
	  typename MSRC::edge_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::edge_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::edge_index idx2;
	  find_closest_edge(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::edge_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::FACE:
    {
      typename MDST::face_iterator itr = mdst->face_begin();
      while (itr != mdst->face_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::edge_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::edge_array array;
	  msrc->get_edges(array, idx);
  
	  typename MSRC::edge_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::edge_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::edge_index idx2;
	  find_closest_edge(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::edge_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::CELL:
    {
      typename MDST::cell_iterator itr = mdst->cell_begin();
      while (itr != mdst->cell_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::edge_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::edge_array array;
	  msrc->get_edges(array, idx);
  
	  typename MSRC::edge_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::edge_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::edge_index idx2;
	  find_closest_edge(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::edge_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  default:
    // no data at value for destination field
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Interpolant");
  ofp->send(fh);
}


template <class FOUT, class MDST, class MSRC>
void
BuildInterpolant::dispatch_src_face(MDST *mdst, MSRC *msrc,
				    Field::data_location mdstloc,
				    FOUT *)
{
  FOUT *ofield = new FOUT(mdst, mdstloc);
  FieldHandle fh(ofield);
  
  switch(mdstloc)
  {
  case Field::NODE:
    {
      typename MDST::node_iterator itr = mdst->node_begin();
      while (itr != mdst->node_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::face_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::face_array array;
	  msrc->get_faces(array, idx);
  
	  typename MSRC::face_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::face_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::face_index idx2;
	  find_closest_face(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::face_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::EDGE:
    {
      typename MDST::edge_iterator itr = mdst->edge_begin();
      while (itr != mdst->edge_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::face_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::face_array array;
	  msrc->get_faces(array, idx);
  
	  typename MSRC::face_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::face_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::face_index idx2;
	  find_closest_face(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::face_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::FACE:
    {
      typename MDST::face_iterator itr = mdst->face_begin();
      while (itr != mdst->face_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::face_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::face_array array;
	  msrc->get_faces(array, idx);
  
	  typename MSRC::face_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::face_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::face_index idx2;
	  find_closest_face(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::face_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::CELL:
    {
      typename MDST::cell_iterator itr = mdst->cell_begin();
      while (itr != mdst->cell_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::face_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  typename MSRC::face_array array;
	  msrc->get_faces(array, idx);
  
	  typename MSRC::face_array::iterator array_itr = array.begin();
	  while (array_itr != array.end())
	  {
	    Point p2;
	    msrc->get_center(p2, *array_itr);
	  
	    v.push_back(pair<typename MSRC::face_index, double>
			(*array_itr, point_distance(p, p2)));
	    ++array_itr;
	  }
	}
	else
	{
	  typename MSRC::face_index idx2;
	  find_closest_face(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::face_index, double>(idx2, 1.0));
	}
	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  default:
    // no data at value for destination field
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Interpolant");
  ofp->send(fh);
}


template <class FOUT, class MDST, class MSRC>
void
BuildInterpolant::dispatch_src_cell(MDST *mdst, MSRC *msrc,
				    Field::data_location mdstloc,
				    FOUT *)
{
  FOUT *ofield = new FOUT(mdst, mdstloc);
  FieldHandle fh(ofield);
  
  switch(mdstloc)
  {
  case Field::NODE:
    {
      typename MDST::node_iterator itr = mdst->node_begin();
      while (itr != mdst->node_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::cell_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  v.push_back(pair<typename MSRC::cell_index, double>(idx, 1.0));
	}
	else
	{
	  typename MSRC::cell_index idx2;
	  find_closest_cell(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::cell_index, double>(idx2, 1.0));
	}

	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::EDGE:
    {
      typename MDST::edge_iterator itr = mdst->edge_begin();
      while (itr != mdst->edge_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::cell_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  v.push_back(pair<typename MSRC::cell_index, double>(idx, 1.0));
	}
	else
	{
	  typename MSRC::cell_index idx2;
	  find_closest_cell(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::cell_index, double>(idx2, 1.0));
	}

	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::FACE:
    {
      typename MDST::face_iterator itr = mdst->face_begin();
      while (itr != mdst->face_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::cell_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  v.push_back(pair<typename MSRC::cell_index, double>(idx, 1.0));
	}
	else
	{
	  typename MSRC::cell_index idx2;
	  find_closest_cell(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::cell_index, double>(idx2, 1.0));
	}

	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  case Field::CELL:
    {
      typename MDST::cell_iterator itr = mdst->cell_begin();
      while (itr != mdst->cell_end())
      {
	Point p;
	mdst->get_center(p, *itr);
    
	vector<pair<typename MSRC::cell_index, double> > v;
	typename MSRC::cell_index idx;
	if (msrc->locate(idx, p))
	{
	  v.push_back(pair<typename MSRC::cell_index, double>(idx, 1.0));
	}
	else
	{
	  typename MSRC::cell_index idx2;
	  find_closest_cell(msrc, idx2, p);
	  v.push_back(pair<typename MSRC::cell_index, double>(idx2, 1.0));
	}

	ofield->set_value(v, *itr);
	++itr;
      }
    }
    break;

  default:
    // no data at value for destination field
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Interpolant");
  ofp->send(fh);
}


void
BuildInterpolant::execute()
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
  
  const string dst_mesh_name = dst_field->get_type_name(0);
  const string src_mesh_name = src_field->get_type_name(0);

#if 0
  if (dst_mesh_name == "TetVol" &&
      src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TetVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TetVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TetVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TetVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "TetVol" &&
	   src_mesh_name == "LatticeVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TetVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<LatVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TetVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<LatVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TetVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<LatVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TetVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TetVol<vector<pair<LatVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "LatticeVol" &&
	   src_mesh_name == "LatticeVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((LatVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<LatVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((LatVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<LatVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((LatVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<LatVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((LatVolMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<LatVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "LatticeVol" &&
	   src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((LatVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((LatVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((LatVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((LatVolMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(LatticeVol<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "TriSurf" &&
	   src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "TriSurf" &&
	   src_mesh_name == "LatticeVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TriSurfMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<LatVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TriSurfMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<LatVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TriSurfMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<LatVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TriSurfMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<LatVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "ContourField" &&
	   src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "ContourField" &&
	   src_mesh_name == "LatticeVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((ContourMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<LatVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((ContourMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<LatVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((ContourMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<LatVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((ContourMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<LatVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "TriSurf" &&
	   src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else if (dst_mesh_name == "PointCloud" &&
	   src_mesh_name == "LatticeVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((PointCloudMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<LatVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((PointCloudMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<LatVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((PointCloudMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<LatVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((PointCloudMesh *)dst_field->mesh().get_rep(),
			(LatVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<LatVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else
  {
    // Can't build that kind of interpolant.
  }
#else
  if (dst_mesh_name == "TriSurf" &&
      src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((TriSurfMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(TriSurf<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  if (dst_mesh_name == "ContourField" &&
      src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((ContourMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(ContourField<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  if (dst_mesh_name == "PointCloud" &&
      src_mesh_name == "TetVol")
  {
    switch (src_field->data_at())
    {
    case Field::NODE:
      dispatch_src_node((PointCloudMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<TetVolMesh::node_index, double> > > *) 0);
      break;

    case Field::EDGE:
      dispatch_src_edge((PointCloudMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<TetVolMesh::edge_index, double> > > *) 0);
      break;

    case Field::FACE:
      dispatch_src_face((PointCloudMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<TetVolMesh::face_index, double> > > *) 0);
      break;

    case Field::CELL:
      dispatch_src_cell((PointCloudMesh *)dst_field->mesh().get_rep(),
			(TetVolMesh *)src_field->mesh().get_rep(),
			dst_field->data_at(),
			(PointCloud<vector<pair<TetVolMesh::cell_index, double> > > *) 0);
      break;

    default:
      return;
    }
  }
  else
  {
    // Can't build that kind of interpolant.
  }
#endif
}

} // End namespace SCIRun
