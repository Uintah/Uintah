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

//    File   : Probe.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(Probe_h)
#define Probe_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/PointCloud.h>
#include <sstream>

namespace SCIRun {

class ProbeLocateAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle mesh_h,
		       const Point &p,
		       string &nodestr,
		       string &edgestr,
		       string &facestr,
		       string &cellstr) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class ProbeLocateAlgoT : public ProbeLocateAlgo
{
public:
  virtual void execute(MeshHandle mesh_h,
		       const Point &p,
		       string &nodestr,
		       string &edgestr,
		       string &facestr,
		       string &cellstr);
};


template <class MESH>
void
ProbeLocateAlgoT<MESH>::execute(MeshHandle mesh_h,
				const Point &p,
				string &nodestr,
				string &edgestr,
				string &facestr,
				string &cellstr)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  {
    typename MESH::Node::index_type index;
//    typename MESH::Node::index_type index = 0;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist = 1.0e6;
      typename MESH::Node::iterator bi; mesh->begin(bi);
      typename MESH::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	  found_p = true;
	}
	++bi;
      }
    }
    if (found_p)
    {
      std::ostringstream ostr;
      ostr << index;
      nodestr = ostr.str();
    }
  }

  {
    typename MESH::Edge::index_type index = 0;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist = 1.0e6;
      typename MESH::Edge::iterator bi; mesh->begin(bi);
      typename MESH::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	  found_p = true;
	}
	++bi;
      }
    }
    if (found_p)
    {
      std::ostringstream ostr;
      ostr << index;
      edgestr = ostr.str();
    }
  }

  {
    typename MESH::Face::index_type index = 0;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist = 1.0e6;
      typename MESH::Face::iterator bi; mesh->begin(bi);
      typename MESH::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	  found_p = true;
	}
	++bi;
      }
    }
    if (found_p)
    {
      std::ostringstream ostr;
      ostr << index;
      facestr = ostr.str();
    }
  }

  {
    typename MESH::Cell::index_type index;
//    typename MESH::Cell::index_type index = 0;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist = 1.0e6;
      typename MESH::Cell::iterator bi; mesh->begin(bi);
      typename MESH::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (dist < mindist)
	{
	  mindist = dist;
	  index = *bi;
	  found_p = true;
	}
	++bi;
      }
    }
    if (found_p)
    {
      std::ostringstream ostr;
      ostr << index;
      cellstr = ostr.str();
    }
  }
}



class ProbeCenterAlgo : public DynamicAlgoBase
{
public:
  virtual bool get_node(MeshHandle mesh_h, const string &index, Point &p) = 0;
  virtual bool get_edge(MeshHandle mesh_h, const string &index, Point &p) = 0;
  virtual bool get_face(MeshHandle mesh_h, const string &index, Point &p) = 0;
  virtual bool get_cell(MeshHandle mesh_h, const string &index, Point &p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class ProbeCenterAlgoT : public ProbeCenterAlgo
{
public:
  virtual bool get_node(MeshHandle mesh_h, const string &index, Point &p);
  virtual bool get_edge(MeshHandle mesh_h, const string &index, Point &p);
  virtual bool get_face(MeshHandle mesh_h, const string &index, Point &p);
  virtual bool get_cell(MeshHandle mesh_h, const string &index, Point &p);
};


template <class MESH>
bool
ProbeCenterAlgoT<MESH>::get_node(MeshHandle mesh_h, const string &indexstr,
				 Point &p)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  unsigned int i = atoi(indexstr.c_str());
  typename MESH::Node::index_type index(i);

  typename MESH::Node::size_type size;
  mesh->size(size);

  if (index < size)
  {
    mesh->get_center(p, index);
    return true;
  }
  return false;
}

template <class MESH>
bool
ProbeCenterAlgoT<MESH>::get_edge(MeshHandle mesh_h, const string &indexstr,
				 Point &p)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  unsigned int i = atoi(indexstr.c_str());
  typename MESH::Edge::index_type index(i);
  
  typename MESH::Edge::size_type size;
  mesh->size(size);

  if (index < size)
  {
    mesh->get_center(p, index);
    return true;
  }
  return false;
}

template <class MESH>
bool
ProbeCenterAlgoT<MESH>::get_face(MeshHandle mesh_h, const string &indexstr,
				 Point &p)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  unsigned int i = atoi(indexstr.c_str());
  typename MESH::Face::index_type index(i);
  
  typename MESH::Face::size_type size;
  mesh->size(size);

  if (index < size)
  {
    mesh->get_center(p, index);
    return true;
  }
  return false;
}

template <class MESH>
bool
ProbeCenterAlgoT<MESH>::get_cell(MeshHandle mesh_h, const string &indexstr,
				 Point &p)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  unsigned int i = atoi(indexstr.c_str());
  typename MESH::Cell::index_type index(i);
  
  typename MESH::Cell::size_type size;
  mesh->size(size);

  if (index < size)
  {
    mesh->get_center(p, index);
    return true;
  }
  return false;
}


} // end namespace SCIRun

#endif // Probe_h
