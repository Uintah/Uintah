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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <sstream>

namespace SCIRun {

class ProbeLocateAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle mesh_h,
		       const Point &p,
		       bool shownode, string &nodestr,
		       bool showedge, string &edgestr,
		       bool showface, string &facestr,
		       bool showcell, string &cellstr) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class ProbeLocateAlgoT : public ProbeLocateAlgo
{
public:
  virtual void execute(MeshHandle mesh_h,
		       const Point &p,
		       bool shownode, string &nodestr,
		       bool showedge, string &edgestr,
		       bool showface, string &facestr,
		       bool showcell, string &cellstr);
};


template <class MESH>
void
ProbeLocateAlgoT<MESH>::execute(MeshHandle mesh_h,
				const Point &p,
				bool shownode, string &nodestr,
				bool showedge, string &edgestr,
				bool showface, string &facestr,
				bool showcell, string &cellstr)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());
  mesh->synchronize(Mesh::LOCATE_E);

  if (shownode)
  {
    mesh->synchronize(Mesh::NODES_E);
    typename MESH::Node::index_type index;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist;
      typename MESH::Node::iterator bi; mesh->begin(bi);
      typename MESH::Node::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (!found_p || (dist < mindist))
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

  if (showedge)
  {
    mesh->synchronize(Mesh::EDGES_E);
    typename MESH::Edge::index_type index;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist;
      typename MESH::Edge::iterator bi; mesh->begin(bi);
      typename MESH::Edge::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (!found_p || dist < mindist)
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

  if (showface)
  {
    mesh->synchronize(Mesh::FACES_E);
    typename MESH::Face::index_type index;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist;
      typename MESH::Face::iterator bi; mesh->begin(bi);
      typename MESH::Face::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (!found_p || dist < mindist)
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

  if (showcell)
  {
    mesh->synchronize(Mesh::CELLS_E);
    typename MESH::Cell::index_type index;
    bool found_p = true;
    if (!mesh->locate(index, p))
    {
      found_p = false;
      double mindist;
      typename MESH::Cell::iterator bi; mesh->begin(bi);
      typename MESH::Cell::iterator ei; mesh->end(ei);
      while (bi != ei)
      {
	Point c;
	mesh->get_center(c, *bi);
	const double dist = (p - c).length2();
	if (!found_p || dist < mindist)
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
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc);
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


template <class M, class L, class S>
bool
probe_center_compute_index(L &index, S &size,
			   const M *mesh, const string &indexstr)
{
  unsigned int i = atoi(indexstr.c_str());
  index = i;
  mesh->size(size);
  return index < size;
}


template <>
bool
probe_center_compute_index(LatVolMesh::Node::index_type &index,
			   LatVolMesh::Node::size_type &size,
			   const LatVolMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(LatVolMesh::Cell::index_type &index,
			   LatVolMesh::Cell::size_type &size,
			   const LatVolMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(StructHexVolMesh::Node::index_type &index,
			   StructHexVolMesh::Node::size_type &size,
			   const StructHexVolMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(StructHexVolMesh::Cell::index_type &index,
			   StructHexVolMesh::Cell::size_type &size,
			   const StructHexVolMesh *m, const string &indexstr);


template <>
bool
probe_center_compute_index(ImageMesh::Node::index_type &index,
			   ImageMesh::Node::size_type &size,
			   const ImageMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(ImageMesh::Face::index_type &index,
			   ImageMesh::Face::size_type &size,
			   const ImageMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(StructQuadSurfMesh::Node::index_type &index,
			   StructQuadSurfMesh::Node::size_type &size,
			   const StructQuadSurfMesh *m,
			   const string &indexstr);

template <>
bool
probe_center_compute_index(StructQuadSurfMesh::Face::index_type &index,
			   StructQuadSurfMesh::Face::size_type &size,
			   const StructQuadSurfMesh *m,
			   const string &indexstr);


template <class MESH>
bool
ProbeCenterAlgoT<MESH>::get_node(MeshHandle mesh_h, const string &indexstr,
				 Point &p)
{
  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());
  mesh->synchronize(Mesh::NODES_E);

  typename MESH::Node::index_type index;
  typename MESH::Node::size_type size;
  if (probe_center_compute_index(index, size, mesh, indexstr))
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
  mesh->synchronize(Mesh::EDGES_E);

  typename MESH::Edge::size_type size;
  typename MESH::Edge::index_type index;
  if (probe_center_compute_index(index, size, mesh, indexstr))
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
  mesh->synchronize(Mesh::FACES_E);

  typename MESH::Face::size_type size;
  typename MESH::Face::index_type index;
  if (probe_center_compute_index(index, size, mesh, indexstr))
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
  mesh->synchronize(Mesh::CELLS_E);

  typename MESH::Cell::size_type size;
  typename MESH::Cell::index_type index;
  if (probe_center_compute_index(index, size, mesh, indexstr))
  {
    mesh->get_center(p, index);
    return true;
  }
  return false;
}


} // end namespace SCIRun

#endif // Probe_h
