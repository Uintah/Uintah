/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : Probe.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(Probe_h)
#define Probe_h

#include <Core/Geometry/Tensor.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <sstream>

namespace SCIRun {

typedef LatVolMesh<HexTrilinearLgn<Point> >         LVMesh;
typedef ImageMesh<QuadBilinearLgn<Point> >          IMesh;
typedef StructHexVolMesh<HexTrilinearLgn<Point> >   SHVMesh;
typedef StructQuadSurfMesh<QuadBilinearLgn<Point> > SQSMesh;

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
probe_center_compute_index(LVMesh::Node::index_type &index,
			   LVMesh::Node::size_type &size,
			   const LVMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(LVMesh::Cell::index_type &index,
			   LVMesh::Cell::size_type &size,
			   const LVMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(SHVMesh::Node::index_type &index,
			   SHVMesh::Node::size_type &size,
			   const SHVMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(SHVMesh::Cell::index_type &index,
			   SHVMesh::Cell::size_type &size,
			   const SHVMesh *m, const string &indexstr);


template <>
bool
probe_center_compute_index(IMesh::Node::index_type &index,
			   IMesh::Node::size_type &size,
			   const IMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(IMesh::Face::index_type &index,
			   IMesh::Face::size_type &size,
			   const IMesh *m, const string &indexstr);

template <>
bool
probe_center_compute_index(SQSMesh::Node::index_type &index,
			   SQSMesh::Node::size_type &size,
			   const SQSMesh *m,
			   const string &indexstr);

template <>
bool
probe_center_compute_index(SQSMesh::Face::index_type &index,
			   SQSMesh::Face::size_type &size,
			   const SQSMesh *m,
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
