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
 *  FieldBoundary.cc:  Build a surface field from a volume field
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/ContourMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/DispatchMesh1.h>
#include <iostream>

namespace SCIRun {

//! Module to build a surface field from a volume field.
class FieldBoundary : public Module {
public:
  FieldBoundary(const string& id);
  virtual ~FieldBoundary();
  virtual void execute();

private:
  
  //! Iterates over mesh, and build TriSurf at the boundary
  template <class Msh> void boundary(const Msh *mesh);
  template <class NIndex>
  void add_ordered_tri(const Point p[3], NIndex nidx[3],
		       const Point &inside, TriSurfMeshHandle tmesh);

  void add_face(const Point &p0, const Point &p1, const Point &p2, 
		MaterialHandle m0, MaterialHandle m1, 
		MaterialHandle m2, GeomTriangles *g);

  //! Input should be a volume field.
  FieldIPort*              infield_;
  int                      infield_gen_;

  //! TriSurf field output.
  FieldOPort*              osurf_;
  
  //! Handle on the generated surface.
  FieldHandle             *tri_fh_;
};

extern "C" Module* make_FieldBoundary(const string& id)
{
  return scinew FieldBoundary(id);
}

FieldBoundary::FieldBoundary(const string& id) : 
  Module("FieldBoundary", id, Filter, "Fields", "SCIRun"),
  infield_gen_(-1),
  tri_fh_(scinew FieldHandle(scinew TriSurf<double>))
{
  // Create the input port
  infield_ = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(infield_);

  osurf_ = scinew FieldOPort(this, "TriSurf", FieldIPort::Atomic);
  add_oport(osurf_);
}
void 
FieldBoundary::add_face(const Point &p0, const Point &p1, const Point &p2, 
			MaterialHandle m0, MaterialHandle m1, 
			MaterialHandle m2, GeomTriangles *g) 
{
  g->add(p0, m0, 
	 p1, m2, 
	 p2, m1);
}

FieldBoundary::~FieldBoundary()
{
}

template <class NIndex>
void 
FieldBoundary::add_ordered_tri(const Point p[3], NIndex nidx[3], 
			       const Point &inside, TriSurfMeshHandle tmesh)
{
  const Vector v1 = p[1] - p[0];
  const Vector v2 = p[2] - p[1];
  const Vector norm = Cross(v1, v2);

  const Vector tmp = inside - p[0];
  const double val = Dot(norm, tmp);
  if (val > 0.0L) {
    // normal points inside, reverse the order.
    tmesh->add_triangle(nidx[2], nidx[1], nidx[0]);
  } else {
    // normal points outside.
    tmesh->add_triangle(nidx[0], nidx[1], nidx[2]);
  }
}

template <> void FieldBoundary::boundary(const ContourMesh *) {
  error("FieldBoundary::boundary can't extract a surface from a ContourMesh");
}

template <> void FieldBoundary::boundary(const PointCloudMesh *) {
  error("FieldBoundary::boundary can't extract a surface from a PointCloudMesh");
}

template <> void FieldBoundary::boundary(const TriSurfMesh *mesh) {
  // Casting away const.  We need const correct handles.
  if (tri_fh_) delete tri_fh_;
  tri_fh_ = scinew FieldHandle(scinew TriSurf<double>(TriSurfMeshHandle((
				        TriSurfMesh *)mesh), Field::NODE));
}

template <class Msh>
void 
FieldBoundary::boundary(const Msh *mesh)
{
  map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type> vertex_map_;
  map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type>::iterator node_iter;
  TriSurfMesh::Node::index_type node_idx[3];

  TriSurfMeshHandle tmesh = scinew TriSurfMesh;
  // Walk all the cells in the mesh.
  Point center;
  typename Msh::Cell::iterator citer = mesh->cell_begin();
  while (citer != mesh->cell_end()) {
    typename Msh::Cell::index_type ci = *citer;
    ++citer;
    mesh->get_center(center, ci);
    // Get all the faces in the cell.
    typename Msh::Face::array_type faces;
    mesh->get_faces(faces, ci);
    // Check each face for neighbors
    typename Msh::Face::array_type::iterator fiter = faces.begin();
    while (fiter != faces.end()) {
      typename Msh::Cell::index_type nci;
      typename Msh::Face::index_type fi = *fiter;
      ++fiter;
      if (! mesh->get_neighbor(nci , ci, fi)) {
	// Faces with no neighbors are on the boundary, build a tri.
	typename Msh::Node::array_type nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	Point p[3]; // cache points off
	typename Msh::Node::array_type::iterator niter = nodes.begin();

	for (int i=0; i<3; i++) {
	  node_iter = vertex_map_.find(*niter);
	  mesh->get_point(p[i], *niter);
	  if (node_iter == vertex_map_.end()) {
	    node_idx[i] = tmesh->add_point(p[i]);
	    vertex_map_[*niter] = node_idx[i];
	  } else {
	    node_idx[i] = (*node_iter).second;
	  }
	  ++niter;
	}
	add_ordered_tri(p, node_idx, center, tmesh);

	while (niter != nodes.end()) {
	  node_idx[1] = node_idx[2];
	  p[1] = p[2];
	  node_iter = vertex_map_.find(*niter);
	  mesh->get_point(p[2], *niter);
	  if (node_iter == vertex_map_.end()) {
	    node_idx[2] = tmesh->add_point(p[2]);
	    vertex_map_[*niter] = node_idx[2];
	  } else {
	    node_idx[2] = (*node_iter).second;
	  }
	  ++niter;
	  add_ordered_tri(p, node_idx, center, tmesh);
	} 
      }
    }
  }
  TriSurf<double> *ts = scinew TriSurf<double>(tmesh, Field::NODE);

  if (tri_fh_) delete tri_fh_;
  tri_fh_ = scinew FieldHandle(ts);
}



void 
FieldBoundary::execute()
{
  FieldHandle input;
  if (!infield_->get(input)) return;
  if (!input.get_rep()) {
    error("FieldBoundary Error: No input data.");
    return;
  } else if (infield_gen_ != input->generation) {
    infield_gen_ = input->generation;
    MeshBaseHandle mesh = input->mesh();
    mesh->finish_mesh();
    dispatch_mesh1(input->mesh(), boundary);
  }
  osurf_->send(*tri_fh_);
}



} // End namespace SCIRun


