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
 *   Peter-Pike Sloan and David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1998
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
using std::cerr;

namespace SCIRun {

//! Module to build a surface field from a volume field.
class FieldBoundary : public Module {
public:
  FieldBoundary(const clString& id);
  virtual ~FieldBoundary();
  virtual void execute();

private:
  
  //! Iterates over mesh, and build TriSurf at the boundary
  template <class Msh> void boundary(const Msh *mesh);

  void add_ordered_tri(const Point &p1, const Point &p2, 
		       const Point &p3, const Point &inside,
		       TriSurfMeshHandle tmesh);

  void add_face(const Point &p0, const Point &p1, const Point &p2, 
		MaterialHandle m0, MaterialHandle m1, 
		MaterialHandle m2, GeomTriangles *g);

  //! Input should be a volume field.
  FieldIPort*              infield_;
  int                      infield_gen_;
  //! Scene graph output.
  GeometryOPort*           viewer_;
  //! TriSurf field output.
  FieldOPort*              osurf_;
  
  //! Handle on the generated surface.
  FieldHandle             *tri_fh_;
  GeomTriangles           *geom_tris_;
  int                      tris_id_;
};

extern "C" Module* make_FieldBoundary(const clString& id)
{
  return scinew FieldBoundary(id);
}

FieldBoundary::FieldBoundary(const clString& id) : 
  Module("FieldBoundary", id, Filter, "Fields", "SCIRun"),
  infield_gen_(-1),
  tri_fh_(scinew FieldHandle(scinew TriSurf<double>)),
  geom_tris_(0),
  tris_id_(0)
{
  // Create the input port
  infield_ = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(infield_);
  viewer_ = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(viewer_);
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

void 
FieldBoundary::add_ordered_tri(const Point &p1, const Point &p2, 
			       const Point &p3, const Point &inside,
			       TriSurfMeshHandle tmesh)
{
  Vector v1 = p2 - p1;
  Vector v2 = p3 - p1;
  Vector norm = Cross(v1, v2);

  Vector tmp = inside - p1;
  double val = Dot(norm, tmp);
  if (val > 0) {
    tmesh->add_triangle_unconnected(p1, p2, p3);
  } else {
    tmesh->add_triangle_unconnected(p3, p2, p1);
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

  
  TriSurf<double> *ts = scinew TriSurf<double>;
  TriSurfMeshHandle tmesh = ts->get_typed_mesh();
  if (geom_tris_) delete geom_tris_;
  geom_tris_ = scinew GeomTriangles;  
  // Walk all the cells in the mesh.
  Point center;
  typename Msh::cell_iterator citer = mesh->cell_begin();
  while (citer != mesh->cell_end()) {
    typename Msh::cell_index ci = *citer;
    ++citer;
    mesh->get_center(center, ci);
    // Get all the faces in the cell.
    typename Msh::face_array faces;
    mesh->get_faces(faces, ci);
    // Check each face for neighbors
    typename Msh::face_array::iterator fiter = faces.begin();
    while (fiter != faces.end()) {
      typename Msh::cell_index nci;
      typename Msh::face_index fi = *fiter;
      ++fiter;
      if (! mesh->get_neighbor(nci , ci, fi)) {
	// Faces with no neighbors are on the boundary, build a tri.
	typename Msh::node_array nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	Point p1, p2, p3;
	typename Msh::node_array::iterator niter = nodes.begin();
	mesh->get_point(p1, *niter);
	++niter;
	mesh->get_point(p2, *niter);
	++niter;
	mesh->get_point(p3, *niter);
	++niter;

	geom_tris_->add(p1, p2, p3);
	add_ordered_tri(p1, p2, p3, center, tmesh);
	while (niter != nodes.end()) {
	  p2 = p3;
	  mesh->get_point(p3, *niter);
	  ++niter;
	  
	  geom_tris_->add(p1, p2, p3);
	  add_ordered_tri(p1, p2, p3, center, tmesh);
	}
      }
    }
  }
  //tmesh->connect();
  if (tris_id_) viewer_->delObj(tris_id_);
  tris_id_ = viewer_->addObj(geom_tris_, "Boundary Surface");
  viewer_->flushViews();
  
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


