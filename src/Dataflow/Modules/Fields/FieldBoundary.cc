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
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
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
  template <class Field>
  void                                     
  find_boundary(Field *field, GeomTrianglesP *tris, 
		TriSurf<typename Field::value_type> *osurf);

  //! Input should be a volume field.
  FieldIPort*              inport_;
  //! Scene graph output.
  GeometryOPort*           outport_;
  //! TriSurf field output.
  FieldOPort*              osurf_;
};

extern "C" Module* make_FieldBoundary(const clString& id)
{
  return scinew FieldBoundary(id);
}

FieldBoundary::FieldBoundary(const clString& id)
  : Module("FieldBoundary", id, Filter)
{
  // Create the input port
  inport_ = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(inport_);
  outport_ = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(outport_);
  osurf_ = scinew FieldOPort(this, "TriSurf", FieldIPort::Atomic);
  add_oport(osurf_);
}

FieldBoundary::~FieldBoundary()
{
}



void 
FieldBoundary::execute()
{
  FieldHandle input;

  if (!inport_->get(input)) return;
  if (!input.get_rep()) {
    cerr << "Error: empty field" << endl;
    return;
  }
    
  GeomTrianglesP *tris= scinew GeomTrianglesP;  

  bool error = false;
  string msg;
  string name = input->get_type_name(0);
  if (name == "TetVol") {
    if (input->get_type_name(1) == "double") {
      TetVol<double> *tv = 0;
      TriSurf<double> *ts = scinew TriSurf<double>;
      tv = dynamic_cast<TetVol<double>*>(input.get_rep());
      if (tv) { 
	// need faces and edges.
	tv->finish_mesh();
	find_boundary(tv, tris, ts);
	FieldHandle ts_handle(ts);
	osurf_->send(ts_handle);
      }
      else { error = true; msg = "Not a valid TetVol."; }
    } else {
      error = true; msg ="TetVol of unknown type.";
    }
  } else if (error) {
    cerr << "FieldBoundary Error: " << msg << endl;
    return;
  }

  outport_->delAll();
  outport_->addObj(tris,"Boundary Triangles");

}

template <class Field>
void                                                         
FieldBoundary::find_boundary(Field *f, GeomTrianglesP *tris, 
			     TriSurf<typename Field::value_type> *osurf)
{
  typedef typename Field::mesh_type Mesh;
  typename Field::mesh_handle_type mesh = f->get_typed_mesh();
  // Walk all the cells in the mesh.
  typename Mesh::cell_iterator citer = mesh->cell_begin();
  while (citer != mesh->cell_end()) {
    typename Mesh::cell_index ci = *citer;
    ++citer;
    // Get all the faces in the cell.
    typename Mesh::face_array faces;
    mesh->get_faces(faces, ci);
    // Check each face for neighbors
    typename Mesh::face_array::iterator fiter = faces.begin();
    while (fiter != faces.end()) {
      typename Mesh::cell_index nci;
      typename Mesh::face_index fi = *fiter;
      if (! mesh->get_neighbor(nci , ci, fi)) {
	// Faces with no neighbors are on the boundary, build a tri.
	typename Mesh::node_array nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	Point p1, p2, p3;
	typename Mesh::node_array::iterator niter = nodes.begin();
	mesh->get_point(p1, *niter);
	++niter;
	mesh->get_point(p2, *niter);
	++niter;
	mesh->get_point(p3, *niter);
	++niter;

	tris->add(p1, p2, p3);
	// FIX_ME add to TriSurf
	//osurf->add_tri(p1,p2,p3);
	while (niter != nodes.end()) {
	  p2 = p3;
	  mesh->get_point(p3, *niter);
	  ++niter;
	  
	  tris->add(p1, p2, p3);
	  // FIX_ME add to TriSurf
	  //osurf->add_tri(p1,p2,p3);
	}
      }
    }
  }
  // FIX_ME remove duplicates and build neighbors
  // osurf->resolve_surf();
}

} // End namespace SCIRun


