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

//    File   : FieldBoundary.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(FieldBoundary_h)
#define FieldBoundary_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {

//! This supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! FieldBoundaryAlgoAux from the DynamicAlgoBase they will have a pointer to.
class FieldBoundaryAlgoAux : public DynamicAlgoBase
{
public:
  virtual void execute(const MeshHandle mesh, FieldHandle &bndry, FieldHandle &intrp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *mesh);
  static bool determine_tri_order(const Point p[3], const Point &inside);
};


template <class Msh>
class FieldBoundaryAlgoAuxT : public FieldBoundaryAlgoAux
{
public:
  //! virtual interface. 
  virtual void execute(const MeshHandle mesh, FieldHandle &boundary, FieldHandle &interp);

private:
};



template <class Msh>
void 
FieldBoundaryAlgoAuxT<Msh>::execute(const MeshHandle mesh_untyped,
				    FieldHandle &boundary_fh,
				    FieldHandle &interp_fh)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;

  TriSurfMesh::Node::index_type node_idx[3];

  TriSurfMeshHandle tmesh = scinew TriSurfMesh;

  mesh->synchronize(Mesh::FACES_E | Mesh::FACE_NEIGHBORS_E);

  // Walk all the cells in the mesh.
  Point center;
  typename Msh::Cell::iterator citer; mesh->begin(citer);
  typename Msh::Cell::iterator citere; mesh->end(citere);
  while (citer != citere) {
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
	  node_iter = vertex_map.find(*niter);
	  mesh->get_point(p[i], *niter);
	  if (node_iter == vertex_map.end()) {
	    node_idx[i] = tmesh->add_point(p[i]);
	    vertex_map[*niter] = node_idx[i];
	    reverse_map.push_back(*niter);
	  } else {
	    node_idx[i] = (*node_iter).second;
	  }
	  ++niter;
	}
	if (determine_tri_order(p, center)) {
	  tmesh->add_triangle(node_idx[0], node_idx[1], node_idx[2]);
	} else {
	  tmesh->add_triangle(node_idx[2], node_idx[1], node_idx[0]);
	}

	while (niter != nodes.end()) {
	  node_idx[1] = node_idx[2];
	  p[1] = p[2];
	  node_iter = vertex_map.find(*niter);
	  mesh->get_point(p[2], *niter);
	  if (node_iter == vertex_map.end()) {
	    node_idx[2] = tmesh->add_point(p[2]);
	    vertex_map[*niter] = node_idx[2];
	    reverse_map.push_back(*niter);
	  } else {
	    node_idx[2] = (*node_iter).second;
	  }
	  ++niter;
	  if (determine_tri_order(p, center)) {
	    tmesh->add_triangle(node_idx[0], node_idx[1], node_idx[2]);
	  } else {
	    tmesh->add_triangle(node_idx[2], node_idx[1], node_idx[0]);
	  }
	} 
      }
    }
  }
  TriSurfField<double> *ts = scinew TriSurfField<double>(tmesh, Field::NODE);
  TriSurfField<vector<pair<typename Msh::Node::index_type, double> > >* interp =
    scinew TriSurfField<vector<pair<typename Msh::Node::index_type, double> > >(tmesh, Field::NODE);
  for (int i=0; i<reverse_map.size(); i++)
    interp->fdata()[i].push_back(pair<typename Msh::Node::index_type, double>(reverse_map[i], 1.0));

  boundary_fh = ts;
  interp_fh = interp;
}

//! DirectInterpBaseBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! DirectInterpBaseBase from the DynamicAlgoBase they will have a pointer to.
class FieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(Module *m, const MeshHandle mesh,
		       FieldHandle &bndry, FieldHandle &intrp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *mesh);
};


template <class Msh>
class FieldBoundaryAlgoT : public FieldBoundaryAlgo
{
public:
  //! virtual interface. 
  virtual void execute(Module *m, const MeshHandle mesh,
		       FieldHandle &boundary, FieldHandle &interp);
};


template <class Msh>
void 
FieldBoundaryAlgoT<Msh>::execute(Module *mod, const MeshHandle mesh,
				 FieldHandle &boundary, FieldHandle &interp)
{
  if (get_type_description((typename Msh::Elem *)0)->get_name() !=
      get_type_description((typename Msh::Cell *)0)->get_name())
  {
    mod->error("Boundary module only works on volumes.");
    return;
  }
  else
  {
    const TypeDescription *mtd = get_type_description((Msh *)0);
    CompileInfo *ci = FieldBoundaryAlgoAux::get_compile_info(mtd);
    Handle<FieldBoundaryAlgoAux> algo;
    if (!mod->module_dynamic_compile(*ci, algo))
    {
      return;
    }
    algo->execute(mesh, boundary, interp);
  }
}


} // end namespace SCIRun

#endif // FieldBoundary_h
