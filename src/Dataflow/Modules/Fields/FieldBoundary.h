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
#include <Core/Util/ProgressReporter.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/CurveField.h>
#include <algorithm>

namespace SCIRun {

//! This supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! FieldBoundaryAlgoAux from the DynamicAlgoBase they will have a pointer to.
class FieldBoundaryAlgoAux : public DynamicAlgoBase
{
public:
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &bndry,
		       FieldHandle &intrp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh,
					    const string &algo);

protected:
  static bool determine_tri_order(const Point &p0, const Point &p1,
				  const Point &p2, const Point &inside);
};


template <class Msh>
class FieldBoundaryAlgoTriT : public FieldBoundaryAlgoAux
{
public:

  //! virtual interface. 
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &boundary,
		       FieldHandle &interp);

};


template <class Msh>
void 
FieldBoundaryAlgoTriT<Msh>::execute(const MeshHandle mesh_untyped,
				    FieldHandle &boundary_fh,
				    FieldHandle &interp_fh)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, typename TriSurfMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;

  TriSurfMeshHandle tmesh = scinew TriSurfMesh;

  mesh->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  // Walk all the cells in the mesh.
  Point center;
  typename Msh::Cell::iterator citer; mesh->begin(citer);
  typename Msh::Cell::iterator citere; mesh->end(citere);

  while (citer != citere)
  {
    typename Msh::Cell::index_type ci = *citer;
    ++citer;
  
    mesh->get_center(center, ci);

    // Get all the faces in the cell.
    typename Msh::Face::array_type faces;
    mesh->get_faces(faces, ci);

    // Check each face for neighbors.
    typename Msh::Face::array_type::iterator fiter = faces.begin();

    while (fiter != faces.end())
    {
      typename Msh::Cell::index_type nci;
      typename Msh::Face::index_type fi = *fiter;
      ++fiter;

      if (! mesh->get_neighbor(nci , ci, fi))
      {
	// Faces with no neighbors are on the boundary, build a tri.
	typename Msh::Node::array_type nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	vector<Point> p(nodes.size()); // cache points off
	TriSurfMesh::Node::array_type node_idx(nodes.size());

	typename Msh::Node::array_type::iterator niter = nodes.begin();

	for (unsigned int i=0; i<nodes.size(); i++)
	{
	  node_iter = vertex_map.find(*niter);
	  mesh->get_point(p[i], *niter);
	  if (node_iter == vertex_map.end())
	  {
	    node_idx[i] = tmesh->add_point(p[i]);
	    vertex_map[*niter] = node_idx[i];
	    reverse_map.push_back(*niter);
	  }
	  else
	  {
	    node_idx[i] = (*node_iter).second;
	  }
	  if (i >= 2)
	  {
	    if (determine_tri_order(p[0], p[i-1], p[i], center))
	    {
	      tmesh->add_triangle(node_idx[0], node_idx[i-1], node_idx[i]);
	    }
	    else
	    {
	      tmesh->add_triangle(node_idx[0], node_idx[i], node_idx[i-1]);
	    }
	  }
	  ++niter;
	}
      }
    }
  }
  TriSurfField<double> *ts = scinew TriSurfField<double>(tmesh, Field::NODE);
  TriSurfField<vector<pair<typename Msh::Node::index_type, double> > >* interp =
    scinew TriSurfField<vector<pair<typename Msh::Node::index_type, double> > >(tmesh, Field::NODE);
  for (unsigned int i=0; i<reverse_map.size(); i++)
    interp->fdata()[i].push_back(pair<typename Msh::Node::index_type, double>(reverse_map[i], 1.0));

  boundary_fh = ts;
  interp_fh = interp;
}



template <class Msh>
class FieldBoundaryAlgoQuadT : public FieldBoundaryAlgoAux
{
public:

  //! virtual interface. 
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &boundary,
		       FieldHandle &interp);

};



template <class Msh>
void 
FieldBoundaryAlgoQuadT<Msh>::execute(const MeshHandle mesh_untyped,
				     FieldHandle &boundary_fh,
				     FieldHandle &interp_fh)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, typename QuadSurfMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, typename QuadSurfMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;

  QuadSurfMeshHandle tmesh = scinew QuadSurfMesh;

  mesh->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  // Walk all the cells in the mesh.
  Point center;
  typename Msh::Cell::iterator citer; mesh->begin(citer);
  typename Msh::Cell::iterator citere; mesh->end(citere);

  while (citer != citere)
  {
    typename Msh::Cell::index_type ci = *citer;
    ++citer;
  
    mesh->get_center(center, ci);

    // Get all the faces in the cell.
    typename Msh::Face::array_type faces;
    mesh->get_faces(faces, ci);

    // Check each face for neighbors.
    typename Msh::Face::array_type::iterator fiter = faces.begin();

    while (fiter != faces.end())
    {
      typename Msh::Cell::index_type nci;
      typename Msh::Face::index_type fi = *fiter;
      ++fiter;

      if (! mesh->get_neighbor(nci , ci, fi))
      {
	// Faces with no neighbors are on the boundary, build a tri.
	typename Msh::Node::array_type nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	vector<Point> p(nodes.size()); // cache points off
	QuadSurfMesh::Node::array_type node_idx(nodes.size());

	typename Msh::Node::array_type::iterator niter = nodes.begin();

	for (unsigned int i=0; i<nodes.size(); i++)
	{
	  node_iter = vertex_map.find(*niter);
	  mesh->get_point(p[i], *niter);
	  if (node_iter == vertex_map.end())
	  {
	    node_idx[i] = tmesh->add_point(p[i]);
	    vertex_map[*niter] = node_idx[i];
	    reverse_map.push_back(*niter);
	  }
	  else
	  {
	    node_idx[i] = (*node_iter).second;
	  }
	  ++niter;
	}

	if (determine_tri_order(p[0], p[1], p[2], center))
	{
	  tmesh->add_elem(node_idx);
	}
	else
	{
	  std::reverse(node_idx.begin(), node_idx.end());
	  tmesh->add_elem(node_idx);
	}
      }
    }
  }
  QuadSurfField<double> *ts = scinew QuadSurfField<double>(tmesh, Field::NODE);
  QuadSurfField<vector<pair<typename Msh::Node::index_type, double> > >* interp =
    scinew QuadSurfField<vector<pair<typename Msh::Node::index_type, double> > >(tmesh, Field::NODE);
  for (unsigned int i=0; i<reverse_map.size(); i++)
    interp->fdata()[i].push_back(pair<typename Msh::Node::index_type, double>(reverse_map[i], 1.0));

  boundary_fh = ts;
  interp_fh = interp;
}



template <class Msh>
class FieldBoundaryAlgoCurveT : public FieldBoundaryAlgoAux
{
public:

  //! virtual interface. 
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &boundary,
		       FieldHandle &interp);

};


template <class Msh>
void 
FieldBoundaryAlgoCurveT<Msh>::execute(const MeshHandle mesh_untyped,
				      FieldHandle &boundary_fh,
				      FieldHandle &interp_fh)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, typename CurveMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, typename CurveMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;

  CurveMeshHandle tmesh = scinew CurveMesh;

  mesh->synchronize(Mesh::EDGE_NEIGHBORS_E | Mesh::EDGES_E);

  // Walk all the faces in the mesh.
  Point center;
  typename Msh::Face::iterator citer; mesh->begin(citer);
  typename Msh::Face::iterator citere; mesh->end(citere);

  while (citer != citere)
  {
    typename Msh::Face::index_type ci = *citer;
    ++citer;
  
    mesh->get_center(center, ci);

    // Get all the edges in the face.
    typename Msh::Edge::array_type edges;
    mesh->get_edges(edges, ci);

    // Check each edge for neighbors.
    typename Msh::Edge::array_type::iterator fiter = edges.begin();

    while (fiter != edges.end())
    {
      typename Msh::Face::index_type nci;
      typename Msh::Edge::index_type fi = *fiter;
      ++fiter;

      if (! mesh->get_neighbor(nci , ci, fi))
      {
	// Edges with no neighbors are on the boundary, build a tri.
	typename Msh::Node::array_type nodes;
	mesh->get_nodes(nodes, fi);
	// Creating triangles, so fan if more than 3 nodes.
	vector<Point> p(nodes.size()); // cache points off
	CurveMesh::Node::array_type node_idx(nodes.size());

	typename Msh::Node::array_type::iterator niter = nodes.begin();

	for (unsigned int i=0; i<nodes.size(); i++)
	{
	  node_iter = vertex_map.find(*niter);
	  mesh->get_point(p[i], *niter);
	  if (node_iter == vertex_map.end())
	  {
	    node_idx[i] = tmesh->add_point(p[i]);
	    vertex_map[*niter] = node_idx[i];
	    reverse_map.push_back(*niter);
	  }
	  else
	  {
	    node_idx[i] = (*node_iter).second;
	  }
	  ++niter;
	}

	tmesh->add_elem(node_idx);
      }
    }
  }
  CurveField<double> *ts = scinew CurveField<double>(tmesh, Field::NODE);
  CurveField<vector<pair<typename Msh::Node::index_type, double> > >* interp =
    scinew CurveField<vector<pair<typename Msh::Node::index_type, double> > >(tmesh, Field::NODE);
  for (unsigned int i=0; i<reverse_map.size(); i++)
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
  virtual void execute(ProgressReporter *m, const MeshHandle mesh,
		       FieldHandle &bndry, FieldHandle &intrp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh);
};


template <class Msh>
class FieldBoundaryAlgoT : public FieldBoundaryAlgo
{
public:
  //! virtual interface. 
  virtual void execute(ProgressReporter *m, const MeshHandle mesh,
		       FieldHandle &boundary, FieldHandle &interp);
};


template <class Msh>
void 
FieldBoundaryAlgoT<Msh>::execute(ProgressReporter *mod, const MeshHandle mesh,
				 FieldHandle &boundary, FieldHandle &interp)
{
  if (get_type_description((typename Msh::Elem *)0)->get_name() ==
      get_type_description((typename Msh::Cell *)0)->get_name())
  {
    string algoname = "Tri";

    mesh->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);
    Msh *typedmesh = dynamic_cast<Msh *>(mesh.get_rep());
    typename Msh::Face::iterator face_iter, face_iter_end;
    typedmesh->begin(face_iter);
    typedmesh->end(face_iter_end);
    if (face_iter != face_iter_end)
    {
      // Note that we only test the face size of the first element
      // in the field.  This assumes that all of the elements have
      // a consistent size of 4, and may break on irregular polyhedral
      // meshes (unsupported at this time).
      typename Msh::Node::array_type nodes;
      typedmesh->get_nodes(nodes, *face_iter);
      if (nodes.size() == 4)
      {
	algoname = "Quad";
      }
    }

    const TypeDescription *mtd = get_type_description((Msh *)0);
    CompileInfoHandle ci =
      FieldBoundaryAlgoAux::get_compile_info(mtd, algoname);
    Handle<FieldBoundaryAlgoAux> algo;
    if (DynamicCompilation::compile(ci, algo, true, mod))
    {
      algo->execute(mesh, boundary, interp);
    }
  }
  else if (get_type_description((typename Msh::Elem *)0)->get_name() ==
	   get_type_description((typename Msh::Face *)0)->get_name())
  {
    const TypeDescription *mtd = get_type_description((Msh *)0);
    CompileInfoHandle ci =
      FieldBoundaryAlgoAux::get_compile_info(mtd, "Curve");
    Handle<FieldBoundaryAlgoAux> algo;
    if (DynamicCompilation::compile(ci, algo, true, mod))
    {
      algo->execute(mesh, boundary, interp);
    }
    else
    {
      mod->error("Fields of '" + mtd->get_name() + 
		 "' type are not currently supported.");
    }
  }
  else
  {
    mod->error("Boundary module only works on volumes and surfaces.");
  }

  // Set the source range for the interpolation field.
  if (interp.get_rep())
  {
    Msh *typedmesh = dynamic_cast<Msh *>(mesh.get_rep());
    if (typedmesh)
    {
      typename Msh::Node::size_type msize;
      typedmesh->size(msize);
      unsigned int range = (unsigned int)msize;
      interp->set_property("interp-source-range", range, false);
    }
  }
}


} // end namespace SCIRun

#endif // FieldBoundary_h
