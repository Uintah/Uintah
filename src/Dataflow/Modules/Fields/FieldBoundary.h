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


//    File   : FieldBoundary.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(FieldBoundary_h)
#define FieldBoundary_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/SparseRowMatrix.h>
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
		       MatrixHandle &intrp,
		       int basis_order) = 0;

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
		       MatrixHandle &interp,
		       int basis_order);

};


template <class Msh>
void 
FieldBoundaryAlgoTriT<Msh>::execute(const MeshHandle mesh_untyped,
				    FieldHandle &boundary_fh,
				    MatrixHandle &interp,
				    int basis_order)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, TriSurfMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, TriSurfMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;
  vector<unsigned int> face_map;

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
	    face_map.push_back(ci);
	  }
	  ++niter;
	}
      }
    }
  }

  if (basis_order == 0)
  {
    TriSurfField<double> *ts = scinew TriSurfField<double>(tmesh, 0);
    boundary_fh = ts;

    typename Msh::Elem::size_type elemsize;
    mesh->size(elemsize);
    const int nrows = face_map.size();
    const int ncols = elemsize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < face_map.size(); i++)
    {
      cc[i] = face_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (basis_order == 1)
  {
    TriSurfField<double> *ts = scinew TriSurfField<double>(tmesh, 1);
    boundary_fh = ts;

    typename Msh::Node::size_type nodesize;
    mesh->size(nodesize);
    const int nrows = reverse_map.size();
    const int ncols = nodesize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < reverse_map.size(); i++)
    {
      cc[i] = reverse_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else
  {
    TriSurfField<double> *ts = scinew TriSurfField<double>(tmesh, -1);
    boundary_fh = ts;

    interp = 0;
  }
}



template <class Msh>
class FieldBoundaryAlgoQuadT : public FieldBoundaryAlgoAux
{
public:

  //! virtual interface. 
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &boundary,
		       MatrixHandle &interp,
		       int basis_order);

};



template <class Msh>
void 
FieldBoundaryAlgoQuadT<Msh>::execute(const MeshHandle mesh_untyped,
				     FieldHandle &boundary_fh,
				     MatrixHandle &interp,
				     int basis_order)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, QuadSurfMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, QuadSurfMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;
  vector<unsigned int> face_map;

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
	face_map.push_back(ci);
      }
    }
  }

  if (basis_order == 0)
  {
    QuadSurfField<double> *ts =
      scinew QuadSurfField<double>(tmesh, 0);
    boundary_fh = ts;
    
    typename Msh::Elem::size_type nodesize;
    mesh->size(nodesize);
    const int nrows = face_map.size();
    const int ncols = nodesize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < face_map.size(); i++)
    {
      cc[i] = face_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (basis_order == 1)
  {
    QuadSurfField<double> *ts =
      scinew QuadSurfField<double>(tmesh, 1);
    boundary_fh = ts;
    
    typename Msh::Node::size_type nodesize;
    mesh->size(nodesize);
    const int nrows = reverse_map.size();
    const int ncols = nodesize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < reverse_map.size(); i++)
    {
      cc[i] = reverse_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else
  {
    QuadSurfField<double> *ts =
      scinew QuadSurfField<double>(tmesh, -1);
    boundary_fh = ts;

    interp = 0;
  }
}



template <class Msh>
class FieldBoundaryAlgoCurveT : public FieldBoundaryAlgoAux
{
public:

  //! virtual interface. 
  virtual void execute(const MeshHandle mesh,
		       FieldHandle &boundary,
		       MatrixHandle &interp,
		       int basis_order);

};


template <class Msh>
void 
FieldBoundaryAlgoCurveT<Msh>::execute(const MeshHandle mesh_untyped,
				      FieldHandle &boundary_fh,
				      MatrixHandle &interp,
				      int basis_order)
{
  Msh *mesh = dynamic_cast<Msh *>(mesh_untyped.get_rep());
  map<typename Msh::Node::index_type, CurveMesh::Node::index_type> vertex_map;
  typename map<typename Msh::Node::index_type, CurveMesh::Node::index_type>::iterator node_iter;
  vector<typename Msh::Node::index_type> reverse_map;
  vector<unsigned int> edge_map;

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
	edge_map.push_back(ci);
      }
    }
  }

  if (basis_order == 0)
  {
    CurveField<double> *ts = scinew CurveField<double>(tmesh, 1);
    boundary_fh = ts;

    typename Msh::Elem::size_type nodesize;
    mesh->size(nodesize);
    const int nrows = edge_map.size();
    const int ncols = nodesize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < edge_map.size(); i++)
    {
      cc[i] = edge_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (basis_order == 1)
  {
    CurveField<double> *ts = scinew CurveField<double>(tmesh, 1);
    boundary_fh = ts;

    typename Msh::Node::size_type nodesize;
    mesh->size(nodesize);
    const int nrows = reverse_map.size();
    const int ncols = nodesize;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (unsigned int i = 0; i < reverse_map.size(); i++)
    {
      cc[i] = reverse_map[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

    interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else
  {
    CurveField<double> *ts = scinew CurveField<double>(tmesh, -1);
    boundary_fh = ts;

    interp = 0;
  }
}


//! DirectInterpBaseBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! DirectInterpBaseBase from the DynamicAlgoBase they will have a pointer to.
class FieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(ProgressReporter *m, const MeshHandle mesh,
		       FieldHandle &bndry, MatrixHandle &intrp,
		       int basis_order) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh);
};


template <class Msh>
class FieldBoundaryAlgoT : public FieldBoundaryAlgo
{
public:
  //! virtual interface. 
  virtual void execute(ProgressReporter *m, const MeshHandle mesh,
		       FieldHandle &boundary, MatrixHandle &interp,
		       int basis_order);
};


template <class Msh>
void 
FieldBoundaryAlgoT<Msh>::execute(ProgressReporter *mod, const MeshHandle mesh,
				 FieldHandle &boundary, MatrixHandle &interp,
				 int basis_order)
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
      algo->execute(mesh, boundary, interp, basis_order);
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
      algo->execute(mesh, boundary, interp, basis_order);
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
