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


/*
 *  PrismMC.h
 *
 *  \author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef PrismMC_h
#define PrismMC_h

#include <Core/Algorithms/Visualization/mcube2.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <sci_hash_map.h>

namespace SCIRun {

struct PrismMCBase {
  virtual ~PrismMCBase() {}
  static const string& get_h_file_path();
};
//! A Macrching Cube tesselator for a tetrahedral cell     

template<class Field>
class PrismMC : public PrismMCBase
{
public:
  typedef Field                                  field_type;
  typedef typename Field::mesh_type::Cell::index_type  cell_index_type;
  typedef typename Field::mesh_type::Node::index_type  node_index_type;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;

private:
  LockingHandle<Field> field_;
  mesh_handle_type mesh_;
  GeomFastTriangles *triangles_;
  bool build_field_;
  bool build_geom_;
  TriSurfMeshHandle trisurf_;
  int nnodes_;

  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
    double dfirst;
  };

#ifdef HAVE_HASH_MAP
  struct edgepairequal
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b) const
    {
      return a.first == b.first && a.second == b.second;
    }
  };

  struct edgepairhash
  {
    unsigned int operator()(const edgepair_t &a) const
    {
      hash<unsigned int> h;
      return h(a.first ^ a.second);
    }
  };

  typedef hash_map<edgepair_t,
		   TriSurfMesh::Node::index_type,
		   edgepairhash,
		   edgepairequal> edge_hash_type;
#else
  struct edgepairless
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b) const
    {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    }
  };

  typedef map<edgepair_t,
	      TriSurfMesh::Node::index_type,
	      edgepairless> edge_hash_type;
#endif

  edge_hash_type   edge_map_;  // Unique edge cuts when surfacing node data
  vector<long int> node_map_;  // Unique nodes when surfacing cell data.

  TriSurfMesh::Node::index_type find_or_add_edgepoint(int n0, int n1,
						      double d0,
						      const Point &p);
  TriSurfMesh::Node::index_type find_or_add_nodepoint(node_index_type &);

  int n_;

public:
  PrismMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~PrismMC();
	
  void extract( cell_index_type, double );
  void extract_n( cell_index_type, double );
  void extract_c( cell_index_type, double );
  void reset( int, bool build_field, bool build_geom);
  GeomHandle get_geom() { return triangles_; }
  FieldHandle get_field(double val);
  MatrixHandle get_interpolant();
};
  

template<class Field>    
PrismMC<Field>::~PrismMC()
{
}
    

template<class Field>
void PrismMC<Field>::reset( int n, bool build_field, bool build_geom )
{
  n_ = 0;

  build_field_ = build_field;
  build_geom_ = build_geom;

  edge_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;
  if (field_->basis_order() == 0)
  {
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
    node_map_ = vector<long int>(nsize, -1);
  }

  triangles_ = 0;
  if (build_geom_)
  {
    triangles_ = scinew GeomFastTriangles;
  }

  trisurf_ = 0;
  if (build_field_)
  {
    trisurf_ = scinew TriSurfMesh; 
  }
}


template<class Field>
TriSurfMesh::Node::index_type
PrismMC<Field>::find_or_add_edgepoint(int u0, int u1, double d0,
				      const Point &p) 
{
  if (d0 <= 0.0) { u1 = -1; }
  if (d0 >= 1.0) { u0 = -1; }
  edgepair_t np;
  if (u0 < u1)  { np.first = u0; np.second = u1; np.dfirst = d0; }
  else { np.first = u1; np.second = u0; np.dfirst = 1.0 - d0; }
  const typename edge_hash_type::iterator loc = edge_map_.find(np);
  if (loc == edge_map_.end())
  {
    const TriSurfMesh::Node::index_type nodeindex = trisurf_->add_point(p);
    edge_map_[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return (*loc).second;
  }
}


template<class Field>
TriSurfMesh::Node::index_type
PrismMC<Field>::find_or_add_nodepoint(node_index_type &tet_node_idx) {
  TriSurfMesh::Node::index_type surf_node_idx;
  long int i = node_map_[(long int)(tet_node_idx)];
  if (i != -1) surf_node_idx = (TriSurfMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, tet_node_idx);
    surf_node_idx = trisurf_->add_point(p);
    node_map_[(long int)tet_node_idx] = (long int)surf_node_idx;
  }
  return surf_node_idx;
}

template<class Field>
void PrismMC<Field>::extract( cell_index_type cell, double v )
{
  if (field_->basis_order() == 1)
    extract_n(cell, v);
  else
    extract_c(cell, v);
}

template<class Field>
void
PrismMC<Field>::extract_c( cell_index_type cell, double iso )
{
  value_type selfvalue, nbrvalue;
  if (!field_->value( selfvalue, cell )) return;
  typename mesh_type::Face::array_type faces;
  mesh_->get_faces(faces, cell);

  cell_index_type nbr;
  Point p[4];
  typename mesh_type::Node::array_type nodes;
  TriSurfMesh::Node::index_type vertices[3];
  unsigned int i, j;
  for (i = 0; i < faces.size(); i++)
  {
    if (mesh_->get_neighbor(nbr, cell, faces[i]) &&
	field_->value(nbrvalue, nbr) &&
	(selfvalue > nbrvalue) &&
	((selfvalue-iso) * (nbrvalue-iso) < 0)) {
      mesh_->get_nodes(nodes, faces[i]);

      for (j=0; j < nodes.size(); j++) { mesh_->get_center(p[j], nodes[j]); }

      if (build_geom_)
      {
	triangles_->add(p[0], p[1], p[2]);
      
	if( nodes.size() == 4 )
	  triangles_->add(p[0], p[2], p[3]);
      }

      if (build_field_)
      {
	for (j=0; j <  nodes.size(); j ++)
	  vertices[j] = find_or_add_nodepoint(nodes[j]);

	trisurf_->add_triangle(vertices[0], vertices[1], vertices[2]);
	
	if( nodes.size() == 4 )
	  trisurf_->add_triangle(vertices[0], vertices[2], vertices[3]);
      }
    }
  }
}


template<class Field>
void
PrismMC<Field>::extract_n( cell_index_type cell, double iso )
{
  typename mesh_type::Node::array_type node;
  Point p[8];
  value_type value[8];
  int code = 0;

  mesh_->get_nodes( node, cell );

  typename mesh_type::Node::array_type nodes;
  mesh_->get_nodes( nodes, cell );

  // Fake having 8 nodes and use the HexMC algorithm.
  node.resize(8);
  
  node[0] = nodes[0];
  node[1] = nodes[1];
  node[2] = nodes[2];
  node[3] = nodes[0];
  node[4] = nodes[3];
  node[5] = nodes[4];
  node[6] = nodes[5];
  node[7] = nodes[3];

  for (int i=node.size()-1; i>=0; i--) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code = code*2+(value[i] < iso );
  }

  if ( code == 0 || code == 255 )
    return;

  TRIANGLE_CASES *tcase=&triCases[code];
  int *vertex = tcase->edges;
  
  Point q[12];
  TriSurfMesh::Node::index_type surf_node[12];

  // interpolate and project vertices
  int v = 0;
  vector<bool> visited(12, false);
  while (vertex[v] != -1) {
    int i = vertex[v++];
    if (visited[i]) continue;
    visited[i]=true;
    const int v1 = edge_tab[i][0];
    const int v2 = edge_tab[i][1];
    const double d = (value[v1]-iso)/double(value[v1]-value[v2]);
    q[i] = Interpolate(p[v1], p[v2], d);
		       
    if (build_field_)
    {
      surf_node[i] = find_or_add_edgepoint(node[v1], node[v2], d, q[i]);
    }
  }    
  
  v = 0;
  while(vertex[v] != -1) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];

    // Degenerate triangle can be built since ponit were duplicated
    // above in order to make a hexvol for MC. 
    if( v0 != v1 && v0 != v2 && v1 != v2 ) {
      if (build_geom_)
      {
	triangles_->add(q[v0], q[v1], q[v2]);
      }
      if (build_field_)
      {
        if (surf_node[v0] != surf_node[v1] &&
            surf_node[v1] != surf_node[v2] &&
            surf_node[v2] != surf_node[v0])
        {
          trisurf_->add_triangle(surf_node[v0], surf_node[v1], surf_node[v2]);
        }
      }
    }
  }
}

template<class Field>
FieldHandle
PrismMC<Field>::get_field(double value)
{
  TriSurfField<double> *fld = 0;
  if (trisurf_.get_rep())
  {
    fld = scinew TriSurfField<double>(trisurf_, 1);
    vector<double>::iterator iter = fld->fdata().begin();
    while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
  }
  return fld;
}


template<class Field>
MatrixHandle
PrismMC<Field>::get_interpolant()
{
  if (field_->basis_order() == 1)
  {
    const int nrows = edge_map_.size();
    const int ncols = nnodes_;
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows*2];
    double *dd = scinew double[nrows*2];

    typename edge_hash_type::iterator eiter = edge_map_.begin();
    while (eiter != edge_map_.end())
    {
      const int ei = (*eiter).second;

      cc[ei * 2 + 0] = (*eiter).first.first;
      cc[ei * 2 + 1] = (*eiter).first.second;
      dd[ei * 2 + 0] = 1.0 - (*eiter).first.dfirst;
      dd[ei * 2 + 1] = (*eiter).first.dfirst;
      
      ++eiter;
    }

    int nnz = 0;
    int i;
    for (i = 0; i < nrows; i++)
    {
      rr[i] = nnz;
      if (cc[i * 2 + 0] > 0)
      {
        cc[nnz] = cc[i * 2 + 0];
        dd[nnz] = dd[i * 2 + 0];
        nnz++;
      }
      if (cc[i * 2 + 1] > 0)
      {
        cc[nnz] = cc[i * 2 + 1];
        dd[nnz] = dd[i * 2 + 1];
        nnz++;
      }
    }
    rr[i] = nnz;

    return scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, dd);
  }
  else
  {
    return 0;
  }
}

     
} // End namespace SCIRun

#endif // PrismMC_h
