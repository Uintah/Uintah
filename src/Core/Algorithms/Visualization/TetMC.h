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
 *  TetMC.h
 *
 *  \author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef TetMC_h
#define TetMC_h

#include <Core/Geom/GeomTriangles.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <sci_hash_map.h>

namespace SCIRun {

struct TetMCBase {
  virtual ~TetMCBase() {}
  static const string& get_h_file_path();
};
//! A Macrching Cube tesselator for a tetrahedral cell     

template<class Field>
class TetMC : public TetMCBase
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
  TriSurfMeshHandle trisurf_;
  int nnodes_;

  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
    double dfirst;

    inline bool operator<(const edgepair_t& e2) const {
      return first < e2.first || (first == e2.first && second < e2.second);
    }
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
#ifdef __ECC
    // These are particularly needed by ICC's hash stuff
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;
    
    // This is a less than function.
    bool operator()(const edgepair_t& ei1, const edgepair_t& ei2) const {
      return ei1 < ei2;
    }
#endif
    
  };

#ifndef __ECC
  typedef hash_map<edgepair_t,
		   TriSurfMesh::Node::index_type,
		   edgepairhash,
		   edgepairequal> edge_hash_type;
#else
  typedef hash_map<edgepair_t,
                   TriSurfMesh::Node::index_type,
                   edgepairhash> edge_hash_type;
#endif // __ECC
  
#else
  struct edgepairless
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b) const
    {
      return a < b;
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
  TriSurfMesh::Node::index_type find_or_add_nodepoint(node_index_type &n0);

  void extract_n( cell_index_type, double );
  void extract_c( cell_index_type, double );

public:
  TetMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()),
			  triangles_(0), trisurf_(0) {}
  virtual ~TetMC();
	
  void extract( cell_index_type, double );
  void reset( int, bool build_field, bool build_geom );
  GeomHandle get_geom() { return triangles_; }
  FieldHandle get_field(double val);
  MatrixHandle get_interpolant();
};
  

template<class Field>
TetMC<Field>::~TetMC()
{
}
    

template<class Field>
void TetMC<Field>::reset( int n, bool build_field, bool build_geom )
{
  edge_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;

  if (field_->basis_order() == 0)
  {
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
    if (build_field) { node_map_ = vector<long int>(nsize, -1); }
  }

  triangles_ = 0;
  if (build_geom)
  {
    triangles_ = scinew GeomFastTriangles;
  }
  
  trisurf_ = 0;
  if (build_field)
  {
    trisurf_ = scinew TriSurfMesh;
  }
}


template<class Field>
TriSurfMesh::Node::index_type
TetMC<Field>::find_or_add_edgepoint(int u0, int u1, double d0, const Point &p) 
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
TetMC<Field>::find_or_add_nodepoint(node_index_type &tet_node_idx) {
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
void TetMC<Field>::extract( cell_index_type cell, double v )
{
  if (field_->basis_order() == 1)
    extract_n(cell, v);
  else
    extract_c(cell, v);
}

template<class Field>
void TetMC<Field>::extract_c( cell_index_type cell, double iso )
{
  value_type selfvalue, nbrvalue;
  if (!field_->value( selfvalue, cell )) return;
  typename mesh_type::Face::array_type faces;
  mesh_->get_faces(faces, cell);

  cell_index_type nbr;
  Point p[3];
  typename mesh_type::Node::array_type nodes;
  TriSurfMesh::Node::index_type vertices[3];
  unsigned int i, j;
  for (i = 0; i < faces.size(); i++)
  {
    if (mesh_->get_neighbor(nbr, cell, faces[i]) &&
	field_->value(nbrvalue, nbr) &&
	(selfvalue > nbrvalue) &&
	((selfvalue-iso) * (nbrvalue-iso) < 0))
    {
      mesh_->get_nodes(nodes, faces[i]);
      for (j=0; j < 3; j++) { mesh_->get_center(p[j], nodes[j]); }

      if (triangles_)
      {
	triangles_->add(p[0], p[1], p[2]);
      }
      if (trisurf_.get_rep())
      {
	for (j=0; j < 3;j ++)
	{
	  vertices[j] = find_or_add_nodepoint(nodes[j]);
	}
	trisurf_->add_triangle(vertices[0], vertices[1], vertices[2]);
      }
    }
  }
}


template<class Field>
void TetMC<Field>::extract_n( cell_index_type cell, double v )
{
  static int num[16] = { 0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0 };
  static int order[16][4] = {
    {0, 0, 0, 0},   /* none - ignore */
    {3, 2, 0, 1},   /* 3 */
    {2, 1, 0, 3},   /* 2 */
    {2, 1, 0, 3},   /* 2, 3 */
    {1, 3, 0, 2},   /* 1 */
    {1, 0, 2, 3},   /* 1, 3 */
    {1, 3, 0, 2},   /* 1, 2 */
    {0, 2, 3, 1},   /* 1, 2, 3 */
    {0, 2, 1, 3},   /* 0 */
    {2, 3, 0, 1},   /* 0, 3 - reverse of 1, 2 */
    {3, 0, 2, 1},   /* 0, 2 - reverse of 1, 3 */
    {1, 0, 3, 2},   /* 0, 2, 3 - reverse of 1 */
    {3, 1, 0, 2},   /* 0, 1 - reverse of 2, 3 */
    {2, 3, 0, 1},   /* 0, 1, 3 - reverse of 2 */
    {3, 2, 1, 0},   /* 0, 1, 2 - reverse of 3 */
    {0, 0, 0, 0}    /* all - ignore */
  };
    
    
  typename mesh_type::Node::array_type node;
  Point p[4];
  value_type value[4];

  mesh_->get_nodes( node, cell );
  int i;
  for (i=0; i<4; i++)
    mesh_->get_point( p[i], node[i] );

// fix the node[i] ordering so tet is orientationally consistent
//  if (Dot(Cross(p[0]-p[1],p[0]-p[2]),p[0]-p[3])>0) {
//    typename mesh_type::Node::index_type nd=node[0];
//    node[0]=node[1];
//    node[1]=nd;
//  }

  int code = 0;
  for (int i=0; i<4; i++) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code = code*2+(value[i] > v );
  }

  //  if ( show_case != -1 && (code != show_case) ) return;
  switch ( num[code] ) {
  case 1: 
    {
      // make a single triangle
      int o = order[code][0];
      int i = order[code][1];
      int j = order[code][2];
      int k = order[code][3];

      const double v1 = (v-value[o])/double(value[i]-value[o]);
      const double v2 = (v-value[o])/double(value[j]-value[o]);
      const double v3 = (v-value[o])/double(value[k]-value[o]);
      const Point p1(Interpolate( p[o],p[i], v1));
      const Point p2(Interpolate( p[o],p[j], v2));
      const Point p3(Interpolate( p[o],p[k], v3));

      if (triangles_)
      {
	triangles_->add( p1, p2, p3 );
      }
      if (trisurf_.get_rep())
      {
	TriSurfMesh::Node::index_type i1, i2, i3;
	i1 = find_or_add_edgepoint(node[o], node[i], v1, p1);
	i2 = find_or_add_edgepoint(node[o], node[j], v2, p2);
	i3 = find_or_add_edgepoint(node[o], node[k], v3, p3);
        if (i1 != i2 && i2 != i3 && i3 != i1)
          trisurf_->add_triangle(i1, i2, i3);
      }
    }
    break;
  case 2: 
    {
      // make order triangles
      const int o = order[code][0];
      const int i = order[code][1];
      const int j = order[code][2];
      const int k = order[code][3];
      const double v1 = (v-value[o])/double(value[i]-value[o]);
      const double v2 = (v-value[o])/double(value[j]-value[o]);
      const double v3 = (v-value[k])/double(value[j]-value[k]);
      const double v4 = (v-value[k])/double(value[i]-value[k]);
      const Point p1(Interpolate( p[o],p[i], v1));
      const Point p2(Interpolate( p[o],p[j], v2));
      const Point p3(Interpolate( p[k],p[j], v3));
      const Point p4(Interpolate( p[k],p[i], v4));

      if (triangles_)
      {
	triangles_->add( p1, p2, p3 );
	triangles_->add( p1, p3, p4 );
      }
      if (trisurf_.get_rep())
      {
	TriSurfMesh::Node::index_type i1, i2, i3, i4;
	i1 = find_or_add_edgepoint(node[o], node[i], v1, p1);
	i2 = find_or_add_edgepoint(node[o], node[j], v2, p2);
	i3 = find_or_add_edgepoint(node[k], node[j], v3, p3);
	i4 = find_or_add_edgepoint(node[k], node[i], v4, p4);
        if (i1 != i2 && i2 != i3 && i3 != i1)
          trisurf_->add_triangle(i1, i2, i3);
        if (i1 != i3 && i3 != i4 && i4 != i1)
          trisurf_->add_triangle(i1, i3, i4);
      }
    }
    break;
  default:
    // do nothing. 
    // MarchingCubes calls extract on each and every cell. i.e., this is
    // not an error
    break;
  }
}

template<class Field>
FieldHandle
TetMC<Field>::get_field(double value)
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
TetMC<Field>::get_interpolant()
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

#endif // TetMC_h
