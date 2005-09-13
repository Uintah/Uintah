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
 *  EdgeMC.h
 *
 *   \author Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   \date September 2002
 *
 */


#ifndef EdgeMC_h
#define EdgeMC_h

#include <Core/Geom/GeomPoint.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <sci_hash_map.h>

namespace SCIRun {

struct EdgeMCBase {
  virtual ~EdgeMCBase() {}
  static const string& get_h_file_path();
};

//! A Macrching Square tesselator for a curve line

template<class Field>
class EdgeMC : public EdgeMCBase
{
public:
  typedef Field                                  field_type;
  typedef typename Field::mesh_type::Edge::index_type  edge_index_type;
  typedef typename Field::mesh_type::Node::index_type  node_index_type;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;

private:
  LockingHandle<Field> field_;
  mesh_handle_type mesh_;
  GeomPoints *points_;
  PointCloudMeshHandle out_mesh_;
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
		   CurveMesh::Node::index_type,
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
	      CurveMesh::Node::index_type,
	      edgepairless> edge_hash_type;
#endif

  edge_hash_type   edge_map_;  // Unique edge cuts when surfacing node data
  vector<long int> node_map_;  // Unique nodes when surfacing cell data.

  PointCloudMesh::Node::index_type find_or_add_edgepoint(unsigned int n0,
							 unsigned int n1,
							 double d0,
							 const Point &p);

  PointCloudMesh::Node::index_type find_or_add_nodepoint(node_index_type &idx);

  void extract_n( edge_index_type, double );
  void extract_e( edge_index_type, double );

public:
  EdgeMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~EdgeMC();
	
  void extract( edge_index_type, double );
  void reset( int, bool build_field, bool build_geom );
  GeomHandle get_geom() { return points_; }
  FieldHandle get_field(double val);
  MatrixHandle get_interpolant();
};
  

template<class Field>    
EdgeMC<Field>::~EdgeMC()
{
}
    

template<class Field>
void EdgeMC<Field>::reset( int n, bool build_field, bool build_geom )
{
  edge_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;

  if (field_->basis_order() == 0)
  {
    mesh_->synchronize(Mesh::EDGES_E);
    mesh_->synchronize(Mesh::EDGE_NEIGHBORS_E);
    if (build_field) { node_map_ = vector<long int>(nsize, -1); }
  }

  points_ = 0;
  if (build_geom)
  {
    points_ = scinew GeomPoints;
  }

  out_mesh_ = 0;
  if (build_field)
  {
    out_mesh_ = scinew PointCloudMesh;
  }
}


template<class Field>
PointCloudMesh::Node::index_type
EdgeMC<Field>::find_or_add_edgepoint(unsigned int u0, unsigned int u1,
				      double d0, const Point &p) 
{
  if (d0 <= 0.0) { u1 = (unsigned int)-1; }
  if (d0 >= 1.0) { u0 = (unsigned int)-1; }
  edgepair_t np;
  if (u0 < u1)  { np.first = u0; np.second = u1; np.dfirst = d0; }
  else { np.first = u1; np.second = u0; np.dfirst = 1.0 - d0; }
  const typename edge_hash_type::iterator loc = edge_map_.find(np);
  if (loc == edge_map_.end())
  {
    const CurveMesh::Node::index_type nodeindex = out_mesh_->add_point(p);
    edge_map_[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return (*loc).second;
  }
}


template<class Field>
PointCloudMesh::Node::index_type
EdgeMC<Field>::find_or_add_nodepoint(node_index_type &curve_node_idx)
{
  PointCloudMesh::Node::index_type point_node_idx;
  long int i = node_map_[(long int)(curve_node_idx)];
  if (i != -1) point_node_idx = (PointCloudMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, curve_node_idx);
    point_node_idx = out_mesh_->add_point(p);
    node_map_[(long int)curve_node_idx] = (long int) point_node_idx;
  }
  return curve_node_idx;
}


template<class Field>
void EdgeMC<Field>::extract_n( edge_index_type edge, double v )
{
  typename mesh_type::Node::array_type node;
  Point p[2];
  value_type value[2];

  mesh_->get_nodes( node, edge );

  static int num[4] = { 0, 1, 1, 0 };

  int code = 0;
  for (int i=0; i<2; i++) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code |= (value[i] > v ) << i;
  }

  //  if ( show_case != -1 && (code != show_case) ) return;
  if (num[code])
  {
    const double d0 = (v-value[0])/double(value[1]-value[0]);
    const Point p0(Interpolate(p[0], p[1], d0));

    if (points_)
      points_->add( p0 );

    if (out_mesh_.get_rep())
    {
      PointCloudMesh::Node::array_type cnode(1);
      cnode[0] = find_or_add_edgepoint(node[0], node[1], d0, p0);
      out_mesh_->add_elem(cnode);
    }
  }
}


template<class Field>
void EdgeMC<Field>::extract_e( edge_index_type edge, double iso )
{
  value_type selfvalue, nbrvalue;
  if (!field_->value( selfvalue, edge )) return;
  typename mesh_type::Node::array_type nodes;
  mesh_->get_nodes(nodes, edge);

  edge_index_type nbr;
  Point p0;
  PointCloudMesh::Node::array_type vertices(1);
  unsigned int i;

  for (i = 0; i < nodes.size(); i++)
  {
    if (mesh_->get_neighbor(nbr, edge, nodes[i]) &&
	field_->value(nbrvalue, nbr) &&
	//(selfvalue > nbrvalue) &&
	((selfvalue-iso) * (nbrvalue-iso) < 0))
    {
      mesh_->get_center(p0, nodes[i]);
      
      if (points_)
	points_->add(p0);

      if (out_mesh_.get_rep())
      {
	vertices[0] = find_or_add_nodepoint(nodes[i]);
	out_mesh_->add_elem(vertices);
      }
    }
  }
}


template<class Field>
void EdgeMC<Field>::extract( edge_index_type edge, double v )
{
  if (field_->basis_order() == 1)
    extract_n(edge, v);
  else
    extract_e(edge, v);
}



template<class Field>
FieldHandle
EdgeMC<Field>::get_field(double value)
{
  PointCloudField<double> *fld = 0;
  if (out_mesh_.get_rep())
  {
    fld = scinew PointCloudField<double>(out_mesh_, 0);
    vector<double>::iterator iter = fld->fdata().begin();
    while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
  }
  return fld;
}


template<class Field>
MatrixHandle
EdgeMC<Field>::get_interpolant()
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

#endif // EdgeMC_h
