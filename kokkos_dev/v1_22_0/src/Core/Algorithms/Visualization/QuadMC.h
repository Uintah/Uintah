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
 *  QuadMC.h
 *
 *   \author Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   \date September 2002
 *
 */


#ifndef QuadMC_h
#define QuadMC_h

#include <Core/Geom/GeomLine.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <sci_hash_map.h>

namespace SCIRun {

struct QuadMCBase {
  virtual ~QuadMCBase() {}
  static const string& get_h_file_path();
};

//! A Macrching Cube tesselator for a triangle face

template<class Field>
class QuadMC : public QuadMCBase
{
public:
  typedef Field                                  field_type;
  typedef typename Field::mesh_type::Face::index_type  cell_index_type;
  typedef typename Field::mesh_type::Node::index_type  node_index_type;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;

private:
  LockingHandle<Field> field_;
  mesh_handle_type mesh_;
  GeomLines *lines_;
  CurveMeshHandle out_mesh_;
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

  CurveMesh::Node::index_type find_or_add_edgepoint(unsigned int n0,
						    unsigned int n1,
						    double d0,
						    const Point &p);
  CurveMesh::Node::index_type find_or_add_nodepoint(node_index_type &idx);

  void extract_n( cell_index_type, double );
  void extract_f( cell_index_type, double );

public:
  QuadMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~QuadMC();
	
  void extract( cell_index_type, double );
  void reset( int, bool build_field, bool build_geom);
  GeomHandle get_geom() { return lines_; }
  FieldHandle get_field(double val);
  MatrixHandle get_interpolant();
};
  

template<class Field>    
QuadMC<Field>::~QuadMC()
{
}
    

template<class Field>
void QuadMC<Field>::reset( int n, bool build_field, bool build_geom )
{
  edge_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;

  if (field_->data_at() == Field::FACE)
  {
    mesh_->synchronize(Mesh::EDGES_E);
    mesh_->synchronize(Mesh::EDGE_NEIGHBORS_E);
    if (build_field) { node_map_ = vector<long int>(nsize, -1); }
  }

  lines_ = 0;
  if (build_geom)
  {
    lines_ = scinew GeomLines;
  }

  out_mesh_ = 0;
  if (build_field)
  {
    out_mesh_ = scinew CurveMesh;
  }
}


template<class Field>
CurveMesh::Node::index_type
QuadMC<Field>::find_or_add_edgepoint(unsigned int u0, unsigned int u1,
				     double d0, const Point &p) 
{
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
CurveMesh::Node::index_type
QuadMC<Field>::find_or_add_nodepoint(node_index_type &tri_node_idx)
{
  CurveMesh::Node::index_type curve_node_idx;
  long int i = node_map_[(long int)(tri_node_idx)];
  if (i != -1) curve_node_idx = (CurveMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, tri_node_idx);
    curve_node_idx = out_mesh_->add_point(p);
    node_map_[(long int)tri_node_idx] = (long int)curve_node_idx;
  }
  return curve_node_idx;
}


template<class Field>
void QuadMC<Field>::extract_n( cell_index_type cell, double v )
{
  typename mesh_type::Node::array_type node;
  Point p[4];
  value_type value[4];

  mesh_->get_nodes( node, cell );

  static int num[16] = { 0, 1, 1, 1,
			 1, 2, 1, 1,
			 1, 1, 2, 1,
			 1, 1, 1, 0 };
  static int order[16][4] =
    {
      { 0, 0, 0, 0 },
      { 0, 1, 0, 3 },
      { 0, 1, 1, 2 },
      { 0, 3, 1, 2 },

      { 1, 2, 2, 3 },
      { 0, 0, 0, 0 },
      { 0, 1, 2, 3 },
      { 0, 3, 2, 3 },

      // Reverse direction, mirrors 0-7
      { 0, 3, 2, 3 },
      { 0, 1, 2, 3 },
      { 0, 0, 0, 0 },
      { 1, 2, 2, 3 },

      { 0, 3, 1, 2 },
      { 0, 1, 1, 2 },
      { 0, 1, 0, 3 },
      { 0, 0, 0, 0 },
    };

  int code = 0;
  for (int i=0; i<4; i++) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code |= (value[i] > v ) << i;
  }

  if (num[code] == 1)
  {
    const int a = order[code][0];
    const int b = order[code][1];
    const int c = order[code][2];
    const int d = order[code][3];
    const double d0 = (v-value[a])/double(value[b]-value[a]);
    const double d1 = (v-value[c])/double(value[d]-value[c]);
    const Point p0(Interpolate(p[a], p[b], d0));
    const Point p1(Interpolate(p[c], p[d], d1));

    if (lines_)
    {
      lines_->add( p0, p1 );
    }
    if (out_mesh_.get_rep())
    {
      CurveMesh::Node::array_type cnode(2);
      cnode[0] = find_or_add_edgepoint(node[a], node[b], d0, p0);
      cnode[1] = find_or_add_edgepoint(node[c], node[d], d1, p1);
      out_mesh_->add_elem(cnode);
    }
  }
  else if (code == 5)
  {
    {
      const int a = order[1][0];
      const int b = order[1][1];
      const int c = order[1][2];
      const int d = order[1][3];
      const double d0 = (v-value[a])/double(value[b]-value[a]);
      const double d1 = (v-value[c])/double(value[d]-value[c]);
      const Point p0(Interpolate(p[a], p[b], d0));
      const Point p1(Interpolate(p[c], p[d], d1));

      if (lines_)
      {
	lines_->add( p0, p1 );
      }
      if (out_mesh_.get_rep())
      {
	CurveMesh::Node::array_type cnode(2);
	cnode[0] = find_or_add_edgepoint(node[a], node[b], d0, p0);
	cnode[1] = find_or_add_edgepoint(node[c], node[d], d0, p1);
	out_mesh_->add_elem(cnode);
      }
    }
    {
      const int a = order[4][0];
      const int b = order[4][1];
      const int c = order[4][2];
      const int d = order[4][3];
      const double d0 = (v-value[a])/double(value[b]-value[a]);
      const double d1 = (v-value[c])/double(value[d]-value[c]);
      const Point p0(Interpolate(p[a], p[b], d0));
      const Point p1(Interpolate(p[c], p[d], d1));

      if (lines_)
      {
	lines_->add( p0, p1 );
      }
      if (out_mesh_.get_rep())
      {
	CurveMesh::Node::array_type cnode(2);
	cnode[0] = find_or_add_edgepoint(node[a], node[b], d0, p0);
	cnode[1] = find_or_add_edgepoint(node[c], node[d], d1, p1);
	out_mesh_->add_elem(cnode);
      }
    }
  }
  else if (code == 10)
  {
    {
      const int a = order[2][0];
      const int b = order[2][1];
      const int c = order[2][2];
      const int d = order[2][3];
      const double d0 = (v-value[a])/double(value[b]-value[a]);
      const double d1 = (v-value[c])/double(value[d]-value[c]);
      const Point p0(Interpolate(p[a], p[b], d0));
      const Point p1(Interpolate(p[c], p[d], d1));

      if (lines_)
      {
	lines_->add( p0, p1 );
      }
      if (out_mesh_.get_rep())
      {
	CurveMesh::Node::array_type cnode(2);
	cnode[0] = find_or_add_edgepoint(node[a], node[b], d0, p0);
	cnode[1] = find_or_add_edgepoint(node[c], node[d], d1, p1);
	out_mesh_->add_elem(cnode);
      }
    }
    {
      const int a = order[8][0];
      const int b = order[8][1];
      const int c = order[8][2];
      const int d = order[8][3];
      const double d0 = (v-value[a])/double(value[b]-value[a]);
      const double d1 = (v-value[c])/double(value[d]-value[c]);
      Point p0(Interpolate(p[a], p[b], d0));
      Point p1(Interpolate(p[c], p[d], d1));

      if (lines_)
      {
	lines_->add( p0, p1 );
      }
      if (out_mesh_.get_rep())
      {
	CurveMesh::Node::array_type cnode(2);
	cnode[0] = find_or_add_edgepoint(node[a], node[b], d0, p0);
	cnode[1] = find_or_add_edgepoint(node[c], node[d], d1, p1);
	out_mesh_->add_elem(cnode);
      }
    }
  }
}


template<class Field>
void QuadMC<Field>::extract_f( cell_index_type cell, double iso )
{
  value_type selfvalue, nbrvalue;
  if (!field_->value( selfvalue, cell )) return;
  typename mesh_type::Edge::array_type edges;
  mesh_->get_edges(edges, cell);

  cell_index_type nbr;
  Point p[2];
  typename mesh_type::Node::array_type nodes;
  CurveMesh::Node::array_type vertices(2);
  unsigned int i, j;
  for (i = 0; i < edges.size(); i++)
  {
    if (mesh_->get_neighbor(nbr, cell, edges[i]) &&
	field_->value(nbrvalue, nbr) &&
	(selfvalue > nbrvalue) &&
	((selfvalue-iso) * (nbrvalue-iso) < 0))
    {
      mesh_->get_nodes(nodes, edges[i]);
      for (j=0; j < 2; j++) { mesh_->get_center(p[j], nodes[j]); }
      if (lines_)
      {
	lines_->add(p[0], p[1]);
      }
      if (out_mesh_.get_rep())
      {
	for (j=0; j < 2; j ++)
	{
	  vertices[j] = find_or_add_nodepoint(nodes[j]);
	}
	out_mesh_->add_elem(vertices);
      }
    }
  }
}


template<class Field>
void QuadMC<Field>::extract( cell_index_type cell, double v )
{
  if (field_->data_at() == Field::NODE)
    extract_n(cell, v);
  else
    extract_f(cell, v);
}


template<class Field>
FieldHandle
QuadMC<Field>::get_field(double value)
{
  CurveField<double> *fld = 0;
  if (out_mesh_.get_rep())
  {
    fld = scinew CurveField<double>(out_mesh_, Field::NODE);
    vector<double>::iterator iter = fld->fdata().begin();
    while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
  }
  return fld;
}

     
template<class Field>
MatrixHandle
QuadMC<Field>::get_interpolant()
{
  if (field_->data_at() == Field::NODE)
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

    for (int i = 0; i <= nrows; i++)
    {
      rr[i] = i * 2;
    }

    return scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows*2, dd);
  }
  else
  {
    return 0;
  }
}

     
} // End namespace SCIRun

#endif // QuadMC_h
