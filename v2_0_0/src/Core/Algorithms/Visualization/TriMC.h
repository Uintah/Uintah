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
 *  TriMC.h
 *
 *   \author Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   \date September 2002
 *
 */


#ifndef TriMC_h
#define TriMC_h

#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {

struct TriMCBase {
  virtual ~TriMCBase() {}
  static const string& get_h_file_path();
};

//! A Macrching Cube tesselator for a triangle face

template<class Field>
class TriMC : public TriMCBase
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
  bool build_mesh_;
  CurveMeshHandle out_mesh_;
  map<long int, CurveMesh::Node::index_type> vertex_map_;
  int nnodes_;
  int n_;
  vector<long int> node_vector_;

  CurveMesh::Node::index_type find_or_add_edgepoint(int, int, const Point &);
  CurveMesh::Node::index_type find_or_add_nodepoint(node_index_type &idx);

  void extract_n( cell_index_type, double );
  void extract_f( cell_index_type, double );

public:
  TriMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~TriMC();
	
  void extract( cell_index_type, double );
  void reset( int, bool build_mesh=false);
  GeomObj *get_geom() { return lines_->size() ? lines_ : 0; };
  FieldHandle get_field(double val);
};
  

template<class Field>    
TriMC<Field>::~TriMC()
{
}
    

template<class Field>
void TriMC<Field>::reset( int n, bool build_mesh )
{
  n_ = 0;

  build_mesh_ = build_mesh;

  lines_ = new GeomLines;

  vertex_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;

  if (field_->data_at() == Field::FACE)
  {
    mesh_->synchronize(Mesh::EDGES_E);
    mesh_->synchronize(Mesh::EDGE_NEIGHBORS_E);
    node_vector_ = vector<long int>(nsize, -1);
  }

  if (build_mesh)
    out_mesh_ = new CurveMesh; 
  else 
    out_mesh_=0;
}


template<class Field>
CurveMesh::Node::index_type
TriMC<Field>::find_or_add_edgepoint(int n0, int n1, const Point &p) 
{
  map<long int, CurveMesh::Node::index_type>::iterator node_iter;
  CurveMesh::Node::index_type node_idx;
  long int key = (n0 < n1) ? n0*nnodes_+n1 : n1*nnodes_+n0;
  node_iter = vertex_map_.find(key);
  if (node_iter == vertex_map_.end()) { // first time to see this node
    node_idx = out_mesh_->add_node(p);
    vertex_map_[key] = node_idx;
  } else {
    node_idx = (*node_iter).second;
  }
  return node_idx;
}


template<class Field>
CurveMesh::Node::index_type
TriMC<Field>::find_or_add_nodepoint(node_index_type &tri_node_idx)
{
  CurveMesh::Node::index_type curve_node_idx;
  long int i = node_vector_[(long int)(tri_node_idx)];
  if (i != -1) curve_node_idx = (CurveMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, tri_node_idx);
    curve_node_idx = out_mesh_->add_point(p);
    node_vector_[(long int)tri_node_idx] = (long int)curve_node_idx;
  }
  return curve_node_idx;
}


template<class Field>
void TriMC<Field>::extract_n( cell_index_type cell, double v )
{
  typename mesh_type::Node::array_type node;
  Point p[3];
  value_type value[3];

  mesh_->get_nodes( node, cell );

  static int num[8] = { 0, 1, 1, 1, 1, 1, 1, 0 };
  static int clip[8] = { 0, 0, 1, 2, 2, 1, 0, 0 };

  int code = 0;
  for (int i=0; i<3; i++) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code |= (value[i] > v ) << i;
  }

  //  if ( show_case != -1 && (code != show_case) ) return;
  if (num[code])
  {
    const int a = clip[code];
    const int b = (a + 1) % 3;
    const int c = (a + 2) % 3;

    Point p0(Interpolate(p[a], p[b], (v-value[a])/double(value[b]-value[a])));
    Point p1(Interpolate(p[a], p[c], (v-value[a])/double(value[c]-value[a])));

    lines_->add( p0, p1 );
    n_++;

    if (build_mesh_)
    {
      CurveMesh::Node::array_type cnode(2);
      cnode[0] = find_or_add_edgepoint(node[a], node[b], p0);
      cnode[1] = find_or_add_edgepoint(node[a], node[c], p1);
      out_mesh_->add_elem(cnode);
    }
  }
}


template<class Field>
void TriMC<Field>::extract_f( cell_index_type cell, double iso )
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
	//(selfvalue > nbrvalue) &&
	((selfvalue-iso) * (nbrvalue-iso) < 0))
    {
      mesh_->get_nodes(nodes, edges[i]);
      for (j=0; j < 2; j++) { mesh_->get_center(p[j], nodes[j]); }
      lines_->add(p[0], p[1]);

      if (build_mesh_)
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
void TriMC<Field>::extract( cell_index_type cell, double v )
{
  if (field_->data_at() == Field::NODE)
    extract_n(cell, v);
  else
    extract_f(cell, v);
}



template<class Field>
FieldHandle
TriMC<Field>::get_field(double value)
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


     
} // End namespace SCIRun

#endif // TriMC_h
