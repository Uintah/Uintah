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
 *  HexMC.h
 *
 *  \author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef HexMC_h
#define HexMC_h

#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
//#include <Core/Algorithms/Visualization/mc_table.h>
#include <Core/Algorithms/Visualization/mcube2.h>
#include <Core/Datatypes/TriSurfMesh.h>

namespace SCIRun {

struct HexMCBase {
  virtual ~HexMCBase() {}
  static const string& get_h_file_path();
};
//! A Macrching Cube teselator for a Hexagon cell     

template<class Field>
class HexMC : public HexMCBase
{
public:
  typedef Field                                  field_type;
  typedef typename Field::mesh_type::Cell::index_type  cell_index_type;
  typedef typename Field::mesh_type::Node::index_type  node_index_type;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;
  typedef typename mesh_type::Node::array_type         node_array_type;
private:
  Field *field_;
  mesh_handle_type mesh_;
  GeomTrianglesP *triangles_;
  bool build_trisurf_;
  TriSurfMeshHandle trisurf_;
  map<long int, TriSurfMesh::Node::index_type> vertex_map_;
  int nx_, ny_, nz_;
  TriSurfMesh::Node::index_type find_or_add_edgepoint(node_index_type, node_index_type, const Point &);

public:
  HexMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~HexMC();
	
  void extract( const cell_index_type &, double);
  void reset( int, bool build_trisurf=false );
  GeomObj *get_geom() { return triangles_; };
  TriSurfMeshHandle get_trisurf() { return trisurf_; };
};
  

template<class Field>    
HexMC<Field>::~HexMC()
{
}
    

template<class Field>
void HexMC<Field>::reset( int n, bool build_trisurf )
{
  build_trisurf_ = build_trisurf;
  triangles_ = new GeomTrianglesP;
  triangles_->reserve_clear((int)(n*2.5));
  vertex_map_.clear();
  nx_ = mesh_->get_nx();
  ny_ = mesh_->get_ny();
  nz_ = mesh_->get_nz();
  if (build_trisurf_)
    trisurf_ = new TriSurfMesh; 
  else 
    trisurf_=0;
}

template<class Field>
TriSurfMesh::Node::index_type
HexMC<Field>::find_or_add_edgepoint(node_index_type n0, node_index_type n1, const Point &p)
{
  map<long int, TriSurfMesh::Node::index_type>::iterator node_iter;
  long int key0 = 
    (long int) n0.i_ + nx_ * ((long int) n0.j_ + (ny_ * (long int) n0.k_));
  long int key1 = 
    (long int) n1.i_ + nx_ * ((long int) n1.j_ + (ny_ * (long int) n1.k_));
  long int small_key;
  int dir;
  if (n0.k_ != n1.k_) dir=2;
  else if (n0.j_ != n1.j_) dir=1;
  else dir=0;
  if (n0.k_ < n1.k_) small_key = key0;
  else if (n0.k_ > n1.k_) small_key = key1;
  else if (n0.j_ < n1.j_) small_key = key0;
  else if (n0.j_ > n1.j_) small_key = key1;
  else if (n0.i_ < n1.i_) small_key = key0;
  else small_key = key1;
  long int key = (small_key * 4) + dir;
  TriSurfMesh::Node::index_type node_idx;
  node_iter = vertex_map_.find(key);
  if (node_iter == vertex_map_.end()) { // first time to see this node
    node_idx = trisurf_->add_point(p);
    vertex_map_[key] = node_idx;
  } else {
    node_idx = (*node_iter).second;
  }
  return node_idx;
}

template<class Field>
void HexMC<Field>::extract( const cell_index_type& cell, double iso )
{
  node_array_type node(8);
  Point p[8];
  value_type value[8];
  int code = 0;

  mesh_->get_nodes( node, cell );

  for (int i=7; i>=0; i--) {
    mesh_->get_point( p[i], node[i] );
    if (!field_->value( value[i], node[i] )) return;
    code = code*2+(value[i] < iso );
  }

  if ( code == 0 || code == 255 )
    return;

//  TriangleCase *tcase=&tri_case[code];
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
    int v1 = edge_tab[i][0];
    int v2 = edge_tab[i][1];
    q[i] = Interpolate(p[v1], p[v2], 
		       (value[v1]-iso)/double(value[v1]-value[v2]));
    if (build_trisurf_)
      surf_node[i] = find_or_add_edgepoint(node[v1], node[v2], q[i]);
  }    
  
  v = 0;
  while(vertex[v] != -1) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    triangles_->add(q[v0], q[v1], q[v2]);
    if (build_trisurf_)
      trisurf_->add_triangle(surf_node[v0], surf_node[v1], surf_node[v2]);
  }
}


     
} // End namespace SCIRun

#endif // HexMC_H
