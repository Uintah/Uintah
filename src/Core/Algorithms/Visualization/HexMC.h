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
#include <Core/Algorithms/Visualization/mcube2.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/QuadSurfField.h>

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
  LockingHandle<Field> field_;
  mesh_handle_type mesh_;
  GeomTrianglesP *triangles_;
  bool build_trisurf_;
  TriSurfMeshHandle trisurf_;
  QuadSurfMeshHandle quadsurf_;
  map<long int, TriSurfMesh::Node::index_type> vertex_map_;
  int nx_, ny_, nz_;
  vector<long int> node_vector_;
  TriSurfMesh::Node::index_type find_or_add_edgepoint(node_index_type, node_index_type, const Point &);
  QuadSurfMesh::Node::index_type find_or_add_nodepoint(node_index_type &);
  void extract_c( const cell_index_type &, double);
  void extract_n( const cell_index_type &, double);

public:
  HexMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~HexMC();
	
  void extract( const cell_index_type &, double);
  void reset( int, bool build_trisurf=false );
  GeomObj *get_geom() { return triangles_; };
  FieldHandle get_field(double val);
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
  nx_ = mesh_->get_ni();
  ny_ = mesh_->get_nj();
  nz_ = mesh_->get_nk();
  if (field_->data_at() == Field::CELL)
  {
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
    node_vector_ = vector<long int>(nx_ * ny_ * nz_, -1);
  }
  trisurf_ = 0;
  quadsurf_ = 0;
  if (build_trisurf_)
  {
    if (field_->data_at() == Field::CELL)
    {
      quadsurf_ = new QuadSurfMesh;
    }
    else
    {
      trisurf_ = new TriSurfMesh; 
    }
  }
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
QuadSurfMesh::Node::index_type
HexMC<Field>::find_or_add_nodepoint(node_index_type &tet_node_idx)
{
  QuadSurfMesh::Node::index_type surf_node_idx;
  long int i = node_vector_[(long int)(tet_node_idx)];
  if (i != -1) surf_node_idx = (QuadSurfMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, tet_node_idx);
    surf_node_idx = quadsurf_->add_point(p);
    node_vector_[(long int)tet_node_idx] = (long int)surf_node_idx;
  }
  return surf_node_idx;
}


template<class Field>
void
HexMC<Field>::extract( const cell_index_type& cell, double iso )
{
  if (field_->data_at() == Field::NODE)
    extract_n(cell, iso);
  else
    extract_c(cell, iso);
}


template<class Field>
void
HexMC<Field>::extract_c( const cell_index_type& cell, double iso )
{
  value_type selfvalue, nbrvalue;
  if (!field_->value( selfvalue, cell )) return;
  typename mesh_type::Face::array_type faces;
  mesh_->get_faces(faces, cell);

  cell_index_type nbr_cell;
  Point p[4];
  node_array_type face_nodes;
  QuadSurfMesh::Node::index_type verts[4];
  unsigned int f, n;
  for (f=0; f<faces.size(); f++)
  {
    if (mesh_->get_neighbor(nbr_cell, cell, faces[f]) &&
	field_->value(nbrvalue, nbr_cell) &&
	(selfvalue > nbrvalue) &&
	((selfvalue - iso) * (nbrvalue - iso) < 0 ))
    {
      mesh_->get_nodes(face_nodes, faces[f]);
      for (n=0; n<4; n++) { mesh_->get_center(p[n], face_nodes[n]); }
      triangles_->add(p[0], p[1], p[2]);
      triangles_->add(p[2], p[3], p[0]);

      if (build_trisurf_)
      {
	for (n=0; n<4; n++)
	{
	  verts[n]=find_or_add_nodepoint(face_nodes[n]);
	}
	quadsurf_->add_quad(verts[0], verts[1], verts[2], verts[3]);
      }
    }
  }
}


template<class Field>
void HexMC<Field>::extract_n( const cell_index_type& cell, double iso )
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


template<class Field>
FieldHandle
HexMC<Field>::get_field(double value)
{
  if (field_->data_at() == Field::CELL)
  {
    QuadSurfField<double> *fld = 0;
    if (quadsurf_.get_rep())
    {
      fld = scinew QuadSurfField<double>(quadsurf_, Field::NODE);
      vector<double>::iterator iter = fld->fdata().begin();
      while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
    }
    return fld;
  }
  else
  {
    TriSurfField<double> *fld = 0;
    if (trisurf_.get_rep())
    {
      fld = scinew TriSurfField<double>(trisurf_, Field::NODE);
      vector<double>::iterator iter = fld->fdata().begin();
      while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
    }
    return fld;
  }
}

     
} // End namespace SCIRun

#endif // HexMC_H
