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
#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfField.h>

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
  GeomTrianglesP *triangles_;
  bool build_trisurf_;
  TriSurfMeshHandle trisurf_;
  map<long int, TriSurfMesh::Node::index_type> vertex_map_;
  vector<long int> node_vector_;
  int nnodes_;
  TriSurfMesh::Node::index_type find_or_add_edgepoint(int, int, const Point &);
  TriSurfMesh::Node::index_type find_or_add_nodepoint(node_index_type &);

  int n_;

public:
  PrismMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~PrismMC();
	
  void extract( cell_index_type, double );
  void extract_n( cell_index_type, double );
  void extract_c( cell_index_type, double );
  void reset( int, bool build_trisurf=false);
  GeomObj *get_geom() { return triangles_->size() ? triangles_ : 0; };
  FieldHandle get_field(double val);
};
  

template<class Field>    
PrismMC<Field>::~PrismMC()
{
}
    

template<class Field>
void PrismMC<Field>::reset( int n, bool build_trisurf )
{
  n_ = 0;

  build_trisurf_ = build_trisurf;

  triangles_ = new GeomTrianglesP;
  triangles_->reserve_clear((int)(1.3*n));

  vertex_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;
  if (field_->data_at() == Field::CELL)
  {
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->synchronize(Mesh::FACE_NEIGHBORS_E);
    node_vector_ = vector<long int>(nsize, -1);
  }

  if (build_trisurf_)
    trisurf_ = new TriSurfMesh; 
  else 
    trisurf_=0;
}


template<class Field>
TriSurfMesh::Node::index_type
PrismMC<Field>::find_or_add_edgepoint(int n0, int n1, const Point &p) 
{
  map<long int, TriSurfMesh::Node::index_type>::iterator node_iter;
  TriSurfMesh::Node::index_type node_idx;
  long int key = (n0 < n1) ? n0*nnodes_+n1 : n1*nnodes_+n0;
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
TriSurfMesh::Node::index_type
PrismMC<Field>::find_or_add_nodepoint(node_index_type &tet_node_idx) {
  TriSurfMesh::Node::index_type surf_node_idx;
  long int i = node_vector_[(long int)(tet_node_idx)];
  if (i != -1) surf_node_idx = (TriSurfMesh::Node::index_type) i;
  else {
    Point p;
    mesh_->get_point(p, tet_node_idx);
    surf_node_idx = trisurf_->add_point(p);
    node_vector_[(long int)tet_node_idx] = (long int)surf_node_idx;
  }
  return surf_node_idx;
}

template<class Field>
void PrismMC<Field>::extract( cell_index_type cell, double v )
{
  if (field_->data_at() == Field::NODE)
    extract_n(cell, v);
  else
    extract_c(cell, v);
}

template<class Field>
void PrismMC<Field>::extract_c( cell_index_type cell, double iso )
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

      triangles_->add(p[0], p[1], p[2]);
      
      if( nodes.size() == 4 )
	triangles_->add(p[0], p[2], p[3]);

      if (build_trisurf_) {
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
void PrismMC<Field>::extract_n( cell_index_type cell, double iso )
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

    // Degenerate triangle can be built since ponit were duplicated
    // above in order to make a hexvol for MC. 
    if( v0 != v1 && v0 != v2 && v1 != v2 ) {
      triangles_->add(q[v0], q[v1], q[v2]);
      if (build_trisurf_)
	trisurf_->add_triangle(surf_node[v0], surf_node[v1], surf_node[v2]);
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
    fld = scinew TriSurfField<double>(trisurf_, Field::NODE);
    vector<double>::iterator iter = fld->fdata().begin();
    while (iter != fld->fdata().end()) { (*iter)=value; ++iter; }
  }
  return fld;
}

     
} // End namespace SCIRun

#endif // PrismMC_h
