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

#include <Core/Geometry/Point.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Datatypes/TriSurfMesh.h>

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
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;
private:
  Field *field_;
  mesh_handle_type mesh_;
  GeomTrianglesP *triangles_;
  bool build_trisurf_;
  TriSurfMeshHandle trisurf_;
  map<long int, TriSurfMesh::Node::index_type> vertex_map_;
  int nnodes_;
  TriSurfMesh::Node::index_type find_or_add_edgepoint(int, int, const Point &);

  int n_;

public:
  TetMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~TetMC();
	
  void extract( cell_index_type, double );
  void reset( int, bool build_trisurf=false);
  GeomObj *get_geom() { return triangles_->size() ? triangles_ : 0; };
  TriSurfMeshHandle get_trisurf() { return trisurf_; };
};
  

template<class Field>    
TetMC<Field>::~TetMC()
{
}
    

template<class Field>
void TetMC<Field>::reset( int n, bool build_trisurf )
{
  n_ = 0;

  build_trisurf_ = build_trisurf;

  triangles_ = new GeomTrianglesP;
  triangles_->reserve_clear((int)(1.3*n));

  vertex_map_.clear();
  typename Field::mesh_type::Node::size_type nsize;
  mesh_->size(nsize);
  nnodes_ = nsize;
  if (build_trisurf_)
    trisurf_ = new TriSurfMesh; 
  else 
    trisurf_=0;
}


template<class Field>
TriSurfMesh::Node::index_type
TetMC<Field>::find_or_add_edgepoint(int n0, int n1, const Point &p) 
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
void TetMC<Field>::extract( cell_index_type cell, double v )
{
  static int num[16] = { 0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0 };
  static int order[16][4] = {
    {0, 0, 0, 0},   /* none - ignore */
    {3, 0, 2, 1},   /* 3 */
    {2, 0, 1, 3},   /* 2 */
    {2, 0, 1, 3},   /* 2, 3 */
    {1, 0, 3, 2},   /* 1 */
    {1, 2, 0, 3},   /* 1, 3 */
    {1, 0, 3, 2},   /* 1, 2 */
    {0, 3, 2, 1},   /* 1, 2, 3 */
    {0, 1, 2, 3},   /* 0 */
    {2, 3, 0, 1},   /* 0, 3 - reverse of 1, 2 */
    {3, 0, 2, 1},   /* 0, 2 - reverse of 1, 3 */
    {1, 3, 0, 2},   /* 0, 2, 3 - reverse of 1 */
    {3, 1, 0, 2},   /* 0, 1 - reverse of 2, 3 */
    {2, 3, 0, 1},   /* 0, 1, 3 - reverse of 2 */
    {3, 1, 2, 0},   /* 0, 1, 2 - reverse of 3 */
    {0, 0, 0, 0}    /* all - ignore */
  };
    
    
  typename mesh_type::Node::array_type node;
  Point p[4];
  value_type value[4];

  mesh_->get_nodes( node, cell );
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
      
      Point p1(Interpolate( p[o],p[i],(v-value[o])/double(value[i]-value[o])));
      Point p2(Interpolate( p[o],p[j],(v-value[o])/double(value[j]-value[o])));
      Point p3(Interpolate( p[o],p[k],(v-value[o])/double(value[k]-value[o])));
      
      triangles_->add( p1, p2, p3 );
      n_++;

      if (build_trisurf_) {
	TriSurfMesh::Node::index_type i1, i2, i3;
	i1 = find_or_add_edgepoint(node[o], node[i], p1);
	i2 = find_or_add_edgepoint(node[o], node[j], p2);
	i3 = find_or_add_edgepoint(node[o], node[k], p3);
	trisurf_->add_triangle(i1, i2, i3);
      }
    }
    break;
  case 2: 
    {
      // make order triangles
      int o = order[code][0];
      int i = order[code][1];
      int j = order[code][2];
      int k = order[code][3];
      
      Point p1(Interpolate( p[o],p[i],(v-value[o])/double(value[i]-value[o])));
      Point p2(Interpolate( p[o],p[j],(v-value[o])/double(value[j]-value[o])));
      Point p3(Interpolate( p[k],p[j],(v-value[k])/double(value[j]-value[k])));
      
      triangles_->add( p1, p2, p3 );

      Point p4(Interpolate( p[k],p[i],(v-value[k])/double(value[i]-value[k])));

      triangles_->add( p1, p3, p4 );
      n_ += 2;

      if (build_trisurf_) {
	TriSurfMesh::Node::index_type i1, i2, i3, i4;
	i1 = find_or_add_edgepoint(node[o], node[i], p1);
	i2 = find_or_add_edgepoint(node[o], node[j], p2);
	i3 = find_or_add_edgepoint(node[k], node[j], p3);
	i4 = find_or_add_edgepoint(node[k], node[i], p4);
	trisurf_->add_triangle(i1, i2, i3);
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


     
} // End namespace SCIRun

#endif // TetMC_h
