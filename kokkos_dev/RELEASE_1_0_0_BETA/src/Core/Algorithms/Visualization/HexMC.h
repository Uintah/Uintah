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
#include <Core/Algorithms/Visualization/mc_table.h>

namespace SCIRun {

//! A Macrching Cube teselator for a Hexagon cell     

template<class Field>
class HexMC
{
public:
  typedef Field                                  field_type;
  typedef typename Field::mesh_type::cell_index  cell_index;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;
  typedef typename mesh_type::node_array         node_array;
private:
  Field *field_;
  mesh_handle_type mesh_;
  GeomTrianglesP *triangles_;

public:
  HexMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~HexMC();
	
  void extract( const cell_index &, double);
  void reset( int );
  GeomObj *get_geom() { return triangles_; };
};
  

template<class Field>    
HexMC<Field>::~HexMC()
{
}
    

template<class Field>
void HexMC<Field>::reset( int n )
{
  triangles_ = new GeomTrianglesP;
  triangles_->reserve_clear(n*2.5);
}

template<class Field>
void HexMC<Field>::extract( const cell_index& cell, double iso )
{
  node_array node(8);
  Point p[8];
  value_type value[8];
  int code = 0;

  mesh_->get_nodes( node, cell );

  for (int i=7; i>=0; i--) {
    mesh_->get_point( p[i], node[i] );
    field_->value( value[i], node[i] );
    code = code*2+(value[i] < iso );
  }

  if ( code == 0 || code == 255 )
    return;

  TriangleCase *tcase=&tri_case[code];
  int *vertex = tcase->vertex;
  
  Point q[12];
  
  // interpolate and project vertices
  int v = 0;
  for (int t=0; t<tcase->n; t++) {
    int i = vertex[v++];
    for ( ; i != -1; i=vertex[v++] ) {
      int v1 = edge_table[i][0];
      int v2 = edge_table[i][1];
      q[i] = Interpolate(p[v1], p[v2], 
			 (value[v1]-iso)/double(value[v1]-value[v2]));
    }
  }
  
  v = 0;
  for ( int i=0; i<tcase->n; i++) {
    int v0 = vertex[v++];
    int v1 = vertex[v++];
    int v2 = vertex[v++];
    
    for (; v2 != -1; v1=v2,v2=vertex[v++]) {
      triangles_->add(q[v0], q[v1], q[v2]);
    }
    
  }
}


     
} // End namespace SCIRun

#endif
