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

namespace SCIRun {

//! A Macrching Cube tesselator for a tetrahedral cell     

template<class Field>
class TetMC
{
public:
  typedef Field                         field_type;
  typedef typename Field::mesh_type::cell_index  cell_index;
  typedef typename Field::value_type             value_type;
  typedef typename Field::mesh_type              mesh_type;
  typedef typename Field::mesh_handle_type       mesh_handle_type;
private:
  Field *field_;
  mesh_handle_type mesh_;
  GeomTrianglesP *triangles_;

public:
  TetMC( Field *field ) : field_(field), mesh_(field->get_typed_mesh()) {}
  virtual ~TetMC();
	
  void extract( cell_index, double);
  void reset( int );
  GeomObj *get_geom() { return triangles_->size() ? triangles_ : 0; };
};
  

template<class Field>    
TetMC<Field>::~TetMC()
{
}
    

template<class Field>
void TetMC<Field>::reset( int n )
{
  triangles_ = new GeomTrianglesP;
  triangles_->reserve_clear( 1.3*n );
}

template<class Field>
void TetMC<Field>::extract( cell_index cell, double v )
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
    
    
  typename mesh_type::node_array node;
  Point p[4];
  value_type value[4];

  mesh_->get_nodes( node, cell );
  int code = 0;

  for (int i=0; i<4; i++) {
    mesh_->get_point( p[i], node[i] );
    value[i] = field_->value( node[i] );
    code = code*2+(value[i] > v );
  }

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
