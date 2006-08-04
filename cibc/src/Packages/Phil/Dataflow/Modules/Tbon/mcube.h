
/*  mcube.h
    Marching Cubes style interpolation for structured and unstructured grids

    Packages/Philip Sutton
    May 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __MCUBE_H__
#define __MCUBE_H__

#include "TriGroup.h"
#include "mcube_table.h"

//#include <Core/Geometry/Point.h>

#include <stdio.h>
#include <stdlib.h>


namespace Phil {
using namespace SCIRun;


struct Tetra;
struct TetraVertex;
struct VertexCL; 
template<class T> struct Data;
template<class T> struct DataUG;
template<class T> struct DataCL;

// MCubes 
// Marching Cubes class for regular (structured) grids
template<class T>
class MCubes {
public:
  GeomTriGroup *triangles;
  T *data;

  MCubes( Data<T>* field, int* X, int* Y, int* Z, 
	  float d1=1.0, float d2=1.0, float d3=1.0 );
  
  void reset( int n );
  void interp( int x, int y, int z, int branching, float iso );
  void setResolution( int res );

  friend pPoint interpolate( const pPoint &p, const pPoint& q, float f);

protected:
private:
  int* Xarray;
  int* Yarray;
  int* Zarray;
  int nx, ny, nz;
  float dx, dy, dz;
  
  int offseti;
  float offsetdx, offsetdy, offsetdz;

  int *lookup, *insert;
  pPoint shared[30];

  static int lookup1[12]; static int insert1[12];
  static int lookup2[12]; static int insert2[12];
  static int lookup4[12]; static int insert4[12];
  static int lookup3[3][12]; static int insert3[3][12];
  static int lookup5[3][12]; static int insert5[3][12];
  static int lookup6[3][12]; static int insert6[3][12];
  static int lookup7[7][12]; static int insert7[7][12];
  
  void interp1( int x, int y, int z, float iso );
  void interp2( int x, int y, int z, float iso, int casenum );
  void interp4( int x, int y, int z, float iso, int casenum );
  void interp8( int x, int y, int z, float iso );
};

// MCubesUG
// Marching Cubes class for unstructured grids
template<class T>
class MCubesUG {
public:
  MCubesUG( DataUG<T> *d );
  
  void reset( int n );
  void interp( int cell, float iso );
  void shift( float x, float y, float z );

  friend pPoint interpolate( const float* p, const float* q, float f);

public:
  GeomTriGroup* triangles;
  TetraVertex* verts;
  Tetra* tets;
  T* values;
};

// MCubesCL
// Marching Cubes class for curvilinear grids
template<class T>
class MCubesCL {
public:
  MCubesCL( DataCL<T>** d, int n, int** X, int** Y, int** Z );

  void reset( int n );
  void interpCircular( int zone, int x, int y, int z, float iso, 
		       int branching );
  void interpRegular( int zone, int x, int y, int z, float iso, 
		      int branching );
  void setResolution( int res );

  friend pPoint interpolate( const float* p, const float* q, float f);
  
public:
  GeomTriGroup* triangles;

private:
  DataCL<T>** data;
  int** Xarray;
  int** Yarray;
  int** Zarray;

  int offseti;

  void interpCircular1( int zone, int x, int y, int z, float iso );
  void interpCircular2( int zone, int x, int y, int z, float iso, 
			int casenum );
  void interpCircular4( int zone, int x, int y, int z, float iso, 
			int casenum );
  void interpCircular8( int zone, int x, int y, int z, float iso );
  void interpRegular1( int zone, int x, int y, int z, float iso );
  void interpRegular2( int zone, int x, int y, int z, float iso, 
		       int casenum );
  void interpRegular4( int zone, int x, int y, int z, float iso, 
		       int casenum );
  void interpRegular8( int zone, int x, int y, int z, float iso ); 
  
};


// interpolate (MCubes)
// interpolate between points p and q, by a fraction f
inline
pPoint interpolate( const pPoint &p, const pPoint &q, float f )
{
  return pPoint( p.x() + f*(q.x()-p.x()), 
		 p.y() + f*(q.y()-p.y()), 
		 p.z() + f*(q.z()-p.z()) );
}

// interpolate (MCubesUG, MCubesCL)
// interpolate between points p and q, by a fraction f
inline
pPoint interpolate( const float* p, const float* q, float f) {
  return pPoint( p[0] + f*(q[0] - p[0]),
		p[1] + f*(q[1] - p[1]),
		p[2] + f*(q[2] - p[2]) );
}


// tables for spatial locality (regular grid)
static const int N = -1;  // null entry - makes tables look pretty
template<class T>
int MCubes<T>::lookup1[] = { N, N, N, 0, N, N, N, 1, 2, N, 3, N };
template<class T>
int MCubes<T>::insert1[] = { N, 0, N, N, N, 1, N, N, N, 2, N, 3 };
template<class T>
int MCubes<T>::lookup2[] = { 0, N, N, N, 1, N, N, N, 2, 3, N, N };
template<class T>
int MCubes<T>::insert2[] = { N, N, 0, N, N, N, 1, N, N, N, 2, 3 };
template<class T>
int MCubes<T>::lookup4[] = { 0, 1, 2, 3, N, N, N, N, N, N, N, N };
template<class T>
int MCubes<T>::insert4[] = { N, N, N, N, 0, 1, 2, 3, N, N, N, N };
template<class T>
int MCubes<T>::lookup3[][12] = {{ N, N, N, 0, N, N, N, 2, 4, N, 5, N },
				{ 7, N, N, N, 9, N, N, N, 5, 8, N, N },
				{ 1, 10, N, N, 3, 12, N, N, 6, 5, N, 11 } };
template<class T>
int MCubes<T>::insert3[][12] = {{ N, 0, 1, N, N, 2, 3, N, N, 4, 6, 5 },
				{ N, N, 7, N, N, N, 9, N, N, N, N, 8 },
				{ N, N, N, 10, N, N, N, 12, N, N, 11, N }};
template<class T>
int MCubes<T>::lookup5[][12] = {{ N, N, N, 5, N, N, N, 1, 4, N, 6, N },
				{ 7, 8, 9, 1, N, N, N, N, N, N, N, N },
				{ 0, 1, 2, 3, N, 11, N, N, N, 10, N, 12 }};
template<class T>
int MCubes<T>::insert5[][12] = {{ N, 5, N, N, 0, 1, 2, 3, N, 4, N, 6 },
				{ N, N, N, N, 7, 8, 9, N, N, N, N, N },
				{ N, N, N, N, N, N, N, 11, 10, N, 12, N }};
template<class T>
int MCubes<T>::lookup6[][12] = {{ 5, N, N, N, 2, N, N, N, 4, 6, N, N },
				{ 2, 7, 8, 9, N, N, N, N, N, N, N, N },
				{ 0, 1, 2, 3, N, N, 12, N, N, N, 10, 11 }};
template<class T>
int MCubes<T>::insert6[][12] = {{ N, N, 5, N, 0, 1, 2, 3, N, N, 4, 6 },
				{ N, N, N, N, N, 7, 8, 9, N, N, N, N },
				{ N, N, N, N, 12, N, N, N, 10, 11, N, N }};
template<class T>
int MCubes<T>::lookup7[][12] = {{ N, N, N, 5, N, N, N, 1, 4, N, 8, N },
				{ 12, N, N, N, 11, N, N, N, 8, 13, N, N },
				{ 6, 17, N, N, 2, 14, N, N, 7, 8, N, 18 },
				{ 2, 14, 20, 19, N, N, N, N, N, N, N, N },
				{ 11, 16, 15, 14, N, N, N, 21, 24, N, 25, N },
				{ 9, 10, 11, 1, N, N, 26, N, N, N, 24, 27 },
				{ 0, 1, 2, 3, N, 28, 23, N, N, 29, 22, 24 }};
template<class T>
int MCubes<T>::insert7[][12] = {{ N, 5, 6, N, 0, 1, 2, 3, N, 4, 7, 8 },
				{ N, N, 12, N, 9, 10, 11, N, N, N, N, 13 },
				{ N, N, N, 17, N, 16, 15, 14, N, N, 18, N },
				{ N, N, N, N, N, N, 20, 19, N, N, N, N },
				{ N, N, N, N, 23, 21, N, N, 22, 24, N, 25 },
				{ N, N, N, N, 26, N, N, N, N, 27, N, N },
				{ N, N, N, N, N, N, N, 28, 29, N, N, N }};



// MCubes
// Constructor
template<class T>
MCubes<T>::MCubes( Data<T> *field, int* X, int* Y, int* Z, 
		   float d1, float d2, float d3 )
  : data(field->values), Xarray(X), Yarray(Y), Zarray(Z)
{
  nx = field->nx;
  ny = field->ny;
  nz = field->nz;
  dx = d1; dy = d2; dz = d3;

  offseti = 1;
  offsetdx = dx;
  offsetdy = dy;
  offsetdz = dz;
  triangles = new GeomTriGroup();
} // MCubes

// reset
// given n (number of isosurface cells), create a GeomTriGroup that can
// hold the maximum number of triangles for that n.
template<class T>
void 
MCubes<T>::reset( int n ) {
  int numcells = ( n / (offseti*offseti)) + 1;
  triangles->reserve_clear( (int)(2.5*(float)numcells) );
} // reset

// setResolution
// sets the stride offset for multires traversal
template<class T>
void
MCubes<T>::setResolution( int res ) {
  offseti = (int)powf( 2.0, (float)res );
  float d = powf( 2.0, (float)res );
  offsetdx = dx*d;
  offsetdy = dy*d;
  offsetdz = dz*d;
  //  offsetd = powf( 2.0, (float)res );
} // setResolution

// interp
// choose which interpx function to call, based on how many cells 
// are contained in the leaf node  
template<class T>
void
MCubes<T>::interp( int x, int y, int z, int branching, float iso ) {
  switch( branching ) {
  case 0:
    interp1( x, y, z, iso );
    break;
  case 1:
    lookup = lookup1;  insert = insert1;
    interp2( x, y, z, iso, 1 );
    break;
  case 2:
    lookup = lookup2;  insert = insert2;
    interp2( x, y, z, iso, 2 );
    break;
  case 3:
    lookup = lookup3[0];  insert = insert3[0];
    interp4( x, y, z, iso, 3 );
    break;
  case 4:
    lookup = lookup4;  insert = insert4;
    interp2( x, y, z, iso, 4 );
    break;
  case 5:
    lookup = lookup5[0];  insert = insert5[0];
    interp4( x, y, z, iso, 5 );
    break;
  case 6:
    lookup = lookup6[0];  insert = insert6[0];
    interp4( x, y, z, iso, 6 );
    break;
  case 7:
    lookup = lookup7[0];  insert = insert7[0];
    interp8( x, y, z, iso );
    break;
  }; // switch
} // interp

// interp1
// interpolate leaf node with 1 cell
template<class T>
void
MCubes<T>::interp1( int x, int y, int z, float iso ) {
  int lox = Xarray[x];
  int hix = Xarray[x+offseti];
  int loy = Yarray[y];
  int hiy = Yarray[y+offseti];
  int loz = Zarray[z];
  int hiz = Zarray[z+offseti];
  
  //  T val[8];
  float val[8];

  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  float px = (float)x*dx;
  float py = (float)y*dy;
  float pz = (float)z*dz;
  
  pPoint v[8];
  v[0] = pPoint(px,            py,            pz);
  v[1] = pPoint((px+offsetdx), py,            pz);
  v[2] = pPoint((px+offsetdx), (py+offsetdy), pz);
  v[3] = pPoint(px,            (py+offsetdy), pz);
  v[4] = pPoint(px,            py,            (pz+offsetdz));
  v[5] = pPoint((px+offsetdx), py,            (pz+offsetdz));
  v[6] = pPoint((px+offsetdx), (py+offsetdy), (pz+offsetdz));
  v[7] = pPoint(px,            (py+offsetdy), (pz+offsetdz));

  TRIANGLE_CASES *tcase=triCases+mask;
  EDGE_LIST *edges=tcase->edges;
  pPoint p[12];

  int i;
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interp1

// interp2
// interpolate leaf node with 2 cells
template<class T>
void
MCubes<T>::interp2( int x, int y, int z, float iso, int casenum ) {
  int lox = Xarray[x];
  int hix = Xarray[x+offseti];
  int loy = Yarray[y];
  int hiy = Yarray[y+offseti];
  int loz = Zarray[z];
  int hiz = Zarray[z+offseti];
  
  //  T val[8];
  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;
 
  // do cell 0
  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  float px = (float)x*dx;
  float py = (float)y*dy;
  float pz = (float)z*dz;
  pPoint v[8];
  v[0] = pPoint(px,            py,            pz);
  v[1] = pPoint((px+offsetdx), py,            pz);
  v[2] = pPoint((px+offsetdx), (py+offsetdy), pz);
  v[3] = pPoint(px,            (py+offsetdy), pz);
  v[4] = pPoint(px,            py,            (pz+offsetdz));
  v[5] = pPoint((px+offsetdx), py,            (pz+offsetdz));
  v[6] = pPoint((px+offsetdx), (py+offsetdy), (pz+offsetdz));
  v[7] = pPoint(px,            (py+offsetdy), (pz+offsetdz));

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;
    
    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
      if( insert[ edges[i] ] != -1 )
	shared[ insert[ edges[i] ] ] = p[i];
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if (mask>0 && mask<255)

  // do cell 1
  switch( casenum ) {
  case 1: { x+=offseti; lox = Xarray[x]; hix = Xarray[x+offseti]; } break;
  case 2: { y+=offseti; loy = Yarray[y]; hiy = Yarray[y+offseti]; } break;
  case 4: { z+=offseti; loz = Zarray[z]; hiz = Zarray[z+offseti]; } break;
  };

  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  switch( casenum ) {
  case 1: { 
    v[0].x(v[0].x() + offsetdx); v[1].x(v[1].x() + offsetdx); 
    v[2].x(v[2].x() + offsetdx); v[3].x(v[3].x() + offsetdx); 
    v[4].x(v[4].x() + offsetdx); v[5].x(v[5].x() + offsetdx); 
    v[6].x(v[6].x() + offsetdx); v[7].x(v[7].x() + offsetdx); 
  } break;
  case 2: { 
    v[0].y(v[0].y() + offsetdy); v[1].y(v[1].y() + offsetdy); 
    v[2].y(v[2].y() + offsetdy); v[3].y(v[3].y() + offsetdy); 
    v[4].y(v[4].y() + offsetdy); v[5].y(v[5].y() + offsetdy); 
    v[6].y(v[6].y() + offsetdy); v[7].y(v[7].y() + offsetdy); 
  } break;
  case 4: { 
    v[0].z(v[0].z() + offsetdz); v[1].z(v[1].z() + offsetdz); 
    v[2].z(v[2].z() + offsetdz); v[3].z(v[3].z() + offsetdz); 
    v[4].z(v[4].z() + offsetdz); v[5].z(v[5].z() + offsetdz); 
    v[6].z(v[6].z() + offsetdz); v[7].z(v[7].z() + offsetdz); 
  } break;
  }; // switch( casenum )

  tcase=triCases+mask;
  edges=tcase->edges;

  for (i = 0; edges[i]>-1; i++) {
    if( lookup[ edges[i] ] != -1 )
      p[i] = shared[ lookup[ edges[i] ] ];
    else {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
    }
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interp2

// interp4
// interpolate leaf node with 4 cells
template<class T>
void
MCubes<T>::interp4( int x, int y, int z, float iso, int casenum ) {
  int lox = Xarray[x];
  int hix = Xarray[x+offseti];
  int loy = Yarray[y];
  int hiy = Yarray[y+offseti];
  int loz = Zarray[z];
  int hiz = Zarray[z+offseti];
  
  //  T val[8];
  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;
 
  // do cell 0
  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  float px = (float)x*dx;
  float py = (float)y*dy;
  float pz = (float)z*dz;
  pPoint v[8];
  v[0] = pPoint(px,            py,            pz);
  v[1] = pPoint((px+offsetdx), py,            pz);
  v[2] = pPoint((px+offsetdx), (py+offsetdy), pz);
  v[3] = pPoint(px,            (py+offsetdy), pz);
  v[4] = pPoint(px,            py,            (pz+offsetdz));
  v[5] = pPoint((px+offsetdx), py,            (pz+offsetdz));
  v[6] = pPoint((px+offsetdx), (py+offsetdy), (pz+offsetdz));
  v[7] = pPoint(px,            (py+offsetdy), (pz+offsetdz));

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;
    
    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
      if( insert[ edges[i] ] != -1 )
	shared[ insert[ edges[i] ] ] = p[i];
    }
    
    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if (mask>0 && mask<255)
  insert += 12;

  // do cells 1..2
  for( int iteration = 1; iteration < 3; iteration++ ) {
    if( iteration == 1 ) {
      switch( casenum ) {
      case 3:
      case 5: {
	x+=offseti; lox = Xarray[x]; hix = Xarray[x+offseti];
	v[0].x(v[0].x() + offsetdx); v[1].x(v[1].x() + offsetdx); 
	v[2].x(v[2].x() + offsetdx); v[3].x(v[3].x() + offsetdx); 
	v[4].x(v[4].x() + offsetdx); v[5].x(v[5].x() + offsetdx); 
	v[6].x(v[6].x() + offsetdx); v[7].x(v[7].x() + offsetdx); 
      } break;
      case 6: {
	y+=offseti; loy = Yarray[y]; hiy = Yarray[y+offseti];	
	v[0].y(v[0].y() + offsetdy); v[1].y(v[1].y() + offsetdy); 
	v[2].y(v[2].y() + offsetdy); v[3].y(v[3].y() + offsetdy); 
	v[4].y(v[4].y() + offsetdy); v[5].y(v[5].y() + offsetdy); 
	v[6].y(v[6].y() + offsetdy); v[7].y(v[7].y() + offsetdy); 
      } break;
      };
    } else {
      switch( casenum ) {
      case 3: {
	y+=offseti; 
	loy = Yarray[y]; hiy = Yarray[y+offseti];
	v[0].y(v[0].y() + offsetdy); v[1].y(v[1].y() + offsetdy); 
	v[2].y(v[2].y() + offsetdy); v[3].y(v[3].y() + offsetdy); 
	v[4].y(v[4].y() + offsetdy); v[5].y(v[5].y() + offsetdy); 
	v[6].y(v[6].y() + offsetdy); v[7].y(v[7].y() + offsetdy); 
      } break;
      case 5: 
      case 6: {
	z+=offseti;
	loz = Zarray[z]; hiz = Zarray[z+offseti];
	v[0].z(v[0].z() + offsetdz); v[1].z(v[1].z() + offsetdz); 
	v[2].z(v[2].z() + offsetdz); v[3].z(v[3].z() + offsetdz); 
	v[4].z(v[4].z() + offsetdz); v[5].z(v[5].z() + offsetdz); 
	v[6].z(v[6].z() + offsetdz); v[7].z(v[7].z() + offsetdz); 
      } break;
      };
    } // if( iteration == 1 )
    
    val[0]=(float)data[lox+loy+loz] - iso;
    val[1]=(float)data[hix+loy+loz] - iso;
    val[2]=(float)data[hix+hiy+loz] - iso;
    val[3]=(float)data[lox+hiy+loz] - iso;
    val[4]=(float)data[lox+loy+hiz] - iso;
    val[5]=(float)data[hix+loy+hiz] - iso;
    val[6]=(float)data[hix+hiy+hiz] - iso;
    val[7]=(float)data[lox+hiy+hiz] - iso;

    mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }
    
    if (mask>0 && mask<255) {
      
      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	if( lookup[ edges[i] ] != -1 )
	  p[i] = shared[ lookup[ edges[i] ] ];
	else {
	  int v1 = edge_table[edges[i]][0];
	  int v2 = edge_table[edges[i]][1];
	  p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
	  if( insert[ edges[i] ] != -1 )
	    shared[ insert[ edges[i] ] ] = p[i];
	}
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if (mask>0 && mask<255)
    lookup += 12;
    insert += 12;
  } // iteration = 1..2
  

  // do cell 3
  switch( casenum ) {
  case 3: 
  case 5: { x-=offseti; lox = Xarray[x]; hix = Xarray[x+offseti]; } break;
  case 6: { y-=offseti; loy = Yarray[y]; hiy = Yarray[y+offseti]; } break;
  };

  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  switch( casenum ) {
  case 3:
  case 5: { 
    v[0].x(v[0].x() - offsetdx); v[1].x(v[1].x() - offsetdx); 
    v[2].x(v[2].x() - offsetdx); v[3].x(v[3].x() - offsetdx); 
    v[4].x(v[4].x() - offsetdx); v[5].x(v[5].x() - offsetdx); 
    v[6].x(v[6].x() - offsetdx); v[7].x(v[7].x() - offsetdx); 
  } break;
  case 6: { 
    v[0].y(v[0].y() - offsetdy); v[1].y(v[1].y() - offsetdy); 
    v[2].y(v[2].y() - offsetdy); v[3].y(v[3].y() - offsetdy); 
    v[4].y(v[4].y() - offsetdy); v[5].y(v[5].y() - offsetdy); 
    v[6].y(v[6].y() - offsetdy); v[7].y(v[7].y() - offsetdy); 
  } break;
  }; // switch( casenum )

  tcase=triCases+mask;
  edges=tcase->edges;

  for (i = 0; edges[i]>-1; i++) {
    if( lookup[ edges[i] ] != -1 )
      p[i] = shared[ lookup[ edges[i] ] ];
    else {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
    }
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interp4

// interp8
// interpolate leaf node with 8 cells
template<class T>
void
MCubes<T>::interp8( int x, int y, int z, float iso ) {
  int lox = Xarray[x];
  int hix = Xarray[x+offseti];
  int loy = Yarray[y];
  int hiy = Yarray[y+offseti];
  int loz = Zarray[z];
  int hiz = Zarray[z+offseti];

  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  float px = (float)x*dx;
  float py = (float)y*dy;
  float pz = (float)z*dz;
  pPoint v[8];
  v[0] = pPoint(px,            py,            pz);
  v[1] = pPoint((px+offsetdx), py,            pz);
  v[2] = pPoint((px+offsetdx), (py+offsetdy), pz);
  v[3] = pPoint(px,            (py+offsetdy), pz);
  v[4] = pPoint(px,            py,            (pz+offsetdz));
  v[5] = pPoint((px+offsetdx), py,            (pz+offsetdz));
  v[6] = pPoint((px+offsetdx), (py+offsetdy), (pz+offsetdz));
  v[7] = pPoint(px,            (py+offsetdy), (pz+offsetdz));

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;
    
    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
      if( insert[ edges[i] ] != -1 )
	shared[ insert[ edges[i] ] ] = p[i];
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if (mask>0 && mask<255)
  insert += 12;

  // do cells 1..6
  float xa[]={0,offsetdx,0,       -offsetdx, 0,        offsetdx, 0        };
  float ya[]={0,0,       offsetdy, 0,        0,        0,        -offsetdy };
  float za[]={0,0,       0,        0,        offsetdz, 0,        0        };

  for( int iteration = 1; iteration < 7; iteration++ ) {
    if( xa[iteration] != 0 ) {
      x += (int)(xa[iteration]/dx);
      lox = Xarray[x]; 
      hix = ( x+offseti >= nx ) ? Xarray[nx-1] : Xarray[x+offseti];
      v[0].x(v[0].x() + xa[iteration]); v[1].x( v[1].x() + xa[iteration]); 
      v[2].x(v[2].x() + xa[iteration]); v[3].x( v[3].x() + xa[iteration]); 
      v[4].x(v[4].x() + xa[iteration]); v[5].x( v[5].x() + xa[iteration]); 
      v[6].x(v[6].x() + xa[iteration]); v[7].x( v[7].x() + xa[iteration]);
    }

    if( ya[iteration] != 0 ) {
      y += (int)(ya[iteration]/dy);
      loy = Yarray[y]; 
      hiy = ( y+offseti >= ny ) ? Yarray[ny-1] : Yarray[y+offseti];
      v[0].y(v[0].y() + ya[iteration]); v[1].y(v[1].y() + ya[iteration]); 
      v[2].y(v[2].y() + ya[iteration]); v[3].y(v[3].y() + ya[iteration]); 
      v[4].y(v[4].y() + ya[iteration]); v[5].y(v[5].y() + ya[iteration]); 
      v[6].y(v[6].y() + ya[iteration]); v[7].y(v[7].y() + ya[iteration]);
    }

    if( za[iteration] != 0 ) {
      z += (int)(za[iteration]/dz);
      loz = Zarray[z]; 
      hiz = ( z+offseti >= nz ) ? Zarray[nz-1] : Zarray[z+offseti];
      v[0].z(v[0].z() + za[iteration]); v[1].z(v[1].z() + za[iteration]); 
      v[2].z(v[2].z() + za[iteration]); v[3].z(v[3].z() + za[iteration]); 
      v[4].z(v[4].z() + za[iteration]); v[5].z(v[5].z() + za[iteration]); 
      v[6].z(v[6].z() + za[iteration]); v[7].z(v[7].z() + za[iteration]);
    }
    
    val[0]=(float)data[lox+loy+loz] - iso;
    val[1]=(float)data[hix+loy+loz] - iso;
    val[2]=(float)data[hix+hiy+loz] - iso;
    val[3]=(float)data[lox+hiy+loz] - iso;
    val[4]=(float)data[lox+loy+hiz] - iso;
    val[5]=(float)data[hix+loy+hiz] - iso;
    val[6]=(float)data[hix+hiy+hiz] - iso;
    val[7]=(float)data[lox+hiy+hiz] - iso;

    mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }

    if (mask>0 && mask<255) {

      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	if( lookup[ edges[i] ] != -1 ) 
	  p[i] = shared[ lookup[ edges[i] ] ];
	else {
	  int v1 = edge_table[edges[i]][0];
	  int v2 = edge_table[edges[i]][1];
	  p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
	  if( insert[ edges[i] ] != -1 ) 
	    shared[ insert[ edges[i] ] ] = p[i];
	}
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if (mask>0 && mask<255)
    lookup += 12;
    insert += 12;
  } // iteration = 1..6
  
  // do cell 7
  x -= offseti;
  lox = Xarray[x]; hix = Xarray[x+offseti];
  
  val[0]=(float)data[lox+loy+loz] - iso;
  val[1]=(float)data[hix+loy+loz] - iso;
  val[2]=(float)data[hix+hiy+loz] - iso;
  val[3]=(float)data[lox+hiy+loz] - iso;
  val[4]=(float)data[lox+loy+hiz] - iso;
  val[5]=(float)data[hix+loy+hiz] - iso;
  val[6]=(float)data[hix+hiy+hiz] - iso;
  val[7]=(float)data[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0].x(v[0].x() - offsetdx); v[1].x(v[1].x() - offsetdx); 
  v[2].x(v[2].x() - offsetdx); v[3].x(v[3].x() - offsetdx); 
  v[4].x(v[4].x() - offsetdx); v[5].x(v[5].x() - offsetdx); 
  v[6].x(v[6].x() - offsetdx); v[7].x(v[7].x() - offsetdx);

  tcase=triCases+mask;
  edges=tcase->edges;

  for (i = 0; edges[i]>-1; i++) {
    if( lookup[ edges[i] ] != -1 ) 
      p[i] = shared[ lookup[ edges[i] ] ];
    else {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
    }
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interp8


// MCubesUG
// Constructor
template <class T>
MCubesUG<T>::MCubesUG( DataUG<T> *d ) : tets(d->tets), verts(d->verts),
  values(d->values)
{
} // MCubesUG

// reset
// given n (number of isosurface cells), create a GeomTriGroup that can
// hold the maximum number of triangles for that n.
template <class T>
void
MCubesUG<T>::reset( int n ) {
  triangles = new GeomTriGroup(2*n);
} // reset

// interp
// interpolate a single tetrahedral cell
template <class T>
void
MCubesUG<T>::interp( int cell, float iso ) {
  T val[4];
  val[0] = values[ tets[cell].v[0] ] - iso;
  val[1] = values[ tets[cell].v[1] ] - iso;
  val[2] = values[ tets[cell].v[2] ] - iso;
  val[3] = values[ tets[cell].v[3] ] - iso;

  int mask = 0;
  for( int idx = 0; idx < 4; idx++ ) {
    if( val[idx] < 0 )
      mask |= 1 << idx;
  }

  if( mask == 0 || mask == 15 ) 
    return;

  TRIANGLE_CASES_UG* tcase = triCasesUG + mask;
  EDGE_LIST* edges = tcase->edges;
  pPoint p[4];
  
  int j;
  for( j = 0; edges[j] > -1; j++ ) {
    int v1 = edge_tableUG[ edges[j] ][0];
    int v2 = edge_tableUG[ edges[j] ][1];
    p[j] = interpolate( verts[ tets[cell].v[v1] ].pos,
			verts[ tets[cell].v[v2] ].pos,
			val[v1]/(val[v1] -  val[v2]));
  }
  
  for( j = 0; j < tcase->numTris; j++ ) {
    int tri = 3*j;
    triangles->add( p[ tcase->points[tri] ],
		    p[ tcase->points[tri+1] ],
		    p[ tcase->points[tri+2] ] );
  } // j = 0 .. tcase->numTris-1

} // interp

template <class T>
void
MCubesUG<T>::shift( float x, float y, float z ) {
  triangles->shift( x, y, z );
}


// MCubesCL - constructor
template <class T>
MCubesCL<T>::MCubesCL( DataCL<T>** d, int n, int** X, int** Y, int** Z ) 
  : data(d), Xarray(X), Yarray(Y), Zarray(Z) {
  triangles = new GeomTriGroup();
}

// reset
// given n (number of isosurface cells), create a GeomTriGroup that can
// hold the maximum number of triangles for that n.
template <class T>
void
MCubesCL<T>::reset( int n ) {
  int numcells = (n / (offseti*offseti)) + 1;
  //  triangles = new GeomTriGroup(4*numcells);
  triangles->reserve_clear( (int)(2.5*(float)numcells) );
} // reset

// setResolution
// sets the stride offset for multires traversal
template<class T>
void
MCubesCL<T>::setResolution( int res ) {
  offseti = (int)powf( 2.0, (float)res );
} // setResolution


// interpCircular
// determine which interpCircularx function to call, based on branching
template <class T>
void
MCubesCL<T>::interpCircular( int zone, int x, int y, int z, float iso,
			     int branching ) {
  switch( branching ) {
  case 0:
    interpCircular1( zone, x, y, z, iso );
    break;
  case 1:
    interpCircular2( zone, x, y, z, iso, 1 );
    break;
  case 2:
    interpCircular2( zone, x, y, z, iso, 2 );
    break;
  case 3:
    interpCircular4( zone, x, y, z, iso, 3 );
    break;
  case 4:
    interpCircular2( zone, x, y, z, iso, 4 );
    break;
  case 5:
    interpCircular4( zone, x, y, z, iso, 5 );
    break;
  case 6:
    interpCircular4( zone, x, y, z, iso, 6 );
    break;
  case 7:
    interpCircular8( zone, x, y, z, iso );
    break;
  }; // switch  
} // interpCircular

// interpRegular
// determine which interpRegularx function to call, based on branching
template <class T>
void
MCubesCL<T>::interpRegular( int zone, int x, int y, int z, float iso,
			    int branching ) {
  switch( branching ) {
  case 0:
    interpRegular1( zone, x, y, z, iso );
    break;
  case 1:
    interpRegular2( zone, x, y, z, iso, 1 );
    break;
  case 2:
    interpRegular2( zone, x, y, z, iso, 2 );
    break;
  case 3:
    interpRegular4( zone, x, y, z, iso, 3 );
    break;
  case 4:
    interpRegular2( zone, x, y, z, iso, 4 );
    break;
  case 5:
    interpRegular4( zone, x, y, z, iso, 5 );
    break;
  case 6:
    interpRegular4( zone, x, y, z, iso, 6 );
    break;
  case 7:
    interpRegular8( zone, x, y, z, iso );
    break;
  }; // switch  
} // interpRegular


// interpCircular1
// interpolate leaf node with 1 cell (possibly circular)
template <class T>
void
MCubesCL<T>::interpCircular1( int zone, int x, int y, int z, float iso ) {
  int lox = Xarray[zone][x];
  int hix = ( x+offseti == data[zone]->nx ) ? 
    Xarray[zone][0] : Xarray[zone][x+offseti];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+offseti];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+offseti];

  T val[8];

  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  TRIANGLE_CASES *tcase=triCases+mask;
  EDGE_LIST *edges=tcase->edges;
  pPoint p[12];

  int i;
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpCircular1

// interpCircular2
// interpolate leaf node with 2 cells (possibly circular)
template <class T>
void
MCubesCL<T>::interpCircular2( int zone, int x, int y, int z, float iso, 
			      int casenum ) {
  int lox = Xarray[zone][x];
  int hix = ( x+offseti == data[zone]->nx ) ? 
    Xarray[zone][0] : Xarray[zone][x+offseti];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+offseti];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+offseti];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cell 1
  switch( casenum ) {
  case 1: { 
    x++; lox = Xarray[zone][x]; 
    hix = ( x+offseti == data[zone]->nx ) ? 
      Xarray[zone][0] : Xarray[zone][x+offseti];
  } break;
  case 2: { y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti]; } break;
  case 4: { z++; loz = Zarray[zone][z]; hiz = Zarray[zone][z+offseti]; } break;
  };
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;
 
  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpCircular2


// interpCircular4
// interpolate leaf node with 4 cells (possibly circular)
template <class T>
void
MCubesCL<T>::interpCircular4( int zone, int x, int y, int z, float iso, 
			      int casenum ) {
  int lox = Xarray[zone][x];
  int hix = ( x+offseti == data[zone]->nx ) ? 
    Xarray[zone][0] : Xarray[zone][x+offseti];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+offseti];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+offseti];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cells 1..2
  for( int iteration = 1; iteration < 3; iteration++ ) {
    if( iteration == 1 ) {
      switch( casenum ) {
      case 3:
      case 5: {
	x++; lox = Xarray[zone][x];
	hix = ( x+offseti == data[zone]->nx ) ? 
	  Xarray[zone][0] : Xarray[zone][x+offseti];
      } break;
      case 6: {
	y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti];
      } break;
      };
    } else {
      switch( casenum ) {
      case 3: {
	x--; y++;
	lox = Xarray[zone][x];
	hix = ( x+offseti == data[zone]->nx ) ? 
	  Xarray[zone][0] : Xarray[zone][x+offseti];
	loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti];
      } break;
      case 5: {
	x--; z++;
	lox = Xarray[zone][x];
	hix = ( x+offseti == data[zone]->nx ) ? 
	  Xarray[zone][0] : Xarray[zone][x+offseti];
	loz = Zarray[zone][z]; hiz = Zarray[zone][z+offseti];
      } break;
      case 6: {
	y--; z++;
	loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti];
	loz = Zarray[zone][z]; hiz = Zarray[zone][z+offseti];
      } break;
      };
    } // if( interation == 1 )    

    v[0]=&data[zone]->verts[lox+loy+loz];
    v[1]=&data[zone]->verts[hix+loy+loz];
    v[2]=&data[zone]->verts[hix+hiy+loz];
    v[3]=&data[zone]->verts[lox+hiy+loz];
    v[4]=&data[zone]->verts[lox+loy+hiz];
    v[5]=&data[zone]->verts[hix+loy+hiz];
    v[6]=&data[zone]->verts[hix+hiy+hiz];
    v[7]=&data[zone]->verts[lox+hiy+hiz];

    val[0]=data[zone]->values[lox+loy+loz] - iso;
    val[1]=data[zone]->values[hix+loy+loz] - iso;
    val[2]=data[zone]->values[hix+hiy+loz] - iso;
    val[3]=data[zone]->values[lox+hiy+loz] - iso;
    val[4]=data[zone]->values[lox+loy+hiz] - iso;
    val[5]=data[zone]->values[hix+loy+hiz] - iso;
    val[6]=data[zone]->values[hix+hiy+hiz] - iso;
    val[7]=data[zone]->values[lox+hiy+hiz] - iso;
   
    int mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }

    if (mask>0 && mask<255) {
      
      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	int v1 = edge_table[edges[i]][0];
	int v2 = edge_table[edges[i]][1];
	p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
      }

      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if(mask>0 && mask<255)

  } // iteration = 1..2

  // do cell 3
  switch( casenum ) {
  case 3: 
  case 5: {
    x++; lox = Xarray[zone][x]; 
    hix = ( x+offseti == data[zone]->nx ) ? 
      Xarray[zone][0] : Xarray[zone][x+offseti];
  } break;
  case 6: { y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti]; } break;
  };
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpCircular4


// interpCircular8
// interpolate leaf node with 8 cells (possibly circular)
template <class T>
void
MCubesCL<T>::interpCircular8( int zone, int x, int y, int z, float iso ) {
  int lox = Xarray[zone][x];
  int hix = ( x+offseti == data[zone]->nx ) ? 
    Xarray[zone][0] : Xarray[zone][x+offseti];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+offseti];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+offseti];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cells 1..6
  int xa[] = {0, offseti, -offseti, offseti, -offseti, offseti, -offseti};
  int ya[] = {0, 0, offseti, 0, -offseti, 0, offseti};
  int za[] = {0, 0, 0, 0, offseti, 0, 0};

  for( int iteration = 1; iteration < 7; iteration++ ) {
    x += xa[iteration];
    lox = Xarray[zone][x]; 
    hix = ( x+offseti == data[zone]->nx ) ? 
      Xarray[zone][0] : Xarray[zone][x+offseti];
    
    if( ya[iteration] != 0 ) {
      y += ya[iteration];
      loy = Yarray[zone][y]; hiy = Yarray[zone][y+offseti];
    }
    if( za[iteration] != 0 ) {
      z += (int)za[iteration];
      loz = Zarray[zone][z]; hiz = Zarray[zone][z+offseti];
    }

    v[0]=&data[zone]->verts[lox+loy+loz];
    v[1]=&data[zone]->verts[hix+loy+loz];
    v[2]=&data[zone]->verts[hix+hiy+loz];
    v[3]=&data[zone]->verts[lox+hiy+loz];
    v[4]=&data[zone]->verts[lox+loy+hiz];
    v[5]=&data[zone]->verts[hix+loy+hiz];
    v[6]=&data[zone]->verts[hix+hiy+hiz];
    v[7]=&data[zone]->verts[lox+hiy+hiz];
    
    val[0]=data[zone]->values[lox+loy+loz] - iso;
    val[1]=data[zone]->values[hix+loy+loz] - iso;
    val[2]=data[zone]->values[hix+hiy+loz] - iso;
    val[3]=data[zone]->values[lox+hiy+loz] - iso;
    val[4]=data[zone]->values[lox+loy+hiz] - iso;
    val[5]=data[zone]->values[hix+loy+hiz] - iso;
    val[6]=data[zone]->values[hix+hiy+hiz] - iso;
    val[7]=data[zone]->values[lox+hiy+hiz] - iso;

    int mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }

    if (mask>0 && mask<255) {
      
      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	int v1 = edge_table[edges[i]][0];
	int v2 = edge_table[edges[i]][1];
	p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if(mask>0 && mask<255)

  } // iteration = 1..6

  // do cell 7
  x++; lox = Xarray[zone][x]; 
  hix = ( x+offseti == data[zone]->nx ) ? 
    Xarray[zone][0] : Xarray[zone][x+offseti];
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpCircular8

// interpRegular1
// interpolate leaf node with 1 cell (not circular)
template <class T>
void
MCubesCL<T>::interpRegular1( int zone, int x, int y, int z, float iso ) {
  int lox = Xarray[zone][x];
  int hix = Xarray[zone][x+1];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+1];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+1];

  T val[8];

  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  TRIANGLE_CASES *tcase=triCases+mask;
  EDGE_LIST *edges=tcase->edges;
  pPoint p[12];

  int i;
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }

  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpRegular1

// interpRegular2
// interpolate leaf node with 2 cells (not circular)
template <class T>
void
MCubesCL<T>::interpRegular2( int zone, int x, int y, int z, float iso, 
			     int casenum ) {
  int lox = Xarray[zone][x];
  int hix = Xarray[zone][x+1];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+1];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+1];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cell 1
  switch( casenum ) {
  case 1: { x++; lox = Xarray[zone][x]; hix = Xarray[zone][x+1]; } break;
  case 2: { y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+1]; } break;
  case 4: { z++; loz = Zarray[zone][z]; hiz = Zarray[zone][z+1]; } break;
  };
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;
 
  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpRegular2

// interpRegular4
// interpolate leaf node with 4 cells (not circular)
template <class T>
void
MCubesCL<T>::interpRegular4( int zone, int x, int y, int z, float iso, 
			     int casenum ) {
  int lox = Xarray[zone][x];
  int hix = Xarray[zone][x+1];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+1];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+1];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cells 1..2
  for( int iteration = 1; iteration < 3; iteration++ ) {
    if( iteration == 1 ) {
      switch( casenum ) {
      case 3:
      case 5: {
	x++; lox = Xarray[zone][x]; hix = Xarray[zone][x+1];
      } break;
      case 6: {
	y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+1];
      } break;
      };
    } else {
      switch( casenum ) {
      case 3: {
	x--; y++;
	lox = Xarray[zone][x]; hix = Xarray[zone][x+1];
	loy = Yarray[zone][y]; hiy = Yarray[zone][y+1];
      } break;
      case 5: {
	x--; z++;
	lox = Xarray[zone][x]; hix = Xarray[zone][x+1];
	loz = Zarray[zone][z]; hiz = Zarray[zone][z+1];
      } break;
      case 6: {
	y--; z++;
	loy = Yarray[zone][y]; hiy = Yarray[zone][y+1];
	loz = Zarray[zone][z]; hiz = Zarray[zone][z+1];
      } break;
      };
    } // if( interation == 1 )    

    v[0]=&data[zone]->verts[lox+loy+loz];
    v[1]=&data[zone]->verts[hix+loy+loz];
    v[2]=&data[zone]->verts[hix+hiy+loz];
    v[3]=&data[zone]->verts[lox+hiy+loz];
    v[4]=&data[zone]->verts[lox+loy+hiz];
    v[5]=&data[zone]->verts[hix+loy+hiz];
    v[6]=&data[zone]->verts[hix+hiy+hiz];
    v[7]=&data[zone]->verts[lox+hiy+hiz];

    val[0]=data[zone]->values[lox+loy+loz] - iso;
    val[1]=data[zone]->values[hix+loy+loz] - iso;
    val[2]=data[zone]->values[hix+hiy+loz] - iso;
    val[3]=data[zone]->values[lox+hiy+loz] - iso;
    val[4]=data[zone]->values[lox+loy+hiz] - iso;
    val[5]=data[zone]->values[hix+loy+hiz] - iso;
    val[6]=data[zone]->values[hix+hiy+hiz] - iso;
    val[7]=data[zone]->values[lox+hiy+hiz] - iso;
   
    int mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }

    if (mask>0 && mask<255) {
      
      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	int v1 = edge_table[edges[i]][0];
	int v2 = edge_table[edges[i]][1];
	p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if(mask>0 && mask<255)

  } // iteration = 1..2

  // do cell 3
  switch( casenum ) {
  case 3: 
  case 5: { x++; lox = Xarray[zone][x]; hix = Xarray[zone][x+1]; } break;
  case 6: { y++; loy = Yarray[zone][y]; hiy = Yarray[zone][y+1]; } break;
  };
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpRegular4

// interpRegular8
// interpolate leaf node with 8 cells (not circular)
template <class T>
void
MCubesCL<T>::interpRegular8( int zone, int x, int y, int z, float iso ) {
  int lox = Xarray[zone][x];
  int hix = Xarray[zone][x+1];
  int loy = Yarray[zone][y];
  int hiy = Yarray[zone][y+1];
  int loz = Zarray[zone][z];
  int hiz = Zarray[zone][z+1];

  T val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  VertexCL* v[8];
  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

    for (i = 0; edges[i]>-1; i++) {
      int v1 = edge_table[edges[i]][0];
      int v2 = edge_table[edges[i]][1];
      p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
    }

    for( i = 0; i < tcase->numTris; i++ ) {
      triangles->add( p[ tcase->triangles[i][0] ],
		      p[ tcase->triangles[i][1] ],
		      p[ tcase->triangles[i][2] ] );
    }
  } // if(mask>0 && mask<255)

  // do cells 1..6
  int xa[] = {0, 1, -1, 1, -1, 1, -1};
  int ya[] = {0, 0, 1, 0, -1, 0, 1};
  int za[] = {0, 0, 0, 0, 1, 0, 0};

  for( int iteration = 1; iteration < 7; iteration++ ) {
    x += xa[iteration];
    lox = Xarray[zone][x]; hix = Xarray[zone][x+1];
    
    if( ya[iteration] != 0 ) {
      y += ya[iteration];
      loy = Yarray[zone][y]; hiy = Yarray[zone][y+1];
    }
    if( za[iteration] != 0 ) {
      z += (int)za[iteration];
      loz = Zarray[zone][z]; hiz = Zarray[zone][z+1];
    }

    v[0]=&data[zone]->verts[lox+loy+loz];
    v[1]=&data[zone]->verts[hix+loy+loz];
    v[2]=&data[zone]->verts[hix+hiy+loz];
    v[3]=&data[zone]->verts[lox+hiy+loz];
    v[4]=&data[zone]->verts[lox+loy+hiz];
    v[5]=&data[zone]->verts[hix+loy+hiz];
    v[6]=&data[zone]->verts[hix+hiy+hiz];
    v[7]=&data[zone]->verts[lox+hiy+hiz];
    
    val[0]=data[zone]->values[lox+loy+loz] - iso;
    val[1]=data[zone]->values[hix+loy+loz] - iso;
    val[2]=data[zone]->values[hix+hiy+loz] - iso;
    val[3]=data[zone]->values[lox+hiy+loz] - iso;
    val[4]=data[zone]->values[lox+loy+hiz] - iso;
    val[5]=data[zone]->values[hix+loy+hiz] - iso;
    val[6]=data[zone]->values[hix+hiy+hiz] - iso;
    val[7]=data[zone]->values[lox+hiy+hiz] - iso;

    int mask=0;
    for(idx=0;idx<8;idx++){
      if(val[idx]<0)
	mask|=1<<idx;
    }

    if (mask>0 && mask<255) {
      
      tcase=triCases+mask;
      edges=tcase->edges;
      
      for (i = 0; edges[i]>-1; i++) {
	int v1 = edge_table[edges[i]][0];
	int v2 = edge_table[edges[i]][1];
	p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if(mask>0 && mask<255)

  } // iteration = 1..6

  // do cell 7
  x++; lox = Xarray[zone][x]; hix = Xarray[zone][x+1];
  
  val[0]=data[zone]->values[lox+loy+loz] - iso;
  val[1]=data[zone]->values[hix+loy+loz] - iso;
  val[2]=data[zone]->values[hix+hiy+loz] - iso;
  val[3]=data[zone]->values[lox+hiy+loz] - iso;
  val[4]=data[zone]->values[lox+loy+hiz] - iso;
  val[5]=data[zone]->values[hix+loy+hiz] - iso;
  val[6]=data[zone]->values[hix+hiy+hiz] - iso;
  val[7]=data[zone]->values[lox+hiy+hiz] - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0]=&data[zone]->verts[lox+loy+loz];
  v[1]=&data[zone]->verts[hix+loy+loz];
  v[2]=&data[zone]->verts[hix+hiy+loz];
  v[3]=&data[zone]->verts[lox+hiy+loz];
  v[4]=&data[zone]->verts[lox+loy+hiz];
  v[5]=&data[zone]->verts[hix+loy+hiz];
  v[6]=&data[zone]->verts[hix+hiy+hiz];
  v[7]=&data[zone]->verts[lox+hiy+hiz];

  tcase=triCases+mask;
  edges=tcase->edges;
  
  for (i = 0; edges[i]>-1; i++) {
    int v1 = edge_table[edges[i]][0];
    int v2 = edge_table[edges[i]][1];
    p[i]=interpolate(v[v1]->pos, v[v2]->pos, val[v1]/(val[v1] -  val[v2]));
  }
  
  for( i = 0; i < tcase->numTris; i++ ) {
    triangles->add( p[ tcase->triangles[i][0] ],
		    p[ tcase->triangles[i][1] ],
		    p[ tcase->triangles[i][2] ] );
  }
} // interpRegular8

} // End namespace Phil


#endif

