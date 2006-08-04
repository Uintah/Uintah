
/*
  mcubeBONO.h
  declarations for the class that does Marching Cubes-style
    interpolation and triangle construction for the BONO classes.

  Packages/Philip Sutton

  Copyright (C) 2000 SCI Group, University of Utah

*/

#ifndef mcube_bono_h
#define mcube_bono_h

#include "TriGroup.h"
#include "mcube_table.h"

#include <stdio.h>
#include <stdlib.h>

namespace Phil {
using namespace SCIRun;


template<class T> struct Data;
struct VertexCL;
template<class T> struct DataCL;

// MCubesBONO
// Marching Cubes class for BONO
template<class T>
class MCubesBono {
public:
  int nx, ny, nz;
 
  int i001, i010, i011, i100, i101, i110, i111;

  GeomTriGroup *triangles;
  T *data;

public:
  MCubesBono();
  MCubesBono( Data<T> * );
  
  void reset( int n );
  void interp( int* array, int branching, float iso, int x, int y, int z );

  friend pPoint interpolate( const Point &p, const Point& q, float f);
private:
  void interp1( int *array, float iso, int x, int y, int z );
  void interp2( int *array, float iso, int x, int y, int z, int casenum );
  void interp4( int *array, float iso, int x, int y, int z, int casenum);
  void interp8( int *array, float iso, int x, int y, int z );
};

// MCubesCL
// Marching Cubes class for curvilinear grids
template<class T>
class MCubesBonoCL {
public:
  MCubesBonoCL( DataCL<T>** d, int n );

  void reset( int n );
  void interpCircular( int zone, int cell, double iso );
  void interpRegular( int zone, int cell, double iso );

  friend pPoint interpolate( const float* p, const float* q, double f);
  
public:
  GeomTriGroup* triangles;

private:
  DataCL<T>** data;
  int numzones;

  int* i001; int* i010; int* i011; int* i100; int* i101; int* i110; int* i111;
  int* c001; int* c010; int* c011; int* c100; int* c101; int* c110; int* c111;
  float* inv_nx;
};


inline
pPoint interpolate( const pPoint &p, const pPoint &q, float f )
{
  return pPoint( p.x() + f*(q.x()-p.x()), 
		 p.y() + f*(q.y()-p.y()), 
		 p.z() + f*(q.z()-p.z()) );
}

// included .cc file

/*
template<class T>
int MCubes<T>::lookup1[] = {-1, -1, -1, 0, -1, -1, -1, 2, 1, -1, 3, -1};
template<class T>
int MCubes<T>::putin1[]  = {-1, 0, -1, -1, -1, 2, -1, -1, -1, 1, -1, 3};
template<class T>
int MCubes<T>::lookup2[] = {0, -1, -1, -1, 2, -1, -1, -1, 3, 1, -1, -1};
template<class T>
int MCubes<T>::putin2[]  = {-1, -1, 0, -1, -1, -1, 2, -1, -1, -1, 3, 1};
template<class T>
int MCubes<T>::lookup4[] = {0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1};
template<class T>
int MCubes<T>::putin4[]  = {-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1};
template<class T>
int MCubes<T>::lookup3[][12] = {{-1, -1, -1, 1, -1, -1, -1, 2, 0, -1, 3, -1},
				{4, -1, -1, -1, 6, -1, -1, -1, 5, 3, -1, -1},
				{7, -1, -1, 10, 9, -1, -1, 12, 3, 8, 11, -1} };
template<class T>
int MCubes<T>::putin3[][12]={{-1, 1, 4, -1, -1, 2, 6, -1, -1, 0, 5, 3},
			     {-1, -1, 7, -1, -1, -1, 9, -1, -1, -1, -1, 8},
			     {-1, 10, -1, -1, -1, 12, -1, -1, -1, -1, -1, 11}};
template<class T>
int MCubes<T>::lookup5[][12] = {{-1, -1, -1, 1, -1, -1, -1, 2, 0, -1, 3, -1},
				{4, 2, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1},
				{7, 8, 9, 2, -1, -1, -1, 11, 10, -1, 12, -1}};
template<class T>
int MCubes<T>::putin5[][12]={{-1, 1, -1, -1, 4, 2, 6, 5, -1, 0, -1, 3},
			     {-1, -1, -1, -1, 7, 8, 9, -1, -1, -1, -1, -1},
			     {-1, -1, -1, -1, -1, 11, -1, -1, -1, 10, -1, 12}};
template<class T>
int MCubes<T>::lookup6[][12] = {{5, -1, -1, -1, 2, -1, -1, -1, 4, 6, -1, -1},
				{0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1},
				{2, 9, 8, 7, 11, -1, -1, -1, 10, 12, -1, -1}};
template<class T>
int MCubes<T>::putin6[][12]={{-1, -1, 5, -1, 0, 1, 2, 3, -1, -1, 4, 6},
			     {-1, -1, -1, -1, -1, 9, 8, 7, -1, -1, -1, -1},
			     {-1, -1, -1, -1, -1, -1, 11, -1, -1, -1, 10, 12}};
template<class T>
int MCubes<T>::lookup7[][12]={{-1, -1, -1, 1, -1, -1, -1, 3, 0, -1, 2, -1},
			      {6, -1, -1, -1, 8, -1, -1, -1, 7, 2, -1, -1},
			      {1, -1, -1, 14, 11, -1, -1, 16, 2, 10, 15, -1},
			      {4, 3, 8, 5, -1, -1, -1, -1, -1, -1, -1, -1},
			      {13, 12, 11, 3, -1, -1, -1, 22, 21, -1, 23, -1},
			      {8, 16, 17, 18, 24, -1, -1, -1, 25, 23, -1, -1},
			      {11, 20, 19, 16, 27, -1, -1, 28, 23, 26, 29, -1}};
template<class T>
int MCubes<T>::putin7[][12] ={{-1, 1, 6, -1, 4, 3, 8, 5, -1, 0, 7, 2},
			      {-1, -1, 9, -1, 13, 12, 11, -1, -1, -1, -1, 10},
			      {-1, 14, -1, -1, -1, 16, 17, 18, -1, -1, -1, 15},
			      {-1, -1, -1, -1, -1, 20, 19, -1, -1, -1, -1, -1},
			      {-1, -1, -1, -1, -1, 22, 24, -1, -1, 21, 25, 23},
			      {-1, -1, -1, -1, -1, -1, 27, -1, -1, -1, -1, 26},
			      {-1, -1, -1, -1, -1, 28, -1, -1, -1, -1, -1, 29}};
*/



template<class T>
MCubesBono<T>::MCubesBono()
{
}
    
template<class T>
MCubesBono<T>::MCubesBono( Data<T> *field ) 
  : data(field->values)
{
  nx = field->nx;
  ny = field->ny;
  nz = field->nz;

  i001 = 1;
  i010 = nx+1;
  i011 = nx;
  i100 = nx * ny;
  i101 = nx * ny + 1;
  i110 = nx * ny + nx + 1;
  i111 = nx * ny + nx;

  triangles = new GeomTriGroup();
}

template<class T>
void 
MCubesBono<T>::reset( int n )
{
  triangles->reserve_clear( (int)(2.5*(float)n ) );
//   triangles = new GeomTriGroup(4*n);
//   if( triangles == 0 )
//     printf("triangles got bad malloc\n");
}  
  

template<class T>
void
MCubesBono<T>::interp( int *array, int branching, float iso, int x, int y, int z )
{
  switch( branching ) {
  case 0: {
    interp1( array, iso, x, y, z );
  } break;
  case 1: {
    //    lookup = lookup1; putin = putin1;
    interp2( array, iso, x, y, z, 1 );
  } break; 
  case 2: {
    //    lookup = lookup2; putin = putin2;
    interp2( array, iso, x, y, z, 2 );
  } break;
  case 3: {
    //    lookup = &lookup3[0][0]; putin = &putin3[0][0];
    interp4( array, iso, x, y, z, 3 );
  } break;
  case 4: {
    //    lookup = lookup4; putin = putin4;
    interp2( array, iso, x, y, z, 4 );
  } break;
  case 5: {
    //    lookup = &lookup5[0][0]; putin = &putin5[0][0];
    interp4( array, iso, x, y, z, 5 );
  } break;
  case 6: {
    //    lookup = &lookup6[0][0]; putin = &putin6[0][0];
    interp4( array, iso, x, y, z, 6 );
  } break;
  case 7: {
    //    lookup = &lookup7[0][0]; putin = &putin7[0][0];
    interp8( array, iso, x, y, z );
  } break;
  }; // switch( branch )
}


template <class T>
void
MCubesBono<T>::interp1( int* array, float isoval, int x, int y, int z )
{
  float val[8];
  int cell = array[0];

  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }


  pPoint v[8];
  v[0]=pPoint((float)x,   (float)y,   (float)z);
  v[1]=pPoint((float)(x+1), (float)y,   (float)z);
  v[2]=pPoint((float)(x+1), (float)(y+1),   (float)z);
  v[3]=pPoint((float)x,   (float)(y+1),   (float)z);
  v[4]=pPoint((float)x,   (float)y, (float)(z+1));
  v[5]=pPoint((float)(x+1), (float)y, (float)(z+1));
  v[6]=pPoint((float)(x+1), (float)(y+1), (float)(z+1));
  v[7]=pPoint((float)x,   (float)(y+1), (float)(z+1));
  
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
}

template <class T>
void
MCubesBono<T>::interp2( int* array, float isoval, int x, int y, int z, 
		    int casenum )
{
  float val[8];
  int idx, i;
  int cell = array[0];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];

  // do cell 0
  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0]=pPoint((float)x,   (float)y,   (float)z);
  v[1]=pPoint((float)(x+1), (float)y,   (float)z);
  v[2]=pPoint((float)(x+1), (float)(y+1),   (float)z);
  v[3]=pPoint((float)x,   (float)(y+1),   (float)z);
  v[4]=pPoint((float)x,   (float)y, (float)(z+1));
  v[5]=pPoint((float)(x+1), (float)y, (float)(z+1));
  v[6]=pPoint((float)(x+1), (float)(y+1), (float)(z+1));
  v[7]=pPoint((float)x,   (float)(y+1), (float)(z+1));
  
  if (mask>0 && mask<255) {
    tcase=triCases+mask;
    edges=tcase->edges;

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
 } // if (mask>0 && mask<255)

  // do cell 1
  cell = array[1];

  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

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
    v[0].x(v[0].x() + 1); v[1].x(v[1].x() + 1); 
    v[2].x(v[2].x() + 1); v[3].x(v[3].x() + 1); 
    v[4].x(v[4].x() + 1); v[5].x(v[5].x() + 1); 
    v[6].x(v[6].x() + 1); v[7].x(v[7].x() + 1); 
  } break;
  case 2: { 
    v[0].y(v[0].y() + 1); v[1].y(v[1].y() + 1); 
    v[2].y(v[2].y() + 1); v[3].y(v[3].y() + 1); 
    v[4].y(v[4].y() + 1); v[5].y(v[5].y() + 1); 
    v[6].y(v[6].y() + 1); v[7].y(v[7].y() + 1); 
  } break;
  case 4: { 
    v[0].z(v[0].z() + 1); v[1].z(v[1].z() + 1); 
    v[2].z(v[2].z() + 1); v[3].z(v[3].z() + 1); 
    v[4].z(v[4].z() + 1); v[5].z(v[5].z() + 1); 
    v[6].z(v[6].z() + 1); v[7].z(v[7].z() + 1); 
  } break;
  }; // switch( casenum )

  tcase=triCases+mask;
  edges=tcase->edges;

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
}

template <class T>
void
MCubesBono<T>::interp4( int* array, float isoval, int x, int y, int z, 
		    int casenum )
{
  float val[8];
  int idx, i;
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];

  int cell = array[0];

  // do cell 0
  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0]=pPoint((float)x,   (float)y,   (float)z);
  v[1]=pPoint((float)(x+1), (float)y,   (float)z);
  v[2]=pPoint((float)(x+1), (float)(y+1),   (float)z);
  v[3]=pPoint((float)x,   (float)(y+1),   (float)z);
  v[4]=pPoint((float)x,   (float)y, (float)(z+1));
  v[5]=pPoint((float)(x+1), (float)y, (float)(z+1));
  v[6]=pPoint((float)(x+1), (float)(y+1), (float)(z+1));
  v[7]=pPoint((float)x,   (float)(y+1), (float)(z+1));
  
  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

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
  } // if (mask>0 && mask<255)

  // do cells 1..2
  for( int iteration = 1; iteration < 3; iteration++ ) {
    cell = array[iteration];
    if( iteration == 1 ) {
      switch( casenum ) {
      case 3:
      case 5: {
	v[0].x(v[0].x() + 1); v[1].x(v[1].x() + 1); 
	v[2].x(v[2].x() + 1); v[3].x(v[3].x() + 1); 
	v[4].x(v[4].x() + 1); v[5].x(v[5].x() + 1); 
	v[6].x(v[6].x() + 1); v[7].x(v[7].x() + 1); 
      } break;
      case 6: {
	v[0].y(v[0].y() + 1); v[1].y(v[1].y() + 1); 
	v[2].y(v[2].y() + 1); v[3].y(v[3].y() + 1); 
	v[4].y(v[4].y() + 1); v[5].y(v[5].y() + 1); 
	v[6].y(v[6].y() + 1); v[7].y(v[7].y() + 1); 
      } break;
      };
    } else {
      switch( casenum ) {
      case 3: {
	v[0].x(v[0].x() - 1); v[1].x(v[1].x() - 1); 
	v[2].x(v[2].x() - 1); v[3].x(v[3].x() - 1); 
	v[4].x(v[4].x() - 1); v[5].x(v[5].x() - 1); 
	v[6].x(v[6].x() - 1); v[7].x(v[7].x() - 1); 
	v[0].y(v[0].y() + 1); v[1].y(v[1].y() + 1); 
	v[2].y(v[2].y() + 1); v[3].y(v[3].y() + 1); 
	v[4].y(v[4].y() + 1); v[5].y(v[5].y() + 1); 
	v[6].y(v[6].y() + 1); v[7].y(v[7].y() + 1); 
      } break;
      case 5: {
	v[0].x(v[0].x() - 1); v[1].x(v[1].x() - 1); 
	v[2].x(v[2].x() - 1); v[3].x(v[3].x() - 1); 
	v[4].x(v[4].x() - 1); v[5].x(v[5].x() - 1); 
	v[6].x(v[6].x() - 1); v[7].x(v[7].x() - 1); 
	v[0].z(v[0].z() + 1); v[1].z(v[1].z() + 1); 
	v[2].z(v[2].z() + 1); v[3].z(v[3].z() + 1); 
	v[4].z(v[4].z() + 1); v[5].z(v[5].z() + 1); 
	v[6].z(v[6].z() + 1); v[7].z(v[7].z() + 1); 
      } break;
      case 6: {
	v[0].y(v[0].y() - 1); v[1].y(v[1].y() - 1); 
	v[2].y(v[2].y() - 1); v[3].y(v[3].y() - 1); 
	v[4].y(v[4].y() - 1); v[5].y(v[5].y() - 1); 
	v[6].y(v[6].y() - 1); v[7].y(v[7].y() - 1); 
	v[0].z(v[0].z() + 1); v[1].z(v[1].z() + 1); 
	v[2].z(v[2].z() + 1); v[3].z(v[3].z() + 1); 
	v[4].z(v[4].z() + 1); v[5].z(v[5].z() + 1); 
	v[6].z(v[6].z() + 1); v[7].z(v[7].z() + 1); 
      } break;
      };
    } // if( iteration == 1 )
    
    val[0]=(float)data[cell]-isoval;
    val[1]=(float)data[cell+i001] - isoval;
    val[2]=(float)data[cell+i010] - isoval;
    val[3]=(float)data[cell+i011] - isoval;
    val[4]=(float)data[cell+i100] - isoval;
    val[5]=(float)data[cell+i101] - isoval;
    val[6]=(float)data[cell+i110] - isoval;
    val[7]=(float)data[cell+i111] - isoval;
    
    mask=0;
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
	p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if (mask>0 && mask<255)
  } // for( iteration )

  // do cell 3
  cell = array[3];

  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

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
    v[0].x(v[0].x() + 1); v[1].x(v[1].x() + 1); 
    v[2].x(v[2].x() + 1); v[3].x(v[3].x() + 1); 
    v[4].x(v[4].x() + 1); v[5].x(v[5].x() + 1); 
    v[6].x(v[6].x() + 1); v[7].x(v[7].x() + 1); 
  } break;
  case 6: { 
    v[0].y(v[0].y() + 1); v[1].y(v[1].y() + 1); 
    v[2].y(v[2].y() + 1); v[3].y(v[3].y() + 1); 
    v[4].y(v[4].y() + 1); v[5].y(v[5].y() + 1); 
    v[6].y(v[6].y() + 1); v[7].y(v[7].y() + 1); 
  } break;
  }; // switch( casenum )

  tcase=triCases+mask;
  edges=tcase->edges;

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
}

template <class T>
void
MCubesBono<T>::interp8( int* array, float isoval, int x, int y, int z )
{
  float val[8];
  int idx, i;
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];

  int cell = array[0];

  // do cell 0
  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0]=pPoint((float)x,   (float)y,   (float)z);
  v[1]=pPoint((float)(x+1), (float)y,   (float)z);
  v[2]=pPoint((float)(x+1), (float)(y+1), (float)z);
  v[3]=pPoint((float)x,   (float)(y+1), (float)z);
  v[4]=pPoint((float)x,   (float)y,   (float)(z+1));
  v[5]=pPoint((float)(x+1), (float)y,   (float)(z+1));
  v[6]=pPoint((float)(x+1), (float)(y+1), (float)(z+1));
  v[7]=pPoint((float)x,   (float)(y+1), (float)(z+1));
  
  if (mask>0 && mask<255) {

    tcase=triCases+mask;
    edges=tcase->edges;

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
  } // if (mask>0 && mask<255)

  // do cells 1..6
  float xa[] = {0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  float ya[] = {0.0, 0.0,  1.0, 0.0, -1.0, 0.0,  1.0};
  float za[] = {0.0, 0.0,  0.0, 0.0,  1.0, 0.0,  0.0};
  for( int iteration = 1; iteration < 7; iteration++ ) {
    cell = array[iteration];
    if( xa[iteration] != 0 ) {
      v[0].x(v[0].x() + xa[iteration]); v[1].x( v[1].x() + xa[iteration]); 
      v[2].x(v[2].x() + xa[iteration]); v[3].x( v[3].x() + xa[iteration]); 
      v[4].x(v[4].x() + xa[iteration]); v[5].x( v[5].x() + xa[iteration]); 
      v[6].x(v[6].x() + xa[iteration]); v[7].x( v[7].x() + xa[iteration]);
    }

    if( ya[iteration] != 0 ) {
      v[0].y(v[0].y() + ya[iteration]); v[1].y(v[1].y() +  ya[iteration]); 
      v[2].y(v[2].y() + ya[iteration]); v[3].y(v[3].y() + ya[iteration]); 
      v[4].y(v[4].y() + ya[iteration]); v[5].y(v[5].y() + ya[iteration]); 
      v[6].y(v[6].y() + ya[iteration]); v[7].y(v[7].y() + ya[iteration]);
    }

    if( za[iteration] != 0 ) {
      v[0].z(v[0].z() + za[iteration]); v[1].z(v[1].z() + za[iteration]); 
      v[2].z(v[2].z() + za[iteration]); v[3].z(v[3].z() + za[iteration]); 
      v[4].z(v[4].z() + za[iteration]); v[5].z(v[5].z() + za[iteration]); 
      v[6].z(v[6].z() + za[iteration]); v[7].z(v[7].z() + za[iteration]);
    }

    val[0]=(float)data[cell]-isoval;
    val[1]=(float)data[cell+i001] - isoval;
    val[2]=(float)data[cell+i010] - isoval;
    val[3]=(float)data[cell+i011] - isoval;
    val[4]=(float)data[cell+i100] - isoval;
    val[5]=(float)data[cell+i101] - isoval;
    val[6]=(float)data[cell+i110] - isoval;
    val[7]=(float)data[cell+i111] - isoval;
    
    mask=0;
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
	p[i]=interpolate(v[v1], v[v2], val[v1]/(val[v1] -  val[v2]));
      }
      
      for( i = 0; i < tcase->numTris; i++ ) {
	triangles->add( p[ tcase->triangles[i][0] ],
			p[ tcase->triangles[i][1] ],
			p[ tcase->triangles[i][2] ] );
      }
    } // if (mask>0 && mask<255)
  } // for( iteration )

  // do cell 7
  cell = array[7];

  val[0]=(float)data[cell]-isoval;
  val[1]=(float)data[cell+i001] - isoval;
  val[2]=(float)data[cell+i010] - isoval;
  val[3]=(float)data[cell+i011] - isoval;
  val[4]=(float)data[cell+i100] - isoval;
  val[5]=(float)data[cell+i101] - isoval;
  val[6]=(float)data[cell+i110] - isoval;
  val[7]=(float)data[cell+i111] - isoval;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0].x(v[0].x() + 1); v[1].x(v[1].x() + 1); 
  v[2].x(v[2].x() + 1); v[3].x(v[3].x() + 1); 
  v[4].x(v[4].x() + 1); v[5].x(v[5].x() + 1); 
  v[6].x(v[6].x() + 1); v[7].x(v[7].x() + 1);

  tcase=triCases+mask;
  edges=tcase->edges;

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
}

// interpolate between points p and q, by a fraction f
inline
pPoint interpolate( const float* p, const float* q, double f) {
  return pPoint( p[0] + f*(q[0] - p[0]),
		 p[1] + f*(q[1] - p[1]),
		 p[2] + f*(q[2] - p[2]) );
}
// MCubesBonoCL - constructor
template <class T>
MCubesBonoCL<T>::MCubesBonoCL( DataCL<T>** d, int n ) {
  numzones = n;
  data = d;
  i001 = new int[numzones]; i010 = new int[numzones]; i011 = new int[numzones];
  i100 = new int[numzones]; i101 = new int[numzones]; i110 = new int[numzones];
  i111 = new int[numzones];
  c001 = new int[numzones]; c010 = new int[numzones]; c011 = new int[numzones];
  c100 = new int[numzones]; c101 = new int[numzones]; c110 = new int[numzones];
  c111 = new int[numzones];
  inv_nx = new float[numzones];

  for( int zone = 0; zone < numzones; zone++ ) {
    inv_nx[zone] = 1.0 / (float)d[zone]->nx;

    i001[zone] = 1;
    i010[zone] = d[zone]->nx+1;
    i011[zone] = d[zone]->nx;
    i100[zone] = d[zone]->nx * d[zone]->ny;
    i101[zone] = d[zone]->nx * d[zone]->ny + 1;
    i110[zone] = d[zone]->nx * d[zone]->ny + d[zone]->nx + 1;
    i111[zone] = d[zone]->nx * d[zone]->ny + d[zone]->nx;

    c001[zone] = -d[zone]->nx + 1;
    c010[zone] = 1;
    c011[zone] = d[zone]->nx;
    c100[zone] = d[zone]->nx * d[zone]->ny;
    c101[zone] = d[zone]->nx * d[zone]->ny - d[zone]->nx + 1;
    c110[zone] = d[zone]->nx * d[zone]->ny + 1;
    c111[zone] = d[zone]->nx * d[zone]->ny + d[zone]->nx;
  }
  triangles = new GeomTriGroup();
}

// reset
// given n (number of isosurface cells), create a GeomTriGroup that can
// hold the maximum number of triangles for that n.
template <class T>
void
MCubesBonoCL<T>::reset( int n ) {
  //  triangles = new GeomTriGroup(4*n);
  triangles->reserve_clear( (int)(2.5*(float)n ) );
} // reset

// interpCircular
// interpolate a single curvilinear cell in a circular grid
template <class T>
void
MCubesBonoCL<T>::interpCircular( int zone, int cell, double iso ) {
  T val[8];

  float n = (float)(cell+1) * inv_nx[zone];
  int notatedge = ( n == (int)n ) ? 0 : 1;
  val[0]=data[zone]->values[cell] - iso;

  if( notatedge ) {
    val[1]=data[zone]->values[cell+i001[zone]] - iso;
    val[2]=data[zone]->values[cell+i010[zone]] - iso;
    val[3]=data[zone]->values[cell+i011[zone]] - iso;
    val[4]=data[zone]->values[cell+i100[zone]] - iso;
    val[5]=data[zone]->values[cell+i101[zone]] - iso;
    val[6]=data[zone]->values[cell+i110[zone]] - iso;
    val[7]=data[zone]->values[cell+i111[zone]] - iso;
  } else {
    val[1]=data[zone]->values[cell+c001[zone]] - iso;
    val[2]=data[zone]->values[cell+c010[zone]] - iso;
    val[3]=data[zone]->values[cell+c011[zone]] - iso;
    val[4]=data[zone]->values[cell+c100[zone]] - iso;
    val[5]=data[zone]->values[cell+c101[zone]] - iso;
    val[6]=data[zone]->values[cell+c110[zone]] - iso;
    val[7]=data[zone]->values[cell+c111[zone]] - iso;
  }

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  VertexCL* v[8];
  v[0] = &data[zone]->verts[cell];
  if( notatedge ) {
    v[1] = &data[zone]->verts[cell+i001[zone]];
    v[2] = &data[zone]->verts[cell+i010[zone]];
    v[3] = &data[zone]->verts[cell+i011[zone]];
    v[4] = &data[zone]->verts[cell+i100[zone]];
    v[5] = &data[zone]->verts[cell+i101[zone]];
    v[6] = &data[zone]->verts[cell+i110[zone]];
    v[7] = &data[zone]->verts[cell+i111[zone]];
  } else {
    v[1] = &data[zone]->verts[cell+c001[zone]];
    v[2] = &data[zone]->verts[cell+c010[zone]];
    v[3] = &data[zone]->verts[cell+c011[zone]];
    v[4] = &data[zone]->verts[cell+c100[zone]];
    v[5] = &data[zone]->verts[cell+c101[zone]];
    v[6] = &data[zone]->verts[cell+c110[zone]];
    v[7] = &data[zone]->verts[cell+c111[zone]];
  }

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

} // interpCircular

// interpRegular
// interpolate a single curvilinear cell in a noncircular grid
template <class T>
void
MCubesBonoCL<T>::interpRegular( int zone, int cell, double iso ) {
  T val[8];

  val[0]=data[zone]->values[cell] - iso;
  val[1]=data[zone]->values[cell+i001[zone]] - iso;
  val[2]=data[zone]->values[cell+i010[zone]] - iso;
  val[3]=data[zone]->values[cell+i011[zone]] - iso;
  val[4]=data[zone]->values[cell+i100[zone]] - iso;
  val[5]=data[zone]->values[cell+i101[zone]] - iso;
  val[6]=data[zone]->values[cell+i110[zone]] - iso;
  val[7]=data[zone]->values[cell+i111[zone]] - iso;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  VertexCL* v[8];
  v[0] = &data[zone]->verts[cell];
  v[1] = &data[zone]->verts[cell+i001[zone]];
  v[2] = &data[zone]->verts[cell+i010[zone]];
  v[3] = &data[zone]->verts[cell+i011[zone]];
  v[4] = &data[zone]->verts[cell+i100[zone]];
  v[5] = &data[zone]->verts[cell+i101[zone]];
  v[6] = &data[zone]->verts[cell+i110[zone]];
  v[7] = &data[zone]->verts[cell+i111[zone]];

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

} // interpRegular
} // End namespace Phil




#endif

