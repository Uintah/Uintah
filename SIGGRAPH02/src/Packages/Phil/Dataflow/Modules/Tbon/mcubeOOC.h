
/*  mcubeOOC.h
    Marching Cubes style interpolation for structured and unstructured grids
      (for out-of-core algorithms)

    Packages/Philip Sutton
    May 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __MCUBE_OOC_H__
#define __MCUBE_OOC_H__

#include "TriGroup.h"
#include "mcube_table.h"

#include <stdio.h>
#include <stdlib.h>


namespace Phil {
using namespace SCIRun;


template<class T> struct Data;
template<class T> class DataList;

// MCubesOOC 
// Marching Cubes class for regular (structured) grids
template<class T>
class MCubesOOC {
public:
  GeomTriGroup *triangles;
  T *data;

  MCubesOOC( Data<T>* field, DataList<T>* dlist );
  
  void reset( int n );
  void interp( int x, int y, int z, int branching, float iso );
  void setResolution( int res );

  friend pPoint interpolate( const pPoint &p, const pPoint& q, float f);

protected:
private:
  DataList<T>* datalist;
  int offseti;
  float offsetd;

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

// interpolate (MCubesOOC)
// interpolate between points p and q, by a fraction f
inline
pPoint interpolate( const pPoint &p, const pPoint &q, float f )
{
  return pPoint( p.x() + f*(q.x()-p.x()), 
		 p.y() + f*(q.y()-p.y()), 
		 p.z() + f*(q.z()-p.z()) );
}



// tables for spatial locality (regular grid)
static const int N = -1;  // null entry - makes tables look pretty
template<class T>
int MCubesOOC<T>::lookup1[] = { N, N, N, 0, N, N, N, 1, 2, N, 3, N };
template<class T>
int MCubesOOC<T>::insert1[] = { N, 0, N, N, N, 1, N, N, N, 2, N, 3 };
template<class T>
int MCubesOOC<T>::lookup2[] = { 0, N, N, N, 1, N, N, N, 2, 3, N, N };
template<class T>
int MCubesOOC<T>::insert2[] = { N, N, 0, N, N, N, 1, N, N, N, 2, 3 };
template<class T>
int MCubesOOC<T>::lookup4[] = { 0, 1, 2, 3, N, N, N, N, N, N, N, N };
template<class T>
int MCubesOOC<T>::insert4[] = { N, N, N, N, 0, 1, 2, 3, N, N, N, N };
template<class T>
int MCubesOOC<T>::lookup3[][12] = {{ N, N, N, 0, N, N, N, 2, 4, N, 5, N },
				{ 7, N, N, N, 9, N, N, N, 5, 8, N, N },
				{ 1, 10, N, N, 3, 12, N, N, 6, 5, N, 11 } };
template<class T>
int MCubesOOC<T>::insert3[][12] = {{ N, 0, 1, N, N, 2, 3, N, N, 4, 6, 5 },
				{ N, N, 7, N, N, N, 9, N, N, N, N, 8 },
				{ N, N, N, 10, N, N, N, 12, N, N, 11, N }};
template<class T>
int MCubesOOC<T>::lookup5[][12] = {{ N, N, N, 5, N, N, N, 1, 4, N, 6, N },
				{ 7, 8, 9, 1, N, N, N, N, N, N, N, N },
				{ 0, 1, 2, 3, N, 11, N, N, N, 10, N, 12 }};
template<class T>
int MCubesOOC<T>::insert5[][12] = {{ N, 5, N, N, 0, 1, 2, 3, N, 4, N, 6 },
				{ N, N, N, N, 7, 8, 9, N, N, N, N, N },
				{ N, N, N, N, N, N, N, 11, 10, N, 12, N }};
template<class T>
int MCubesOOC<T>::lookup6[][12] = {{ 5, N, N, N, 2, N, N, N, 4, 6, N, N },
				{ 2, 7, 8, 9, N, N, N, N, N, N, N, N },
				{ 0, 1, 2, 3, N, N, 12, N, N, N, 10, 11 }};
template<class T>
int MCubesOOC<T>::insert6[][12] = {{ N, N, 5, N, 0, 1, 2, 3, N, N, 4, 6 },
				{ N, N, N, N, N, 7, 8, 9, N, N, N, N },
				{ N, N, N, N, 12, N, N, N, 10, 11, N, N }};
template<class T>
int MCubesOOC<T>::lookup7[][12] = {{ N, N, N, 5, N, N, N, 1, 4, N, 8, N },
				{ 12, N, N, N, 11, N, N, N, 8, 13, N, N },
				{ 6, 17, N, N, 2, 14, N, N, 7, 8, N, 18 },
				{ 2, 14, 20, 19, N, N, N, N, N, N, N, N },
				{ 11, 16, 15, 14, N, N, N, 21, 24, N, 25, N },
				{ 9, 10, 11, 1, N, N, 26, N, N, N, 24, 27 },
				{ 0, 1, 2, 3, N, 28, 23, N, N, 29, 22, 24 }};
template<class T>
int MCubesOOC<T>::insert7[][12] = {{ N, 5, 6, N, 0, 1, 2, 3, N, 4, 7, 8 },
				{ N, N, 12, N, 9, 10, 11, N, N, N, N, 13 },
				{ N, N, N, 17, N, 16, 15, 14, N, N, 18, N },
				{ N, N, N, N, N, N, 20, 19, N, N, N, N },
				{ N, N, N, N, 23, 21, N, N, 22, 24, N, 25 },
				{ N, N, N, N, 26, N, N, N, N, 27, N, N },
				{ N, N, N, N, N, N, N, 28, 29, N, N, N }};



// MCubesOOC
// Constructor
template<class T>
MCubesOOC<T>::MCubesOOC( Data<T> *field, DataList<T>* dlist ) 
  : data(field->values), datalist(dlist)
{
  offseti = 1;
  offsetd = 1.0;
  triangles = new GeomTriGroup();
} // MCubesOOC

// reset
// given n (number of isosurface cells), create a GeomTriGroup that can
// hold the maximum number of triangles for that n.
template<class T>
void 
MCubesOOC<T>::reset( int n ) {
  int numcells = ( n / (offseti*offseti)) + 1;
  //  triangles = new GeomTriGroup((int)(2.5*(float)numcells));
  triangles->reserve_clear( (int)(2.5*(float)numcells) );
} // reset

// setResolution
// sets the stride offset for multires traversal
template<class T>
void
MCubesOOC<T>::setResolution( int res ) {
  offseti = (int)powf( 2.0, (float)res );
  offsetd = powf( 2.0, (float)res );
} // setResolution

// interp
// choose which interpx function to call, based on how many cells 
// are contained in the leaf node  
template<class T>
void
MCubesOOC<T>::interp( int x, int y, int z, int branching, float iso ) {
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
MCubesOOC<T>::interp1( int x, int y, int z, float iso ) {
  float val[8];

  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

  int mask=0;
  for(int idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  pPoint v[8];
  v[0] = pPoint((float)x,           (float)y,           (float)z);
  v[1] = pPoint((float)(x+offseti), (float)y,           (float)z);
  v[2] = pPoint((float)(x+offseti), (float)(y+offseti), (float)z);
  v[3] = pPoint((float)x,           (float)(y+offseti), (float)z);
  v[4] = pPoint((float)x,           (float)y,           (float)(z+offseti));
  v[5] = pPoint((float)(x+offseti), (float)y,           (float)(z+offseti));
  v[6] = pPoint((float)(x+offseti), (float)(y+offseti), (float)(z+offseti));
  v[7] = pPoint((float)x,           (float)(y+offseti), (float)(z+offseti));

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
MCubesOOC<T>::interp2( int x, int y, int z, float iso, int casenum ) {
  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;
 
  // do cell 0
  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0] = pPoint((float)x,           (float)y,           (float)z);
  v[1] = pPoint((float)(x+offseti), (float)y,           (float)z);
  v[2] = pPoint((float)(x+offseti), (float)(y+offseti), (float)z);
  v[3] = pPoint((float)x,           (float)(y+offseti), (float)z);
  v[4] = pPoint((float)x,           (float)y,           (float)(z+offseti));
  v[5] = pPoint((float)(x+offseti), (float)y,           (float)(z+offseti));
  v[6] = pPoint((float)(x+offseti), (float)(y+offseti), (float)(z+offseti));
  v[7] = pPoint((float)x,           (float)(y+offseti), (float)(z+offseti));

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
  case 1: { x+=offseti; } break;
  case 2: { y+=offseti; } break;
  case 4: { z+=offseti; } break;
  };

  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

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
    v[0].x(v[0].x() + offsetd); v[1].x(v[1].x() + offsetd); 
    v[2].x(v[2].x() + offsetd); v[3].x(v[3].x() + offsetd); 
    v[4].x(v[4].x() + offsetd); v[5].x(v[5].x() + offsetd); 
    v[6].x(v[6].x() + offsetd); v[7].x(v[7].x() + offsetd); 
  } break;
  case 2: { 
    v[0].y(v[0].y() + offsetd); v[1].y(v[1].y() + offsetd); 
    v[2].y(v[2].y() + offsetd); v[3].y(v[3].y() + offsetd); 
    v[4].y(v[4].y() + offsetd); v[5].y(v[5].y() + offsetd); 
    v[6].y(v[6].y() + offsetd); v[7].y(v[7].y() + offsetd); 
  } break;
  case 4: { 
    v[0].z(v[0].z() + offsetd); v[1].z(v[1].z() + offsetd); 
    v[2].z(v[2].z() + offsetd); v[3].z(v[3].z() + offsetd); 
    v[4].z(v[4].z() + offsetd); v[5].z(v[5].z() + offsetd); 
    v[6].z(v[6].z() + offsetd); v[7].z(v[7].z() + offsetd); 
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
MCubesOOC<T>::interp4( int x, int y, int z, float iso, int casenum ) {
  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;
 
  // do cell 0
  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0] = pPoint((float)x,           (float)y,           (float)z);
  v[1] = pPoint((float)(x+offseti), (float)y,           (float)z);
  v[2] = pPoint((float)(x+offseti), (float)(y+offseti), (float)z);
  v[3] = pPoint((float)x,           (float)(y+offseti), (float)z);
  v[4] = pPoint((float)x,           (float)y,           (float)(z+offseti));
  v[5] = pPoint((float)(x+offseti), (float)y,           (float)(z+offseti));
  v[6] = pPoint((float)(x+offseti), (float)(y+offseti), (float)(z+offseti));
  v[7] = pPoint((float)x,           (float)(y+offseti), (float)(z+offseti));

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
	x+=offseti; 
	v[0].x(v[0].x() + offsetd); v[1].x(v[1].x() + offsetd); 
	v[2].x(v[2].x() + offsetd); v[3].x(v[3].x() + offsetd); 
	v[4].x(v[4].x() + offsetd); v[5].x(v[5].x() + offsetd); 
	v[6].x(v[6].x() + offsetd); v[7].x(v[7].x() + offsetd); 
      } break;
      case 6: {
	y+=offseti; 
	v[0].y(v[0].y() + offsetd); v[1].y(v[1].y() + offsetd); 
	v[2].y(v[2].y() + offsetd); v[3].y(v[3].y() + offsetd); 
	v[4].y(v[4].y() + offsetd); v[5].y(v[5].y() + offsetd); 
	v[6].y(v[6].y() + offsetd); v[7].y(v[7].y() + offsetd); 
      } break;
      };
    } else {
      switch( casenum ) {
      case 3: {
	y+=offseti; 
	v[0].y(v[0].y() + offsetd); v[1].y(v[1].y() + offsetd); 
	v[2].y(v[2].y() + offsetd); v[3].y(v[3].y() + offsetd); 
	v[4].y(v[4].y() + offsetd); v[5].y(v[5].y() + offsetd); 
	v[6].y(v[6].y() + offsetd); v[7].y(v[7].y() + offsetd); 
      } break;
      case 5: 
      case 6: {
	z+=offseti;
	v[0].z(v[0].z() + offsetd); v[1].z(v[1].z() + offsetd); 
	v[2].z(v[2].z() + offsetd); v[3].z(v[3].z() + offsetd); 
	v[4].z(v[4].z() + offsetd); v[5].z(v[5].z() + offsetd); 
	v[6].z(v[6].z() + offsetd); v[7].z(v[7].z() + offsetd); 
      } break;
      };
    } // if( iteration == 1 )
    
    val[0]=datalist->getValue(x,         y,         z        ) - iso;
    val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
    val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
    val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
    val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
    val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
    val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
    val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

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
  case 5: { x-=offseti; } break;
  case 6: { y-=offseti; } break;
  };

  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

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
    v[0].x(v[0].x() - offsetd); v[1].x(v[1].x() - offsetd); 
    v[2].x(v[2].x() - offsetd); v[3].x(v[3].x() - offsetd); 
    v[4].x(v[4].x() - offsetd); v[5].x(v[5].x() - offsetd); 
    v[6].x(v[6].x() - offsetd); v[7].x(v[7].x() - offsetd); 
  } break;
  case 6: { 
    v[0].y(v[0].y() - offsetd); v[1].y(v[1].y() - offsetd); 
    v[2].y(v[2].y() - offsetd); v[3].y(v[3].y() - offsetd); 
    v[4].y(v[4].y() - offsetd); v[5].y(v[5].y() - offsetd); 
    v[6].y(v[6].y() - offsetd); v[7].y(v[7].y() - offsetd); 
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
MCubesOOC<T>::interp8( int x, int y, int z, float iso ) {
  float val[8];
  TRIANGLE_CASES *tcase;
  EDGE_LIST *edges;
  pPoint p[12];
  int i, idx;

  // do cell 0
  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

  int mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  pPoint v[8];
  v[0] = pPoint((float)x,           (float)y,           (float)z);
  v[1] = pPoint((float)(x+offseti), (float)y,           (float)z);
  v[2] = pPoint((float)(x+offseti), (float)(y+offseti), (float)z);
  v[3] = pPoint((float)x,           (float)(y+offseti), (float)z);
  v[4] = pPoint((float)x,           (float)y,           (float)(z+offseti));
  v[5] = pPoint((float)(x+offseti), (float)y,           (float)(z+offseti));
  v[6] = pPoint((float)(x+offseti), (float)(y+offseti), (float)(z+offseti));
  v[7] = pPoint((float)x,           (float)(y+offseti), (float)(z+offseti));

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
  static float xa[]={0,offsetd,0,       -offsetd, 0,       offsetd, 0        };
  static float ya[]={0,0,      offsetd, 0,        0,       0,       -offsetd };
  static float za[]={0,0,      0,       0,        offsetd, 0,       0        };

  for( int iteration = 1; iteration < 7; iteration++ ) {
    if( xa[iteration] != 0 ) {
      x += (int)xa[iteration];
      v[0].x(v[0].x() + xa[iteration]); v[1].x( v[1].x() + xa[iteration]); 
      v[2].x(v[2].x() + xa[iteration]); v[3].x( v[3].x() + xa[iteration]); 
      v[4].x(v[4].x() + xa[iteration]); v[5].x( v[5].x() + xa[iteration]); 
      v[6].x(v[6].x() + xa[iteration]); v[7].x( v[7].x() + xa[iteration]);
    }

    if( ya[iteration] != 0 ) {
      y += (int)ya[iteration];
      v[0].y(v[0].y() + ya[iteration]); v[1].y(v[1].y() + ya[iteration]); 
      v[2].y(v[2].y() + ya[iteration]); v[3].y(v[3].y() + ya[iteration]); 
      v[4].y(v[4].y() + ya[iteration]); v[5].y(v[5].y() + ya[iteration]); 
      v[6].y(v[6].y() + ya[iteration]); v[7].y(v[7].y() + ya[iteration]);
    }

    if( za[iteration] != 0 ) {
      z += (int)za[iteration];
      v[0].z(v[0].z() + za[iteration]); v[1].z(v[1].z() + za[iteration]); 
      v[2].z(v[2].z() + za[iteration]); v[3].z(v[3].z() + za[iteration]); 
      v[4].z(v[4].z() + za[iteration]); v[5].z(v[5].z() + za[iteration]); 
      v[6].z(v[6].z() + za[iteration]); v[7].z(v[7].z() + za[iteration]);
    }
    
    val[0]=datalist->getValue(x,         y,         z        ) - iso;
    val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
    val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
    val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
    val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
    val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
    val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
    val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

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

  val[0]=datalist->getValue(x,         y,         z        ) - iso;
  val[1]=datalist->getValue(x+offseti, y,         z        ) - iso;
  val[2]=datalist->getValue(x+offseti, y+offseti, z        ) - iso;
  val[3]=datalist->getValue(x,         y+offseti, z        ) - iso;
  val[4]=datalist->getValue(x,         y,         z+offseti) - iso;
  val[5]=datalist->getValue(x+offseti, y,         z+offseti) - iso;
  val[6]=datalist->getValue(x+offseti, y+offseti, z+offseti) - iso;
  val[7]=datalist->getValue(x        , y+offseti, z+offseti) - iso;

  mask=0;
  for(idx=0;idx<8;idx++){
    if(val[idx]<0)
      mask|=1<<idx;
  }

  if (mask==0 || mask==255) {
    return;
  }

  v[0].x(v[0].x() - offsetd); v[1].x(v[1].x() - offsetd); 
  v[2].x(v[2].x() - offsetd); v[3].x(v[3].x() - offsetd); 
  v[4].x(v[4].x() - offsetd); v[5].x(v[5].x() - offsetd); 
  v[6].x(v[6].x() - offsetd); v[7].x(v[7].x() - offsetd);

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

} // End namespace Phil


#endif

