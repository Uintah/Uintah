
/*
  datalist.h

  Dynamic data structure containing data bricks for out-of-core algorithms

  Packages/Philip Sutton

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_DATALIST_H__
#define __TBON_DATALIST_H__

#include <math.h>
#include <strings.h>

namespace Phil {
template <class T>
class DataList {
public:
  DataList( int listsize, int bricksize, int* X, int* Y, int* Z );
  ~DataList();

  void addBricks( int* barray, FILE* brickfile );
  T& getValue( int x, int y, int z );
  void reset();

protected:
private:
  int lsize;
  int bsize;
  int shiftamt;
  T** list;
  int* indices;
  int* lastused;
  int gen;
  int numbricks;

  int* Xarray;
  int* Yarray;
  int* Zarray;

  void insert( int bricknum, FILE* brickfile );
}; // class DataList

template <class T>
DataList<T>::DataList( int listsize, int bricksize, int* X, int* Y, int* Z ):
  lsize( listsize ), bsize( bricksize ), Xarray(X), Yarray(Y), Zarray(Z)
{
  int i;
  // guarantee minimum size
  if( lsize < 8 ) 
    lsize = 8;
  shiftamt = log((float)bsize)/log(2.0) - log((float)sizeof(T))/log(2.0);
  
  list = new T*[ lsize ];
  for( i = 0; i < lsize; i++ )
    list[i] = new T[ bsize/sizeof(T) ];
  indices = new int[ lsize ];
  lastused = new int[ lsize ];
  bzero( lastused, lsize*sizeof(int) );

  gen = 1;
  numbricks = 0;
} // DataList

template <class T>
DataList<T>::~DataList() {
  for( int i = 0; i < lsize; i++ )
    delete [] list[i];
  delete [] list;
  delete [] indices;
  delete [] lastused;
} // ~DataList

template <class T>
void
DataList<T>::reset() {
  bzero( lastused, lsize*sizeof(int) );
  numbricks = 0;
  gen = 1;
} // reset

template <class T>
void
DataList<T>::addBricks( int* barray, FILE* brickfile ) {
  int i, j;
  for( i = 0; i < 8; i++ ) {
    for( j = 0; j < numbricks && indices[j] != barray[i]; j++ );
    if( j == numbricks ) {
      // we don't have this brick
    } else {
      // we do have this brick
      lastused[j] = gen;
    }
  }
  
  for( i = 0; i < 8; i++ ) {
    for( j = 0; j < numbricks && indices[j] != barray[i]; j++ );
    if( j == numbricks ) {
      // we don't have this brick
      insert( barray[i], brickfile );
    } else {
      // we do have this brick
    }
  }

  gen++;
} // addBricks

template <class T> 
void 
DataList<T>::insert( int bricknum, FILE* brickfile ) {
  int i, slot;
  if( numbricks < lsize ) {
    // there's an empty slot
    slot = numbricks;
    numbricks++;
  } else {
    // no empty slots - blow away oldest brick
    int minvalue = gen+1;
    int minindex = -1;
    for( i = 0; i < lsize; i++ ) {
      if( lastused[i] < minvalue ) {
	minvalue = lastused[i];
	minindex = i;
      }
    }
    slot = minindex;
  }

  // read brick into slot
  fseek( brickfile, bricknum*bsize, SEEK_SET );
  fread( list[slot], bsize, 1, brickfile );
  indices[slot] = bricknum;
  lastused[slot] = gen;
} // insert

template <class T>
T&
DataList<T>::getValue( int x, int y, int z ) {
  int index = Xarray[x] + Yarray[y] + Zarray[z];
  int brick = index >> shiftamt;
  int mask = (1 << shiftamt) - 1;
  int offset = index & mask;
  int i;
  for( i = 0; indices[i] != brick; i++ );
  return list[i][offset];
} // getValue

} // End namespace Phil


#endif

