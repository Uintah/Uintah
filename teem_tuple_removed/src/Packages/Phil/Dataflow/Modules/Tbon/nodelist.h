
/*
  nodelist.h
  Dynamic data structure containing node bricks for out-of-core algorithms

  Packages/Philip Sutton

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_NODELIST_H__
#define __TBON_NODELIST_H__

#include <math.h>
#include <strings.h>

namespace Phil {
template <class T>
class NodeList {
public:
  NodeList( int listsize, int bricksize );
  ~NodeList();

  void addBrick( int bricknum, FILE* brickfile );
  Node<T>& getNode( int bricknum, int nodenum );
  void reset();

protected:
private:
  int lsize;
  int bsize;
  Node<T>** list;
  int* indices;
  int* lastused;
  int gen;
  int numbricks;

  void insert( int bricknum, FILE* brickfile );
}; // class NodeList

template <class T>
NodeList<T>::NodeList( int listsize, int bricksize ):
  lsize( listsize ), bsize( bricksize )
{
  int i;
  
  list = new Node<T>*[ lsize ];
  for( i = 0; i < lsize; i++ )
    list[i] = new Node<T>[ bsize ];
  indices = new int[ lsize ];
  lastused = new int[ lsize ];
  bzero( lastused, lsize*sizeof(int) );

  gen = 1;
  numbricks = 0;
} // NodeList

template <class T>
NodeList<T>::~NodeList() {
  for( int i = 0; i < lsize; i++ )
    delete [] list[i];
  delete [] list;
  delete [] indices;
  delete [] lastused;
} // ~NodeList

template <class T>
void
NodeList<T>::reset() {
  bzero( lastused, lsize*sizeof(int) );
  numbricks = 0;
  gen = 1;
} // reset

template <class T>
void
NodeList<T>::addBrick( int bricknum, FILE* brickfile ) {
  int j;
  for( j = 0; j < numbricks && indices[j] != bricknum; j++ );
  if( j == numbricks ) {
    // we don't have this brick
    insert( bricknum, brickfile );
  } else {
    // we do have this brick
    lastused[j] = gen;
  }

  gen++;
} // addBricks

template <class T> 
void 
NodeList<T>::insert( int bricknum, FILE* brickfile ) {
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
Node<T>&
NodeList<T>::getNode( int bricknum, int nodenum ) {
  int i;
  for( i = 0; indices[i] != bricknum; i++ );
  return list[i][nodenum];
} // getValue

} // End namespace Phil


#endif

