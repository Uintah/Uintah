
/* BonoTreeP.h
   class declarations for the parallel BONO tree

   Packages/Philip Sutton
   July/October 1999

   Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __BONO_TREE_P_H__
#define __BONO_TREE_P_H__

#include "mcubeBONO.h"
#include "WorkQueue.h"

#include <stdio.h>
#include <math.h>
#include <strings.h>

namespace Phil {
template <class T>
struct Data {
  int nx, ny, nz;
  int size;
  int cells;

  T* values;
};

template <class T>
struct Node {
  char branching;
  T min, max;
  int sibling;
};

template <class T>
class BonoTreeP {
public:
  // constructor for preprocessing
  BonoTreeP( int nx, int ny, int nz );
  // constructor for execution
  BonoTreeP( const char* filename, int b, int np );
  // destructor
  ~BonoTreeP();

  // operations on the tree
  void readData( const char* filename );
  void fillTree( );
  void writeTree( const char* meta, const char* base, int num );
  int search1( double iso, const char* treefile, const char* datafile, 
	       int timechanged, int level, int np );
  void resize( int isocells, int np );
  GeomTriGroup* search3( double iso, int level, int rank, int isocells );
  void cleanup();

  // accessors
  int getDepth() { return depth; }

  // constants
  static const int TIME_CHANGED;
  static const int TIME_SAME;
protected:

private:
  // structure of the tree
  Data<T> *data;
  Node<T> *tree;
  int *indices;
  
  // properties
  int depth;
  int numnodes;

  // parallel stuff
  int numProcessors;
  WorkQueue* workqueue;
  MCubesBono<T>** mcube;
  int* first;
  
  // auxiliary stuff
  static const int BRANCHTABLE[8];

  FILE* currtree;
  FILE* currdata;

  // private methods
  void fillArray( char* array, int num );
  void countNodes( char* xa, char* ya, char* za, int idx );
  void createTree( char* xa, char* ya, char* za, int idx, int* myindex );
  void fill( int myindex, int d, int vnum, T* min, T* max );
  void getMinMax( int index, T* min, T* max );

  int searchFirstPass( int myindex, int d, double iso, int level );
  void searchSecondPass( int myindex, int rank, int d, double iso, 
			 int x, int y, int z, MCubesBono<T>* mc );
  void assignPoints( int myindex, int d, int x, int y, int z, int level,
		     int* currjob, double iso );
};


// define constants
template <class T>
const int BonoTreeP<T>::TIME_CHANGED = 0;
template <class T>
const int BonoTreeP<T>::TIME_SAME = 1;
template <class T>
const int BonoTreeP<T>::BRANCHTABLE[] = {1, 2, 2, 4, 2, 4, 4, 8};

// globals for this file
static int curr;
static char* lo;
static int nodecount;
static int numjobs;

// BonoTreeP - constructor for preprocessing
template <class T>
BonoTreeP<T>::BonoTreeP( int nx, int ny, int nz ) {
  int i;
  int idx;

  // set up data
  data = new Data<T>;
  data->nx = nx; data->ny = ny; data->nz = nz;
  data->size = nx * ny * nz;
  data->cells = (nx - 1) * (ny - 1) * (nz - 1);
  data->values = new T[data->size];

  // these char arrays hold a representation of the spatial range of the data
  // (in binary)
  char* xa = new char[8*sizeof(T)];
  char* ya = new char[8*sizeof(T)];
  char* za = new char[8*sizeof(T)];
  lo = new char[8*sizeof(T)];

  fillArray( xa, data->nx - 2 );
  fillArray( ya, data->ny - 2 );
  fillArray( za, data->nz - 2 );
  for( i = 0; i < 8*sizeof(T); i++ )
    lo[i] = 1;

  // find first non-zero entry - that corresponds to the depth of the tree
  for( idx = 8*sizeof(T) - 1; 
       idx >= 0 && xa[idx] != 1 && ya[idx] != 1 && za[idx] != 1;
       idx-- );
  depth = idx;

  // find how many nodes are needed
  numnodes = 1;
  countNodes( xa, ya, za, idx );
  cout << "Tree has " << numnodes << " nodes" << endl;

  // allocate tree structure
  tree = new Node<T>[numnodes];
  indices = new int[data->cells];
  
  // construct tree skeleton
  int rootindex = 0;
  createTree( xa, ya, za, idx, &rootindex );
       
  // clean up
  delete [] xa;
  delete [] ya;
  delete [] za;
  delete [] lo;
} // BonoTreeP

// BonoTreeP - constructor for execution
template <class T>
BonoTreeP<T>::BonoTreeP( const char* filename, int b, int np ) {
  FILE* metafile = fopen( filename, "r" );
  if( !metafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // read tree parameters
  fscanf( metafile, "%d\n%d\n", &numnodes, &depth );

  // set up and read data parameters
  data = new Data<T>;
  fscanf( metafile, "%d %d %d\n", &(data->nx), &(data->ny), &(data->nz) );
  fscanf( metafile, "%d %d\n", &(data->size), &(data->cells) );
  data->values = new T[data->size];
  
  // allocate tree structure
  tree = new Node<T>[numnodes];
  indices = new int[data->cells];
  
  // read in tree skeleton
  int i;
  for( i = 0; i < numnodes; i++ ) {
    fread( &(tree[i].branching), sizeof(char), 1, metafile );
    fread( &(tree[i].sibling), sizeof(int), 1, metafile );
  }
  fread( indices, sizeof(int), data->cells, metafile );

  fclose( metafile );

  // create and initialize the structures that contain the triangles
  mcube = new MCubesBono<T>*[np];
  first = new int[np];
  for( i = 0; i < np; i++ )
    first[i] = 1;

} // BonoTreeP

// ~BonoTreeP - Destructor
template <class T>
BonoTreeP<T>::~BonoTreeP() {
  // common to preprocessing and execution
  delete [] data->values;
  delete data;
  delete [] tree;
  delete [] indices;

} // ~BonoTreeP

// readData - read data from file into the "data" structure
template <class T>
void 
BonoTreeP<T>::readData( const char* filename ) {
  FILE* datafile = fopen( filename, "r" );
  if( !datafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // read values
  int n = fread( data->values, sizeof(T), data->size, datafile );

  // make sure all values were read
  if( n != data->size ) {
    cerr << "Error: only " << n << "/" << data->size << " objects read from "
	 << filename << endl;
  }

  fclose( datafile );
} // readData

// fillTree - fill in skeleton tree with min, max, etc.
template <class T>
void 
BonoTreeP<T>::fillTree( ) {
  curr = 0;
  fill( 0, depth, 0, &(tree[0].min), &(tree[0].max) );
} // fillTree

// writeTree - write the tree to disk
template <class T>
void 
BonoTreeP<T>::writeTree( const char* meta, const char* base, int num ) {
  char filename[80];
  int i;
  FILE* out;
  cout << "Writing tree" << endl;

  // if this is the first write, write the tree metafile, too
  //  (skeleton)
  if( num == 0 ) {
    out = fopen( meta, "w" );
    if( !out ) {
      cerr << "Error: cannot open file " << meta << endl;
      return;
    }

    fprintf(out, "%d\n%d\n", numnodes, depth);
    fprintf(out,"%d %d %d\n%d %d\n", data->nx, data->ny, data->nz,
	    data->size, data->cells );
    for( i = 0; i < numnodes; i++ ) {
      fwrite( &(tree[i].branching), sizeof(char), 1, out );
      fwrite( &(tree[i].sibling), sizeof(int), 1, out );
    }
    fwrite( indices, sizeof(int), data->cells, out );
    
    fclose(out);
  } 

  sprintf(filename, "%s%d", base, num );
  out = fopen( filename, "w" );
  if( !out ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }
  
  for( i = 0; i < numnodes; i++ ) {
    fwrite( &(tree[i].min), sizeof(T), 1, out );
    fwrite( &(tree[i].max), sizeof(T), 1, out );
  }
  
  fclose( out );
} // writeTree

// search1 - first phase of search - read in nodes and data 
//           and set up work queue
template <class T>
int
BonoTreeP<T>::search1( double iso, const char* treefile, const char* datafile, 
		       int timechanged, int level, int np ) {
  int i;
  numProcessors = np;
  if( timechanged == TIME_CHANGED ) {
    currtree = fopen( treefile, "r" );
    currdata = fopen( datafile, "r" );

    // read nodes
    for( i = 0; i < numnodes; i++ ) {
      fread( &(tree[i].min), sizeof(T), 1, currtree );
      fread( &(tree[i].max), sizeof(T), 1, currtree );
    } 

    // read data
    fread( data->values, sizeof(T), data->size, currdata );
  }

  fclose( currtree );
  fclose( currdata );

  nodecount = 0;
  numjobs = 0;
  int n = searchFirstPass( 0, depth, iso, depth-level );
  workqueue = new WorkQueue( numProcessors, nodecount );
  static int firsttime = 1;
  if( firsttime ) {
    for( i = 0; i < np; i++ ) {
      mcube[i] = new MCubesBono<T>( data );
    }
    firsttime = 0;
  }

  if( n > 0 ) {
    int currjob = 0;
    assignPoints( 0, depth, 0, 0, 0, depth-level, &currjob, iso );
    workqueue->prepare(numjobs);
  }

  return n;
} // search1

template <class T>
void
BonoTreeP<T>::resize( int isocells, int np ) {
  for( int i = 0; i < np; i++ ) {
    mcube[i]->reset( isocells / np );
  }
} // resize

// search3 - last phase of search - get jobs from work queue
//           and construct isosurface
template <class T>
GeomTriGroup*
BonoTreeP<T>::search3( double iso, int level, int rank, int isocells ) {
  int start, end;
  while( workqueue->getWork( start, end ) ) {
    for( int i = start; i < end; i++ ) {
      searchSecondPass( workqueue->jobs[i].index, rank, depth-level-1, iso,
      			workqueue->jobs[i].x, workqueue->jobs[i].y, 
			workqueue->jobs[i].z, mcube[rank] ); 
    }
  }

  if( first[rank] && isocells > 0 ) {
    first[rank] = 0;
    return mcube[rank]->triangles; 
  }

  return 0;

} // search3

template <class T>
void
BonoTreeP<T>::cleanup() {
  delete workqueue;
}


// fillArray -  fill array with binary representation of num
template <class T>
void 
BonoTreeP<T>::fillArray( char* array, int num ) {
  int i;
  for( i = 8*sizeof(T)-1; i >= 0; i-- ) {
    int pow2 = (int)powf(2.0,(float)i);
    if( pow2 <= num ) { array[i] = 1; num -= pow2; }
    else { array[i] = 0; }
  }
} // fillArray


// countNodes - find number of nodes needed for tree
template <class T>
void 
BonoTreeP<T>::countNodes( char* xa, char* ya, char* za, int idx ) {
  if( idx == 0 )
    return;

  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];
  numnodes += BRANCHTABLE[branching];
  idx--;

  switch( branching ) {
  case 0: {
    countNodes( xa, ya, za, idx );
  } break;
  case 1: {
    countNodes( lo, ya, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 2: {
    countNodes( xa, lo, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 3: {
    countNodes( lo, lo, za, idx );
    countNodes( xa, lo, za, idx );
    countNodes( lo, ya, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 4: {
    countNodes( xa, ya, lo, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 5: {
    countNodes( lo, ya, lo, idx );
    countNodes( xa, ya, lo, idx );
    countNodes( lo, ya, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 6: {
    countNodes( xa, lo, lo, idx );
    countNodes( xa, ya, lo, idx );
    countNodes( xa, lo, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  case 7: {
    countNodes( lo, lo, lo, idx );
    countNodes( xa, lo, lo, idx );
    countNodes( lo, ya, lo, idx );
    countNodes( xa, ya, lo, idx );
    countNodes( lo, lo, za, idx );
    countNodes( xa, lo, za, idx );
    countNodes( lo, ya, za, idx );
    countNodes( xa, ya, za, idx );
  } break;
  };
  
} // countNodes

// createTree - construct tree skeleton
template <class T>
void 
BonoTreeP<T>::createTree( char* xa, char* ya, char* za, int idx, int* myindex ){
  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];

  if( idx == 0 ) {
    tree[*myindex].branching = (char)branching;
    *myindex = *myindex + 1;
    return;
  }

  int counter = *myindex + 1;
  idx--;
  
  switch( branching ) {
  case 0: {
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 1: {
    createTree( lo, ya, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 2: {
    createTree( xa, lo, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 3: {
    createTree( lo, lo, za, idx, &counter );
    createTree( xa, lo, za, idx, &counter );
    createTree( lo, ya, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 4: {
    createTree( xa, ya, lo, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 5: {
    createTree( lo, ya, lo, idx, &counter );
    createTree( xa, ya, lo, idx, &counter );
    createTree( lo, ya, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 6: {
    createTree( xa, lo, lo, idx, &counter );
    createTree( xa, ya, lo, idx, &counter );
    createTree( xa, lo, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  case 7: {
    createTree( lo, lo, lo, idx, &counter );
    createTree( xa, lo, lo, idx, &counter );
    createTree( lo, ya, lo, idx, &counter );
    createTree( xa, ya, lo, idx, &counter );
    createTree( lo, lo, za, idx, &counter );
    createTree( xa, lo, za, idx, &counter );
    createTree( lo, ya, za, idx, &counter );
    createTree( xa, ya, za, idx, &counter );
  } break;
  };

  tree[*myindex].sibling = counter;
  tree[*myindex].branching = (char)branching;
  *myindex = counter;

} // createTree

// fill - recursively fill each node in the tree 
template <class T>
void
BonoTreeP<T>::fill( int myindex, int d, int vnum, T* min, T* max ) {
  int j;
  T mins[8], maxs[8];
  int branching = (int)tree[myindex].branching;

  if( d == 0 ) {
    switch( branching ) {
    case 0: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
    } break;
    case 1: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum+1;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 2: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + data->nx;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 3: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + 1;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = vnum + data->nx;
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = vnum + data->nx + 1;
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 4: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + data->nx * data->ny;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 5: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + 1;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = vnum + data->nx * data->ny;
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = vnum + data->nx * data->ny + 1;
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 6: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + data->nx;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = vnum + data->nx * data->ny;
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = vnum + data->nx * data->ny + data->nx;
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 7: {
      indices[curr] = vnum;
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = vnum + 1;
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = vnum + data->nx;
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = vnum + data->nx + 1;
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
      indices[curr] = vnum + data->nx * data->ny;
      getMinMax( indices[curr++], &mins[4], &maxs[4] );
      indices[curr] = vnum + data->nx * data->ny + 1;
      getMinMax( indices[curr++], &mins[5], &maxs[5] );
      indices[curr] = vnum + data->nx * data->ny + data->nx;
      getMinMax( indices[curr++], &mins[6], &maxs[6] );
      indices[curr] = vnum + data->nx * data->ny + data->nx + 1;
      getMinMax( indices[curr++], &mins[7], &maxs[7] );
    } break;
    };  // switch(branching)

    tree[myindex].sibling = curr - BRANCHTABLE[branching];

    *min = mins[0];
    *max = maxs[0];
    for( j = 1; j < BRANCHTABLE[branching]; j++ ) {
      if( mins[j] < *min ) *min = mins[j];
      if( maxs[j] > *max ) *max = maxs[j];
    }
    tree[myindex].min = *min;
    tree[myindex].max = *max;
    return;
  } // if( d == 0 )

  //  int xstep = pow( 2, d );
  int xstep = 1 << d;
  int ystep = data->nx * xstep;
  int zstep = data->ny * ystep;
  d--;

  fill( myindex+1, d, vnum, &mins[0], &maxs[0] );
  if( d == 0 ) {
    switch( branching ) {
    case 0: break;
    case 1: {
      fill( myindex+2, d, vnum+xstep, &mins[1], &maxs[1] );
    } break;
    case 2: {
      fill( myindex+2, d, vnum+ystep, &mins[1], &maxs[1] );
    } break;
    case 3: {
      fill( myindex+2, d, vnum+xstep, &mins[1], &maxs[1] );
      fill( myindex+3, d, vnum+ystep, &mins[2], &maxs[2] );
      fill( myindex+4, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
    } break;
    case 4: {
      fill( myindex+2, d, vnum+zstep, &mins[1], &maxs[1] );
    } break;
    case 5: {
      fill( myindex+2, d, vnum+xstep, &mins[1], &maxs[1] );
      fill( myindex+3, d, vnum+zstep, &mins[2], &maxs[2] );
      fill( myindex+4, d, vnum+xstep+zstep, &mins[3], &maxs[3] );
    } break;
    case 6: {
      fill( myindex+2, d, vnum+ystep, &mins[1], &maxs[1] );
      fill( myindex+3, d, vnum+zstep, &mins[2], &maxs[2] );
      fill( myindex+4, d, vnum+ystep+zstep, &mins[3], &maxs[3] );
    } break;
    case 7: {
      fill( myindex+2, d, vnum+xstep, &mins[1], &maxs[1] );
      fill( myindex+3, d, vnum+ystep, &mins[2], &maxs[2] );
      fill( myindex+4, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
      fill( myindex+5, d, vnum+zstep, &mins[4], &maxs[4] );
      fill( myindex+6, d, vnum+xstep+zstep, &mins[5], &maxs[5] );
      fill( myindex+7, d, vnum+ystep+zstep, &mins[6], &maxs[6] );
      fill( myindex+8, d, vnum+xstep+ystep+zstep, &mins[7], &maxs[7] );
    } break; 
    }; // switch( branching )

  } else {
    
    switch( branching ) {
    case 0: break;
    case 1: {
      fill( tree[myindex+1].sibling, d, vnum+xstep, &mins[1], &maxs[1] );
    } break;
    case 2: {
      fill( tree[myindex+1].sibling, d, vnum+ystep, &mins[1], &maxs[1] );
    } break;
    case 3: {
      fill( tree[myindex+1].sibling, d, vnum+xstep, &mins[1], &maxs[1] );
      int s = tree[myindex+1].sibling;
      fill( tree[s].sibling, d, vnum+ystep, &mins[2], &maxs[2] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
    } break;
    case 4: {
      fill( tree[myindex+1].sibling, d, vnum+zstep, &mins[1], &maxs[1] );
    } break;
    case 5: {
      fill( tree[myindex+1].sibling, d, vnum+xstep, &mins[1], &maxs[1] );
      int s = tree[myindex+1].sibling;
      fill( tree[s].sibling, d, vnum+zstep, &mins[2], &maxs[2] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+xstep+zstep, &mins[3], &maxs[3] );
    } break;
    case 6: {
      fill( tree[myindex+1].sibling, d, vnum+ystep, &mins[1], &maxs[1] );
      int s = tree[myindex+1].sibling;
      fill( tree[s].sibling, d, vnum+zstep, &mins[2], &maxs[2] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+ystep+zstep, &mins[3], &maxs[3] );
    } break;
    case 7: {
      fill( tree[myindex+1].sibling, d, vnum+xstep, &mins[1], &maxs[1] );
      int s = tree[myindex+1].sibling;
      fill( tree[s].sibling, d, vnum+ystep, &mins[2], &maxs[2] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+zstep, &mins[4], &maxs[4] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+xstep+zstep, &mins[5], &maxs[5] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d, vnum+ystep+zstep, &mins[6], &maxs[6] );
      s = tree[s].sibling;
      fill( tree[s].sibling, d,vnum+xstep+ystep+zstep, &mins[7],&maxs[7]);
    } break;
    }; // switch( branching )
  } // else (if d == 0)

  *min = mins[0];
  *max = maxs[0];
  for( j = 1; j < BRANCHTABLE[branching]; j++ ) {
    if( mins[j] < *min ) *min = mins[j];
    if( maxs[j] > *max ) *max = maxs[j];
  }
  tree[myindex].min = *min;
  tree[myindex].max = *max;

} // fill

// getMinMax - find the min and max values of the voxel 
//   based at data->values[index]
template <class T>
void
BonoTreeP<T>::getMinMax( int index, T* min, T* max ) {
  T v;
  *min = *max = data->values[index];
  
  v = data->values[index+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx*data->ny];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx*data->ny+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx*data->ny+data->nx];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data->values[index+data->nx*data->ny+data->nx+1];  
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;
} // getMinMax


// searchFirstPass - first pass of the tree traversal
template <class T>
int
BonoTreeP<T>::searchFirstPass( int myindex, int d, double iso, int level ) {
  static const int NX = data->nx;
  static const int NXNY = data->ny * data->nx;
  static const int OFFSET = 2*sizeof(T);
  int i;
  int branching = (int)tree[myindex].branching;
  
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {
    
    if( d == level ) {
      int j;
      int retval = 0;

      for( j = 1; j <= BRANCHTABLE[branching]; j++ ) {
	if( tree[myindex+j].min <= iso && tree[myindex+j].max >= iso ) {
	  retval += BRANCHTABLE[(int)tree[myindex+j].branching];
	  nodecount++;
	}
      }
      return retval; 
    }
    
    d--;
    int c = searchFirstPass( myindex+1, d, iso, level );

    switch( branching ) {
    case 0: break;
    case 1:
    case 2:
    case 4: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso, level);
    } break;
    case 3:
    case 5:
    case 6: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso, level );
      int s = tree[myindex+1].sibling;
      c+=searchFirstPass( tree[s].sibling, d, iso, level );
      s = tree[s].sibling;
      c+=searchFirstPass( tree[s].sibling, d, iso, level );
    } break;
    case 7: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso, level );
      int s = tree[myindex+1].sibling;
      for( int j = 0; j < 6; j++ ) {
	c+=searchFirstPass( tree[s].sibling, d, iso, level );
	s = tree[s].sibling;
      }
    } break;
    };
    
    return c;
    
  } // if( tree[myindex].min <= iso && tree[myindex].min >= iso )
  return 0;
} // searchFirstPass

// searchSecondPass - second pass of tree traversal
//   perform interpolations on cells
template <class T>
void
BonoTreeP<T>::searchSecondPass( int myindex, int rank, int d, double iso, 
				int x, int y, int z, MCubesBono<T>* mc ) {
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {

    if( d == 0 ) {
      mc->interp( &indices[tree[myindex].sibling],
		  (int)tree[myindex].branching, (float)iso, x, y, z );  
      return;
    }

    int step = 1 << d;
    d--;
    searchSecondPass( myindex+1, rank, d, iso, x, y, z, mc );
    if( d == 0 ) {
      switch( (int)tree[myindex].branching ) {
      case 0:
	break;
      case 1: {
	searchSecondPass( myindex+2, rank, d, iso, x+step, y, z , mc);
      } break;
      case 2:  {
	searchSecondPass( myindex+2, rank, d, iso, x, y+step, z , mc);
      } break;
      case 3:{
	searchSecondPass( myindex+2, rank, d, iso, x+step, y, z , mc);
	searchSecondPass( myindex+3, rank, d, iso, x, y+step, z , mc);
	searchSecondPass( myindex+4, rank, d, iso, x+step, y+step, z , mc);
      } break;
      case 4: {
	searchSecondPass( myindex+2, rank, d, iso, x, y, z+step , mc);
      } break;
      case 5:{
	searchSecondPass( myindex+2, rank, d, iso, x+step, y, z , mc);
	searchSecondPass( myindex+3, rank, d, iso, x, y, z+step , mc);
	searchSecondPass( myindex+4, rank, d, iso, x+step, y, z+step , mc);
      } break;
      case 6:{
	searchSecondPass( myindex+2, rank, d, iso, x, y+step, z , mc);
	searchSecondPass( myindex+3, rank, d, iso, x, y, z +step, mc);
	searchSecondPass( myindex+4, rank, d, iso, x, y+step, z+step , mc);
      } break;
      case 7: {
	searchSecondPass( myindex+2, rank, d, iso, x+step, y, z , mc);
	searchSecondPass( myindex+3, rank, d, iso, x, y+step, z , mc);
	searchSecondPass( myindex+4, rank, d, iso, x+step, y+step, z , mc);
	searchSecondPass( myindex+5, rank, d, iso, x, y, z+step , mc);
	searchSecondPass( myindex+6, rank, d, iso, x+step, y, z+step , mc);
	searchSecondPass( myindex+7, rank, d, iso, x, y+step, z+step , mc);
	searchSecondPass( myindex+8, rank, d, iso, x+step, y+step, z+step ,mc);
      } break;
      }; // switch( branching )
      
    } else {

      switch( (int)tree[myindex].branching ) {
      case 0: break;
      case 1: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x+step, y, z,mc);
      } break;
      case 2: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x, y+step, z,mc);
      } break;
      case 3: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x+step, y, z,mc);
	int s = tree[myindex+1].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y+step, z, mc);
	s = tree[s].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x+step, y+step, z, mc);
      } break;
      case 4: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x, y, z+step,mc);
      } break;
      case 5: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x+step, y, z,mc);
	int s = tree[myindex+1].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y, z+step, mc);
	s = tree[s].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x+step, y, z+step, mc);
      } break;
      case 6: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x, y+step, z,mc);
	int s = tree[myindex+1].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y, z+step, mc);
	s = tree[s].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y+step, z+step, mc);
      } break;
      case 7: {
	searchSecondPass(tree[myindex+1].sibling,rank,d, iso, x+step, y, z,mc);
	int s = tree[myindex+1].sibling;
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y+step, z, mc);
	s = tree[s].sibling;	
	searchSecondPass(tree[s].sibling,rank,d, iso, x+step, y+step, z, mc);
	s = tree[s].sibling;	
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y, z+step, mc);
	s = tree[s].sibling;	
	searchSecondPass(tree[s].sibling,rank,d, iso, x+step, y, z+step, mc);
	s = tree[s].sibling;	
	searchSecondPass(tree[s].sibling,rank,d, iso, x, y+step, z+step, mc);
	s = tree[s].sibling;	
	searchSecondPass(tree[s].sibling,rank,d,iso,x+step,y+step,z+step, mc);
      } break;
      };
    }
     
  } // if( tree[myindex].min <= iso && tree[myindex].min >= iso )

} // searchSecondPass

// assignPoints - traverse tree down to level, assign coordinates to
//   nodes in the work queue
template <class T> 
void 
BonoTreeP<T>::assignPoints( int myindex, int d, int x, int y, int z, 
			    int level, int* currjob, double iso ) {
  int step;
  if( d == level ) {
    step = 1 << d;
    if( tree[myindex+1].min <= iso && tree[myindex+1].max >= iso ) {
      workqueue->addjob(*currjob, myindex+1, x, y, z );
      *currjob = *currjob + 1;
      numjobs++;
    }
    switch( (int)tree[myindex].branching ) {
    case 0: {
    } break;
    case 1: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x+step, y, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 2: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 3: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x+step, y, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+3].min <= iso && tree[myindex+3].max >= iso ) {
	workqueue->addjob(*currjob, myindex+3, x, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+4].min <= iso && tree[myindex+4].max >= iso ) {
	workqueue->addjob(*currjob, myindex+4, x+step, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 4: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 5: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x+step, y, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+3].min <= iso && tree[myindex+3].max >= iso ) {
	workqueue->addjob(*currjob, myindex+3, x, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+4].min <= iso && tree[myindex+4].max >= iso ) {
	workqueue->addjob(*currjob, myindex+4, x+step, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 6: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+3].min <= iso && tree[myindex+3].max >= iso ) {
	workqueue->addjob(*currjob, myindex+3, x, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+4].min <= iso && tree[myindex+4].max >= iso ) {
	workqueue->addjob(*currjob, myindex+4, x, y+step, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    case 7: {
      if( tree[myindex+2].min <= iso && tree[myindex+2].max >= iso ) {
	workqueue->addjob(*currjob, myindex+2, x+step, y, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+3].min <= iso && tree[myindex+3].max >= iso ) {
	workqueue->addjob(*currjob, myindex+3, x, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+4].min <= iso && tree[myindex+4].max >= iso ) {
	workqueue->addjob(*currjob, myindex+4, x+step, y+step, z );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+5].min <= iso && tree[myindex+5].max >= iso ) {
	workqueue->addjob(*currjob, myindex+5, x, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+6].min <= iso && tree[myindex+6].max >= iso ) {
	workqueue->addjob(*currjob, myindex+6, x+step, y, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+7].min <= iso && tree[myindex+7].max >= iso ) {
	workqueue->addjob(*currjob, myindex+7, x, y+step, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
      if( tree[myindex+8].min <= iso && tree[myindex+8].max >= iso ) {
	workqueue->addjob(*currjob, myindex+8, x+step, y+step, z+step );
	*currjob = *currjob + 1;
	numjobs++;
      }
    } break;
    };

    return;
  }

  step = 1 << d;
  d--;

  assignPoints( myindex+1, d, x, y, z, level, currjob, iso );
  switch( (int)tree[myindex].branching ) {
  case 0:
    break;
  case 1: {
    assignPoints(tree[myindex+1].sibling,d,x+step, y, z, level, currjob, iso);
  } break;
  case 2: {
    assignPoints(tree[myindex+1].sibling,d,x, y+step, z, level, currjob, iso);
  } break;
  case 4: {
    assignPoints(tree[myindex+1].sibling,d,x, y, z+step, level, currjob, iso);
  } break;
  case 3: {
    assignPoints(tree[myindex+1].sibling,d,x+step,y, z, level, currjob, iso);
    int s = tree[myindex+1].sibling;
    assignPoints( tree[s].sibling, d, x, y+step, z, level, currjob, iso);
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x+step, y+step, z, level, currjob, iso);
  } break;
  case 5: {
    assignPoints(tree[myindex+1].sibling,d,x+step, y, z, level, currjob, iso );
    int s = tree[myindex+1].sibling;
    assignPoints( tree[s].sibling, d, x, y, z+step, level, currjob, iso );
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x+step, y, z+step, level, currjob, iso);
  } break;
  case 6: {
    assignPoints(tree[myindex+1].sibling,d, x,y+step, z, level, currjob, iso);
    int s = tree[myindex+1].sibling;
    assignPoints( tree[s].sibling, d, x, y, z+step, level, currjob, iso);
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x, y+step, z+step, level, currjob, iso);
  } break;
  case 7: {
    assignPoints(tree[myindex+1].sibling,d,x+step, y, z, level, currjob, iso);
    int s = tree[myindex+1].sibling;
    assignPoints( tree[s].sibling, d, x, y+step, z, level, currjob, iso );
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x+step, y+step, z, level, currjob, iso);
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x, y, z+step, level, currjob, iso );
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x+step, y, z+step, level, currjob, iso);
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x, y+step, z+step, level, currjob, iso);
    s = tree[s].sibling;
    assignPoints( tree[s].sibling, d, x+step, y+step, z+step, level, 
		  currjob, iso );
  } break;
  }; // switch

} // assignPoints
} // End namespace Phil


#endif

