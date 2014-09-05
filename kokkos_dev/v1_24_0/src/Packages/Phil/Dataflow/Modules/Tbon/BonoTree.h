
/* BonoTree.h
   class declarations for the BONO tree

   Packages/Philip Sutton
   July 1999

   Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __BONO_TREE_H__
#define __BONO_TREE_H__

#include "mcubeBONO.h"

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
class BonoTree {
public:
  // constructor for preprocessing
  BonoTree( int nx, int ny, int nz );
  // constructor for execution
  BonoTree( const char* filename, int b );
  // destructor
  ~BonoTree();

  // operations on the tree
  void readData( const char* filename );
  void fillTree( );
  void writeTree( const char* meta, const char* base, int num );
  GeomTriGroup* search( double iso, const char* treefile, 
			const char* datafile, int timechanged );

  // constants
  static const int TIME_CHANGED;
  static const int TIME_SAME;
protected:

private:
  // structure of the tree
  Data<T> *data;
  Node<T> *tree;
  int *indices;
  MCubesBono<T> *mcube;
  
  // properties
  int depth;
  int numnodes;
  
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

  int searchFirstPass( int myindex, int d, double iso );
  void searchSecondPass( int myindex, int d, double iso, int x, int y, int z );
};


// define constants
template <class T>
const int BonoTree<T>::TIME_CHANGED = 0;
template <class T>
const int BonoTree<T>::TIME_SAME = 1;
template <class T>
const int BonoTree<T>::BRANCHTABLE[] = {1, 2, 2, 4, 2, 4, 4, 8};

// globals for this file
static int curr;
static char* lo;

// BonoTree - constructor for preprocessing
template <class T>
BonoTree<T>::BonoTree( int nx, int ny, int nz ) {
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

} // BonoTree

// BonoTree - constructor for execution
template <class T>
BonoTree<T>::BonoTree( const char* filename, int b ) {
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
  mcube = new MCubesBono<T>(data);
  
  // read in tree skeleton
  int i;
  for( i = 0; i < numnodes; i++ ) {
    fread( &(tree[i].branching), sizeof(char), 1, metafile );
    fread( &(tree[i].sibling), sizeof(int), 1, metafile );
  }
  fread( indices, sizeof(int), data->cells, metafile );

  fclose( metafile );

} // BonoTree

// ~BonoTree - Destructor
template <class T>
BonoTree<T>::~BonoTree() {
  // common to preprocessing and execution
  delete [] data->values;
  delete data;
  delete [] tree;
  delete [] indices;
} // ~BonoTree

// readData - read data from file into the "data" structure
template <class T>
void 
BonoTree<T>::readData( const char* filename ) {
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
BonoTree<T>::fillTree( ) {
  curr = 0;
  fill( 0, depth, 0, &(tree[0].min), &(tree[0].max) );
} // fillTree

// writeTree - write the tree to disk
template <class T>
void 
BonoTree<T>::writeTree( const char* meta, const char* base, int num ) {
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

// search - find isosurface
template <class T>
GeomTriGroup*
BonoTree<T>::search( double iso, const char* treefile, const char* datafile,
		     int timechanged ) {
  int i;
  
  // can't reuse nodes/data if the time value changed
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

  // find # of isosurface cells and which data to read
  int n = searchFirstPass( 0, depth, iso );

  mcube->reset( n );
  searchSecondPass( 0, depth, iso, 0, 0, 0 );

  fclose( currtree );
  fclose( currdata );

  return mcube->triangles;
} // search


// fillArray -  fill array with binary representation of num
template <class T>
void 
BonoTree<T>::fillArray( char* array, int num ) {
  int i;
  for( i = 31; i >= 0; i-- ) {
    if( (int)powf(2.,(float)i) <= num ) { 
      array[i] = 1; num -= (int)powf(2.,(float)i); 
    }
    else { array[i] = 0; }
  }
} // fillArray


// countNodes - find number of nodes needed for tree
template <class T>
void 
BonoTree<T>::countNodes( char* xa, char* ya, char* za, int idx ) {
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
BonoTree<T>::createTree( char* xa, char* ya, char* za, int idx, int* myindex ){
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
BonoTree<T>::fill( int myindex, int d, int vnum, T* min, T* max ) {
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

  int xstep = (int)powf( 2., (float)d );
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
BonoTree<T>::getMinMax( int index, T* min, T* max ) {
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
BonoTree<T>::searchFirstPass( int myindex, int d, double iso ) {
  static const int NX = data->nx;
  static const int NXNY = data->ny * data->nx;
  static const int OFFSET = 2*sizeof(T);
  int i;
  int branching = (int)tree[myindex].branching;
  
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {
    
    // stop one level above leaves
    if( d == 1 ) {
      int j;
      int retval = 0;

      for( j = 1; j <= BRANCHTABLE[branching]; j++ )
	if( tree[myindex+j].min <= iso && tree[myindex+j].max >= iso ) 
	  retval += BRANCHTABLE[(int)tree[myindex+j].branching];
      return retval; 
    }
    
    // recursively search children
    d--;
    int c = searchFirstPass( myindex+1, d, iso );

    switch( branching ) {
    case 0: break;
    case 1:
    case 2:
    case 4: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso);
    } break;
    case 3:
    case 5:
    case 6: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso );
      int s = tree[myindex+1].sibling;
      c+=searchFirstPass( tree[s].sibling, d, iso );
      s = tree[s].sibling;
      c+=searchFirstPass( tree[s].sibling, d, iso );
    } break;
    case 7: {
      c+=searchFirstPass( tree[myindex+1].sibling, d, iso );
      int s = tree[myindex+1].sibling;
      for( int j = 0; j < 6; j++ ) {
	c+=searchFirstPass( tree[s].sibling, d, iso );
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
BonoTree<T>::searchSecondPass( int myindex, int d, double iso, 
			       int x, int y, int z ) {
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {

    // leaf node - perform interpolation
    if( d == 0 ) {
      mcube->interp( &indices[tree[myindex].sibling],
		     (int)tree[myindex].branching , iso, x, y, z );  
      return;
    }

    // recursively search children
    int step = 1 << d;
    d--;
    searchSecondPass( myindex+1, d, iso, x, y, z );
    if( d == 0 ) {
      switch( (int)tree[myindex].branching ) {
      case 0:
	break;
      case 1: {
	searchSecondPass( myindex+2, d, iso, x+step, y, z  );
      } break;
      case 2:  {
	searchSecondPass( myindex+2, d, iso, x, y+step, z  );
      } break;
      case 3:{
	searchSecondPass( myindex+2, d, iso, x+step, y, z  );
	searchSecondPass( myindex+3, d, iso, x, y+step, z  );
	searchSecondPass( myindex+4, d, iso, x+step, y+step, z  );
      } break;
      case 4: {
	searchSecondPass( myindex+2, d, iso, x, y, z+step  );
      } break;
      case 5:{
	searchSecondPass( myindex+2, d, iso, x+step, y, z  );
	searchSecondPass( myindex+3, d, iso, x, y, z+step  );
	searchSecondPass( myindex+4, d, iso, x+step, y, z+step  );
      } break;
      case 6:{
	searchSecondPass( myindex+2, d, iso, x, y+step, z  );
	searchSecondPass( myindex+3, d, iso, x, y, z +step );
	searchSecondPass( myindex+4, d, iso, x, y+step, z+step  );
      } break;
      case 7: {
	searchSecondPass( myindex+2, d, iso, x+step, y, z  );
	searchSecondPass( myindex+3, d, iso, x, y+step, z  );
	searchSecondPass( myindex+4, d, iso, x+step, y+step, z  );
	searchSecondPass( myindex+5, d, iso, x, y, z+step  );
	searchSecondPass( myindex+6, d, iso, x+step, y, z+step  );
	searchSecondPass( myindex+7, d, iso, x, y+step, z+step  );
	searchSecondPass( myindex+8, d, iso, x+step, y+step, z+step  );
      } break;
      }; // switch( branching )
      
    } else {

      switch( (int)tree[myindex].branching ) {
      case 0: break;
      case 1: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x+step, y, z );
      } break;
      case 2: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x, y+step, z );
      } break;
      case 3: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x+step, y, z );
	int s = tree[myindex+1].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x, y+step, z );
	s = tree[s].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x+step, y+step, z );
      } break;
      case 4: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x, y, z+step );
      } break;
      case 5: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x+step, y, z );
	int s = tree[myindex+1].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x, y, z+step );
	s = tree[s].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x+step, y, z+step );
      } break;
      case 6: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x, y+step, z );
	int s = tree[myindex+1].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x, y, z+step );
	s = tree[s].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x, y+step, z+step );
      } break;
      case 7: {
	searchSecondPass( tree[myindex+1].sibling, d, iso, x+step, y, z );
	int s = tree[myindex+1].sibling;
	searchSecondPass( tree[s].sibling, d, iso, x, y+step, z );
	s = tree[s].sibling;	
	searchSecondPass( tree[s].sibling, d, iso, x+step, y+step, z );
	s = tree[s].sibling;	
	searchSecondPass( tree[s].sibling, d, iso, x, y, z+step );
	s = tree[s].sibling;	
	searchSecondPass( tree[s].sibling, d, iso, x+step, y, z+step );
	s = tree[s].sibling;	
	searchSecondPass( tree[s].sibling, d, iso, x, y+step, z+step );
	s = tree[s].sibling;	
	searchSecondPass( tree[s].sibling, d, iso, x+step, y+step, z+step );
	s = tree[s].sibling;	
	/*	for( int j = 0; j < 6; j++ ) {
	  searchSecondPass( tree[s].sibling, d, iso, x, y, z );
	  s = tree[s].sibling;
	  } */
      } break;
      };
    }
     
  } // if( tree[myindex].min <= iso && tree[myindex].min >= iso )

} // searchSecondPass

} // End namespace Phil


#endif

