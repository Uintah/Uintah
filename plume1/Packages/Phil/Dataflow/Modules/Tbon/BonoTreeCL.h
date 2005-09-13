
/* BonoTreeCL.h
   class declarations and code for the BONO tree, curvilinear implementation
   NOTE: This contains some detritus from the T-BON implementation it's
         based on, but probably not enough to impact performance.

   Packages/Philip Sutton
   June 1999

   Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __Bono_TREE_CL_H__
#define __Bono_TREE_CL_H__

#include "TreeUtils.h"
#include "TriGroup.h"
#include "mcubeBONO.h"
#include <stdio.h>
#include <math.h>
#include <strings.h>

namespace Phil {
using namespace SCIRun;
struct VertexCL {
  float pos[3];
};

template <class T>
struct DataCL {
  int nx, ny, nz;
  int size;
  int cells;

  VertexCL* verts;
  T* values;
};

template <class T>
class BonoTreeCL {
public:
  // constructor for preprocessing
  BonoTreeCL( int* nx, int* ny, int* nz, const char* geomfile, 
	      int numzones, int* c, int nb, int db );
  // constructor for execution
  BonoTreeCL( const char* filename, const char* geomfile );
  // destructor
  ~BonoTreeCL();

  // operations on the tree
  void readData( const char* filename );
  void fillTree( );
  void writeTree( const char* meta, const char* base, int num, 
		  const char* newdatafile );
  GeomTriGroup* search( double iso, const char* treefile, 
			const char* datafile, int timechanged );

  int getDepth() { 
    int min = depth[0];
    for( int i = 1; i < numtrees; i++ )
      min = ( depth[i] < min ) ? depth[i] : min;
    return min;
  }
  // constants
  static const int TIME_CHANGED;
  static const int TIME_SAME;  
  static const int NONCIRCULAR;
  static const int CIRCULAR_X;
  static const int CIRCULAR_Y;
  static const int CIRCULAR_Z;
protected:
private:
  // how many trees (for multi-zoned data sets)
  int numtrees;

  // structure of the trees
  DataCL<T>** data;
  Node<T>** tree;
  int** indices;
  int* circular;
  MCubesBonoCL<T>* mcube;

  // properties
  int* depth;
  int* numnodes;
  int* numNodeBricks;
  int* numDataBricks;

  // auxiliary stuff
  int nodebricksize;
  int databricksize;
  int entriesPerBrick;
  int* previousNodeBricks;
  int* previousDataBricks;
  int*** brickRanges;
  int** brickstoread;
  int shiftamt;
  char** nodeBrickInMem;
  char** datainmem;
  int gen;

  FILE* currtreefile;
  FILE* currdatafile;

  // flag for destructor
  int deleteExecutionStuff;

  // private methods
  void reorderTree( char* branchArray, int* sibArray, int currtree );
  void fill( int currtree, int myindex, int d, int vnum, T* min, T* max );
  void getMinMax( int currtree, int index, T* min, T* max );
  void findBricks();
  int searchFirstPass( double iso );
  void searchSecondPass( int currtree, int myindex, int d, double iso );
};

// define constants
template <class T>
const int BonoTreeCL<T>::TIME_CHANGED = 0;
template <class T>
const int BonoTreeCL<T>::TIME_SAME = 1;
template <class T>
const int BonoTreeCL<T>::NONCIRCULAR = 0;
template <class T>
const int BonoTreeCL<T>::CIRCULAR_X = 1;
template <class T>
const int BonoTreeCL<T>::CIRCULAR_Y = 2;
template <class T>
const int BonoTreeCL<T>::CIRCULAR_Z = 3;

// BonoTreeCL - constructor for preprocessing
template <class T>
BonoTreeCL<T>::BonoTreeCL( int* nx, int* ny, int* nz, const char* geomfile, 
			   int numzones, int* c, int nb, int db ) {
  int idx;
  int currtree;
  int** range;

  numtrees = numzones;
  circular = new int[numtrees];
  range = new int*[numtrees];
  numnodes = new int[numtrees];
  depth = new int[numtrees];
  tree = new Node<T>*[numtrees];
  indices = new int*[numtrees];

  // set up data
  data = new DataCL<T>*[numtrees];
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    circular[currtree] = c[currtree];
    range[currtree] = new int[3];

    data[currtree] = new DataCL<T>;
    data[currtree]->nx = nx[currtree]; 
    data[currtree]->ny = ny[currtree]; 
    data[currtree]->nz = nz[currtree];
    data[currtree]->size = nx[currtree] * ny[currtree] * nz[currtree];
    if( circular[currtree] == NONCIRCULAR ) {
      data[currtree]->cells = 
	(nx[currtree] - 1) * (ny[currtree] - 1) * (nz[currtree] - 1);
      range[currtree][0] = nx[currtree] - 2; 
      range[currtree][1] = ny[currtree] - 2; 
      range[currtree][2] = nz[currtree] - 2;
    } else if( circular[currtree] == CIRCULAR_X ) {
      data[currtree]->cells = 
	nx[currtree] * (ny[currtree] - 1) * (nz[currtree] - 1);
      range[currtree][0] = nx[currtree] - 1; 
      range[currtree][1] = ny[currtree] - 2; 
      range[currtree][2] = nz[currtree] - 2;
    } else if( circular[currtree] == CIRCULAR_Y ) {
      data[currtree]->cells = 
	(nx[currtree] - 1) * ny[currtree] * (nz[currtree] - 1);
      range[currtree][0] = nx[currtree] - 2; 
      range[currtree][1] = ny[currtree] - 1; 
      range[currtree][2] = nz[currtree] - 2;
    } else if( circular[currtree] == CIRCULAR_Z ) {
      data[currtree]->cells = 
	(nx[currtree] - 1) * (ny[currtree] - 1) * nz[currtree];
      range[currtree][0] = nx[currtree] - 2; 
      range[currtree][1] = ny[currtree] - 2; 
      range[currtree][2] = nz[currtree] - 1;
    }
    data[currtree]->values = new T[data[currtree]->size];
    data[currtree]->verts = new VertexCL[data[currtree]->size];
  }

  FILE* geom = fopen( geomfile, "r" );
  if( !geom ) {
    cerr << "Error: cannot open geom file " << geomfile << endl;
    return;
  }
  
  // these char arrays hold a representation of the spatial range of the data
  // (in binary)
  char* xa = new char[8*sizeof(T)];
  char* ya = new char[8*sizeof(T)];
  char* za = new char[8*sizeof(T)];

  // set up information for bricking nodes
  nodebricksize = nb;
  databricksize = db;
  numNodeBricks = new int[numtrees];
  brickRanges = new int**[numtrees];

  for( currtree = 0; currtree < numtrees; currtree++ ) {
    // read geometric data
    for( idx = 0; idx < data[currtree]->size; idx++ ) {
      fread( data[currtree]->verts[idx].pos, sizeof(float), 3, geom );
    }

    fillArray( xa, range[currtree][0], 8*sizeof(T) );
    fillArray( ya, range[currtree][1], 8*sizeof(T) );
    fillArray( za, range[currtree][2], 8*sizeof(T) );

    // find first non-zero entry - that corresponds to the depth of the tree
    for( idx = 8*sizeof(T) - 1; 
	 idx >= 0 && xa[idx] != 1 && ya[idx] != 1 && za[idx] != 1;
	 idx-- );
    depth[currtree] = idx;

    // find how many nodes are needed
    int num = 1;
    int dummy;
    countNodes( xa, ya, za, idx, &num, &dummy );
    numnodes[currtree] = num;
    cout << "Tree[" << currtree << "] has " << numnodes[currtree] 
	 << " nodes" << endl;
    
    // allocate tree structure
    tree[currtree] = new Node<T>[ numnodes[currtree] ];
    indices[currtree] = new int[ data[currtree]->cells ];
    
    // construct tree skeleton
    int rootindex = 0;
    char* branchArray = new char[numnodes[currtree]];
    int* sibArray = new int[numnodes[currtree]];

    // round up to find number of node bricks
    float temp = (float)numnodes[currtree] * 2.0 * (float)sizeof(T) / 
      (float)nodebricksize;
    numNodeBricks[currtree] = ((float)(temp - (int)temp)>= 0.5) ? 
      (int)temp + 2 : (int)temp + 1;
    brickRanges[currtree] = new int*[numNodeBricks[currtree]];
    
    createTree( xa, ya, za, idx, &rootindex, branchArray, sibArray, 0);
    reorderTree( branchArray, sibArray, currtree );
    delete [] branchArray;
    delete [] sibArray;
  }

  // clean up
  fclose( geom );
  delete [] xa;
  delete [] ya;
  delete [] za;
  delete [] lo;
  delete [] range;
  deleteExecutionStuff = 0;  // (for destructor)

} // BonoTreeCL

// BonoTreeCL - constructor for execution
template <class T>
BonoTreeCL<T>::BonoTreeCL( const char* filename, const char* geomfile ) {
  int i;
  int currtree;
  FILE* metafile = fopen( filename, "r" );
  if( !metafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // read tree and data parameters
  fscanf(metafile, "%d\n", &numtrees);
  fscanf(metafile, "%d %d\n", &nodebricksize, &databricksize);
  
  entriesPerBrick = nodebricksize / (int)sizeof(T);
  numnodes = new int[numtrees];
  depth = new int[numtrees];
  circular = new int[numtrees];
  data = new DataCL<T>*[numtrees];
  tree = new Node<T>*[numtrees];
  indices = new int*[numtrees];
  numNodeBricks = new int[numtrees];
  brickRanges = new int**[numtrees];
  nodeBrickInMem = new char*[numtrees];
  previousNodeBricks = new int[numtrees];
  previousDataBricks = new int[numtrees];
  numDataBricks = new int[numtrees];
  brickstoread = new int*[numtrees];
  datainmem = new char*[numtrees];
  
  shiftamt = log((float)databricksize)/log(2.0) - log((float)sizeof(T))/log(2.0);
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    data[currtree] = new DataCL<T>;
    fscanf(metafile, "%d\n%d\n", &(numnodes[currtree]), &(depth[currtree]) );
    fscanf(metafile, "%d\n", &(numNodeBricks[currtree]) );
    fscanf(metafile, "%d %d %d\n%d %d\n", &(data[currtree]->nx), 
	   &(data[currtree]->ny), &(data[currtree]->nz),
	   &(data[currtree]->size), &(data[currtree]->cells) );
    fscanf(metafile, "%d\n", &(circular[currtree]) );

    data[currtree]->values = new T[data[currtree]->size];
    data[currtree]->verts = new VertexCL[data[currtree]->size];

    // allocate tree structure
    tree[currtree] = new Node<T>[ numnodes[currtree] ];
    indices[currtree] = new int[ data[currtree]->cells ];

    // set up bricking info
    brickRanges[currtree] = new int*[numNodeBricks[currtree]];
    for( i = 0; i < numNodeBricks[currtree]; i++ ) {
      brickRanges[currtree][i] = new int[2];
      fread( brickRanges[currtree][i], sizeof(int), 2, metafile );
    }

    nodeBrickInMem[currtree] = new char[numNodeBricks[currtree]];
    bzero( nodeBrickInMem[currtree], numNodeBricks[currtree] );
    previousNodeBricks[currtree] = ( currtree == 0 ) ? 
      0 : previousNodeBricks[currtree-1] + numNodeBricks[currtree-1];

    numDataBricks[currtree] = ((data[currtree]->size-1) >> shiftamt ) + 1;
    brickstoread[currtree] = new int[ numDataBricks[currtree] ];
    datainmem[currtree] = new char[ numDataBricks[currtree] ];
    bzero( brickstoread[currtree], numDataBricks[currtree] * sizeof(int) );
    bzero( datainmem[currtree], numDataBricks[currtree] );
    previousDataBricks[currtree] = ( currtree == 0 ) ?
      0 : previousDataBricks[currtree-1] + 
      numDataBricks[currtree-1]*databricksize;
  }

  mcube = new MCubesBonoCL<T>( data, numtrees );

  // read in tree skeletons
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    for( i = 0; i < numnodes[currtree]; i++ ) {
      fread( &(tree[currtree][i].branching), sizeof(char), 1, metafile );
      fread( &(tree[currtree][i].child), sizeof(int), 1, metafile );
    }
    fread( indices[currtree], sizeof(int), data[currtree]->cells, metafile );
  }

  fclose( metafile );

  // read in geometry information
  FILE* geom = fopen( geomfile, "r" );
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    for( i = 0; i < data[currtree]->size; i++ )
      fread( data[currtree]->verts[i].pos, sizeof(float), 3, geom );
  }
  fclose( geom );

  gen = 1;
  deleteExecutionStuff = 1;
} // BonoTreeCL

// ~BonoTreeCL - destructor
template <class T>
BonoTreeCL<T>::~BonoTreeCL() {
  int currtree;
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    delete [] tree[currtree];
    delete [] indices[currtree];
    delete [] data[currtree]->values;
    delete [] data[currtree]->verts;
    delete data[currtree];
    delete [] brickRanges[currtree];
  }
  delete [] circular;
  delete [] numnodes;
  delete [] depth;
  delete [] brickRanges;
  delete [] numNodeBricks;

  delete [] tree;
  delete [] indices;
  delete [] data;

  if( deleteExecutionStuff ) {
    for( currtree = 0; currtree < numtrees; currtree++ ) {
      delete [] nodeBrickInMem[currtree];
      delete [] brickstoread[currtree];
      delete [] datainmem[currtree];
    }
    delete [] nodeBrickInMem;
    delete [] previousNodeBricks;
    delete [] previousDataBricks;
    delete [] numDataBricks;
    delete [] brickstoread;
    delete [] datainmem;
  }
} // ~BonoTreeCL

// readData - read data from file into the "data" structures
template <class T>
void
BonoTreeCL<T>::readData( const char* filename ) {
  int currtree;
  FILE* datafile = fopen( filename, "r" );
  if( !datafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  for( currtree = 0; currtree < numtrees; currtree++ ) {
    unsigned long n = fread( data[currtree]->values, sizeof(T), 
			     data[currtree]->size, datafile );
    if( n != (unsigned long)data[currtree]->size ) {
      cerr << "Error: only " << n << "/" << data[currtree]->size 
	   << " objects read from " << filename << endl;
    }
  }
} // readData

// fillTree - fill in tree skeleton with min, max
template <class T>
void
BonoTreeCL<T>::fillTree( ) {
  for( int currtree = 0; currtree < numtrees; currtree++ ) {
    curr = 0;
    fill( currtree, 0, depth[currtree], 0, 
	  &(tree[currtree][0].min), &(tree[currtree][0].max) );
  }
} // fillTree

// writeTree - write the tree to disk
template <class T>
void
BonoTreeCL<T>::writeTree( const char* meta, const char* base, int num,
			  const char* newdatafile ) {
  char filename[80];
  int i;
  FILE* out;
  FILE* newdata;
  int currtree;

  cout << "Writing tree" << endl;

  // if this is the first write, write the tree metafile, too
  //  (skeleton)
  if( num == 0 ) {
    out = fopen( meta, "w" );
    if( !out ) {
      cerr << "Error: cannot open file " << meta << endl;
      return;
    }

    // determine brick boundaries
    findBricks();

    // write everything
    fprintf(out, "%d\n", numtrees);
    fprintf(out, "%d %d\n", nodebricksize, databricksize);

    for( currtree = 0; currtree < numtrees; currtree++ ) {
      fprintf(out, "%d\n%d\n", numnodes[currtree], depth[currtree]);
      fprintf(out, "%d\n", numNodeBricks[currtree] );
      fprintf(out,"%d %d %d\n%d %d\n", 
	      data[currtree]->nx, data[currtree]->ny, data[currtree]->nz,
	      data[currtree]->size, data[currtree]->cells );
      fprintf(out,"%d\n",circular[currtree]);

      for( i = 0; i < numNodeBricks[currtree]; i++ ) {
      	fwrite( &brickRanges[currtree][i][0], sizeof(int), 1, out );
      	fwrite( &brickRanges[currtree][i][1], sizeof(int), 1, out );
      }
    }

    for( currtree = 0; currtree < numtrees; currtree++ ) {
      for( i = 0; i < numnodes[currtree]; i++ ) {
	fwrite( &(tree[currtree][i].branching), sizeof(char), 1, out );
	fwrite( &(tree[currtree][i].child), sizeof(int), 1, out );
      }
      fwrite( indices[currtree], sizeof(int), data[currtree]->cells, out );
    }
    fclose(out);
  } 

  // write the data specific to tree #num
  sprintf(filename, "%s%d", base, num );
  out = fopen( filename, "w" );
  if( !out ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }
  newdata = fopen( newdatafile, "w" );
  if( !newdata ) {
    cerr << "Error: cannot open file " << newdatafile << endl;
    return;
  }

  shiftamt = log((float)databricksize)/log(2.0) - log((float)sizeof(T))/log(2.0); 
  for( currtree = 0; currtree < numtrees; currtree++ ) {

    for( i = 0; i < numnodes[currtree]; i++ ) {
      fwrite( &(tree[currtree][i].min), sizeof(T), 1, out );
      fwrite( &(tree[currtree][i].max), sizeof(T), 1, out );
    }
  }
  
  fclose( newdata );
  fclose( out );
} // writeTree

// search - find isosurface
template <class T>
GeomTriGroup*
BonoTreeCL<T>::search( double iso, const char* treefile, 
		       const char* datafile, int timechanged ) {
  static const int SHIFT = (int)(sizeof(T) * 0.5);
  static int firsttime = 1;
  int currtree;
  currtreefile = fopen( treefile, "r" );
  currdatafile = fopen( datafile, "r" );
  iotimer_t t0,t1;

  // can't reuse nodes/data if the time value changed
  if( timechanged == TIME_CHANGED ) {
    for( currtree = 0; currtree < numtrees; currtree++ ) {
      for( int j = 0; j < numnodes[currtree]; j++ ) {
	fread( &(tree[currtree][j].min), sizeof(T), 1, currtreefile );
	fread( &(tree[currtree][j].max), sizeof(T), 1, currtreefile );
      }
      fread( data[currtree]->values, sizeof(T), data[currtree]->size,
	     currdatafile );
    }
  }
  
  // find # of isosurface cells and which data to read
  int n = searchFirstPass( iso );
  //  cout << "n = " << n << endl;
  mcube->reset(n);

  for( currtree = 0; currtree < numtrees; currtree++ ) 
    searchSecondPass( currtree, 0, depth[currtree], iso );

  fclose( currtreefile );
  fclose( currdatafile );
  gen++;

  cout << "#triangles = " << mcube->triangles->getSize() << endl;
  if( firsttime && mcube->triangles->getSize() > 0 ) {
    firsttime = 0;
    return mcube->triangles;
  }

  return 0;
} // search


///////////////////////
// Private Functions //
///////////////////////

// reorderTree - create BFS version of tree
template <class T>
void
BonoTreeCL<T>::reorderTree( char* branchArray, int* sibArray, int currtree ) {
  int currdepth;

  int size;
  int index = 1;
  int curr = 0;
  int node = 0;
  int* queue;
  int i, j;
  int newsize = 1;
  int last;

  // create levels depth..1
  for( currdepth = depth[currtree]; currdepth > 0; currdepth-- ) {
    size = newsize;
    newsize = 0;
    last = curr;
    queue = new int[size];
    for( i = 0; i < size; i++ ) {
      queue[i] = curr;
      tree[currtree][node].branching = branchArray[curr];
      tree[currtree][node].child = index;
      node++;
      index += BRANCHTABLE[ branchArray[curr] ];
      newsize += BRANCHTABLE[ branchArray[curr] ];
      curr = sibArray[curr];
    } // for( i = 0 .. size-1 )
    curr = last + 1;
    if( currdepth > 1 )
      delete [] queue;
  } // for( currdepth = depth .. 1 )

  // create level 0
  for( i = 0; i < size; i++ ) {
    for( j = 1; j <= BRANCHTABLE[ branchArray[queue[i]] ]; j++ )
      tree[currtree][node++].branching = branchArray[ queue[i] + j ];
  }
  delete [] queue;

} // reorderTree

// fill - recursively fill each node in the tree 
template <class T>
void
BonoTreeCL<T>::fill( int currtree, int myindex, int d, int vnum, 
		     T* min, T* max ) {
  int j;
  T mins[8], maxs[8];
  int branching = (int)tree[currtree][myindex].branching;

  if( d == 0 ) {
    switch( branching ) {
    case 0: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
    } break;
    case 1: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum+1;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
    } break;
    case 2: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
    } break;
    case 3: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
      indices[currtree][curr] = vnum + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[2], &maxs[2] );
      indices[currtree][curr] = vnum + data[currtree]->nx + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[3], &maxs[3] );
    } break;
    case 4: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + data[currtree]->nx * data[currtree]->ny;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
    } break;
    case 5: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
      indices[currtree][curr] = vnum + data[currtree]->nx * data[currtree]->ny;
      getMinMax( currtree, indices[currtree][curr++], &mins[2], &maxs[2] );
      indices[currtree][curr] = 
	vnum + data[currtree]->nx * data[currtree]->ny + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[3], &maxs[3] );
    } break;
    case 6: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
      indices[currtree][curr] = vnum + data[currtree]->nx * data[currtree]->ny;
      getMinMax( currtree, indices[currtree][curr++], &mins[2], &maxs[2] );
      indices[currtree][curr] = 
	vnum + data[currtree]->nx * data[currtree]->ny + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[3], &maxs[3] );
    } break;
    case 7: {
      indices[currtree][curr] = vnum;
      getMinMax( currtree, indices[currtree][curr++], &mins[0], &maxs[0] );
      indices[currtree][curr] = vnum + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[1], &maxs[1] );
      indices[currtree][curr] = vnum + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[2], &maxs[2] );
      indices[currtree][curr] = vnum + data[currtree]->nx + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[3], &maxs[3] );
      indices[currtree][curr] = vnum + data[currtree]->nx * data[currtree]->ny;
      getMinMax( currtree, indices[currtree][curr++], &mins[4], &maxs[4] );
      indices[currtree][curr] = 
	vnum + data[currtree]->nx * data[currtree]->ny + 1;
      getMinMax( currtree, indices[currtree][curr++], &mins[5], &maxs[5] );
      indices[currtree][curr] = 
	vnum + data[currtree]->nx * data[currtree]->ny + data[currtree]->nx;
      getMinMax( currtree, indices[currtree][curr++], &mins[6], &maxs[6] );
      indices[currtree][curr] = 
	vnum + data[currtree]->nx * data[currtree]->ny + data[currtree]->nx +1;
      getMinMax( currtree, indices[currtree][curr++], &mins[7], &maxs[7] );
    } break;
    };  // switch(branching)

    tree[currtree][myindex].child = curr - BRANCHTABLE[branching];

    *min = mins[0];
    *max = maxs[0];
    for( j = 1; j < BRANCHTABLE[branching]; j++ ) {
      if( mins[j] < *min ) *min = mins[j];
      if( maxs[j] > *max ) *max = maxs[j];
    }
    tree[currtree][myindex].min = *min;
    tree[currtree][myindex].max = *max;
    return;
  } // if( d == 0 )

  int xstep = 1 << d;
  int ystep = data[currtree]->nx * xstep;
  int zstep = data[currtree]->ny * ystep;
  d--;

  fill( currtree, tree[currtree][myindex].child, d, vnum, &mins[0], &maxs[0] );
  switch( branching ) {
  case 0: break;
  case 1: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+xstep, 
	  &mins[1], &maxs[1] );
  } break;
  case 2: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+ystep, 
	  &mins[1], &maxs[1] );
  } break;
  case 3: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+xstep, 
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, vnum+ystep, 
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, vnum+xstep+ystep, 
	  &mins[3], &maxs[3] );
  } break;
  case 4: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+zstep, 
	  &mins[1], &maxs[1] );
  } break;
  case 5: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+xstep, 
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, vnum+zstep, 
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, vnum+xstep+zstep, 
	  &mins[3], &maxs[3] );
  } break;
  case 6: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+ystep, 
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, vnum+zstep, 
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, vnum+ystep+zstep, 
	  &mins[3], &maxs[3] );
  } break;
  case 7: {
    fill( currtree, tree[currtree][myindex].child + 1, d, vnum+xstep, 
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, vnum+ystep, 
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, vnum+xstep+ystep, 
	  &mins[3], &maxs[3] );
    fill( currtree, tree[currtree][myindex].child + 4, d, vnum+zstep, 
	  &mins[4], &maxs[4] );
    fill( currtree, tree[currtree][myindex].child + 5, d, vnum+xstep+zstep, 
	  &mins[5], &maxs[5] );
    fill( currtree, tree[currtree][myindex].child + 6, d, vnum+ystep+zstep, 
	  &mins[6], &maxs[6] );
    fill( currtree, tree[currtree][myindex].child + 7, d, 
	  vnum+xstep+ystep+zstep, &mins[7], &maxs[7] );
  } break; 
  }; // switch( branching )
  
  *min = mins[0];
  *max = maxs[0];
  for( j = 1; j < BRANCHTABLE[branching]; j++ ) {
    if( mins[j] < *min ) *min = mins[j];
    if( maxs[j] > *max ) *max = maxs[j];
  }
  tree[currtree][myindex].min = *min;
  tree[currtree][myindex].max = *max;

} // fill

// getMinMax - find the min and max values of the voxel 
//   based at data->values[index]
template <class T>
void
BonoTreeCL<T>::getMinMax( int currtree, int index, T* min, T* max ) {
  T v;
  *min = *max = data[currtree]->values[index];
  
  v = data[currtree]->values[index+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx*data[currtree]->ny];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx*data[currtree]->ny+1];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx*data[currtree]->ny + 
			    data[currtree]->nx];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  v = data[currtree]->values[index+data[currtree]->nx*data[currtree]->ny + 
			    data[currtree]->nx + 1];  
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;
} // getMinMax

// findBricks - determine brick boundaries that guarantee all siblings
//     of a node are in the same brick
template <class T>
void
BonoTreeCL<T>::findBricks() {

  for( int currtree = 0; currtree < numtrees; currtree++ ) {
    int brick = 0;
    int currSiblings = 1;
    int oldsize = 1;
    int* oldqueue = new int[oldsize];
    int* newqueue;
    int i;
    
    oldqueue[0] = 0;
    int newsize = BRANCHTABLE[ (int)tree[currtree][0].branching ];
    int bsize = 2 * sizeof(T);
    brickRanges[currtree][0] = new int[2];
    brickRanges[currtree][0][0] = 0;

    for( int currdepth = depth[currtree]; currdepth > 0; currdepth-- ) {
      int index = 0;
      int nextsize = 0;
      newqueue = new int[newsize];
      
      for( i = 0; i < oldsize; i++ ) {
	int j;
	int branch_i = 
	  BRANCHTABLE[ (int)tree[currtree][oldqueue[i]].branching ];
	
	// make sure all children will fit
	if( branch_i * 2 * sizeof(T) > nodebricksize - bsize ) {
	  // done with this brick
	  brickRanges[currtree][brick][1] = currSiblings - 1;
	  brick++;
	  // set up next brick
	  brickRanges[currtree][brick] = new int[2];
	  brickRanges[currtree][brick][0] = currSiblings;
	  bsize = 0;
	}
	
	// add node to brick
	bsize += 2 * sizeof(T) * branch_i;
	
	// add node's children to newqueue
	for( j = 0; j < branch_i; j++ ) {
	  newqueue[index] = tree[currtree][oldqueue[i]].child + j;
	  nextsize += 
	    BRANCHTABLE[ (int)tree[currtree][ newqueue[index] ].branching ];
	  index++;
	}
	currSiblings += branch_i;
	
      } // i = 0 .. oldsize-1
      delete [] oldqueue;
      oldqueue = newqueue;
      oldsize = newsize;
      newsize = nextsize;
      
    } // currdepth = depth .. 1
    
    delete [] newqueue;
    brickRanges[currtree][brick][1] = currSiblings - 1;
    if( brick < numNodeBricks[currtree] - 1 )
      numNodeBricks[currtree]--;
  } // currtree = 0 .. numtrees-1

} // findBricks

// searchFirstPass - first pass of the tree traversal
//   read in nodes and set up data to be read
template <class T>
int
BonoTreeCL<T>::searchFirstPass( double iso ) {
  static const int BRANCH_Y[] = {2, 2, 3, 3, 2, 2, 3, 3};
  static const int BRANCH_Z[] = {2, 2, 2, 2, 3, 3, 3, 3};

  int isoCells = 0;
  for( int currtree = 0; currtree < numtrees; currtree++ ) {
    int i, j;
    int idx;
    int NX = data[currtree]->nx;
    int NXNY = data[currtree]->nx * data[currtree]->ny;
    
    int oldsize = 1;
    int* oldqueue = new int[oldsize];
    int* newqueue;
    
    oldqueue[0] = 0;
    int newsize = BRANCHTABLE[ (int)tree[currtree][0].branching ];
    
    for( int currdepth = depth[currtree]; currdepth > 0; currdepth-- ) {
      idx = 0;
      int nextsize = 0;
      newqueue = new int[newsize];
      
      // find nodes at this level that span the isovalue
      for( i = 0; i < oldsize; i++ ) {
	if( tree[currtree][ oldqueue[i] ].min <= iso &&
	    tree[currtree][ oldqueue[i] ].max >= iso ) {
	  for( j = 0; 
	       j < BRANCHTABLE[ (int)tree[currtree][oldqueue[i]].branching ];
	       j++ ){
	    newqueue[idx] = tree[currtree][oldqueue[i]].child + j;
	    nextsize += 
	      BRANCHTABLE[ (int)tree[currtree][ newqueue[idx] ].branching ];
	    idx++;
	  }
	  
	} // if node oldqueue[i] spans iso
	
      } // i = 0 .. size-1
      
      if( currdepth > 1 ) {
	delete [] oldqueue;
	oldqueue = newqueue;
	oldsize = idx;
	newsize = nextsize;
      }
    } // currdepth = depth .. 1

    newsize = idx;
    // finish with level 0
    for( int loopvar = 0; loopvar < oldsize; loopvar++ ) {
      if( tree[currtree][ oldqueue[loopvar] ].min <= iso &&
	  tree[currtree][ oldqueue[loopvar] ].max >= iso ) {
	int numy, numz;
	int firstchild = tree[currtree][ oldqueue[loopvar] ].child;
	
	switch( (int)tree[currtree][ oldqueue[loopvar] ].branching ) {
	case 0:
	case 1: {
	  numy = BRANCH_Y[ (int)tree[currtree][firstchild].branching ];
	  numz = BRANCH_Z[ (int)tree[currtree][firstchild].branching ];
	} break;
	case 2:
	case 3: {
	  numy = BRANCH_Y[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Y[ (int)tree[currtree][firstchild+1].branching ] - 1;
	  numz = BRANCH_Z[ (int)tree[currtree][firstchild].branching ];
	} break;
	case 4:
	case 5: {
	  numy = BRANCH_Y[ (int)tree[currtree][firstchild].branching ];
	  numz = BRANCH_Z[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Z[ (int)tree[currtree][firstchild+1].branching ] - 1;
	} break;
	case 6: {
	  numy = BRANCH_Y[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Y[ (int)tree[currtree][firstchild+1].branching ] - 1;
	  numz = BRANCH_Z[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Z[ (int)tree[currtree][firstchild+2].branching ] - 1;
	} break;
	case 7: {
	  numy = BRANCH_Y[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Y[ (int)tree[currtree][firstchild+2].branching ] - 1;
	  numz = BRANCH_Z[ (int)tree[currtree][firstchild].branching ] +
	    BRANCH_Z[ (int)tree[currtree][firstchild+4].branching ] - 1;
	} break;
	};
	int start = indices[currtree][ tree[currtree][firstchild].child ];
	for( i = 0; i < numz; i++ ) {
	  idx = start + i*NXNY;
	  for( j = 0; j < numy; j++ ) {
	    brickstoread[currtree][idx>>shiftamt] = gen;
	    brickstoread[currtree][(idx+5)>>shiftamt] = gen;
	    idx += NX;
	  }
	}
      } // if oldqueue[loopvar] spans iso
    } // loopvar = 0 .. oldsize
    for( i = 0; i < newsize; i++ )
      isoCells += BRANCHTABLE[ (int)tree[currtree][ newqueue[i] ].branching ];

    delete [] oldqueue;
    delete [] newqueue;
    
  } // currtree = 0 .. numtrees-1

  return isoCells;
} // searchFirstPass

// searchSecondPass - second pass of tree traversal
//    perform interpolations in cells
template <class T>
void
BonoTreeCL<T>::searchSecondPass( int currtree, int myindex, int d, 
				 double iso ) {

  if( tree[currtree][myindex].min <= iso && 
      tree[currtree][myindex].max >= iso ) {

    if( d == 0 ) {
      int start = tree[currtree][myindex].child;
      for( int i = 0; 
	   i < BRANCHTABLE[ (int)tree[currtree][myindex].branching ];
	   i++ ) {
	if( circular[currtree] == NONCIRCULAR )
	  mcube->interpRegular( currtree, indices[currtree][start+i], iso );
	else
	  mcube->interpCircular( currtree, indices[currtree][start+i], iso );
      }
      return;
    } // if d == 0

    d--;
    searchSecondPass( currtree, tree[currtree][myindex].child, d, iso );
    switch( (int)tree[currtree][myindex].branching ) {
    case 0: 
      break;
    case 1: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
    } break;
    case 2: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
    } break;
    case 3: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso );
    } break;
    case 4: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
    } break;
    case 5: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso );
    } break;
    case 6: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso );
    } break;
    case 7: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 4, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 5, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 6, d, iso );
      searchSecondPass( currtree, tree[currtree][myindex].child + 7, d, iso );
    } break;
    }; // switch

  } // if tree[myindex] spans iso
} // searchSecondPass
} // End namespace Phil


#endif

