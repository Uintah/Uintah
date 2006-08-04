
/* TbonTreeOOC2.h
   class declarations and code for the T-BON tree
   Out-of-Core, algorithm 2

   Packages/Philip Sutton
   April 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_TREE_OOC2_H__
#define __TBON_TREE_OOC2_H__

#include "TreeUtils.h"
#include "mcubeOOC.h"
#include "stack.h"
#include "nodelist.h"
#include "datalist.h"
#include <stdio.h>
#include <math.h>
#include <strings.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

template <class T>
struct Data {
  int nx, ny, nz;
  int size;

  T* values;
};

struct Stackcell {
  int index;
  int x, y, z;
  Stackcell( int i, int u, int v, int w ) : index(i), x(u), y(v), z(w) { }
  Stackcell() : index(0), x(0), y(0), z(0) { }
  void print() {
    cout << index << ": " << x << " " << y << " " << z << endl;
  }
};

template <class T>
class TbonTreeOOC2 {
public:
  // constructor for preprocessing
  TbonTreeOOC2( int nx, int ny, int nz, int nb, int db );
  // constructor for execution
  TbonTreeOOC2( const char* filename, int nlistsize, int dlistsize );
  // destructor
  ~TbonTreeOOC2();

  // operations on the tree
  void readData( const char* filename );
  void fillTree( );
  void writeTree( const char* meta, const char* base, 
		  const char* newdata, int num );
  GeomTriGroup* search( double iso, const char* treefile, 
			const char* datafile, int timechanged );

  // accessors
  int getDepth() { return depth; }

  // constants
  static const int TIME_CHANGED;
  static const int TIME_SAME;
protected:

private:
  // structure of the tree - preprocessing
  Node<T>* tree;
  Data<T>* data;
 
  //structure of the tree -  execution
  NodeList<T>* nodelist;
  DataList<T>* datalist;
  Stack< Stackcell >* nodestack;
  MCubesOOC<T> *mcube;

  // arrays used for indexing into the bricked data
  int* Xarray;
  int* Yarray;
  int* Zarray;
  
  // properties
  int depth;
  int numnodes;
  int numleaves;
  int numNodeBricks;
  int numDataBricks;
  int nodelistsize;
  int datalistsize;

  // auxiliary stuff
  int nodebricksize;
  int databricksize;
  int entriesPerBrick;
  int** brickRanges;
  int currBrick;
  char* nodeBrickInMem;

  // flag for destructor
  int deleteExecutionStuff;

  FILE* currtree;
  FILE* currdata;

  // private methods
  void reorderTree( char* branchArray, int* sibArray );
  void fill( int myindex, int d, int x, int y, int z, T* min, T* max );
  void getMinMax( int x, int y, int z, T* min, T* max );
  void findBricks();
  void reorderBricks();
  void findBrickOrder( int idx, int d, int* order, int& curr );
  void readBricks( int x, int y, int z, int branching );

  int searchFirstPass( double iso );
  void searchSecondPass( int myindex, int d, double iso, 
			 int x, int y, int z );
};

int maxqsize;

// define constants
template <class T>
const int TbonTreeOOC2<T>::TIME_CHANGED = 0;
template <class T>
const int TbonTreeOOC2<T>::TIME_SAME = 1;

// TbonTreeOOC2 - constructor for preprocessing
template <class T>
TbonTreeOOC2<T>::TbonTreeOOC2( int nx, int ny, int nz, int nb, int db ) {
  int idx;

  databricksize = db;
  int n = (int)cbrt( (double)databricksize/(double)sizeof(T) );
  ASSERT( cbrt( (double)databricksize/(double)sizeof(T) ) == n );

  // set up data
  data = new Data<T>;
  data->nx = nx; data->ny = ny; data->nz = nz;

  int numx = (int)(nx/n);
  if( numx != (float)nx/(float)n )
    numx++;
  int numy = (int)(ny/n);
  if( numy != (float)ny/(float)n )
    numy++;
  int numz = (int)(nz/n);
  if( numz != (float)nz/(float)n )
    numz++;

  data->size = (databricksize/(int)sizeof(T))*numx*numy*numz;
  data->values = new T[data->size];
  numDataBricks = numx*numy*numz;

  // these char arrays hold a representation of the spatial range of the data
  // (in binary)
  char* xa = new char[8*sizeof(T)];
  char* ya = new char[8*sizeof(T)];
  char* za = new char[8*sizeof(T)];

  fillArray( xa, data->nx - 2, 8*sizeof(T) );
  fillArray( ya, data->ny - 2, 8*sizeof(T) );
  fillArray( za, data->nz - 2, 8*sizeof(T) );

  // find first non-zero entry - that corresponds to the depth of the tree
  for( idx = 8*sizeof(T) - 1; 
       idx >= 0 && xa[idx] != 1 && ya[idx] != 1 && za[idx] != 1;
       idx-- );
  depth = idx;
  cout << "depth = " << depth << endl;

  // find how many nodes are needed
  int num = 1;
  int leaves = 0;
  countNodes( xa, ya, za, idx, &num, &leaves );
  numnodes = num;
  numleaves = leaves;
  cout << "Tree has " << numnodes << " nodes (" << numleaves << " leaves)"
       << endl;

  // set up information for bricking nodes
  nodebricksize = nb;
  // round up to find number of node bricks
  float temp = (float)numnodes * 2.0 * (float)sizeof(T) / (float)nodebricksize;
  numNodeBricks = ((float)(temp - (int)temp)>= 0.5) ? 
    (int)temp + 2 : (int)temp + 1;
  brickRanges = new int*[numNodeBricks];

  cout << "There are " << numNodeBricks << " node bricks and " 
       << numDataBricks << " data bricks" << endl;

  // set up information for bricking data
  Xarray = new int[data->nx];
  Yarray = new int[data->ny];
  Zarray = new int[data->nz];
  for( int x = 0; x < data->nx; x++ ) 
    Xarray[x] = (x/n)*n*n*n + (x%n);
  for( int y = 0; y < data->ny; y++ )
    Yarray[y] = (y/n)*n*n*data->nx + (y%n)*n;
  for( int z = 0; z < data->nz; z++ )
    Zarray[z] = (z/n)*n*data->nx*data->ny + (z%n)*n*n;

  // allocate tree structure
  tree = new Node<T>[numnodes];
  
  // construct tree skeleton
  int rootindex = 0;
  char* branchArray = new char[numnodes];
  int* sibArray = new int[numnodes];

  createTree( xa, ya, za, idx, &rootindex, branchArray, sibArray, 0);
  reorderTree( branchArray, sibArray );

  // clean up
  delete [] branchArray;
  delete [] sibArray;
  delete [] xa;
  delete [] ya;
  delete [] za;
  delete [] lo;
  deleteExecutionStuff = 0;  // (for destructor)

} // TbonTreeOOC2

// TbonTreeOOC2 - constructor for execution
template <class T>
TbonTreeOOC2<T>::TbonTreeOOC2( const char* filename, int nlistsize, 
			     int dlistsize ) {
  int i;
  nodelistsize = nlistsize;
  datalistsize = dlistsize;

  FILE* metafile = fopen( filename, "r" );
  if( !metafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // read tree parameters
  fscanf( metafile, "%d\n%d\n", &numnodes, &depth );
  fscanf( metafile, "%d %d\n", &nodebricksize, &numNodeBricks);
  fscanf( metafile, "%d %d\n", &databricksize, &numDataBricks );
  entriesPerBrick = nodebricksize / (int)sizeof(T);

  // set up and read data parameters
  data = new Data<T>;
  fscanf( metafile, "%d %d %d\n", &(data->nx), &(data->ny), &(data->nz) );
  fscanf( metafile, "%d %d\n", &(data->size), &numleaves );
  data->values = 0;

  brickRanges = new int*[numNodeBricks];
  for( i = 0; i < numNodeBricks; i++ ) {
    brickRanges[i] = new int[2];
    fread( &brickRanges[i][0], sizeof(int), 1, metafile );
    fread( &brickRanges[i][1], sizeof(int), 1, metafile );
  }

  tree = new Node<T>[numnodes];
  nodeBrickInMem = new char[numNodeBricks];
  bzero( nodeBrickInMem, numNodeBricks );

  for( i = 0; i < numnodes; i++ ) {
    fread( &(tree[i].branching), sizeof(char), 1, metafile );
    fread( &(tree[i].child), sizeof(int), 1, metafile );
  }
  fclose( metafile );

  // set up information for bricking data
  Xarray = new int[data->nx];
  Yarray = new int[data->ny];
  Zarray = new int[data->nz];
  int n = (int)cbrt( (double)databricksize/(double)sizeof(T) );
  ASSERT( cbrt( (double)databricksize/(double)sizeof(T) ) == n );
  for( int x = 0; x < data->nx; x++ ) 
    Xarray[x] = (x/n)*n*n*n + (x%n);
  for( int y = 0; y < data->ny; y++ )
    Yarray[y] = (y/n)*n*n*data->nx + (y%n)*n;
  for( int z = 0; z < data->nz; z++ )
    Zarray[z] = (z/n)*n*data->nx*data->ny + (z%n)*n*n;

  // allocate tree structure
  datalist = new DataList<T>( datalistsize, databricksize, 
			      Xarray, Yarray, Zarray );

  mcube = new MCubesOOC<T>(data, datalist);
  maxqsize = 0;

  deleteExecutionStuff = 1;
} // TbonTreeOOC2

// ~TbonTreeOOC2 - Destructor
template <class T>
TbonTreeOOC2<T>::~TbonTreeOOC2() {
  // common to preprocessing and execution
  delete [] brickRanges;
  delete [] Xarray;
  delete [] Yarray;
  delete [] Zarray;

  if( deleteExecutionStuff ) {
    // for execution only
    //    delete nodelist;
    delete datalist;
  } else {
    // for preprocessing only
    delete [] data->values;
    delete [] tree;
  }

  delete data;
} // ~TbonTreeOOC2

// readData - read data from file into the "data" structure
template <class T>
void 
TbonTreeOOC2<T>::readData( const char* filename ) {
  FILE* datafile = fopen( filename, "r" );
  if( !datafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // read values
  int bufsize = data->nx * data->ny * data->nz;
  T* buffer = new T[bufsize];
  unsigned long n = fread( buffer, sizeof(T), bufsize, datafile );

  // make sure all values were read
  if( n != (unsigned long)bufsize ) {
    cerr << "Error: only " << n << "/" << bufsize << " objects read from "
	 << filename << endl;
  }
  fclose( datafile );

  // reorder values
  int i = 0;
  for( int z = 0; z < data->nz; z++ ) {
    for( int y = 0; y < data->ny; y++ ) {
      for( int x = 0; x < data->nx; x++ ) {
	int index = Xarray[x] + Yarray[y] + Zarray[z];
	data->values[index] = buffer[i++];
      }
    }
  }
  delete buffer;
} // readData

// fillTree - fill in skeleton tree with min, max, etc.
template <class T>
void 
TbonTreeOOC2<T>::fillTree( ) {
  curr = 0;
  fill( 0, depth, 0, 0, 0, &(tree[0].min), &(tree[0].max) );
} // fillTree


// writeTree - write the tree to disk
template <class T>
void 
TbonTreeOOC2<T>::writeTree( const char* meta, const char* base, 
			const char* newdata, int num ) {
  char filename[128];
  int i;
  FILE* out;

  // if this is the first write, write the tree metafile, too
  //  (skeleton)
  if( num == 0 ) {
    cout << "Writing tree metafile" << endl;
    out = fopen( meta, "w" );
    if( !out ) {
      cerr << "Error: cannot open file " << meta << endl;
      return;
    }

    // determine brick boundaries
    findBricks();

    // write everything
    fprintf(out, "%d\n%d\n", numnodes, depth);
    fprintf(out, "%d %d\n", nodebricksize, numNodeBricks );
    fprintf(out, "%d %d\n", databricksize, numDataBricks );
    fprintf(out,"%d %d %d\n%d %d\n", data->nx, data->ny, data->nz,
	    data->size, numleaves );

    for( i = 0; i < numNodeBricks; i++ ) {
      fwrite( &brickRanges[i][0], sizeof(int), 1, out );
      fwrite( &brickRanges[i][1], sizeof(int), 1, out );
    }

    for( i = 0; i < numnodes; i++ ) {
      fwrite( &(tree[i].branching), sizeof(char), 1, out );
      fwrite( &(tree[i].child), sizeof(int), 1, out );
    }

    fclose(out);
  } 

  cout << "Writing tree #" << num << endl;

  // write the data specific to tree #num
  sprintf(filename, "%s%d", base, num );
  out = fopen( filename, "w" );

  if( !out ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  for( i = 0; i < numNodeBricks; i++ ) {
    int bsize = 0;
    // write all the nodes in the brick
    for( int j = brickRanges[i][0]; j <= brickRanges[i][1]; j++ ) {
      fwrite( &(tree[j].min), sizeof(T), 1, out );
      fwrite( &(tree[j].max), sizeof(T), 1, out );
      bsize += 2*sizeof(T);
    }
    // pad if necessary
    if( bsize < nodebricksize ) {
      char* padding = new char[nodebricksize - bsize];
      fwrite( padding, 1, nodebricksize - bsize, out );
      delete [] padding;
    }
  }
  
  fclose( out );

  out = fopen( newdata, "w" );
  if( !out ) {
    cerr << "Error: cannot open file " << newdata << endl;
    return;
  }
  
  cout << "Rewriting data" << endl;

  fwrite( data->values, sizeof(T), data->size, out ); 
  fclose( out );

} // writeTree

// search - find isosurface
template <class T>
GeomTriGroup*
TbonTreeOOC2<T>::search( double iso, const char* treefile, const char* datafile,
			int timechanged ) {
  int i, d;
  int brick;

  currtree = fopen( treefile, "r" );
  currdata = fopen( datafile, "r" );
  
  currBrick = 0;
  if( timechanged == TIME_CHANGED ) {
    bzero( nodeBrickInMem, numNodeBricks );

    // read in first node brick
    T* buffer = new T[entriesPerBrick];
    int numentries = brickRanges[currBrick][1] - 
      brickRanges[currBrick][0] + 1;
    fread( buffer, sizeof(T), entriesPerBrick, currtree );
    int j, k;
    for( j = 0, k = 0; k < numentries; j+=2, k++ ) {
      tree[ brickRanges[currBrick][0] + k ].min = buffer[j];
      tree[ brickRanges[currBrick][0] + k ].max = buffer[j+1];
    }
    delete [] buffer;
    nodeBrickInMem[currBrick] = 1;
 
    datalist->reset();
  }

  iotimer_t t0, t1;
  t0 = read_time();
  int n = searchFirstPass( iso );
  t1 = read_time();
  PrintTime(t0,t1,"First Pass: ");
  //  cout << "n = " << n << endl;

  mcube->setResolution( 0 );
  mcube->reset( n );

  t0 = read_time();
  searchSecondPass( 0, depth, iso, 0, 0, 0 );
  t1 = read_time();
  PrintTime(t0,t1,"SecondPass: ");
  
  fclose( currtree );
  fclose( currdata );
  cout << "#triangles = " << mcube->triangles->getSize() << endl;

  static int firsttime = 1;
  if( firsttime ) {
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
TbonTreeOOC2<T>::reorderTree( char* branchArray, int* sibArray ) {
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
  for( currdepth = depth; currdepth > 0; currdepth-- ) {
    size = newsize;
    newsize = 0;
    last = curr;
    queue = new int[size];
    for( i = 0; i < size; i++ ) {
      queue[i] = curr;
      tree[node].branching = branchArray[curr];
      tree[node].child = index;
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
      tree[node++].branching = branchArray[ queue[i] + j ];
  }
  delete [] queue;

} // reorderTree

// findBricks - determine brick boundaries that guarantee all siblings
//     of a node are in the same brick
template <class T>
void
TbonTreeOOC2<T>::findBricks() {
  int brick = 0;
  int currSiblings = 1;
  int oldsize = 1;
  int* oldqueue = new int[oldsize];
  int* newqueue;
  int i;

  oldqueue[0] = 0;
  int newsize = BRANCHTABLE[ (int)tree[0].branching ];
  int bsize = 2*sizeof(T);
  brickRanges[0] = new int[2];
  brickRanges[0][0] = 0;

  for( int currdepth = depth; currdepth > 0; currdepth-- ) {
    int index = 0;
    int nextsize = 0;
    newqueue = new int[newsize];
    
    for( i = 0; i < oldsize; i++ ) {
      int j;
      int branch_i = BRANCHTABLE[ (int)tree[oldqueue[i]].branching ];

      // make sure all children will fit
      if( branch_i * 2 * sizeof(T) > nodebricksize - bsize ) {
	// done with this brick
	brickRanges[brick][1] = currSiblings - 1;
	brick++;
	// set up next brick
	brickRanges[brick] = new int[2];
	brickRanges[brick][0] = currSiblings;
	bsize = 0;
      }

      // add node to brick
      bsize += 2*sizeof(T) * branch_i;
      
      // add node's children to newqueue
      for( j = 0; j < branch_i; j++ ) {
	newqueue[index] = tree[oldqueue[i]].child + j;
	nextsize += BRANCHTABLE[ (int)tree[ newqueue[index] ].branching ];
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
  brickRanges[brick][1] = currSiblings - 1;
  if( brick < numNodeBricks - 1 )
    numNodeBricks--;

  //  reorderBricks();
} // findBricks

// reorderBricks - reorder bricks in DFS order
template <class T>
void
TbonTreeOOC2<T>::reorderBricks() {
  int** newranges;
  int* neworder;
  int i;

  neworder = new int[numNodeBricks];
  newranges = new int*[numNodeBricks];
  for( i = 0; i < numNodeBricks; i++ )
    newranges[i] = new int[2];

  // traverse tree in DFS order to find neworder
  int curr = 0;
  findBrickOrder( 0, depth, neworder, curr );

  // fill in newranges based on neworder
  for( i = 0; i < numNodeBricks; i++ ) {
    newranges[i][0] = brickRanges[ neworder[i] ][0];
    newranges[i][1] = brickRanges[ neworder[i] ][1];
  }

  // make newranges be the official brickRanges
  delete [] brickRanges;
  brickRanges = newranges;
} // reorderBricks

// findBrickOrder - traverse tree to get DFS order of bricks
template <class T>
void
TbonTreeOOC2<T>::findBrickOrder( int idx, int d, int* order, int& curr ) {
  int i;
  // find the brick that node idx is in
  for( i = 0; i < numNodeBricks; i++ ) {
    if( brickRanges[i][0] <= idx && brickRanges[i][1] >= idx )
      break;
  }
  // if that brick isn't in the order, add it
  if( notinlist( i, order, curr ) ) {
    order[ curr++ ] = i;
  }

  if( d == 0 )
    return;

  d--;

  for( i = 0; i < BRANCHTABLE[ (int)tree[idx].branching ]; i++ ) {
    findBrickOrder( tree[idx].child + i, d, order, curr );    
  }
} // findBrickOrder

// fill - recursively fill each node in the tree 
template <class T>
void
TbonTreeOOC2<T>::fill( int myindex, int d, int x, int y, int z, T* min, T* max ) {
  int j;
  T mins[8], maxs[8];
  int branching = (int)tree[myindex].branching;

  if( d == 0 ) {
    switch( branching ) {
    case 0: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
    } break;
    case 1: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x+1, y, z, &mins[1], &maxs[1] );
    } break;
    case 2: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x, y+1, z, &mins[1], &maxs[1] );
    } break;
    case 3: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x+1, y, z, &mins[1], &maxs[1] );
      getMinMax( x, y+1, z, &mins[2], &maxs[2] );
      getMinMax( x+1, y+1, z, &mins[3], &maxs[3] );
    } break;
    case 4: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x, y, z+1, &mins[1], &maxs[1] );
    } break;
    case 5: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x+1, y, z, &mins[1], &maxs[1] );
      getMinMax( x, y, z+1, &mins[2], &maxs[2] );
      getMinMax( x+1, y, z+1, &mins[3], &maxs[3] );
    } break;
    case 6: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x, y+1, z, &mins[1], &maxs[1] );
      getMinMax( x, y, z+1, &mins[2], &maxs[2] );
      getMinMax( x, y+1, z+1, &mins[3], &maxs[3] );
    } break;
    case 7: {
      getMinMax( x, y, z, &mins[0], &maxs[0] );
      getMinMax( x+1, y, z, &mins[1], &maxs[1] );
      getMinMax( x, y+1, z, &mins[2], &maxs[2] );
      getMinMax( x+1, y+1, z, &mins[3], &maxs[3] );
      getMinMax( x, y, z+1, &mins[4], &maxs[4] );
      getMinMax( x+1, y, z+1, &mins[5], &maxs[5] );
      getMinMax( x, y+1, z+1, &mins[6], &maxs[6] );
      getMinMax( x+1, y+1, z+1, &mins[7], &maxs[7] );
    } break;
    };  // switch(branching)

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

  int step = 1 << d;
  d--;

  fill( tree[myindex].child, d, x, y, z, &mins[0], &maxs[0] );
  switch( branching ) {
  case 0: break;
  case 1: {
    fill( tree[myindex].child + 1, d, x+step, y, z, &mins[1], &maxs[1] );
  } break;
  case 2: {
    fill( tree[myindex].child + 1, d, x, y+step, z, &mins[1], &maxs[1] );
  } break;
  case 3: {
    fill( tree[myindex].child + 1, d, x+step, y, z, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, x, y+step, z, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, x+step, y+step, z, &mins[3], &maxs[3] );
  } break;
  case 4: {
    fill( tree[myindex].child + 1, d, x, y, z+step, &mins[1], &maxs[1] );
  } break;
  case 5: {
    fill( tree[myindex].child + 1, d, x+step, y, z, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, x, y, z+step, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, x+step, y, z+step, &mins[3], &maxs[3] );
  } break;
  case 6: {
    fill( tree[myindex].child + 1, d, x, y+step, z, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, x, y, z+step, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, x, y+step, z+step, &mins[3], &maxs[3] );
  } break;
  case 7: {
    fill( tree[myindex].child + 1, d, x+step, y, z, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, x, y+step, z, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, x+step, y+step, z, &mins[3], &maxs[3] );
    fill( tree[myindex].child + 4, d, x, y, z+step, &mins[4], &maxs[4] );
    fill( tree[myindex].child + 5, d, x+step, y, z+step, &mins[5], &maxs[5] );
    fill( tree[myindex].child + 6, d, x, y+step, z+step, &mins[6], &maxs[6] );
    fill( tree[myindex].child + 7, d, x+step, y+step, z+step, 
	  &mins[7], &maxs[7] );
  } break; 
  }; // switch( branching )
  
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
TbonTreeOOC2<T>::getMinMax( int x, int y, int z, T* min, T* max ) {
  T v;
  int index = Xarray[x] + Yarray[y] + Zarray[z];
  *min = *max = data->values[index];
  
  index = Xarray[x+1] + Yarray[y] + Zarray[z];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x] + Yarray[y+1] + Zarray[z];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x+1] + Yarray[y+1] + Zarray[z];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x] + Yarray[y] + Zarray[z+1];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x+1] + Yarray[y] + Zarray[z+1];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x] + Yarray[y+1] + Zarray[z+1];
  v = data->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[x+1] + Yarray[y+1] + Zarray[z+1];
  v = data->values[index];  
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;
} // getMinMax

template <class T>
void
TbonTreeOOC2<T>::readBricks( int x, int y, int z, int branching ) {
  static const int BX[] = {1, 2, 1, 2, 1, 2, 1, 2};
  static const int BY[] = {1, 1, 2, 2, 1, 1, 2, 2};
  static const int BZ[] = {1, 1, 1, 1, 2, 2, 2, 2};
  static const int shiftamt = log((float)databricksize)/log(2.0) - 
    log((float)sizeof(T))/log(2.0);

  Corner bounds[7];
  bounds[0].set( x+BX[branching], y,               z               );
  bounds[1].set( x,               y+BY[branching], z               );
  bounds[2].set( x+BX[branching], y+BY[branching], z               );
  bounds[3].set( x,               y,               z+BZ[branching] );
  bounds[4].set( x+BX[branching], y,               z+BZ[branching] );
  bounds[5].set( x,               y+BY[branching], z+BZ[branching] );
  bounds[6].set( x+BX[branching], y+BY[branching], z+BZ[branching] );

  int bricks[8];
  int index = Xarray[x] + Yarray[y] + Zarray[z];
  bricks[0] = (index >> shiftamt);
  for( int i = 0; i < 7; i++ ) {
    index = Xarray[ bounds[i].x ] + 
      Yarray[ bounds[i].y ] + 
      Zarray[ bounds[i].z ];
    bricks[i+1] = (index >> shiftamt);
  }
  datalist->addBricks( bricks, currdata );
} // readBricks

// searchFirstPass - first pass of the tree traversal
//   read in nodes and set up data to be read
template <class T>
int
TbonTreeOOC2<T>::searchFirstPass( double iso ) {
  static const int BX[] = {1, 2, 1, 2, 1, 2, 1, 2};
  static const int BY[] = {1, 1, 2, 2, 1, 1, 2, 2};
  static const int BZ[] = {1, 1, 1, 1, 2, 2, 2, 2};
  
  static const int OFFSET = numnodes-numleaves;

  int i, j;
  int isoCells = 0;
  int idx;

  int oldsize = 1;
  int* oldqueue = new int[oldsize];
  int* newqueue;

  oldqueue[0] = 0;
  int newsize = BRANCHTABLE[ (int)tree[0].branching ];

  for( int currdepth = depth; currdepth > 0; currdepth-- ) {
    idx = 0;
    int nextsize = 0;
    newqueue = new int[newsize];
    if( newsize > maxqsize )
      maxqsize = newsize;

    // find nodes at this level that span the isovalue
    for( i = 0; i < oldsize; i++ ) {
      if( tree[ oldqueue[i] ].min <= iso && tree[ oldqueue[i] ].max >= iso ) {
	for( j = 0; j < BRANCHTABLE[ (int)tree[oldqueue[i]].branching ]; j++ ){
	  newqueue[idx] = tree[oldqueue[i]].child + j;
	  nextsize += BRANCHTABLE[ (int)tree[ newqueue[idx] ].branching ];
	  idx++;
	}

	// read in needed node bricks
	// find brick containing child      
	for( ; tree[ oldqueue[i] ].child > brickRanges[ currBrick ][1]; 
	     currBrick++ );
	
	// read brick if not in memory
	if( currBrick < numNodeBricks && !nodeBrickInMem[ currBrick ] ) {
	  int numentries = brickRanges[currBrick][1] - 
	    brickRanges[currBrick][0] + 1;
	  if( ftell(currtree) != currBrick*nodebricksize ) 
	    fseek( currtree, currBrick*nodebricksize, SEEK_SET );
	  T* buffer = new T[ entriesPerBrick ];
	  fread( buffer, sizeof(T), entriesPerBrick, currtree );
	  int k = 0;
	  for( j = 0; k < numentries; j+=2, k++ ) {
	    tree[ brickRanges[currBrick][0] + k ].min = buffer[j];
	    tree[ brickRanges[currBrick][0] + k ].max = buffer[j+1];
	  }
	  delete [] buffer;
	  nodeBrickInMem[currBrick] = 1;
	} // done reading currBrick
	
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
  
  // count number of isosurface cells
  for( i = 0; i < newsize; i++ ) 
    isoCells += BRANCHTABLE[ (int)tree[ newqueue[i] ].branching ];

  delete [] oldqueue;
  delete [] newqueue;
  
  return isoCells;
} // searchFirstPass

template <class T>
void
TbonTreeOOC2<T>::searchSecondPass( int myindex, int d, double iso, 
				  int x, int y, int z ) {
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {
    if( d == 0 ) {
      readBricks( x, y, z, (int)tree[myindex].branching );
      mcube->interp( x, y, z, (int)tree[myindex].branching, (float)iso );
      return;
    } else {
      int step = 1 << d;
      d--;
      searchSecondPass( tree[myindex].child, d, iso, x, y, z );
      switch( (int)tree[myindex].branching ) {
      case 0: 
	break;
      case 1: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x+step, y, z );
      } break;
      case 2: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x, y+step, z );
      } break;
      case 3: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x+step, y, z );
	searchSecondPass( tree[myindex].child + 2, d, iso, x, y+step, z );
	searchSecondPass( tree[myindex].child + 3, d, iso, x+step, y+step, z );
      } break;
      case 4: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x, y, z+step );
      } break;
      case 5: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x+step, y, z );
	searchSecondPass( tree[myindex].child + 2, d, iso, x, y, z+step );
	searchSecondPass( tree[myindex].child + 3, d, iso, x+step, y, z+step );
      } break;
      case 6: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x, y+step, z );
	searchSecondPass( tree[myindex].child + 2, d, iso, x, y, z+step );
	searchSecondPass( tree[myindex].child + 3, d, iso, x, y+step, z+step );
      } break;
      case 7: {
	searchSecondPass( tree[myindex].child + 1, d, iso, x+step, y, z );
	searchSecondPass( tree[myindex].child + 2, d, iso, x, y+step, z );
	searchSecondPass( tree[myindex].child + 3, d, iso, x+step, y+step, z );
	searchSecondPass( tree[myindex].child + 4, d, iso, x, y, z+step );
	searchSecondPass( tree[myindex].child + 5, d, iso, x+step, y, z+step );
	searchSecondPass( tree[myindex].child + 6, d, iso, x, y+step, z+step );
	searchSecondPass( tree[myindex].child + 7, d, iso, 
			  x+step,y+step,z+step );
      } break;
      }; // switch
      
    } // if( d == 0 ) 
  } // if( tree[myindex] spans iso )
} // searchSecondPass
} // End namespace Phil


#endif


