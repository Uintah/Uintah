
/* TbonTreeCL.h
   class declarations and code for the T-BON tree, curvilinear implementation

   Packages/Philip Sutton
   June 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_TREE_CL_H__
#define __TBON_TREE_CL_H__

#include "TreeUtils.h"
#include "TriGroup.h"
#include "mcube.h"
#include <stdio.h>
#include <math.h>
#include <strings.h>

namespace Phil {
using namespace SCIRun;
using namespace std;

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
class TbonTreeCL {
public:
  // constructor for preprocessing
  TbonTreeCL( int* nx, int* ny, int* nz, const char* geomfile, 
	      int numzones, int* c, int nb, int db );
  // constructor for execution
  TbonTreeCL( const char* filename, const char* geomfile );
  // destructor
  ~TbonTreeCL();

  // operations on the tree
  void readData( const char* filename );
  void fillTree( );
  void writeTree( const char* meta, const char* base, int num, 
		  const char* newdatafile, const char* newgeomfile );
  GeomTriGroup* search( double iso, const char* treefile, 
			const char* datafile, int timechanged, int res );

  // accessors
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
  Corner** corners;
  int* circular;
  MCubesCL<T>* mcube;

  // arrays used for indexing into the bricked data
  int** Xarray;
  int** Yarray;
  int** Zarray;
  
  int nodesread;
  int dataread;

  // properties
  int* depth;
  int* numnodes;
  int* numleaves;
  int* numNodeBricks;
  int* numDataBricks;

  // auxiliary stuff
  int nodebricksize;
  int databricksize;
  int entriesPerBrick;
  int* previousNodeBricks;
  int* previousDataBricks;
  int*** brickRanges;
  int currBrick;
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
  void fill( int currtree, int myindex, int d, int x, int y, int z,
	     T* min, T* max );
  void getMinMax( int currtree, int x, int y, int z, T* min, T* max );
  void findBricks();

  int searchFirstPass( double iso );
  void searchSecondPass( int currtree, int myindex, int d, double iso,
			 int x, int y, int z, int res );
};

// define constants
template <class T>
const int TbonTreeCL<T>::TIME_CHANGED = 0;
template <class T>
const int TbonTreeCL<T>::TIME_SAME = 1;
template <class T>
const int TbonTreeCL<T>::NONCIRCULAR = 0;
template <class T>
const int TbonTreeCL<T>::CIRCULAR_X = 1;
template <class T>
const int TbonTreeCL<T>::CIRCULAR_Y = 2;
template <class T>
const int TbonTreeCL<T>::CIRCULAR_Z = 3;

// TbonTreeCL - constructor for preprocessing
template <class T>
TbonTreeCL<T>::TbonTreeCL( int* nx, int* ny, int* nz, const char* geomfile, 
			   int numzones, int* c, int nb, int db ) {
  int idx;
  int currtree;
  int** range;

  databricksize = db;
  int n = (int)cbrt( (double)databricksize/(double)sizeof(T) );
  ASSERT( cbrt( (double)databricksize/(double)sizeof(T) ) == n );

  numtrees = numzones;
  circular = new int[numtrees];
  range = new int*[numtrees];
  numnodes = new int[numtrees];
  numleaves = new int[numtrees];
  depth = new int[numtrees];
  tree = new Node<T>*[numtrees];
  corners = new Corner*[numtrees];
  numDataBricks = new int[numtrees];
  Xarray = new int*[numtrees];
  Yarray = new int*[numtrees];
  Zarray = new int*[numtrees];
  int* numx = new int[numtrees];
  int* numy = new int[numtrees];
  int* numz = new int[numtrees];

  // set up data
  data = new DataCL<T>*[numtrees];
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    circular[currtree] = c[currtree];
    range[currtree] = new int[3];

    data[currtree] = new DataCL<T>;
    data[currtree]->nx = nx[currtree]; 
    data[currtree]->ny = ny[currtree]; 
    data[currtree]->nz = nz[currtree];

    numx[currtree] = (int)(nx[currtree] / n);
    if( numx[currtree] != (float)nx[currtree] / (float)n )
      numx[currtree]++;
    numy[currtree] = (int)(ny[currtree] / n);
    if( numy[currtree] != (float)ny[currtree] / (float)n )
      numy[currtree]++;
    numz[currtree] = (int)(nz[currtree] / n);
    if( numz[currtree] != (float)nz[currtree] / (float)n )
      numz[currtree]++;

    data[currtree]->size = (databricksize/(int)sizeof(T))*
      numx[currtree]*numy[currtree]*numz[currtree];
    numDataBricks[currtree] = numx[currtree]*numy[currtree]*numz[currtree];
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
  numNodeBricks = new int[numtrees];
  brickRanges = new int**[numtrees];

  for( currtree = 0; currtree < numtrees; currtree++ ) {
    // set up information for bricking data
    Xarray[currtree] = new int[data[currtree]->nx];
    Yarray[currtree] = new int[data[currtree]->ny];
    Zarray[currtree] = new int[data[currtree]->nz];
    int NX = numx[currtree]*n;
    int NY = numy[currtree]*n;

    for( int x = 0; x < data[currtree]->nx; x++ )
      Xarray[currtree][x] = (x/n)*n*n*n + (x%n);
    for( int y = 0; y < data[currtree]->ny; y++ )
      Yarray[currtree][y] = (y/n)*n*n*NX + (y%n)*n;
    for( int z = 0; z < data[currtree]->nz; z++ )
      Zarray[currtree][z] = (z/n)*n*NX*NY + (z%n)*n*n;

    // read geometric data
    int bufsize = data[currtree]->nx * data[currtree]->ny * data[currtree]->nz;
    VertexCL* buffer = new VertexCL[bufsize];
    for( idx = 0; idx < bufsize; idx++ ) {
      fread( buffer[idx].pos, sizeof(float), 3, geom );
    }

    // brick geometry
    int i = 0;
    for( int z = 0; z < data[currtree]->nz; z++ ) {
      for( int y = 0; y < data[currtree]->ny; y++ ) {
	for( int x = 0; x < data[currtree]->nx; x++ ) {
	  int index = Xarray[currtree][x] +
	    Yarray[currtree][y] + Zarray[currtree][z];
	  data[currtree]->verts[index].pos[0] = buffer[i].pos[0];
	  data[currtree]->verts[index].pos[1] = buffer[i].pos[1];
	  data[currtree]->verts[index].pos[2] = buffer[i].pos[2];
	  i++;
	}
      }
    }
    delete [] buffer;

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
    int leaves = 0;
    countNodes( xa, ya, za, idx, &num, &leaves );
    numnodes[currtree] = num;
    numleaves[currtree] = leaves;
    cout << "Tree[" << currtree << "] has " << numnodes[currtree] 
	 << " nodes (" << numleaves[currtree] << " leaves)" << endl;
    
    // allocate tree structure
    tree[currtree] = new Node<T>[ numnodes[currtree] ];
    corners[currtree] = new Corner[numleaves[currtree]];
    
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
  delete [] numx;
  delete [] numy;
  delete [] numz;
  delete [] xa;
  delete [] ya;
  delete [] za;
  delete [] lo;
  delete [] range;
  deleteExecutionStuff = 0;  // (for destructor)

} // TbonTreeCL

// TbonTreeCL - constructor for execution
template <class T>
TbonTreeCL<T>::TbonTreeCL( const char* filename, const char* geomfile ) {
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
  int n = (int)cbrt( (double)databricksize/(double)sizeof(T) );
  ASSERT( cbrt( (double)databricksize/(double)sizeof(T) ) == n );
  
  entriesPerBrick = nodebricksize / (int)sizeof(T);
  numnodes = new int[numtrees];
  depth = new int[numtrees];
  circular = new int[numtrees];
  data = new DataCL<T>*[numtrees];
  tree = new Node<T>*[numtrees];
  corners = new Corner*[numtrees];
  numNodeBricks = new int[numtrees];
  brickRanges = new int**[numtrees];
  nodeBrickInMem = new char*[numtrees];
  previousNodeBricks = new int[numtrees];
  previousDataBricks = new int[numtrees];
  numDataBricks = new int[numtrees];
  brickstoread = new int*[numtrees];
  datainmem = new char*[numtrees];
  numleaves = new int[numtrees];
  Xarray = new int*[numtrees];
  Yarray = new int*[numtrees];
  Zarray = new int*[numtrees];

  shiftamt = log((float)databricksize)/log(2.0) - log((float)sizeof(T))/log(2.0);
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    data[currtree] = new DataCL<T>;
    fscanf(metafile, "%d\n%d\n", &(numnodes[currtree]), &(depth[currtree]) );
    fscanf(metafile, "%d %d\n", &(numNodeBricks[currtree]),
	   &(numDataBricks[currtree]) );
    fscanf(metafile, "%d %d %d\n", &(data[currtree]->nx), 
	   &(data[currtree]->ny), &(data[currtree]->nz) );
    fscanf(metafile, "%d %d %d\n", &(data[currtree]->size), 
	   &(data[currtree]->cells), &(numleaves[currtree]) );
    fscanf(metafile, "%d\n", &(circular[currtree]) );

    data[currtree]->values = new T[data[currtree]->size];
    data[currtree]->verts = new VertexCL[data[currtree]->size];

    // allocate tree structure
    tree[currtree] = new Node<T>[ numnodes[currtree] ];
    corners[currtree] = new Corner[ numleaves[currtree] ];

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

    brickstoread[currtree] = new int[ numDataBricks[currtree] ];
    datainmem[currtree] = new char[ numDataBricks[currtree] ];
    bzero( brickstoread[currtree], numDataBricks[currtree] * sizeof(int) );
    bzero( datainmem[currtree], numDataBricks[currtree] );
    previousDataBricks[currtree] = ( currtree == 0 ) ?
      0 : previousDataBricks[currtree-1] + 
      numDataBricks[currtree-1]*databricksize;
  }

  // read in tree skeletons
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    for( i = 0; i < numnodes[currtree]; i++ ) {
      fread( &(tree[currtree][i].branching), sizeof(char), 1, metafile );
      fread( &(tree[currtree][i].child), sizeof(int), 1, metafile );
    }
    fread( corners[currtree], sizeof(Corner), numleaves[currtree], metafile );

    // set up information for bricking data
    Xarray[currtree] = new int[data[currtree]->nx];
    Yarray[currtree] = new int[data[currtree]->ny];
    Zarray[currtree] = new int[data[currtree]->nz];

    int numx = (int)(data[currtree]->nx / n);
    if( numx != (float)data[currtree]->nx / (float)n )
      numx++;
    int numy = (int)(data[currtree]->ny / n);
    if( numy != (float)data[currtree]->ny / (float)n )
      numy++;

    int NX = numx*n;
    int NY = numy*n;

    for( int x = 0; x < data[currtree]->nx; x++ )
      Xarray[currtree][x] = (x/n)*n*n*n + (x%n);
    for( int y = 0; y < data[currtree]->ny; y++ )
      Yarray[currtree][y] = (y/n)*n*n*NX + (y%n)*n;
    for( int z = 0; z < data[currtree]->nz; z++ )
      Zarray[currtree][z] = (z/n)*n*NX*NY + (z%n)*n*n;
  }

  fclose( metafile );

  // read in geometry information
  FILE* geom = fopen( geomfile, "r" );
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    for( i = 0; i < data[currtree]->size; i++ )
      fread( data[currtree]->verts[i].pos, sizeof(float), 3, geom );
  }
  fclose( geom );

  mcube = new MCubesCL<T>( data, numtrees, Xarray, Yarray, Zarray );

  gen = 1;
  deleteExecutionStuff = 1;
} // TbonTreeCL

// ~TbonTreeCL - destructor
template <class T>
TbonTreeCL<T>::~TbonTreeCL() {
  int currtree;
  for( currtree = 0; currtree < numtrees; currtree++ ) {
    delete [] tree[currtree];
    delete [] corners[currtree];
    delete [] Xarray[currtree];
    delete [] Yarray[currtree];
    delete [] Zarray[currtree];
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
  delete [] corners;
  delete [] data;
  delete [] Xarray;
  delete [] Yarray;
  delete [] Zarray;

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
} // ~TbonTreeCL

// readData - read data from file into the "data" structures
template <class T>
void
TbonTreeCL<T>::readData( const char* filename ) {
  int currtree;
  FILE* datafile = fopen( filename, "r" );
  if( !datafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  for( currtree = 0; currtree < numtrees; currtree++ ) {
    int bufsize = data[currtree]->nx * data[currtree]->ny * data[currtree]->nz;
    T* buffer = new T[bufsize];
    unsigned long n = fread( buffer, sizeof(T), bufsize, datafile );

    if( n != (unsigned long)bufsize ) {
      cerr << "Error: only " << n << "/" << bufsize
	   << " objects read from " << filename << endl;
    }

    int i = 0;
    for( int z = 0; z < data[currtree]->nz; z++ ) {
      for( int y = 0; y < data[currtree]->ny; y++ ) {
	for( int x = 0; x < data[currtree]->nx; x++ ) {
	  int index = Xarray[currtree][x] +
	    Yarray[currtree][y] + Zarray[currtree][z];
	  data[currtree]->values[index] = buffer[i++];
	}
      }
    }
    delete [] buffer;
  } // currtree = 0 .. numtrees-1
} // readData

// fillTree - fill in tree skeleton with min, max
template <class T>
void
TbonTreeCL<T>::fillTree( ) {
  for( int currtree = 0; currtree < numtrees; currtree++ ) {
    curr = 0;
    fill( currtree, 0, depth[currtree], 0, 0, 0,
	  &(tree[currtree][0].min), &(tree[currtree][0].max) );
  }
} // fillTree

// writeTree - write the tree to disk
template <class T>
void
TbonTreeCL<T>::writeTree( const char* meta, const char* base, int num,
			  const char* newdatafile, const char* newgeomfile ) {
  char filename[80];
  int i;
  FILE* out;
  FILE* newdata;
  int currtree;

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
    fprintf(out, "%d\n", numtrees);
    fprintf(out, "%d %d\n", nodebricksize, databricksize);

    for( currtree = 0; currtree < numtrees; currtree++ ) {
      fprintf(out, "%d\n%d\n", numnodes[currtree], depth[currtree]);
      fprintf(out, "%d %d\n", numNodeBricks[currtree], 
	      numDataBricks[currtree] );
      fprintf(out,"%d %d %d\n%d %d %d\n", 
	      data[currtree]->nx, data[currtree]->ny, data[currtree]->nz,
	      data[currtree]->size, data[currtree]->cells, 
	      numleaves[currtree] );
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
      fwrite( corners[currtree], sizeof(Corner), numleaves[currtree], out );
    }
    fclose(out);

    // also write the new geometry file
    out = fopen( newgeomfile, "w" );
    for( currtree = 0; currtree < numtrees; currtree++ ) {
      for( i = 0; i < data[currtree]->size; i++ )
	fwrite( data[currtree]->verts[i].pos, sizeof(float), 3, out );
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
  newdata = fopen( newdatafile, "w" );
  if( !newdata ) {
    cerr << "Error: cannot open file " << newdatafile << endl;
    return;
  }

  shiftamt = log((float)databricksize)/log(2.0) - log((float)sizeof(T))/log(2.0); 
  for( currtree = 0; currtree < numtrees; currtree++ ) {

    // write nodes, with padding
    for( i = 0; i < numNodeBricks[currtree]; i++ ) {
      int bsize = 0;
      // write all the nodes in the brick
      for( int j = brickRanges[currtree][i][0]; 
	   j <= brickRanges[currtree][i][1]; 
	   j++ ) {
	fwrite( &(tree[currtree][j].min), sizeof(T), 1, out );
	fwrite( &(tree[currtree][j].max), sizeof(T), 1, out );
	bsize += 2 * sizeof(T);
      }
      // pad if necessary
      if( bsize < nodebricksize ) {
	char* padding = new char[nodebricksize - bsize];
	fwrite( padding, 1, nodebricksize - bsize, out );
	delete [] padding;
      }
    }
    
    // write data, with padding
    fwrite( data[currtree]->values, sizeof(T), data[currtree]->size, newdata );
  }
  
  fclose( newdata );
  fclose( out );
} // writeTree

// search - find isosurface
template <class T>
GeomTriGroup*
TbonTreeCL<T>::search( double iso, const char* treefile, 
		       const char* datafile, int timechanged, int res ) {
  static const int SHIFT = (int)(sizeof(T) * 0.5);
  static int firsttime = 1;
  int currtree;
  currtreefile = fopen( treefile, "r" );
  currdatafile = fopen( datafile, "r" );
  nodesread = 0;
  dataread = 0;

  // can't reuse nodes/data if the time value changed
  currBrick = 0;
  if( timechanged == TIME_CHANGED ) {
    for( currtree = 0; currtree < numtrees; currtree++ ) {
      bzero( nodeBrickInMem[currtree], numNodeBricks[currtree] );
      bzero( datainmem[currtree], numDataBricks[currtree] );
      
      // read in first node brick of each tree
      T* buffer = new T[entriesPerBrick];
      fseek( currtreefile, 
	     (currBrick + previousNodeBricks[currtree])*nodebricksize, 
	     SEEK_SET );
      fread( buffer, sizeof(T), entriesPerBrick, currtreefile );
      int numentries = brickRanges[currtree][currBrick][1] - 
	brickRanges[currtree][currBrick][0] + 1;
      int j, k;
      for( j = 0, k = 0; k < numentries; j+=2, k++ ) {
	nodesread++;
	tree[currtree][ brickRanges[currtree][currBrick][0] + k ].min = 
	  buffer[j];
	tree[currtree][ brickRanges[currtree][currBrick][0] + k ].max = 
	  buffer[j+1];
      }
      delete [] buffer;
      nodeBrickInMem[currtree][currBrick] = 1;
    }
  }
  
  // find # of isosurface cells and which data to read
  int n = searchFirstPass( iso );
  //  cout << "n = " << n << endl;

  register int i;
  register int last;
  register int front;

  for( currtree = 0; currtree < numtrees; currtree++ ) {
    last = front = i = 0;
    // read data bricks
    if( timechanged == TIME_CHANGED ) {
      while( i < numDataBricks[currtree] ) {
	for(; i < numDataBricks[currtree] && brickstoread[currtree][i] != gen;
	    i++ );
	last = i*databricksize;
	fseek( currdatafile, previousDataBricks[currtree] + last, SEEK_SET );
	for(; i < numDataBricks[currtree] && brickstoread[currtree][i] == gen;
	    datainmem[currtree][i]=1, i++ );
	front = i*databricksize;
	if( front > last ) {
	  fread( &(data[currtree]->values[last>>SHIFT]), 1, 
		 (front-last), currdatafile );
	  dataread += (front-last);
	}
      }
    } else {
      while( i < numDataBricks[currtree] ) {
	for( ; i < numDataBricks[currtree] && 
	       (datainmem[currtree][i] || brickstoread[currtree][i] != gen); 
	    i++);
	last = i*databricksize;
	fseek( currdatafile, previousDataBricks[currtree] + last, SEEK_SET );
	for(; i < numDataBricks[currtree] && brickstoread[currtree][i] == gen;
	    datainmem[currtree][i]=1,i++ ); 
	front = i*databricksize;
	if( front > last ) {
	  fread( &data[currtree]->values[last>>SHIFT], 1, (front-last), 
		 currdatafile );
	  dataread += (front-last);
	}
      }
    } // timechanged == TIME_CHANGED?

  } // currtree = 0 .. numtrees-1
  
  mcube->setResolution( res );
  mcube->reset(n);
  
  for( currtree = 0; currtree < numtrees; currtree++ )
    searchSecondPass( currtree, 0, depth[currtree]-res, iso, 0, 0, 0, res );

  fclose( currtreefile );
  fclose( currdatafile );
  gen++;
  cout << "#triangles = " << mcube->triangles->getSize() << endl;
  int totalnodes = 0;
  for( i = 0; i < numtrees; i++ )
    totalnodes += numnodes[i];
  cout << "nodesread = " << nodesread << endl;
  cout << "totalnodes = " << totalnodes << endl;
  cout << "% nodes read: " << 100.0*(float)nodesread/(float)totalnodes << endl;
  cout << "bytes data read: " << dataread << endl;
  
  if( firsttime && n > 0 ) {
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
TbonTreeCL<T>::reorderTree( char* branchArray, int* sibArray, int currtree ) {
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
TbonTreeCL<T>::fill( int currtree, int myindex, int d, int x, int y, int z,
		     T* min, T* max ) {
  int j;
  T mins[8], maxs[8];
  int branching = (int)tree[currtree][myindex].branching;

  if( d == 0 ) {
    corners[currtree][curr++].set( x, y, z );
    switch( branching ) {
    case 0: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
    } break;
    case 1: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x+1, y, z, &mins[1], &maxs[1] );
    } break;
    case 2: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x, y+1, z, &mins[1], &maxs[1] );
    } break;
    case 3: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x+1, y, z, &mins[1], &maxs[1] );
      getMinMax( currtree, x, y+1, z, &mins[2], &maxs[2] );
      getMinMax( currtree, x+1, y+1, z, &mins[3], &maxs[3] );
    } break;
    case 4: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x, y, z+1, &mins[1], &maxs[1] );
    } break;
    case 5: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x+1, y, z, &mins[1], &maxs[1] );
      getMinMax( currtree, x, y, z+1, &mins[2], &maxs[2] );
      getMinMax( currtree, x+1, y, z+1, &mins[3], &maxs[3] );
    } break;
    case 6: {
      getMinMax( currtree, x, y, z, &mins[0], &maxs[0] );
      getMinMax( currtree, x, y+1, z, &mins[1], &maxs[1] );
      getMinMax( currtree, x, y, z+1, &mins[2], &maxs[2] );
      getMinMax( currtree, x, y+1, z+1, &mins[3], &maxs[3] );
    } break;
    case 7: {
      getMinMax( currtree, x,   y,   z, &mins[0], &maxs[0] );
      getMinMax( currtree, x+1, y,   z, &mins[1], &maxs[1] );
      getMinMax( currtree, x,   y+1, z, &mins[2], &maxs[2] );
      getMinMax( currtree, x+1, y+1, z, &mins[3], &maxs[3] );
      getMinMax( currtree, x,   y,   z+1, &mins[4], &maxs[4] );
      getMinMax( currtree, x+1, y,   z+1, &mins[5], &maxs[5] );
      getMinMax( currtree, x,   y+1, z+1, &mins[6], &maxs[6] );
      getMinMax( currtree, x+1, y+1, z+1, &mins[7], &maxs[7] );
    } break;
    };  // switch(branching)

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

  //  int step = pow( 2, d );
  int step = 1 << d;
  d--;

  fill( currtree, tree[currtree][myindex].child, d, x, y, z, 
	&mins[0], &maxs[0] );
  switch( branching ) {
  case 0: break;
  case 1: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x+step, y, z,
	  &mins[1], &maxs[1] );
  } break;
  case 2: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x, y+step, z,
	  &mins[1], &maxs[1] );
  } break;
  case 3: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x+step, y, z,
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, x, y+step, z,
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, x+step, y+step, z,
	  &mins[3], &maxs[3] );
  } break;
  case 4: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x, y, z+step,
	  &mins[1], &maxs[1] );
  } break;
  case 5: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x+step, y, z,
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, x, y, z+step,
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, x+step, y, z+step,
	  &mins[3], &maxs[3] );
  } break;
  case 6: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x, y+step, z,
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, x, y, z+step,
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, x, y+step, z+step,
	  &mins[3], &maxs[3] );
  } break;
  case 7: {
    fill( currtree, tree[currtree][myindex].child + 1, d, x+step, y, z,
	  &mins[1], &maxs[1] );
    fill( currtree, tree[currtree][myindex].child + 2, d, x, y+step, z,
	  &mins[2], &maxs[2] );
    fill( currtree, tree[currtree][myindex].child + 3, d, x+step, y+step, z,
	  &mins[3], &maxs[3] );
    fill( currtree, tree[currtree][myindex].child + 4, d, x, y, z+step,
	  &mins[4], &maxs[4] );
    fill( currtree, tree[currtree][myindex].child + 5, d, x+step, y, z+step,
	  &mins[5], &maxs[5] );
    fill( currtree, tree[currtree][myindex].child + 6, d, x, y+step, z+step,
	  &mins[6], &maxs[6] );
    fill( currtree, tree[currtree][myindex].child + 7, d, 
	  x+step, y+step, z+step, &mins[7], &maxs[7] );
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
TbonTreeCL<T>::getMinMax( int currtree, int x, int y, int z, T* min, T* max ) {
  T v;
  int index = Xarray[currtree][x] + Yarray[currtree][y] + Zarray[currtree][z];
  *min = *max = data[currtree]->values[index];

  int xindex = ( circular[currtree] == CIRCULAR_X && 
		 x+1 == data[currtree]->nx ) ? 
    Xarray[currtree][x+1-data[currtree]->nx] : Xarray[currtree][x+1];

  index = xindex + Yarray[currtree][y] + Zarray[currtree][z];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[currtree][x] + Yarray[currtree][y+1] + Zarray[currtree][z];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = xindex + Yarray[currtree][y+1] + Zarray[currtree][z];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[currtree][x] + Yarray[currtree][y] + Zarray[currtree][z+1];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = xindex + Yarray[currtree][y] + Zarray[currtree][z+1];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = Xarray[currtree][x] + Yarray[currtree][y+1] + Zarray[currtree][z+1];
  v = data[currtree]->values[index];
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;

  index = xindex + Yarray[currtree][y+1] +Zarray[currtree][z+1];
  v = data[currtree]->values[index];  
  if( v < *min ) *min = v;
  if( v > *max ) *max = v;
} // getMinMax

// findBricks - determine brick boundaries that guarantee all siblings
//     of a node are in the same brick
template <class T>
void
TbonTreeCL<T>::findBricks() {

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
TbonTreeCL<T>::searchFirstPass( double iso ) {
  static const int BX[] = {1, 2, 1, 2, 1, 2, 1, 2};
  static const int BY[] = {1, 1, 2, 2, 1, 1, 2, 2};
  static const int BZ[] = {1, 1, 1, 1, 2, 2, 2, 2};

  int isoCells = 0;
  for( int currtree = 0; currtree < numtrees; currtree++ ) {
    currBrick = 0;
    int i, j;
    int idx;
    int OFFSET = numnodes[currtree] - numleaves[currtree];
    
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
	  
	  // read in needed node bricks
	  // find brick containing child      
	  for( ; tree[currtree][ oldqueue[i] ].child > 
		 brickRanges[currtree][ currBrick ][1]; 
	       currBrick++ );
	  
	  // read brick if not in memory
	  if( currBrick < numNodeBricks[currtree] && 
	      !nodeBrickInMem[currtree][ currBrick ] ) {
	    int numentries = brickRanges[currtree][currBrick][1] - 
	      brickRanges[currtree][currBrick][0] + 1;
	    if( ftell(currtreefile) != 
		(currBrick+previousNodeBricks[currtree])*nodebricksize ) {
	      fseek( currtreefile, 
		     (currBrick+previousNodeBricks[currtree])*nodebricksize, 
		     SEEK_SET );
	    }
	    T* buffer = new T[ entriesPerBrick ];
	    fread( buffer, sizeof(T), entriesPerBrick, currtreefile );
	    int k = 0;
	    for( j = 0; k < numentries; j+=2, k++ ) {
	      nodesread++;
	      tree[currtree][ brickRanges[currtree][currBrick][0] + k ].min =
		buffer[j];
	      tree[currtree][ brickRanges[currtree][currBrick][0] + k ].max =
		buffer[j+1];
	    }
	    delete [] buffer;
	    nodeBrickInMem[currtree][currBrick] = 1;
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

    // finish with level 0
    for( int loopvar = 0; loopvar < oldsize; loopvar++ ) {
      if( tree[currtree][ oldqueue[loopvar] ].min <= iso &&
	  tree[currtree][ oldqueue[loopvar] ].max >= iso ) {
	int first = tree[currtree][ oldqueue[loopvar] ].child;
	idx = first - OFFSET;
	Corner bounds[7];

	switch( (int)tree[currtree][ oldqueue[loopvar] ].branching ) {
	case 0: {
	  bounds[1].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z );
	  bounds[3].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y, 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );
	  bounds[5].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );
	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx].x + 
		BX[ (int)tree[currtree][first].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ],
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z );
	    bounds[4].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );
	    bounds[6].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx].x - data[currtree]->nx +
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx].x - data[currtree]->nx +
			   BX[ (int)tree[currtree][first].branching ],
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z );
	    bounds[4].set( corners[currtree][idx].x - data[currtree]->nx +
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );
	    bounds[6].set( corners[currtree][idx].x - data[currtree]->nx +
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );

	  }
	} break;
	case 1:{
	  bounds[1].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z );
	  bounds[3].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y, 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );
	  bounds[5].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx+1].x + 
		BX[ (int)tree[currtree][first+1].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y,
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y,
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	  }
	} break;
	case 2:{
	  bounds[1].set( corners[currtree][idx+1].x, 
			 corners[currtree][idx+1].y + 
			 BY[ (int)tree[currtree][first+1].branching ], 
			 corners[currtree][idx+1].z );
	  bounds[3].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y, 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );
	  bounds[5].set( corners[currtree][idx+1].x, 
			 corners[currtree][idx+1].y + 
			 BY[ (int)tree[currtree][first+1].branching ], 
			 corners[currtree][idx+1].z + 
			 BZ[ (int)tree[currtree][first+1].branching ] );
	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx].x + 
		BX[ (int)tree[currtree][first].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z + 
			   BZ[ (int)tree[currtree][first].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );

	  }
	} break;
	case 3:{
	  bounds[1].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y + 
			 BY[ (int)tree[currtree][first+2].branching ], 
			 corners[currtree][idx+2].z );
	  bounds[3].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y, 
			 corners[currtree][idx].z + 
			 BZ[ (int)tree[currtree][first].branching ] );
	  bounds[5].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y + 
			 BY[ (int)tree[currtree][first+2].branching ], 
			 corners[currtree][idx+2].z + 
			 BZ[ (int)tree[currtree][first+2].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx+1].x + 
		BX[ (int)tree[currtree][first+1].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }
	  
	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ],
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z );
	    bounds[4].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ],
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z );
	    bounds[4].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  }
	} break;
	case 4:{
	  bounds[1].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z );
	  bounds[3].set( corners[currtree][idx+1].x, 
			 corners[currtree][idx+1].y, 
			 corners[currtree][idx+1].z + 
			 BZ[ (int)tree[currtree][first+1].branching ] );
	  bounds[5].set( corners[currtree][idx+1].x, 
			 corners[currtree][idx+1].y + 
			 BY[ (int)tree[currtree][first+1].branching ], 
			 corners[currtree][idx+1].z + 
			 BZ[ (int)tree[currtree][first+1].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx].x + 
		BX[ (int)tree[currtree][first].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ],
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z );
	    bounds[4].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first].branching ],
			   corners[currtree][idx].y + 
			   BY[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].z );
	    bounds[4].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	    bounds[6].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z + 
			   BZ[ (int)tree[currtree][first+1].branching ] );
	  }
	} break;
	case 5:{
	  bounds[1].set( corners[currtree][idx].x, 
			 corners[currtree][idx].y + 
			 BY[ (int)tree[currtree][first].branching ], 
			 corners[currtree][idx].z );
	  bounds[3].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y, 
			 corners[currtree][idx+2].z + 
			 BZ[ (int)tree[currtree][first+2].branching ] );
	  bounds[5].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y + 
			 BY[ (int)tree[currtree][first+2].branching ], 
			 corners[currtree][idx+2].z + 
			 BZ[ (int)tree[currtree][first+2].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx+1].x + 
		BX[ (int)tree[currtree][first+1].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y, 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y, 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  }
	} break;
	case 6:{
	  bounds[1].set( corners[currtree][idx+1].x, 
			 corners[currtree][idx+1].y + 
			 BY[ (int)tree[currtree][first+1].branching ], 
			 corners[currtree][idx+1].z );
	  bounds[3].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y, 
			 corners[currtree][idx+2].z +
			 BZ[ (int)tree[currtree][first+2].branching ] );
	  bounds[5].set( corners[currtree][idx+3].x, 
			 corners[currtree][idx+3].y + 
			 BY[ (int)tree[currtree][first+3].branching ], 
			 corners[currtree][idx+3].z + 
			 BZ[ (int)tree[currtree][first+3].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx].x + 
		BX[ (int)tree[currtree][first].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx].x + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+2].x + 
			   BX[ (int)tree[currtree][first+2].branching ], 
			   corners[currtree][idx+2].y, 
			   corners[currtree][idx+2].z + 
			   BZ[ (int)tree[currtree][first+2].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first].branching ], 
			   corners[currtree][idx].y, 
			   corners[currtree][idx].z );
	    bounds[2].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ],
			   corners[currtree][idx+1].y + 
			   BY[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].z );
	    bounds[4].set( corners[currtree][idx+2].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+2].branching ], 
			   corners[currtree][idx+2].y, 
			   corners[currtree][idx+2].z + 
			   BZ[ (int)tree[currtree][first+2].branching ] );
	    bounds[6].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z + 
			   BZ[ (int)tree[currtree][first+3].branching ] );
	  }
	} break;
	case 7:{
	  bounds[1].set( corners[currtree][idx+2].x, 
			 corners[currtree][idx+2].y + 
			 BY[ (int)tree[currtree][first+2].branching ], 
			 corners[currtree][idx+2].z );
	  bounds[3].set( corners[currtree][idx+4].x, 
			 corners[currtree][idx+4].y, 
			 corners[currtree][idx+4].z + 
			 BZ[ (int)tree[currtree][first+4].branching ] );
	  bounds[5].set( corners[currtree][idx+6].x, 
			 corners[currtree][idx+6].y + 
			 BY[ (int)tree[currtree][first+6].branching ], 
			 corners[currtree][idx+6].z + 
			 BZ[ (int)tree[currtree][first+6].branching ] );

	  int atedge = 0;
	  if( circular[currtree] == CIRCULAR_X ) {
	    if( corners[currtree][idx+1].x + 
		BX[ (int)tree[currtree][first+1].branching ] == 
		data[currtree]->nx )
	      atedge = 1;
	  }

	  if( !atedge ) {
	    bounds[0].set( corners[currtree][idx+1].x + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+3].x + 
			   BX[ (int)tree[currtree][first+3].branching ],
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z );
	    bounds[4].set( corners[currtree][idx+5].x + 
			   BX[ (int)tree[currtree][first+5].branching ], 
			   corners[currtree][idx+5].y, 
			   corners[currtree][idx+5].z + 
			   BZ[ (int)tree[currtree][first+5].branching ] );
	    bounds[6].set( corners[currtree][idx+7].x + 
			   BX[ (int)tree[currtree][first+7].branching ], 
			   corners[currtree][idx+7].y + 
			   BY[ (int)tree[currtree][first+7].branching ], 
			   corners[currtree][idx+7].z + 
			   BZ[ (int)tree[currtree][first+7].branching ] );
	  } else {
	    bounds[0].set( corners[currtree][idx+1].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+1].branching ], 
			   corners[currtree][idx+1].y, 
			   corners[currtree][idx+1].z );
	    bounds[2].set( corners[currtree][idx+3].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+3].branching ],
			   corners[currtree][idx+3].y + 
			   BY[ (int)tree[currtree][first+3].branching ], 
			   corners[currtree][idx+3].z );
	    bounds[4].set( corners[currtree][idx+5].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+5].branching ], 
			   corners[currtree][idx+5].y, 
			   corners[currtree][idx+5].z + 
			   BZ[ (int)tree[currtree][first+5].branching ] );
	    bounds[6].set( corners[currtree][idx+7].x - data[currtree]->nx + 
			   BX[ (int)tree[currtree][first+7].branching ], 
			   corners[currtree][idx+7].y + 
			   BY[ (int)tree[currtree][first+7].branching ], 
			   corners[currtree][idx+7].z + 
			   BZ[ (int)tree[currtree][first+7].branching ] );
	  }
	} break;
	} // switch
	
	int index = Xarray[currtree][corners[currtree][idx].x] + 
	  Yarray[currtree][corners[currtree][idx].y] + 
	  Zarray[currtree][corners[currtree][idx].z];
	brickstoread[currtree][ index >> shiftamt ] = gen;
	for( i = 0; i < 7; i++ ) {
	  index = Xarray[currtree][ bounds[i].x ] + 
	    Yarray[currtree][ bounds[i].y ] + 
	    Zarray[currtree][ bounds[i].z ];
	  brickstoread[currtree][ index >> shiftamt ] = gen;
	}
	
      } // if oldqueue[loopvar] spans iso
    } // loopvar = 0 .. oldsize-1

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
TbonTreeCL<T>::searchSecondPass( int currtree, int myindex, int d, 
				 double iso, int x, int y, int z, int res ) {

  if( tree[currtree][myindex].min <= iso && 
      tree[currtree][myindex].max >= iso ) {

    if( d == 0 ) {
      if( circular[currtree] == NONCIRCULAR )
	mcube->interpRegular( currtree, x, y, z, iso, 
			      (int)tree[currtree][myindex].branching );
      else
	mcube->interpCircular( currtree, x, y, z, iso, 
			      (int)tree[currtree][myindex].branching );
      return;
    } // if d == 0

    //    int step = pow( 2, (d+res) );
    int step = 1 << (d+res);
    d--;

    searchSecondPass( currtree, tree[currtree][myindex].child, d, iso, 
		      x, y, z, res );
    switch( (int)tree[currtree][myindex].branching ) {
    case 0: 
      break;
    case 1: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x+step, y, z, res );
    } break;
    case 2: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x, y+step, z, res );
    } break;
    case 3: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x+step, y, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso, 
			x, y+step, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso, 
			x+step, y+step, z, res );
    } break;
    case 4: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x, y, z+step, res );
    } break;
    case 5: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x+step, y, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso, 
			x, y, z+step, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso, 
			x+step, y, z+step, res );
    } break;
    case 6: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x, y+step, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso, 
			x, y, z+step, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso, 
			x, y+step, z+step, res );
    } break;
    case 7: {
      searchSecondPass( currtree, tree[currtree][myindex].child + 1, d, iso, 
			x+step, y, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 2, d, iso, 
			x, y+step, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 3, d, iso, 
			x+step, y+step, z, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 4, d, iso, 
			x, y, z+step, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 5, d, iso, 
			x+step, y, z+step, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 6, d, iso, 
			x, y+step, z+step, res );
      searchSecondPass( currtree, tree[currtree][myindex].child + 7, d, iso, 
			x+step, y+step, z+step, res );
    } break;
    }; // switch

  } // if tree[myindex] spans iso, x, y, z
} // searchSecondPass
} // End namespace Phil


#endif

