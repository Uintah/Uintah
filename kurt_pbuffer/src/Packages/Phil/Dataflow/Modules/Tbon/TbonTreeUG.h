
/* TbonTreeUG.h
   class declarations and code for the T-BON tree (unstructured grid version)

   Packages/Philip Sutton
   May 1999

  Copyright (C) 2000 SCI Group, University of Utah
*/

#ifndef __TBON_TREE_UG_H__
#define __TBON_TREE_UG_H__

#include "TreeUtils.h"
#include "TriGroup.h"
#include "mcube.h"

#include <limits.h>
#include <strings.h>

namespace Phil {
using namespace SCIRun;

struct Tetra {
  int v[4];
};  

struct TetraVertex {
  float pos[3];
};

template <class T>
struct DataUG {
  int npts;
  int ntets;
  int ncells;
  int listsize;
  int nx, ny, nz;

  TetraVertex* verts;
  Tetra* tets;
  T* values;
  int* cells;
  int* lists;

  int* extremes;
};

template <class T>
class TbonTreeUG {
public:
  // constructor for preprocessing
  TbonTreeUG( int nx, int ny, int nz, const char* geomfile, int b );
  // constructor for execution
  TbonTreeUG( const char* filename, const char* geomfile, int b );
  // destructor
  ~TbonTreeUG();
  
  // operations on the tree
  void readData( const char* filename );
  void fillTree( int threshold );
  void writeTree( const char* meta, const char* base, 
		  const char* geom, int num );
  GeomTriGroup* search( double iso, const char* treefile, 
			const char* datafile, int timechanged );

  // constants
  static const int TIME_CHANGED;
  static const int TIME_SAME;
protected:
private:
  // structure of the tree
  DataUG<T>* data;
  MCubesUG<T>* mcube;
  int* indices;
  Node<T>* tree;

  // properties
  int depth;
  int numnodes;
  int numNodeBricks;
  int numDataBricks;

  // auxiliary stuff
  int* brickstoread;
  int nodebricksize;
  int databricksize;
  int entriesPerBrick;
  int gen;
  int** brickRanges;
  int currBrick;
  char* datainmem;
  char* nodeBrickInMem;
  char* interpolated;

  // flag for destructor
  int deleteExecutionStuff;

  FILE* currtree;
  FILE* currdata;

  // private methods
  void reorderTree( char* branchArray, int* sibArray );
  void fill( int myindex, int d, int vnum, T* min, T* max );
  void getMinMax( int index, T* min, T* max );
  void findBricks();
  void traverse( int root, int d, int* gnum, int** glist );
  void collapse( int root, int d, int* gnum, int** glist, int* indexIndices,
		 int* indexLists, int threshold );
  void condense( Node<T>* newtree );
  void writeGeometry( const char* filename );

  int searchFirstPass( double iso );
  void searchSecondPass( int myindex, double iso );
};

// define constants
template <class T>
const int TbonTreeUG<T>::TIME_CHANGED = 0;
template <class T>
const int TbonTreeUG<T>::TIME_SAME = 1;

// TbonTreeUG
//   Constructor for Preprocessing
template <class T>
TbonTreeUG<T>::TbonTreeUG( int nx, int ny, int nz, 
			   const char* geomfile, int b) {
  int i;
  int idx;

  data = new DataUG<T>;

  FILE* geom = fopen( geomfile, "r" );
  if( !geom ) {
    cerr << "Error: cannot open geom file " << geomfile << endl;
    return;
  }

  // read geometric data
  fscanf( geom, "%d %d\n", &data->npts, &data->ntets );
  fscanf( geom, "%d %d\n", &data->ncells, &data->listsize );
  data->verts = new TetraVertex[data->npts];
  data->tets = new Tetra[data->ntets];
  data->cells = new int[data->ncells];
  data->lists = new int[data->listsize];

  // read points
  for( i = 0; i < data->npts; i++ ) {
    fread( data->verts[i].pos, sizeof(float), 3, geom );
  }

  // read tets
  for( i = 0; i < data->ntets; i++ ) {
    fread( data->tets[i].v, sizeof(int), 4, geom );
  }

  // read list data
  fread( data->lists, sizeof(int), data->listsize, geom );
  fseek( geom, (long)(sizeof(int) * 2 * data->ncells), SEEK_CUR );
  fread( data->cells, sizeof(int), data->ncells, geom );
  fclose(geom);

  data->nx = nx; data->ny = ny; data->nz = nz;

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

  // find how many nodes are needed
  int num = 1;
  int dummyleaves = 0;
  countNodes( xa, ya, za, idx, &num, &dummyleaves );
  numnodes = num;
  cout << "Tree has " << numnodes << " nodes" << endl;

  // set up information for bricking nodes
  nodebricksize = b;

  // allocate tree structure
  tree = new Node<T>[numnodes];
  indices = new int[data->ncells];
  
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

  deleteExecutionStuff = 0;
} // TbonTreeUG

// TbonTreeUG
//   constructor for execution
template <class T>
TbonTreeUG<T>::TbonTreeUG( const char* filename, const char* geomfile, 
			   int b ) {
  int i;
  FILE* metafile = fopen( filename, "r" );
  if( !metafile ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  databricksize = b;

  // read tree parameters
  fscanf( metafile, "%d\n%d\n", &numnodes, &depth );
  fscanf( metafile, "%d %d\n", &nodebricksize, &numNodeBricks);
  entriesPerBrick = nodebricksize / (int)sizeof(T);

  // set up and read data parameters
  data = new DataUG<T>;
  fscanf( metafile, "%d %d %d\n", &(data->nx), &(data->ny), &(data->nz) );
  fscanf( metafile, "%d %d %d %d\n", &(data->npts), &(data->ntets),
	  &(data->ncells), &(data->listsize) );
  data->verts = new TetraVertex[data->npts];
  data->values = new T[data->npts];
  data->tets = new Tetra[data->ntets];
  data->lists = new int[data->listsize];
  data->extremes = new int[2*data->ncells];

  brickRanges = new int*[numNodeBricks];
  for( i = 0; i < numNodeBricks; i++ ) {
    brickRanges[i] = new int[2];
    fread( &brickRanges[i][0], sizeof(int), 1, metafile );
    fread( &brickRanges[i][1], sizeof(int), 1, metafile );
  }

  // allocate tree structure
  tree = new Node<T>[numnodes];
  indices = new int[data->ncells];
  mcube = new MCubesUG<T>(data);

  // read in tree skeleton
  for( i = 0; i < numnodes; i++ ) {
    fread( &(tree[i].branching), sizeof(char), 1, metafile );
    fread( &(tree[i].child), sizeof(int), 1, metafile );
  }
  fread( indices, sizeof(int), data->ncells, metafile );
  fclose( metafile );

  FILE* geom = fopen( geomfile, "r" );
  if( !geom ) {
    cerr << "Error: cannot open geom file " << geomfile << endl;
    return;
  }

  // read in geometry information
  int numpts, numtets;
  int numcells, size;
  fscanf( geom, "%d %d\n", &numpts, &numtets );
  fscanf( geom, "%d %d\n", &numcells, &size );
  
  // make sure things are correct
  ASSERT( numpts == data->npts );
  ASSERT( numtets == data->ntets );
  ASSERT( numcells == data->ncells );
  ASSERT( size == data->listsize );

  // read data from geom file
  fread( data->verts, sizeof(float), 3*data->npts, geom );
  fread( data->tets, sizeof(int), 4*data->ntets, geom );
  fread( data->lists, sizeof(int), data->listsize, geom );
  fread( data->extremes, sizeof(int), 2*data->ncells, geom );

  fclose( geom );

  interpolated = new char[data->ntets];
  bzero( interpolated, data->ntets );

  gen = 1;
  databricksize = b;
  numDataBricks = 1 + data->npts * (int)sizeof(T) / databricksize;
  brickstoread = new int[ numDataBricks ];
  bzero( brickstoread, numDataBricks*sizeof(int) );
  
  datainmem = new char[numDataBricks];
  nodeBrickInMem = new char[numNodeBricks];
  bzero( datainmem, numDataBricks );
  bzero( nodeBrickInMem, numNodeBricks );

  deleteExecutionStuff = 1;
}

// ~TbonTreeUG
//    Destructor
template <class T>
TbonTreeUG<T>::~TbonTreeUG() {
  delete [] data->lists;
  delete [] data->extremes;
  delete [] data->verts;
  delete [] data->tets;
  delete [] tree;
  delete [] indices;
  delete [] brickRanges;

  if( deleteExecutionStuff == 0 ) {
    delete [] data->cells;
  } else {
    delete [] data->values;
    delete [] interpolated;
    delete [] brickstoread;
    delete [] datainmem;
    delete [] nodeBrickInMem;
  }

  delete data;
} // ~TbonTreeUG

// readData
//   Read unstructured data
template <class T>
void
TbonTreeUG<T>::readData( const char* filename ) {
  FILE* currfile = fopen( filename, "r" );
  if( currfile == 0 ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }
  
  data->values = new T[data->npts];
  unsigned long n = fread( data->values, sizeof(T), data->npts, currfile );
  if( n != (unsigned long)data->npts ) {
    cerr << "Error: only " << n << "/" << data->npts << " objects read from "
	 << filename << endl;
  }
  fclose(currfile);
} // readData

// fillTree 
//   fill in skeleton tree with min, max
template <class T>
void
TbonTreeUG<T>::fillTree( int threshold ) {
  curr = 0;
  fill( 0, depth, 0, &(tree[0].min), &(tree[0].max) );
  
  static int firsttime = 1;
  if( firsttime ) {
    cout << "Condensing tree" << endl;
    int* gnum = new int[numnodes];
    int** glist = new int*[numnodes];
    traverse( 0, depth, gnum, glist );
    
    int indexIndices = 0;
    int indexLists = 1;
    collapse( 0, depth, gnum, glist, &indexIndices, &indexLists, threshold );
    data->listsize = indexLists;
    data->ncells = indexIndices;
    
    Node<T>* newtree = new Node<T>[numnodes];
    condense( newtree );
    cout << "New tree has " << numnodes << " nodes"
	 << " and " << data->ncells << " cells" << endl;

    // find maximum number of node bricks
    numNodeBricks = numnodes / 8;
    brickRanges = new int*[numNodeBricks];

    delete [] gnum;
    delete [] glist; 
    firsttime = 0;
  }
} // fillTree

// writeTree 
//   write the tree to disk
template <class T>
void
TbonTreeUG<T>::writeTree( const char* meta, const char* base, 
			  const char* geom, int num ) {
  char filename[80];
  int i;
  FILE* out;

  cout << "Writing tree #" << num << endl;

  // if this is the first write, write the tree metafile (skeleton)
  // and rewrite the geometry file for the condensed nodes.
  if( num == 0 ) {
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
    fprintf(out,"%d %d %d\n%d %d %d %d\n", data->nx, data->ny, data->nz,
	    data->npts, data->ntets, data->ncells, data->listsize );

    for( i = 0; i < numNodeBricks; i++ ) {
      fwrite( &brickRanges[i][0], sizeof(int), 1, out );
      fwrite( &brickRanges[i][1], sizeof(int), 1, out );
    }

    for( i = 0; i < numnodes; i++ ) {
      fwrite( &(tree[i].branching), sizeof(char), 1, out );
      fwrite( &(tree[i].child), sizeof(int), 1, out );
    }
    fwrite( indices, sizeof(int), data->ncells, out );
    
    fclose(out);

    writeGeometry( geom );
  } 

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
      bsize += 2 * sizeof(T);
    }
    // pad if necessary
    if( bsize < nodebricksize ) {
      char* padding = new char[nodebricksize - bsize];
      fwrite( padding, 1, nodebricksize - bsize, out );
      delete [] padding;
    }
  }
  
  delete [] data->values;

  fclose( out );
} // writeTree

// search
//   traverse the octree to find isosurface
template <class T>
GeomTriGroup*
TbonTreeUG<T>::search( double iso, const char* treefile, 
		       const char* datafile, int timechanged ) {
  static const int SHIFT = (int)(sizeof(T) * 0.5);
  int j, k;

  currtree = fopen( treefile, "r" );
  currdata = fopen( datafile, "r" );

  currBrick = 0;
  // can't reuse nodes/data if the time value changed
  if( timechanged == TIME_CHANGED ) {
    bzero( datainmem, numDataBricks );
    bzero( nodeBrickInMem, numNodeBricks );

    // read in first node brick
    T* buffer = new T[entriesPerBrick];
    int numentries = brickRanges[currBrick][1] - 
      brickRanges[currBrick][0] + 1;
    fread( buffer, sizeof(T), entriesPerBrick, currtree );
    for( j = 0, k = 0; k < numentries; j+=2, k++ ) {
      tree[ brickRanges[currBrick][0] + k ].min = buffer[j];
      tree[ brickRanges[currBrick][0] + k ].max = buffer[j+1];
    }
    delete [] buffer;
    nodeBrickInMem[currBrick] = 1; 
  }

  // find # of isosurface cells and which data to read
   int n = searchFirstPass( iso );
  //  cout << "n = " << n << endl;

  register int i = 0;
  register int last = 0;
  register int front = 0;
  
  // read data bricks
  if( timechanged == TIME_CHANGED ) {
    while( i < numDataBricks ) {
      for( ; i < numDataBricks && brickstoread[i] != gen; i++ );
      last = i*databricksize;
      fseek( currdata, last, SEEK_SET );
      for(; i < numDataBricks && brickstoread[i] == gen; datainmem[i]=1, i++ );
      front = i*databricksize;
      if( front > last )
	fread( &(data->values[last>>SHIFT]), 1, (front-last), currdata );
    }
  } else {
    while( i < numDataBricks ) {
      for(;i < numDataBricks && (datainmem[i] || brickstoread[i] != gen); i++);
      last = i*databricksize;
      fseek( currdata, last, SEEK_SET );
      for(; i < numDataBricks && brickstoread[i] == gen; datainmem[i]=1,i++ ); 
      front = i*databricksize;
      if( front > last ) 
	fread( &data->values[last>>SHIFT], 1, (front-last), currdata );
    }
  }  
  

  bzero( interpolated, data->ntets );
  mcube->reset( n );

  searchSecondPass( 0, iso );

  fclose( currtree );
  fclose( currdata );
  
  gen++;
  return mcube->triangles;
} // search

///////////////////////
// Private Functions //
///////////////////////

// reorderTree - create BFS version of tree
template <class T>
void
TbonTreeUG<T>::reorderTree( char* branchArray, int* sibArray ) {
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

// fill - recursively fill each node in the tree 
template <class T>
void
TbonTreeUG<T>::fill( int myindex, int d, int vnum, T* min, T* max ) {
  static int nx = data->nx - 1;
  static int ny = data->ny - 1;
  int j;
  T mins[8], maxs[8];
  int branching = (int)tree[myindex].branching;
  if( d == 0 || branching == 8 ) {
    switch( branching ) {
    case 0: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
    } break;
    case 1: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum+1];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 2: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + nx];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 3: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + 1];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = data->cells[vnum + nx];
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = data->cells[vnum + nx + 1];
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 4: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + nx * ny];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
    } break;
    case 5: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + 1];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = data->cells[vnum + nx * ny];
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = data->cells[vnum + nx * ny + 1];
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 6: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + nx];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = data->cells[vnum + nx * ny];
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = data->cells[vnum + nx * ny + nx];
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
    } break;
    case 7: 
    case 8: {
      indices[curr] = data->cells[vnum];
      getMinMax( indices[curr++], &mins[0], &maxs[0] );
      indices[curr] = data->cells[vnum + 1];
      getMinMax( indices[curr++], &mins[1], &maxs[1] );
      indices[curr] = data->cells[vnum + nx];
      getMinMax( indices[curr++], &mins[2], &maxs[2] );
      indices[curr] = data->cells[vnum + nx + 1];
      getMinMax( indices[curr++], &mins[3], &maxs[3] );
      indices[curr] = data->cells[vnum + nx * ny];
      getMinMax( indices[curr++], &mins[4], &maxs[4] );
      indices[curr] = data->cells[vnum + nx * ny + 1];
      getMinMax( indices[curr++], &mins[5], &maxs[5] );
      indices[curr] = data->cells[vnum + nx * ny + nx];
      getMinMax( indices[curr++], &mins[6], &maxs[6] );
      indices[curr] = data->cells[vnum + nx * ny + nx + 1];
      getMinMax( indices[curr++], &mins[7], &maxs[7] );
    } break;
    };  // switch(branching)

    tree[myindex].child = curr - BRANCHTABLE[branching];

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
  int ystep = nx * xstep;
  int zstep = ny * ystep;
  d--;

  fill( tree[myindex].child, d, vnum, &mins[0], &maxs[0] );
  switch( branching ) {
  case 0: break;
  case 1: {
    fill( tree[myindex].child + 1, d, vnum+xstep, &mins[1], &maxs[1] );
  } break;
  case 2: {
    fill( tree[myindex].child + 1, d, vnum+ystep, &mins[1], &maxs[1] );
  } break;
  case 3: {
    fill( tree[myindex].child + 1, d, vnum+xstep, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, vnum+ystep, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
  } break;
  case 4: {
    fill( tree[myindex].child + 1, d, vnum+zstep, &mins[1], &maxs[1] );
  } break;
  case 5: {
    fill( tree[myindex].child + 1, d, vnum+xstep, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, vnum+zstep, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, vnum+xstep+zstep, &mins[3], &maxs[3] );
  } break;
  case 6: {
    fill( tree[myindex].child + 1, d, vnum+ystep, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, vnum+zstep, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, vnum+ystep+zstep, &mins[3], &maxs[3] );
  } break;
  case 7: {
    fill( tree[myindex].child + 1, d, vnum+xstep, &mins[1], &maxs[1] );
    fill( tree[myindex].child + 2, d, vnum+ystep, &mins[2], &maxs[2] );
    fill( tree[myindex].child + 3, d, vnum+xstep+ystep, &mins[3], &maxs[3] );
    fill( tree[myindex].child + 4, d, vnum+zstep, &mins[4], &maxs[4] );
    fill( tree[myindex].child + 5, d, vnum+xstep+zstep, &mins[5], &maxs[5] );
    fill( tree[myindex].child + 6, d, vnum+ystep+zstep, &mins[6], &maxs[6] );
    fill( tree[myindex].child + 7, d, vnum+xstep+ystep+zstep, 
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

// getMinMax
//   find the min and max values of the voxel whose list of tets 
//   begins at index
template <class T>
void
TbonTreeUG<T>::getMinMax( int index, T* min, T* max ) {
  T v;
  int i, n;
  *min = FLT_MAX;
  *max = -FLT_MAX;

  // if this cell contains no tetrahedra, do nothing
  if( index == 0 )
    return;

  // find # tetrahedra in this list
  n = data->lists[index];

  // for each tetra, check values
  for( i = 1; i <= n; i++ ) {
    Tetra tet_i = data->tets[data->lists[index + i]];

    v = data->values[tet_i.v[0]];
    if( v < *min ) *min = v;
    if( v > *max ) *max = v;

    v = data->values[tet_i.v[1]];
    if( v < *min ) *min = v;
    if( v > *max ) *max = v;

    v = data->values[tet_i.v[2]];
    if( v < *min ) *min = v;
    if( v > *max ) *max = v;

    v = data->values[tet_i.v[3]];
    if( v < *min ) *min = v;
    if( v > *max ) *max = v;
  }  
}

// findBricks - determine brick boundaries that guarantee all siblings
//     of a node are in the same brick
template <class T>
void
TbonTreeUG<T>::findBricks() {
  int brick = 0;
  int currSiblings = 1;
  int oldsize = 1;
  int* oldqueue = new int[oldsize];
  int* newqueue;
  int i;

  oldqueue[0] = 0;
  int newsize = BRANCHTABLE[ (int)tree[0].branching ];
  int bsize = 2 * sizeof(T);
  brickRanges[0] = new int[2];
  brickRanges[0][0] = 0;

  for( int currdepth = depth; currdepth > 0; currdepth-- ) {
    int index = 0;
    int nextsize = 0;
    newqueue = new int[newsize];
    
    for( i = 0; i < oldsize; i++ ) {
      int j;

      if( (int)tree[oldqueue[i]].branching < 8 ) {
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
	
	// add node's children to brick
	bsize += 2 * sizeof(T) * branch_i;
	
	// add node's children to newqueue
	for( j = 0; j < branch_i; j++ ) {
	  newqueue[index] = tree[oldqueue[i]].child + j;
	  nextsize += BRANCHTABLE[ (int)tree[ newqueue[index] ].branching ];
	  index++;
	}
	currSiblings += branch_i;
      } // branching < 8

    } // i = 0 .. oldsize-1
    delete [] oldqueue;
    oldqueue = newqueue;
    oldsize = index;
    newsize = nextsize;

  } // currdepth = depth .. 1

  delete [] newqueue;
  brickRanges[brick++][1] = currSiblings - 1;
  numNodeBricks = brick;

} // findBricks

// traverse - get number and list of tetrahedra per node
//   (for later use in collapsing nodes)
template <class T>
void
TbonTreeUG<T>::traverse( int root, int d, int* gnum, int** glist ) {
  static const int doneTable[] = {1, 3, 3, 15, 3, 15, 15, 255};
  int n, idx;
  int i, j;
  int branching = BRANCHTABLE[(int)tree[root].branching];

  if( d == 0 ) {
    n = 0;

    // find max number of tets for this node
    for( i = 0; i < branching; i++ ) {
      if( data->lists[ indices[ tree[root].child + i ] ] )
	n += data->lists[ indices[ tree[root].child + i ] ];
    }
    if( n == 0 ) {
      gnum[root] = 0;
      return;
    }
    
    // fill in the global list for this node
    glist[root] = new int[n];
    for( i = 0, idx = 0; i < branching; i++ ) {
      int start = indices[ tree[root].child + i ];
      for( j = 1; j <= data->lists[start]; j++ ) {
	if( notinlist( data->lists[start+j], glist[root], idx ) )
	  glist[root][idx++] = data->lists[start+j];
      }
    }
    // sort the list
    qsort( glist[root], idx, sizeof(int), intLess );

    // fill in the number array for this node
    gnum[root] = idx;
    return;
  } // if d == 0

  d--;
  n = 0;
  int** larray = new int*[branching];
  int* p = new int[branching];
  int* num = new int[branching];

  int done = 0;
  // find max number of tets for this node (sum of children)
  for( i = 0; i < branching; i++ ) {
    int child_i = tree[root].child + i;
    traverse( child_i, d, gnum, glist );
    larray[i] = glist[ child_i ];
    num[i] = gnum[ child_i ];
    p[i] = 0;
    if( num[i] == 0 ) done |= (1 << i);
    n += gnum[ child_i ];
  }
  if( n == 0 ) {
    delete [] larray;
    delete [] p;
    delete [] num;
    gnum[root] = 0;
    return;
  }

  // fill in the global list for this node (mergesort-like)
  glist[root] = new int[n];
  idx = 0;
  while( done != doneTable[(int)tree[root].branching] ) {
    int min = INT_MAX;
    for( i = 0; i < branching; i++ ) {
      if( p[i] < num[i] ) {
	min = (larray[i][p[i]] < min) ? larray[i][p[i]] : min;
      }
    }
    glist[root][idx] = min;
    idx++;
    for( i = 0; i < branching; i++ ) {
      if( p[i] < num[i] && larray[i][p[i]] == min ) {
	p[i]++;
	if( p[i] == num[i] )
	  done |= (1 << i);
      }
    }
  }

  // fill in the number array for this node
  gnum[root] = idx;

  delete [] larray;
  delete [] p;
  delete [] num;
} // traverse

// collapse - rewrite the structure of the tree to represent
//    collapsed nodes
template <class T>
void
TbonTreeUG<T>::collapse( int root, int d, int* gnum, int** glist, 
			 int* indexIndices, int* indexLists, int threshold ) {
  int i;
  if( d == 0 ) {
    // mark as leaf
    tree[root].branching = 8;

    // modify tree structure
    tree[root].child = *indexIndices;
    indices[*indexIndices] = *indexLists;
    *indexIndices = (*indexIndices)+1;

    // write list
    data->lists[*indexLists] = gnum[root];
    *indexLists = (*indexLists)+1;
    for( i = 0; i < gnum[root]; i++ ) {
      data->lists[*indexLists] = glist[root][i];
      *indexLists = (*indexLists)+1;
    }
    return;
  }

  // check if any child node is over the threshold
  int anyChildOverTH = 0;
  for( i = 0; i < BRANCHTABLE[ (int)tree[root].branching ]; i++ ) {
    if( gnum[ tree[root].child + i ] >= threshold )
      anyChildOverTH = 1;
  }

  if( anyChildOverTH ) {
    // at least one child is over threshold, must recurse over all children
    d--;
    for( i = 0; i < BRANCHTABLE[ (int)tree[root].branching ]; i++ )
      collapse( tree[root].child + i, d, gnum, glist, indexIndices, 
		indexLists, threshold );
  } else {
    // mark as leaf
    tree[root].branching = 8;
  
    // modify tree structure
    tree[root].child = *indexIndices;
    indices[*indexIndices] = *indexLists;
    *indexIndices = (*indexIndices)+1;

    // write list
    data->lists[*indexLists] = gnum[root];
    *indexLists = (*indexLists)+1;
    for( i = 0; i < gnum[root]; i++ ) {
      data->lists[*indexLists] = glist[root][i];
      *indexLists = (*indexLists)+1;
    }
  }

} // collapse

// condense - create a smaller tree using the condensed nodes
template <class T>
void
TbonTreeUG<T>::condense( Node<T>* newtree ) {
  int oldsize = 1;
  int* oldqueue = new int[oldsize];
  int* newqueue;
  int i, j;
  
  oldqueue[0] = 0;
  newtree[0] = tree[0];
  int newsize = BRANCHTABLE[ (int)tree[0].branching ];
  int last = newsize + 1;
  int curr = 1;
  int branch_i;

  for( int currdepth = depth; currdepth > 0; currdepth-- ) {
    int index = 0;
    int nextsize = 0;
    newqueue = new int[newsize];

    // add children of nodes in oldqueue to newqueue
    for( i = 0; i < oldsize; i++ ) {
      branch_i = (int)tree[oldqueue[i]].branching;
      // add children if this node is not a leaf
      if( branch_i < 8 ) {
	for( j = 0; j < BRANCHTABLE[ branch_i ]; j++ ) {
	  newqueue[index] = tree[oldqueue[i]].child + j;
	  nextsize += BRANCHTABLE[ (int)tree[ newqueue[index] ].branching ];
	  index++;
	}
      }
    } // i = 0 .. oldsize-1
     
    // copy nodes from newqueue into newtree
    for( i = 0; i < index; i++ ) {
      branch_i = (int)tree[ newqueue[i] ].branching;
      newtree[curr] = tree[ newqueue[i] ];
      // if this node is not a leaf, recompute the child pointer
      if( branch_i < 8 ) {
	newtree[curr].child = last;
	last += BRANCHTABLE[ branch_i ];
      }
      curr++;
    } // i = 0 .. index-1
    
    delete [] oldqueue;
    oldqueue = newqueue;
    oldsize = index;
    newsize = nextsize;
  } // currdepth = depth .. 1

  delete [] newqueue;

  delete [] tree;
  tree = newtree;
  numnodes = last;
} // condense

// writeGeometry - rewrite the geometry file, taking the condensed tree
//   into account
template <class T>
void
TbonTreeUG<T>::writeGeometry( const char* filename ) {
  int i;
  FILE* geom = fopen( filename, "w" );
  if( !geom ) {
    cerr << "Error: cannot open file " << filename << endl;
    return;
  }

  // find new extremes
  int* extremes = new int[2*data->ncells];
  for( i = 0; i < data->ncells; i++ ) {
    int min = INT_MAX;
    int max = -INT_MAX;
    int n = data->lists[ indices[i] ];
    if( n > 0 ) {
      for( int j = 1; j <= n; j++ ) {
	for( int k = 0; k < 4; k++ ) {
	  if( data->tets[ data->lists[ indices[i] + j ] ].v[k] < min )
	    min = data->tets[ data->lists[ indices[i] + j ] ].v[k];
	  if( data->tets[ data->lists[ indices[i] + j ] ].v[k] > max )
	    max = data->tets[ data->lists[ indices[i] + j ] ].v[k];
	}
      }
      extremes[2*i] = min;
      extremes[2*i+1] = max;
    } else {
      extremes[2*i] = -1;
      extremes[2*i+1] = -1;
    }
  } // done finding extremes

  fprintf(geom, "%d %d\n%d %d\n", data->npts, data->ntets, 
	  data->ncells, data->listsize);
  for( i = 0; i < data->npts; i++ )
    fwrite( &(data->verts[i].pos[0]), sizeof(float), 3, geom );
  for( i = 0; i < data->ntets; i++ )
    fwrite( &(data->tets[i].v[0]), sizeof(int), 4, geom );
  fwrite( data->lists, sizeof(int), data->listsize, geom );
  fwrite( extremes, sizeof(int), 2*data->ncells, geom );

  fclose(geom);
  delete [] extremes;
} // writeGeometry

// searchFirstPass - first pass of the tree traversal
//   read in nodes and set up data to be read
template <class T>
int
TbonTreeUG<T>::searchFirstPass( double iso ) {
  static int shiftamt = log((float)databricksize)/log(2.0) - log((float)sizeof(T))/log(2.0);
  int i, j;
  int isoCells = 0;
  int idx, index;

  int oldsize = 1;
  int* oldqueue = new int[oldsize];
  int* newqueue;

  oldqueue[0] = 0;
  int newsize = BRANCHTABLE[ (int)tree[0].branching ];

  for( int currdepth = depth; currdepth > 0; currdepth-- ) {
    idx = 0;
    int nextsize = 0;
    newqueue = new int[newsize];

    // find nodes at this level that span the isovalue
    for( i = 0; i < oldsize; i++ ) {
      if( tree[ oldqueue[i] ].min <= iso && tree[ oldqueue[i] ].max >= iso ) {

	if( (int)tree[oldqueue[i]].branching < 8 ) {
	  // add children to newqueue
	  for(j = 0;j < BRANCHTABLE[ (int)tree[oldqueue[i]].branching ];j++ ) {
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

	} else {
	  index = tree[ oldqueue[i] ].child;
	  isoCells += data->lists[ indices[index] ];
	  if( data->extremes[2*index] > -1 ) {
	    for( j = (data->extremes[2*index] >> shiftamt );
		 j <= (data->extremes[2*index + 1] >> shiftamt );
		 j++ ) 
	      brickstoread[j] = gen;
	  }
	} // if node oldqueue[i] is a leaf
      } // if node oldqueue[i] spans iso

    } // i = 0 .. size-1

    if( currdepth > 1 ) {
      delete [] oldqueue;
      oldqueue = newqueue;
      oldsize = idx;
      newsize = nextsize;
    }
  } // currdepth = depth .. 1

  // at depth 0, determine which data bricks to read
  for( int loopvar = 0; loopvar < idx; loopvar++ ) {
    if( tree[ newqueue[loopvar] ].min <= iso &&
	tree[ newqueue[loopvar] ].max >= iso ) {
      index = tree[ newqueue[loopvar] ].child;
      isoCells += data->lists[ indices[index] ];
      for( j = 0; j < BRANCHTABLE[ (int)tree[ newqueue[loopvar] ].branching ];
	   j++ ) {
	if( data->extremes[ 2*(index+j) ] > -1 ) {
	  for( i = (data->extremes[ 2*(index+j) ] >> shiftamt); 
	       i <= (data->extremes[ 2*(index+j) + 1 ] >> shiftamt);
	       i++ )
	    brickstoread[i] = gen;
	}
      } // for all children
    } // if newqueue[loopvar] spans iso
  } // loopvar = 0 .. newsize-1

  delete [] oldqueue;
  delete [] newqueue;

  return isoCells;
} // searchFirstPass

// searchSecondPass - second pass of tree traversal
//    perform interpolations in cells

template <class T>
void 
TbonTreeUG<T>::searchSecondPass( int myindex, double iso ) {
  if( tree[myindex].min <= iso && tree[myindex].max >= iso ) {
    if( tree[myindex].branching == 8 ) {
      for( int j = 0; j < BRANCHTABLE[(int)tree[myindex].branching]; j++ ) {
	int start = indices[tree[myindex].child+j];
	int n = data->lists[ start ];
	if( n != 0 ) {
	  for( int i = 1; i <= n; i++ ) {
	    if( !interpolated[ data->lists[ start+i ] ] ) 
	      mcube->interp( data->lists[ start+i ], iso );
	    interpolated[ data->lists[ start+i ] ] = 1;
	  }
	}
      }
      return;
    }

    searchSecondPass( tree[myindex].child, iso );
    switch( (int)tree[myindex].branching ) {
    case 0: 
      break;
    case 1: {
      searchSecondPass( tree[myindex].child + 1, iso );
    } break;
    case 2: {
      searchSecondPass( tree[myindex].child + 1, iso );
    } break;
    case 3: {
      searchSecondPass( tree[myindex].child + 1, iso );
      searchSecondPass( tree[myindex].child + 2, iso );
      searchSecondPass( tree[myindex].child + 3, iso );
    } break;
    case 4: {
      searchSecondPass( tree[myindex].child + 1, iso );
    } break;
    case 5: {
      searchSecondPass( tree[myindex].child + 1, iso );
      searchSecondPass( tree[myindex].child + 2, iso );
      searchSecondPass( tree[myindex].child + 3, iso );
    } break;
    case 6: {
      searchSecondPass( tree[myindex].child + 1, iso );
      searchSecondPass( tree[myindex].child + 2, iso );
      searchSecondPass( tree[myindex].child + 3, iso );
    } break;
    case 7: {
      searchSecondPass( tree[myindex].child + 1, iso );
      searchSecondPass( tree[myindex].child + 2, iso );
      searchSecondPass( tree[myindex].child + 3, iso );
      searchSecondPass( tree[myindex].child + 4, iso );
      searchSecondPass( tree[myindex].child + 5, iso );
      searchSecondPass( tree[myindex].child + 6, iso );
      searchSecondPass( tree[myindex].child + 7, iso );
    } break;
    }; // switch

  } // if( tree[myindex] spans iso )
} // searchSecondPass

} // End namespace Phil


#endif

