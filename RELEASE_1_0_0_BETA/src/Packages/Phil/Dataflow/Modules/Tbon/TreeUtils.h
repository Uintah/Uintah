
/* TreeUtils.h
 * Routines for TbonTree, TbonTreeUG, etc. (with no class information required)
 *
 * Packages/Philip Sutton
 * May 1999

   Copyright (C) 2000 SCI Group, University of Utah
 */

#ifndef __TREE_UTILS_H__
#define __TREE_UTILS_H__

#include <math.h>


template <class T>
struct Node {
  char branching;
  T min, max;
  int child;
};

struct Corner {
  int x, y, z;
  void set( int _x, int _y, int _z ) { x = _x; y = _y; z = _z; }
};

// globals
static const int BRANCHTABLE[] = {1, 2, 2, 4, 2, 4, 4, 8, 1}; 

static char* lo;
static int curr;

// fillArray -  fill array with binary representation of num
static void 
fillArray( char* array, int num, int size ) {
  static int done_lo = 0;
  int i;
  for( i = size-1; i >= 0; i-- ) {
    //    int pow2 = 1 << i;
    int pow2 = (int)powf(2.0, (float)i);
    if( pow2 <= num ) { array[i] = 1; num -= pow2; }
    else { array[i] = 0; }
  }
  if( done_lo == 0 ) {
    lo = new char[size];
    for( i = 0; i < size; i++ )
      lo[i] = 1;
    done_lo = 1;
  }
} // fillArray

// countNodes - find number of nodes needed for tree
static void 
countNodes( char* xa, char* ya, char* za, int idx, int* numnodes, 
	    int* numleaves ) {
  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];

  if( idx == 0 ) {
    *numleaves = *numleaves + 1;
    return;
  }

  *numnodes += BRANCHTABLE[branching];
  idx--;

  switch( branching ) {
  case 0: {
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 1: {
    countNodes( lo, ya, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 2: {
    countNodes( xa, lo, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 3: {
    countNodes( lo, lo, za, idx, numnodes, numleaves );
    countNodes( xa, lo, za, idx, numnodes, numleaves );
    countNodes( lo, ya, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 4: {
    countNodes( xa, ya, lo, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 5: {
    countNodes( lo, ya, lo, idx, numnodes, numleaves );
    countNodes( xa, ya, lo, idx, numnodes, numleaves );
    countNodes( lo, ya, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 6: {
    countNodes( xa, lo, lo, idx, numnodes, numleaves );
    countNodes( xa, ya, lo, idx, numnodes, numleaves );
    countNodes( xa, lo, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  case 7: {
    countNodes( lo, lo, lo, idx, numnodes, numleaves );
    countNodes( xa, lo, lo, idx, numnodes, numleaves );
    countNodes( lo, ya, lo, idx, numnodes, numleaves );
    countNodes( xa, ya, lo, idx, numnodes, numleaves );
    countNodes( lo, lo, za, idx, numnodes, numleaves );
    countNodes( xa, lo, za, idx, numnodes, numleaves );
    countNodes( lo, ya, za, idx, numnodes, numleaves );
    countNodes( xa, ya, za, idx, numnodes, numleaves );
  } break;
  };
  
} // countNodes

// createTree - construct tree skeleton
static void 
createTree( char* xa, char* ya, char* za, int idx, int* myindex, 
	    char* branchArray, int* sibArray, int last ){
  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];

  if( idx == 0 ) {
    branchArray[*myindex] = (char)branching;
    *myindex = *myindex + 1;
    return;
  }

  int counter = *myindex + 1;
  idx--;
  
  switch( branching ) {
  case 0: {
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 1: {
    createTree( lo, ya, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 2: {
    createTree( xa, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 3: {
    createTree( lo, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( lo, ya, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 4: {
    createTree( xa, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 5: {
    createTree( lo, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( lo, ya, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 6: {
    createTree( xa, lo, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  case 7: {
    createTree( lo, lo, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, lo, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( lo, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, lo, idx, &counter, branchArray, sibArray, 0 );
    createTree( lo, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, lo, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( lo, ya, za, idx, &counter, branchArray, sibArray, 0 );
    createTree( xa, ya, za, idx, &counter, branchArray, sibArray, last+1 );
  } break;
  };

  if( !last )
    sibArray[*myindex] = counter;
  else
    sibArray[*myindex] = counter + last;
  branchArray[*myindex] = (char)branching;
  *myindex = counter;

} // createTree

// notinlist
//   return 1 if element "elem" is not in array "list" (of size "size")
//   return 0 otherwise
static int
notinlist( int elem, int* list, int size ) {
  int i;
  for( i = 0; i < size; i++ ) {
    if( list[i] == elem )
      return 0;
  }
  return 1;
} // notinlist

// intLess
//   comparison function for qsort
inline int
intLess( const void* i1, const void* i2 ) {
  int* a = (int *)i1;
  int* b = (int *)i2;
  return ( *a < *b ) ? -1 : (( *a == *b ) ? 0 : 1);
} // intLess

// countNodesSS - find number of nodes needed for hybrid tree
static void 
countNodesS( char* xa, char* ya, char* za, int idx, int cutoff, 
	     int* numnodes, int* numleaves ) {
  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];

  if( idx == cutoff ) {
    *numleaves = *numleaves + 1;
    return;
  }

  *numnodes += BRANCHTABLE[branching];
  idx--;

  switch( branching ) {
  case 0: {
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 1: {
    countNodesS( lo, ya, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 2: {
    countNodesS( xa, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 3: {
    countNodesS( lo, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( lo, ya, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 4: {
    countNodesS( xa, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 5: {
    countNodesS( lo, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( lo, ya, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 6: {
    countNodesS( xa, lo, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  case 7: {
    countNodesS( lo, lo, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, lo, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( lo, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, lo, idx, cutoff, numnodes, numleaves );
    countNodesS( lo, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, lo, za, idx, cutoff, numnodes, numleaves );
    countNodesS( lo, ya, za, idx, cutoff, numnodes, numleaves );
    countNodesS( xa, ya, za, idx, cutoff, numnodes, numleaves );
  } break;
  };
  
} // countNodesS

// createTreeS - construct hybrid tree skeleton
static void 
createTreeS( char* xa, char* ya, char* za, int idx, int cutoff, 
	     int* myindex, char* branchArray, int* sibArray, int last ) {
  int branching = xa[idx] + 2*ya[idx] + 4*za[idx];

  if( idx == cutoff ) {
    branchArray[*myindex] = (char)branching;
    *myindex = *myindex + 1;
    return;
  }

  int counter = *myindex + 1;
  idx--;
  
  switch( branching ) {
  case 0: {
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 1: {
    createTreeS( lo, ya, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 2: {
    createTreeS( xa, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 3: {
    createTreeS( lo, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( lo, ya, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 4: {
    createTreeS( xa, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 5: {
    createTreeS( lo, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( lo, ya, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 6: {
    createTreeS( xa, lo, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  case 7: {
    createTreeS( lo, lo, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, lo, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( lo, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, lo, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( lo, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, lo, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( lo, ya, za, idx, cutoff, &counter, branchArray, sibArray, 0 );
    createTreeS( xa, ya, za, idx, cutoff, &counter, branchArray, sibArray, last+1 );
  } break;
  };

  if( !last )
    sibArray[*myindex] = counter;
  else
    sibArray[*myindex] = counter + last;
  branchArray[*myindex] = (char)branching;
  *myindex = counter;

} // createTreeS



#endif

