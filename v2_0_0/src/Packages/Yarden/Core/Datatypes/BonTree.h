/*
 *  Bon.cc    View Depended Iso Surface Extraction
 *             for Structures Grids (Bricks)
 *  Written by:
 *    Packages/Yarden Livnat
 *    Department of Computer Science
 *    University of Utah
 *    Oct 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Bon_h
#define Bon_h

#include <Core/Datatypes/ScalarFieldRG.h>

namespace Yarden {

  template< class T >
  struct Node {
    Node<T> *child;
    T min, max;
    unsigned char type;
  };
  
  template< class T, class F > class Tree {
  public:
    F *field;
    Node<T> **tree;
    Node<T> **tree_next;
    int size; 
    int dx, dy, dz;
    int depth;
    int last;
    int mask;
    int ss[20];

  public:
    Tree() {tree=0; size=0;}
    ~Tree() {if ( tree ) delete [] tree; }
    
    //Node &operator[](int i) { return tree[i]; }
    void init( F *);
    void build( int level, Node<T> &, int mask, int x, int y, int z );
    void get_minmax( T v, T &min, T &max );
    void child_minmax( Node<T> &child, 
		       int x, int y, int z, int dx, int dy, int dz,
		       int mask, T &min, T &max );
    void fill( Node<T> &node,
	       int x, int y, int  z, int dx, int dy, int dz, int mask,
	       T &pmin, T &pmax );
  };
  

  // BonTree   
  template<class T, class F>
  void Tree<T,F>::init( F *f )
{
  field = f;
  dx = field->grid.dim1()-1; // number of cells = number of pts -1
  dy = field->grid.dim2()-1;
  dz = field->grid.dim3()-1;
  
  if (tree ) {
    for (int l=0; l<depth+1; l++ )
      delete tree[l];
    delete tree;
    delete tree_next;
  }
  int dim = dx;
  if ( dy > dim ) dim = dy;
  if ( dz > dim ) dim = dz;
  int mdim = dim-1;  //  = number of cells - 1 = number_of pts -2
  for (depth=1,mask=1; mdim > 1 ; mdim>>=1, depth++, mask<<=1);
  
  int new_size = 0;
  tree = new Node<T>*[depth+1];
  tree_next = new Node<T>*[depth+1];
  
  int s = 1;
  printf("depth: %d\n",depth);
  for (int i=depth; i>0; i--) {
    int n = ((dx>>i)+1)*((dy>>i)+1)*((dz>>i)+1);
    ss[depth-i] = s;
    tree_next[depth-i] = tree[depth-i] = scinew Node<T>[s];
    new_size += s;
    printf("Tree [%d]  %d %d   [%3dx%3dx%3d]\n", 
	   depth-i, s, new_size,
	   ((dx>>i)+1),((dy>>i)+1),((dz>>i)+1));
    s = n*8;; 
  }
  
  printf( "Tree: %d / %ld\n", new_size,  long(dim+1)*(dim+1)*(dim+1)*8/7 );
  printf( "\tmemory  node: (2*%d + %d + %d) = %d bytes\n"
	  "\t  tree = %.0fMB   data = %.0fMB\n", 
	  sizeof(T), sizeof(Node<T>*),sizeof(int),sizeof(Node<T>),
	  new_size*sizeof(Node<T>)/float(1024*1024),
	  (dx*dy*dz)*sizeof(T)/float(1024*1024)) ;
  
  build( 1, tree[0][0], mask, dx-2, dy-2, dz-2 );
  
  cerr << "fill [" << dx << ", " << dy << ", " << dz << "]  depth= " 
       << depth << "\n" ;
  
  
  T min, max;
  fill( tree[0][0], 0, 0, 0, dx-2, dy-2, dz-2, mask, min, max );
  cerr << "Min " << int(min) << "  Max " << int(max) << "\n";
  }
  
  template<class T, class F>
  void Tree<T,F>::build( int level, Node<T> &node, int mask, 
		       int dx, int dy, int dz )
{
  if ( !( dx>1 || dy>1 || dz>1 ) ) {
    node.child = 0;
    node.type = 7;
    //     printf("build: end at level %d.   %dx%dx%d\n", level ,dx, dy,dz );
    return;
  }
  
  //assert (ss[level] >= 8 );
  
  node.child = tree_next[level];
  tree_next[level] += 8;
  ss[level]-=8;
  
  unsigned type = 0;
  int dx1, dy1, dz1;
  if ( (mask & dx) && (dx > 1) ) {
    dx1 = dx & ~mask;
    dx = mask-1;
    type = 1;
  }
  if ( (mask & dy) && (dy > 1) ) {
    dy1 = dy & ~mask;
    dy  = mask-1;
    type += 2;
  }
  if ( (mask & dz) &&(dz > 1) ) {
    dz1 = dz & ~mask;
    dz  = mask-1;
    type+=4;
  }
  
  type = (~type) & 0x7;
  node.type = type;
  
  for (int i=0; i<8; i++ )
    if ( ! (i & type) )
      build ( level+1, node.child[i], mask>>1,
	      i & 1 ? dx1 : dx,
	      i & 2 ? dy1 : dy, 
	      i & 4 ? dz1 : dz );
    else
      node.child[i].child = 0;
  }
  
  template<class T, class F> 
  void Tree<T,F>::get_minmax( T v, T &min, T &max )
{
  if ( v < min ) min = v;
  else if ( v > max ) max = v;
  }

  template<class T, class F> void
  Tree<T,F>::child_minmax( Node<T> &child, int x, int y, int z, int dx, int dy, int dz,
		      int mask,  T &min, T &max )
{
  T cmin = 0;
  T cmax = 0;
  
  fill( child, x, y, z, dx, dy, dz, mask, cmin, cmax );
  
  if ( cmin < min ) min = cmin;
  if ( max < cmax ) max = cmax;
}
  
  int iii = 0;

  template<class T, class F>
  void Tree<T,F>::fill( Node<T> &node, int x, int y, int  z, int dx, int dy, int dz,
		   int mask, T &pmin, T &pmax )
{
  Node<T> *child = node.child;
  int type = node.type;
  
  T min, max;
  
  if ( !child ) {
    min = max = field->grid(x,y,z   );
    for (int i=0; i<3; i++ )
      for (int j=0; j<3; j++)
	for (int k=0; k<3; k++)
	  get_minmax( field->grid( x+i, y+j, z+k ), min, max );
  }
  else {
    int dx1, dy1, dz1;
    if ( (mask & dx) && dx>1) {
      dx1 = dx & ~mask;
      dx  = mask-1;
    }
    if ( (mask & dy) && dy>1) {
      dy1 = dy & ~mask;
      dy  = mask-1;
    }
    if ( (mask & dz) && dz>1) {
      dz1 = dz & ~mask;
      dz  = mask-1;
    }
    mask >>= 1;
    
    fill( child[0], x, y, z, dx, dy, dz, mask, min, max );
    if ( !(type & 0x1) ) child_minmax( child[1], x+dx+1, y,    z,
				       dx1, dy, dz, mask, min, max );
    if ( !(type & 0x2) ) child_minmax( child[2], x,    y+dy+1, z,
				       dx, dy1, dz, mask, min, max );
    if ( !(type & 0x3) ) child_minmax( child[3], x+dx+1, y+dy+1, z,
				       dx1, dy1, dz, mask, min, max );
    if ( !(type & 0x4) ) child_minmax( child[4], x,    y,    z+dz+1,
				       dx, dy, dz1, mask, min, max );
    if ( !(type & 0x5) ) child_minmax( child[5], x+dx+1, y,    z+dz+1,
				       dx1, dy, dz1, mask, min, max );
    if ( !(type & 0x6) ) child_minmax( child[6], x,    y+dy+1, z+dz+1,
				       dx, dy1, dz1, mask, min, max );
    if ( !(type & 0x7) ) child_minmax( child[7], x+dx+1, y+dy+1, z+dz+1,
				       dx1, dy1, dz1, mask, min, max );
  }
  
  node.min = min;
  node.max = max;
  pmin = min;
  pmax = max;
} 
} // End namespace Yarden



#endif // BonTree_h


