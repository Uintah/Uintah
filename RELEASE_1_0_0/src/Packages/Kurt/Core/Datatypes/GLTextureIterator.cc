#include "GLTextureIterator.h"
#include "Brick.h"
#include <iostream>

namespace Kurt {
using namespace SCIRun;

int GLTextureIterator::traversalTable[27][8] = { {7,3,5,6,1,2,4,0},
					  {6,7,2,3,4,5,0,1},
					  {6,2,4,7,0,3,5,1},
					  {5,7,1,3,4,6,0,2},
					  {4,5,6,7,0,1,2,3},
					  {4,6,0,2,5,7,1,3},
					  {5,1,4,7,0,3,6,2},
					  {4,5,0,1,6,7,2,3},
					  {4,0,5,6,1,2,7,3},
					  {3,7,1,2,5,6,0,4},
					  {2,3,6,7,0,1,4,5},
					  {2,6,0,3,4,7,1,5},
					  {1,3,5,7,0,2,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,4,6,1,3,5,7},
					  {1,5,0,3,4,7,2,6},
					  {0,1,4,5,2,3,6,7},
					  {0,4,1,2,5,6,3,7},
					  {3,1,2,7,0,5,6,4},
					  {2,3,0,1,6,7,4,5},
					  {2,0,3,6,1,4,7,5},
					  {1,3,0,2,5,7,4,6},
					  {0,1,2,3,4,5,6,7},
					  {0,2,1,3,4,6,5,7},
					  {1,0,3,5,2,4,7,6},
					  {0,1,2,3,4,5,6,7},
					  {0,1,2,4,3,5,6,7}};



GLTextureIterator::GLTextureIterator(const GLTexture3D* tex,
				       Ray view,
				       Point c ):
    tex(tex), view(view), control( c ), done( false )
{

}

GLTextureIterator::~GLTextureIterator()
{
}

deque<int>* 
GLTextureIterator::traversal(const Octree<Brick*>* n)
{
  const Octree<Brick*>& node = *n;
  const Octree<Brick*>& child = *(node[0]);


  int *traversal;
  int traversalIndex, x, y, z;

  Point min, max, mid;
  mid = child()->bbox().max();
  min = child()->bbox().min();

  if( view.origin().x() < mid.x()) x = 0;
  else if( view.origin().x() == mid.x()) x = 1;
  else x = 2;
  if( view.origin().y() < mid.y()) y = 0;
  else if( view.origin().y() == mid.y()) y = 1;
  else y = 2;
  if( view.origin().z() < mid.z()) z = 0;
  else if( view.origin().z() == mid.z()) z = 1;
  else z = 2;
    
  traversalIndex = 9*x + 3*y + z;

  traversal = traversalTable[ traversalIndex ];

  deque< int > *q = scinew deque< int >;
  
  //  std::cerr<<"Child order ";
  for( int i = 0; i < 8; i++){
    if((*(node[ traversal[i] ]))() != 0){
      //      std::cerr<<traversal[i]<<" ";
      q->push_back( traversal[i] );
    }
  }
  //  std::cerr<<std::endl;
  return q;
}
} // End namespace Kurt

