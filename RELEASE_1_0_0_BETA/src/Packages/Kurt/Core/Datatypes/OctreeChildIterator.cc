#include "OctreeChildIterator.h"

namespace Kurt {
int
OctreeChildIterator::defaultOrder = {0,1,2,3,4,5,6,7};

OctreeChildIterator::OctreeChildIterator( Octree<T> *tree,
					  const int *order) :
  tree(tree), current(0), order(order), isDone(false), start(0);
{
  if( tree->type() == Octree< T >::LEAF )
    {
      isDone = true;
    } else {
      int i;
      for( i = 0; i < 8; i++ )
	if( (*tree)[ order[i] ] != 0 ){
	  start = *((*tree)[ order[i] ]);
	  current = start;
	  break;
	}
      if ( i == 8 )
	isDone = true;
    }
}

template<class T>
const Octree< T > &
OctreeChildIterator::Next()
{
  if( current < 7) 
    return *((*tree)[++current]);
  if( current == 7)
    isDone = true;
  if( current >= 8)
    return 0;
}
} // End namespace Kurt

