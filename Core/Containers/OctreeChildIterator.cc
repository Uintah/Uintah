/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include <Core/Containers/OctreeChildIterator.h>

namespace SCIRun {

int
OctreeChildIterator::defaultOrder = {0,1,2,3,4,5,6,7};

OctreeChildIterator::OctreeChildIterator( Octree<T> *tree,
					  const int *order) :
  tree(tree), current(0), order(order), isDone(false), start(0)
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

} // End namespace SCIRun
