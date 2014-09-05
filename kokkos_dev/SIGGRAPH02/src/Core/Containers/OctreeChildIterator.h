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

#ifndef OCTREE_CHILD_ITERATOR_H
#define OCTREE_CHILD_ITERATOR_H

#include <Core/Containers/Octree.h>

namespace SCIRun {

/**************************************

CLASS
   OctreeChildIterator
   
   Class.

GENERAL INFORMATION

   OctreeChildIterator.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   OctreeChildIterator

DESCRIPTION
   OctreeChildIterator class.  This class is only a little strange.
   Given an octree node, iterate through its children in a specific order.
   If no order is given, then iterate through the children via a default
   order [0, ..., 7].  Different types of traversals can be constructed by
   using a child to create another iterator and another traversal order.
   The "order" parameter is an array of 8 values consisting of some 
   arrangement of the integers [0,7]
  
WARNING
  
****************************************/

template<class T>
class OctreeChildIterator {
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  OctreeChildIterator(Octree<T> *tree, const int *order = defaultOrder);
  // GROUP: Destructors
  //////////
  // Destructor
  ~OctreeChildIterator();

  // GROUP: Modify
  //////////
  // Set one of the 8 children

  // GROUP: Access & Info
  //////////
  // Get the first child according to the ordering
  const Octree< T > * Start() { index = 1; return start; }
  //////////
  // Get the next child according to the ordering
  const Octree< T > & Next();
  //////////
  // Have we traversed all of the children?
  bool IsDone(){ return isDone; }


private:
  static const int defaultOrder[8];
  int *order;
  Octree<T> *tree;
  Octree<T> *start;
  Octree<T> *current;
  bool isDone;
};

} // End namespace SCIRun
#endif
