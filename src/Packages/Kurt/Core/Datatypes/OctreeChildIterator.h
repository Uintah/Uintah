#ifndef OCTREE_CHILD_ITERATOR_H
#define OCTREE_CHILD_ITERATOR_H

#include "Octree.h"

namespace Kurt {
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
} // End namespace Kurt

#endif
