#ifndef OCTREE_H
#define OCTREE_H

namespace Kurt {
/**************************************

CLASS
   Octree
   
   Simple Octree Class.

GENERAL INFORMATION

   Octree.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Octree

DESCRIPTION
   Octree class.
  
WARNING
  
****************************************/

template<class T>
class Octree {
public:
  enum nodeType {LEAF, PARENT};

  // GROUP: Constructors:
  //////////
  // Constructor
  Octree(T stored, Octree::nodeType t, const Octree<T> *parent = 0);
  // GROUP: Destructors
  //////////
  // Destructor
  ~Octree();

  // GROUP: Modify
  //////////
  // Set one of the 8 children
  void SetChild(int i, Octree<T>* n);

  // GROUP: Access
  //////////
  // get the ith child
  const Octree<T>* operator[](int i) const;
  //////////
  // access the parent node.
  const Octree<T>* parent() { return parent; }
  //////////
  // Is this node a LEAF or PARENT?  
  nodeType type() const {return t;}
  //////////
  // Return the data stored at this node. 
  T operator()() const;
private:
  T stored;
  nodeType t;
  Octree <T> **children;
  const Octree <T> *Parent;
};
} // End namespace Kurt

#endif
