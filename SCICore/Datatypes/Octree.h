#ifndef OCTREE_H
#define OCTREE_H

namespace SCICore {
namespace Datatypes {

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

template<class T>
Octree<T>::Octree(const T stored, nodeType t, const Octree<T> *parent):
  t(t), stored(stored), Parent(parent)
{
  if( t == LEAF ){
    children = 0;
  } else {
    children = scinew Octree<T>*[8];
    for( int i = 0; i < 8; i++)
      children[i] = 0;
  }
}

template<class T>  
Octree<T>::~Octree()
{
   if (children){
     delete [] children;
   }
  delete stored;
}

template<class T>
T Octree<T>::operator()() const {
  return stored;
}

template<class T>
void Octree<T>::SetChild(int i, Octree<T>* n)
{
  children[i] = n;
}

template<class T>
const Octree<T>* Octree<T>::operator[](int i) const
{
  if( i >= 0 && i < 8 )
    return children[i];
  else 
    return 0;
}

} // end namespace Datatypes
} // end namespace SCICore
#endif
