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

#ifndef OCTREE_H
#define OCTREE_H

#include <Core/Malloc/Allocator.h>

namespace SCIRun {

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
  Octree(T stored, typename Octree::nodeType t, const Octree<T> *parent = 0);
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
  Octree<T>* operator[](int i) const;
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
  stored(stored), t(t), Parent(parent)
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
     for(int i = 0; i < 8; i++){
       delete children[i];
     }
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
Octree<T>* Octree<T>::operator[](int i) const
{
  if( i >= 0 && i < 8 )
    return children[i];
  else 
    return 0;
}

} // End namespace SCIRun
#endif
