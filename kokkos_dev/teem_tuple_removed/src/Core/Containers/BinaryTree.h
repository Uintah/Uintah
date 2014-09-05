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

#ifndef BINARYTREE_H
#define BINARYTREE_H

/************************************************
CLASS
   BinaryTree

   Simple binary tree class
   BinaryTree.h

GENERAL INFORMATION
   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2003 SCI Group
*************************************************/

namespace SCIRun {

//template <class T> BinaryTree<T>;

template<class T>
class BinaryTree {
public:
  enum nodeType{LEAF, PARENT};

  // Constructor
  BinaryTree( const T stored, nodeType t);

  // Destructor
  ~BinaryTree();

  // Modify
  void AddChild(BinaryTree<T>* c, int child); // Child is 0 or 1

  // Access
  BinaryTree<T>* child( int child) const;

  nodeType type() const { return t_; } // LEAF or PARENT?
  T stored() const { return stored_; } // return what is stored here.
private:
  T stored_;
  nodeType t_;
  BinaryTree<T> *child0_;
  BinaryTree<T> *child1_;
    
};

template<class T>
BinaryTree<T>::BinaryTree( const T stored, typename BinaryTree::nodeType t) :
  stored_(stored), t_(t), child0_(0), child1_(0)
{}

  

template<class T>
BinaryTree<T>::~BinaryTree()
{
  if( child0_ ) delete child0_;
  if( child1_ ) delete child1_;
  delete stored_;
}

template<class T>
void
BinaryTree<T>::AddChild( BinaryTree<T> *c, int child )
{
  if( !child ) {  //child = 0
    if( child0_ ) delete child0_;
    child0_ = c;
  } else if( child == 1 ){
    if( child1_ ) delete child1_;
    child1_ = c;
  }    
}

template<class T>
BinaryTree<T>*
BinaryTree<T>::child( int child ) const
{
  if( child == 0 ) return child0_;
  else if( child == 1) return child1_;
  else return 0;
}

} // end namespace SCIRun
#endif BINARYTREE_H
