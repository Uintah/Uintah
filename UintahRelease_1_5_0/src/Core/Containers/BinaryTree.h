/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
#endif // BINARYTREE_H
