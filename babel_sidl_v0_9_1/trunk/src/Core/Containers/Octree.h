/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
