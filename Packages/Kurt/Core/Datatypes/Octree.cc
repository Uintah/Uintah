#include "Octree.h"
#include <iostream>
using std::cerr;

namespace Kurt {
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


} // End namespace Kurt
  
