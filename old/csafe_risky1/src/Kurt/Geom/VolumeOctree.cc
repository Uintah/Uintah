#include "VolumeOctree.h"


namespace Kurt{
  namespace GeomSpace{

template<class T>
VolumeOctree<T>::VolumeOctree(const Point min, const Point max, 
			      const T stored, int nodeId,
			      nodeType t):
  min(min), max(max), t(t), stored(stored), id(nodeId)
{
  if( t == LEAF ){
    children = NULL;
  } else {
    children = new VolumeOctree<T>*[8];
    for( int i = 0; i < 8; i++)
      children[i] = NULL;
  }
}

template<class T>  
VolumeOctree<T>::~VolumeOctree()
{
  if (children){
    for(int i = 0; i < 8; i++)
      {
	delete children[i];
      }
  }
}

template<class T>
T VolumeOctree<T>::operator()() const {
  return stored;
}

template<class T>
void VolumeOctree<T>::SetChild(int i, VolumeOctree<T>* n)
{
  children[i] = n;
}

template<class T>
const VolumeOctree<T>* VolumeOctree<T>::child(int i) const
{
  return children[i];
}


  
} // namespace GeomSpace
} // namespace Kurt
