
/*
 * Manual template instantiations for g++
 */

#include <SCICore/Util/Debug.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/HashTable.h>
#include <SCICore/Containers/Queue.h>
#include <SCICore/Containers/String.h>

//namespace SCICore {

//namespace PersistentSpace {
//  class PersistentTypeID;
//  class Persistent;
//}

//namespace Containers {

using namespace SCICore::Util;
using namespace SCICore::PersistentSpace;
using namespace SCICore::Containers;

template class Array1<int>;
template class Array1<unsigned int>;
template class Array1<float>;
template class Array1<clString>;
template class Array1<DebugSwitch*>;

template class Array2<int>;

template class Array3<int>;

template class AVLTree<clString, DebugVars*>;
template class AVLTreeIter<clString, DebugVars*>;
template class TreeLink<clString, DebugVars*>;

template class HashTable<clString, int>;
template class HashKey<clString, int>;

template class HashTable<clString, PersistentTypeID*>;
template class HashKey<clString, PersistentTypeID*>;
template class HashTable<int, Persistent*>;
template class HashKey<int, Persistent*>;
template class HashTable<Persistent*, int>;
template class HashKey<Persistent*, int>;

template class Queue<int>;
template class Queue<float>;
template class Queue<char*>;

//} // End namespace Containers
//} // End namespace SCICore 

