
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/Array2.cc>
#include <Classlib/Array3.cc>
#include <Classlib/AVLTree.cc>
#include <Classlib/Debug.h>
#include <Classlib/HashTable.cc>
#include <Classlib/Queue.cc>
#include <Classlib/String.h>

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

class PersistentTypeID;
class Persistent;
template class HashTable<clString, PersistentTypeID*>;
template class HashKey<clString, PersistentTypeID*>;
template class HashTable<int, Persistent*>;
template class HashKey<int, Persistent*>;
template class HashTable<Persistent*, int>;
template class HashKey<Persistent*, int>;

template class Queue<int>;
template class Queue<float>;
template class Queue<char*>;
