
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/AVLTree.cc>
#include <Classlib/HashTable.cc>
#include <Classlib/Queue.cc>
#include <Dataflow/ModuleList.h>

class OPort;
template class Array1<OPort*>;
class IPort;
template class Array1<IPort*>;
class Connection;
template class Array1<Connection*>;
class Module;
template class Array1<Module*>;

#include <Multitask/Mailbox.cc>
class MessageBase;
template class Mailbox<MessageBase*>;

template class AVLTree<clString, makeModule>;
template class AVLTreeIter<clString, makeModule>;
template class AVLTree<clString, ModuleCategory*>;
template class AVLTreeIter<clString, ModuleCategory*>;

template class HashTable<clString, clString>;
template class HashTable<clString, void*>;
template class HashTable<clString, Connection*>;
template class HashTable<clString, Module*>;
template class HashTable<int, Module*>;
template class HashTable<int, Connection*>;

template class Queue<Module*>;
