
#pragma implementation "HashTable.h"

#define HASH_OBJECTS
#include <Classlib/HashTable.cc>
#include <Classlib/String.h>


class PersistentTypeID;
class Arg_item;
class Arg_base;
class Module;
class Connection;
class Renderer;

typedef HashTable<clString, PersistentTypeID*> _dummy1_;
typedef HashTable<clString, Arg_item*> _dummy3_;
typedef HashTableIter<clString, Arg_item*> _dummy4_;
typedef HashTable<clString, Arg_base*> _dummy5_;
typedef HashTableIter<clString, Arg_base*> _dummy6_;
typedef HashTable<clString, Module*> _dummy7_;
typedef HashTable<clString, Connection*> _dummy8_;
typedef HashTable<clString, Renderer*> _dummy9_;
typedef HashTable<clString, int> _dummy10_;
