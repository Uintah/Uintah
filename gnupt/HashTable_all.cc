
#pragma implementation "HashTable.h"

#define HASH_OTHER

#include <Classlib/HashTable.cc>
class GeomObj;
class Persistent;

typedef HashTable<int, int> _dummy2_;
typedef HashTable<int, GeomObj*> _dummy3_;
typedef HashTableIter<int, GeomObj*> _dummy4_;
typedef HashTable<int, _dummy3_*> _dummy5_;
typedef HashTableIter<int, _dummy3_*> _dummy6_;
typedef HashTable<Persistent*, int> _dummy7_;
typedef HashTable<int, Persistent*> _dummy8_;
