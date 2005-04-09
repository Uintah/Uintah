
#pragma implementation "HashTable.h"

#define HASH_OBJECTS
#include <Classlib/HashTable.cc>

struct Face {
    int n[3];
    Face(int, int, int);
    int hash(int hash_size) const;
    int operator==(const Face&) const;
};

typedef HashTable<Face, int> _dummy9_;
typedef HashTableIter<Face, int> _dummy10_;
