
/*
 *  HashTable.h: Interface to HashTable type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_Classlib_HashTable_h
#define SCI_Classlib_HashTable_h 1

#include <Classlib/Assert.h>

#ifdef __GNUG__
#pragma interface
#endif

class Piostream;

template<class Key, class Data> class HashKey;
template<class Key, class Data> class HashTableIter;

// Provide default hash functions for ints, longs and void*'s

template<class Key> inline int Hash(const Key& k, int hash_size)
{
    int h=k.hash(hash_size);
    ASSERTRANGE(h, 0, hash_size);
    return h;
}

inline int Hash(const int& k, int hash_size)
{
    return (k^(3*hash_size+1))%hash_size;
}

inline int Hash(const unsigned long& k, int hash_size)
{
    return (k^(3*hash_size+1))%hash_size;
}   

typedef void* HashVoid;

inline int Hash(const HashVoid& k, int hash_size)
{
    return (int)(((long)k^(3*hash_size+1))%hash_size);
}

// The hashtable itself
template<class Key, class Data> class HashTable {
    HashKey<Key,Data>** table;
    int hash_size;
    int nelems;
    
    void rehash(int newsize);
    friend class HashTableIter<Key, Data>;
public:
    HashTable();
    HashTable(const HashTable<Key, Data>&);
    ~HashTable();
    // Inserts the key/data pair into the hash table
    void insert(const Key& key, const Data& data);

    // Looks up key in the hashtable.  Returns 0 if not found.
    // Returns 1, and places the data item in data if it is found.
    // If more than one of "key" exist, it is undefined which it
    // will return.
    int lookup(const Key& key, Data& data);

    // Removes all items with key "key" from the hash table.
    // Returns the number actually removed
    int remove(const Key& key);

    // Empties the hash table
    void remove_all();

    // Returns how many items are stored in the hash table
    int size() const;

    // Persistent io
    friend void Pio(Piostream&, HashTable<Key, Data>&);
};

// Use this class for walking through a hashtable
template<class Key, class Data> class HashTableIter {
    HashTable<Key, Data>* hash_table;
    HashKey<Key, Data>* current_key;
    int current_index;
public:
    // Build a hash table iterator for a specific hash table
    HashTableIter(HashTable<Key, Data>*);
    
    // Reset the iterator to the first item
    void first();

    // Does the iterator point to a valid item?
    int ok();

    // Advance to the next item
    void operator++();

    // Get the key from the current item
    Key& get_key();

    // Get the data from the current item
    Data& get_data();
};

// An element in the HashTable
// Used internally by HashTable ONLY
template<class Key, class Data> class HashKey {
    friend class HashTable<Key, Data>;
    friend class HashTableIter<Key, Data>;
    Key key;
    Data data;
    HashKey<Key, Data>* next;
    
    HashKey();
    HashKey(const Key&, const Data&, HashKey<Key, Data>*);
    HashKey(const HashKey<Key, Data>&, int deep=0);
    friend void Pio(Piostream&, HashTable<Key, Data>&);
};

#endif
