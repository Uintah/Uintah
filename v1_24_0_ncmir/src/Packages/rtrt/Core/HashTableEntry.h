
#ifndef HASHTABLEENTRY_H
#define HASHTABLEENTRY_H 1

namespace rtrt {

template<class Key, class Data> class HashTable;
template<class Key, class Data> class HashTableIter;

template<class Key, class Data> class HashTableEntry {
    /*
     * Copy constructor intentionally left undefined.
     */
    HashTableEntry(const HashTableEntry<Key, Data>&);

    friend class HashTable<Key, Data>;
    friend class HashTableIter<Key, Data>;
    Key key;
    Data data;
    HashTableEntry<Key, Data>* next;
    
    HashTableEntry();
    HashTableEntry(const Key& key, const Data& data,
		   HashTableEntry<Key, Data>* next );
};

  /////////////////////////////////////////
  // All the templated functions

template<class Key, class Data>
HashTableEntry<Key, Data>::HashTableEntry()
    : next(0)
{
}

template<class Key, class Data>
HashTableEntry<Key, Data>::HashTableEntry(const Key& key, const Data& data,
					  HashTableEntry<Key, Data>* next )
    : key(key), data(data), next(next)
{
}
  
} // end namespace rtrt

#endif
