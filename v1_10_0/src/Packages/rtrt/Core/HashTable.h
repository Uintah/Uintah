
#ifndef HASHTABLE_H
#define HASHTABLE_H 1

#include <Packages/rtrt/Core/Hash.h>
#include <Packages/rtrt/Core/HashTableEntry.h>

namespace rtrt {

template<class Key, class Data> class HashTable {
    HashTableEntry<Key,Data>** table;
    unsigned int hash_size;
    unsigned int nelems;
    
    void rehash(unsigned int newsize);
    friend class HashTableIter<Key, Data>;
public:
    HashTable();
    HashTable(const HashTable<Key, Data>&);

    ~HashTable();

    /*
     * Inserts the key/data pair into the hash table
     */
    void insert(const Key& key, const Data& data);

    /* Looks up key in the hashtable.  Returns false if not found.
     * Returns true, and places the data item in data if it is found.
     * If more than one of <em>key<\em> exists, it is undefined which it
     * will return.
     */
    bool lookup(const Key& key, Data& data);

    /*
     * Removes all items with <em>key<\em> from the hash table.
     * Returns the number actually removed
     */
    int remove(const Key& key);

    /*
     * Empties the hash table
     */
    void remove_all();

    /*
     * Returns how many items are stored in the hash table
     */
    int size() const;
};

  /////////////////////////////////////////////////
  // All the templated function

  template <class Key, class Data>
  void HashTable<Key, Data>::rehash(unsigned int newsize)
  {
    HashTableEntry<Key, Data>** oldtab=table;
    table=new HashTableEntry<Key, Data>*[newsize];
    for(unsigned int ii=0;ii<newsize;ii++){
      table[ii]=0;
    }
    unsigned int oldsize=hash_size;
    hash_size=newsize;
    for(unsigned int i=0;i<oldsize;i++){
      HashTableEntry<Key, Data>* p=oldtab[i];
      while(p){
	HashTableEntry<Key, Data>* next=p->next;
	int h=Hash(p->key)%hash_size;
	p->next=table[h];
	table[h]=p;
	p=next;
      }
    }
    if(oldtab)delete[] oldtab;
  }
  
  template <class Key, class Data>
  HashTable<Key, Data>::HashTable()
  {
    table=0;
    hash_size=0;
    nelems=0;
  }
  
  template <class Key, class Data>
  HashTable<Key, Data>::~HashTable()
  {
    if(table){
      for(unsigned int i=0;i<hash_size;i++){
	HashTableEntry<Key,Data>* p=table[i];
	while(p){
	  HashTableEntry<Key, Data>* prev=p;
	  p=p->next;
	  delete prev;
	}
      }
      delete[] table;
    }
  }
  
  /*
   * Inserts the key/data pair into the hash table
   */
  template <class Key, class Data>
  void HashTable<Key, Data>::insert(const Key& key, const Data& data)
  {
    nelems++;
    if(3*nelems >= 2*hash_size){
      /* Build a new table... */
      rehash(2*hash_size+1);
    }
    unsigned int h=Hash(key)%hash_size;
    HashTableEntry<Key, Data>* bin=table[h];
    HashTableEntry<Key, Data>* p=new HashTableEntry<Key, Data>(key, data, bin);
    table[h]=p;
  }
  
  /* Looks up key in the hashtable.  Returns false if not found.
   * Returns true, and places the data item in data if it is found.
   * If more than one of <em>key<\em> exists, it is undefined which it
   * will return.
   */
  template <class Key, class Data>
  bool HashTable<Key, Data>::lookup(const Key& key, Data& data)
  {
    if(table){
      unsigned int h=Hash(key)%hash_size;
      for(HashTableEntry<Key, Data>* p=table[h];p!=0;p=p->next){
	if(p->key == key){
	  data=p->data;
	  return true;
	}
      }
    }
    return false;
  }
  
  /*
   * Removes all items with <em>key<\em> from the hash table.
   * Returns the number actually removed
   */
  template <class Key, class Data>
  int HashTable<Key, Data>::remove(const Key& key)
  {
    int count=0;
    if(table){
      unsigned int h=Hash(key)%hash_size;
      HashTableEntry<Key, Data>* prev=0;
      HashTableEntry<Key, Data>* p=table[h];
      while(p){
	HashTableEntry<Key, Data>* next=p->next;
	if(p->key == key){
	  count++;
	  nelems--;
	  if(prev){
	    prev->next=p->next;
	  } else {
	    table[h]=p->next;
	  }
	  delete p;
	} else {
	  prev=p;
	}
	p=next;
      }
    }
    return count;
  }
  
  /*
   * Empties the hash table
   */
  template <class Key, class Data>
  void HashTable<Key, Data>::remove_all()
  {
    if(table){
      for(unsigned int i=0;i<hash_size;i++){
	HashTableEntry<Key, Data>* p=table[i];
	while(p){
	  HashTableEntry<Key, Data>* next=p->next;
	  delete p;
	  p=next;
	}
	table[i]=0;
      }
      nelems=0;
    }
  }
  
  /*
   * Returns how many items are stored in the hash table
   */
  template <class Key, class Data>
  int HashTable<Key, Data>::size() const
  {
    return nelems;
  }
  
  
} // end namespace rtrt

#endif
