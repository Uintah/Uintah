
/*
 *  HashTable.cc: Implementation of HashTable type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/HashTable.h>
#include <Classlib/Assert.h>
#include <Malloc/Allocator.h>
#include <Tester/RigorousTest.h>



#ifdef __GNUG__
#pragma interface
#endif

// Create a hashtable
template<class Key, class Data>
HashTable<Key, Data>::HashTable()
{
    table=0;
    hash_size=0;
    nelems=0;
}

// Destroy a hashtable
template<class Key, class Data>
HashTable<Key, Data>::~HashTable()
{
    if(table){
	for(int i=0;i<hash_size;i++){
	    HashKey<Key,Data>* p=table[i];
	    while(p){
		HashKey<Key, Data>* prev=p;
		p=p->next;
		delete prev;
	    }
	}
	delete[] table;
    }
}

// Insert an item into a hashtable
template<class Key, class Data>
void HashTable<Key, Data>::insert(const Key& k, const Data& d)
{
    nelems++;
    if(3*nelems >= 2*hash_size){
	/* Build a new table... */
	rehash(2*hash_size+1);
    }
    int h=Hash(k, hash_size);
    HashKey<Key, Data>* bin=table[h];
    HashKey<Key, Data>* p=scinew HashKey<Key, Data>(k, d, bin);
    table[h]=p;
}

// Return an item from the hashtable
template<class Key, class Data>
int HashTable<Key, Data>::lookup(const Key& k, Data& d)
{
    if(table){
	int h=Hash(k, hash_size);
	for(HashKey<Key, Data>* p=table[h];p!=0;p=p->next){
	    if(p->key == k){
		d=p->data;
		return 1;
	    }
	}
    }
    return 0;
}

// Remove an item from the hashtable
template<class Key, class Data>
int HashTable<Key, Data>::remove(const Key& k)
{
    int count=0;
    if(table){
	int h=Hash(k, hash_size);
	HashKey<Key, Data>* prev=0;
	HashKey<Key, Data>* p=table[h];
	while(p){
	    HashKey<Key, Data>* next=p->next;
	    if(p->key == k){
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

// Remove all items from the hashtable
template<class Key, class Data>
void HashTable<Key, Data>::remove_all()
{
    if(table){
	for(int i=0;i<hash_size;i++){
	    HashKey<Key, Data>* p=table[i];
	    while(p){
		HashKey<Key, Data>* next=p->next;
		delete p;
		p=next;
	    }
	    table[i]=0;
	}
	nelems=0;
    }
}

// Internal function - rehash when growing the table
template<class Key, class Data>
void HashTable<Key, Data>::rehash(int newsize)
{
    HashKey<Key, Data>** oldtab=table;
    table=scinew HashKey<Key, Data>*[newsize];
    for(int ii=0;ii<newsize;ii++){
	table[ii]=0;
    }
    int oldsize=hash_size;
    hash_size=newsize;
    for(int i=0;i<oldsize;i++){
	HashKey<Key, Data>* p=oldtab[i];
	while(p){
	    HashKey<Key, Data>* next=p->next;
	    int h=Hash(p->key, hash_size);
	    p->next=table[h];
	    table[h]=p;
	    p=next;
	}
    }
    if(oldtab)delete[] oldtab;
}

// Return the number of elements in the hash table
template<class Key, class Data>
int HashTable<Key, Data>::size() const
{
    return nelems;
}

template<class Key, class Data>
HashKey<Key, Data>::HashKey()
: next(0)
{
}

template<class Key, class Data>
HashKey<Key, Data>::HashKey(const Key& k, const Data& d, HashKey<Key, Data>* n)
: key(k), data(d), next(n)
{
    
}

template<class Key, class Data>
HashTableIter<Key, Data>::HashTableIter(HashTable<Key, Data>* hash_table)
: hash_table(hash_table)
{
    current_key=0;
}

// Set the iterator to the beginning
template<class Key, class Data>
void HashTableIter<Key, Data>::first()
{
    if(hash_table->table){
	current_index=0;
	current_key=hash_table->table[0];
	while(current_key==0 && ++current_index < hash_table->hash_size){
	    current_key=hash_table->table[current_index];
	}
    } else {
	current_key=0;
    }
}

// Use in a for loop like this:
// for(iter.first();iter.ok();++iter)...
template<class Key, class Data>
int HashTableIter<Key, Data>::ok()
{
    return current_key!=0;
}

// Go to the next element (random order)
template<class Key, class Data>
void HashTableIter<Key, Data>::operator++()
{
    ASSERT(current_key!=0);
    current_key=current_key->next;
    while(current_key==0 && ++current_index < hash_table->hash_size){
	current_key=hash_table->table[current_index];
    }
}

// Get the key for the current element
template<class Key, class Data>
Key& HashTableIter<Key, Data>::get_key()
{
    ASSERT(current_key!=0);
    return current_key->key;
}

// Get the data for the current element
template<class Key, class Data>
Data& HashTableIter<Key, Data>::get_data()
{
    ASSERT(current_key!=0);
    return current_key->data;
}

// Copy constructor
template<class Key, class Data>
HashTable<Key, Data>::HashTable(const HashTable<Key, Data>& copy)
: hash_size(copy.hash_size), nelems(copy.nelems)
{
    table=scinew HashKey<Key, Data>*[hash_size];
    for(int i=0;i<hash_size;i++){
	if(copy.table[i])
	    table[i]=scinew HashKey<Key, Data>(*copy.table[i], 1); // Deep copy
	else
	    table[i]=0;
    }
}

// Copy CTOR for keys
template<class Key, class Data>
HashKey<Key, Data>::HashKey(const HashKey<Key, Data>& copy, int deep)
: key(copy.key), data(copy.data),
  next((deep && copy.next)?scinew HashKey<Key, Data>(*copy.next, 1):0)
{
}

#define HASHTABLE_VERSION 1

// Persistent IO for hash tables
template <class Key, class Data>
void Pio(Piostream& stream, HashTable<Key, Data>& t)
{
    stream.begin_class("HashTable", HASHTABLE_VERSION);
    Pio(stream, t.nelems);
    Pio(stream, t.hash_size);
    if(stream.reading())
	t.table=new HashKey<Key,Data>*[t.hash_size];
    for(int i=0;i<t.hash_size;i++){
	stream.begin_cheap_delim();
	int count;
	if(stream.writing()){
	    count=0;
	    for(HashKey<Key, Data>* p=t.table[i];p!=0;p=p->next)
		count++;
	} else {
	    t.table[i]=0;
	}
	Pio(stream, count);
	HashKey<Key, Data>* p;
	for(int ii=0;ii<count;ii++){
	    if(stream.reading()){
		HashKey<Key, Data>* tmp=scinew HashKey<Key, Data>;
		tmp->next=0;
		if(ii==0)
		    t.table[i]=tmp;
		else
		    p->next=tmp;
		p=tmp;
	    } else {
		if(ii==0){
		    p=t.table[i];
		}
		else
		    p=p->next;
	    }
	    Pio(stream, p->key);
	    Pio(stream, p->data);
	}
	stream.end_cheap_delim();
    }
    stream.end_class();
}
