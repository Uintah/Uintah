
/*
 *  FastHashTable.cc: Implementation of FastHashTable type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/FastHashTable.h>
#include <Classlib/Assert.h>
#include <Malloc/Allocator.h>

#ifdef __GNUG__
#pragma interface
#endif

// Create a FastHashTable
template<class Key>
FastHashTable<Key>::FastHashTable()
{
    table=0;
    hash_size=0;
    nelems=0;
}

// Copy constructor
template<class Key>
FastHashTable<Key>::FastHashTable(const FastHashTable<Key>& copy)
: hash_size(copy.hash_size), nelems(copy.nelems)
{
    table=scinew Key*[hash_size];
    for(int i=0;i<hash_size;i++){
	Key* p=copy.table[i];
	Key* prev=0;
	Key* first=0;
	while(p){
	    Key* n=new Key(*p);
	    if(prev)
		prev->next=n;
	    if(!first)
		first=n;
	    prev=n;
	    p=p->next;
	}
	if(prev)
	    prev->next=0;
	table[i]=first;
    }
}

// Destroy a FastHashTable
template<class Key>
FastHashTable<Key>::~FastHashTable()
{
    if(table){
	for(int i=0;i<hash_size;i++){
	    Key* p=table[i];
	    while(p){
		Key* next=p->next;
		delete p;
		p=next;
	    }
	}
	delete[] table;
    }
}

// Remove all items from the FastHashTable
template<class Key>
void FastHashTable<Key>::remove_all()
{
    if(table){
	for(int i=0;i<hash_size;i++){
	    Key* p=table[i];
	    while(p){
		Key* next=p->next;
		delete p;
		p=next;
	    }
	    table[i]=0;
	}
	nelems=0;
    }
}

// Internal function - rehash when growing the table
template<class Key>
void FastHashTable<Key>::rehash(int newsize)
{
    if(newsize<11)newsize=11;
    Key** oldtab=table;
    table=scinew Key*[newsize];
    for(int ii=0;ii<newsize;ii++){
	table[ii]=0;
    }
    int oldsize=hash_size;
    hash_size=newsize;
    for(int i=0;i<oldsize;i++){
	Key* p=oldtab[i];
	while(p){
	    Key* next=p->next;
	    int h=p->hash%hash_size;
	    p->next=table[h];
	    table[h]=p;
	    p=next;
	}
    }
    if(oldtab)delete[] oldtab;
}

// Return the number of elements in the hash table
template<class Key>
int FastHashTable<Key>::size() const
{
    return nelems;
}

template<class Key>
FastHashTableIter<Key>::FastHashTableIter(FastHashTable<Key>* hash_table)
: hash_table(hash_table)
{
    current_key=0;
}

// Set the iterator to the beginning
template<class Key>
void FastHashTableIter<Key>::first()
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
template<class Key>
int FastHashTableIter<Key>::ok()
{
    return current_key!=0;
}

// Go to the next element (random order)
template<class Key>
void FastHashTableIter<Key>::operator++()
{
    ASSERT(current_key!=0);
    current_key=current_key->next;
    while(current_key==0 && ++current_index < hash_table->hash_size){
	current_key=hash_table->table[current_index];
    }
}

// Get the key for the current element
template<class Key>
Key* FastHashTableIter<Key>::get_key()
{
    ASSERT(current_key!=0);
    return current_key;
}

