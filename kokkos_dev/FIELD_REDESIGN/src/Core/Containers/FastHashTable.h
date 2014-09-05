
/*
 *  FastHashTable.h: Interface to a Faster HashTable type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#ifndef SCI_Containers_FastHashTable_h
#define SCI_Containers_FastHashTable_h 1

#include <SCICore/Util/Assert.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Tester {
class RigorousTest;
}
}

namespace SCICore {
namespace Containers {

using SCICore::Tester::RigorousTest;

template<class Key> class FastHashTableIter;

/**************************************

CLASS 
   FastHashTable
   
KEYWORDS
   FastHashTable

DESCRIPTION
   
 
   FastHashTable.h: Interface to Faster HashTable type
 
   Written by:
    Steven G. Parker
    Department of Computer Science
    University of Utah
    Feb. 1994
 
   Copyright (C) 1994 SCI Group
  
 
PATTERNS
   
WARNING
  
****************************************/

template<class Key> class FastHashTable {
    Key** table;
    int hash_size;
    int nelems;
    
    void rehash(int newsize);
    friend class FastHashTableIter<Key>;
public:
    //////////
    //Class constructor
    FastHashTable();
    
    //////////
    //Copy Constructor
    FastHashTable(const FastHashTable<Key>&);
    
    //////////
    //Class Destructor
    ~FastHashTable();

    //////////
    // Inserts the key into the FastHash table
    inline void insert(Key* key);

    //////////
    // Looks up key in the FastHashtable.  Returns 0 if not found.
    // Returns 1, and places the data item in data if it is found.
    // If more than one of "key" exist, it is undefined which it
    // will return.
    inline int lookup(const Key* key, Key*& data);


    //////////
    // Removes all items with key "key" from the FastHash table.
    // Returns the number actually removed
    inline int remove(const Key* key);


    //////////
    // Empties the FastHash table
    void remove_all();

    //////////
    // Returns how many items are stored in the FastHash table
    int size() const;

    //////////
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

};

// Use this class for walking through a FastHashtable
template<class Key> class FastHashTableIter {
    FastHashTable<Key>* hash_table;
    Key* current_key;
    int current_index;
public:
    // Build a FastHash table iterator for a specific FastHash table
    FastHashTableIter(FastHashTable<Key>*);
    
    // Reset the iterator to the first item
    void first();

    // Does the iterator point to a valid item?
    int ok();

    // Advance to the next item
    void operator++();

    // Get the key from the current item
    Key* get_key();
};

// Insert an item into a FastHashTable
template<class Key>
void FastHashTable<Key>::insert(Key* k)
{
    nelems++;
    if(3*nelems >= 2*hash_size){
	/* Build a new table... */
	rehash(2*hash_size+1);
    }
    int h=k->hash%hash_size;
    Key* bin=table[h];
    k->next=bin;
    table[h]=k;
}

// Return an item from the FastHashTable
template<class Key>
inline int FastHashTable<Key>::lookup(const Key* k, Key*& d)
{
    if(table){
	int h=k->hash%hash_size;
	int count=0;
	for(Key* p=table[h];p!=0;p=p->next){
	    count++;
	    if((*p) == (*k)){
		d=p;
		return 1;
	    }
	}
    }
    d=0;
    return 0;
}

// Remove an item from the FastHashTable
template<class Key>
inline int FastHashTable<Key>::remove(const Key* k)
{
    int count=0;
    if(table){
	int h=k->hash%hash_size;
	Key* prev=0;
	Key* p=table[h];
	while(p){
	    Key* next=p->next;
	    if((*p) == (*k)){
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

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.4  1999/09/08 02:26:45  sparker
// Various #include cleanups
//
// Revision 1.3  1999/08/19 23:52:58  sparker
// Removed extraneous includes of iostream.h  Fixed a few NotFinished.h
// problems.  May have broken KCC support.
//
// Revision 1.2  1999/08/17 06:38:35  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:12  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:42  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:30  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif
