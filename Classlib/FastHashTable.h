
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


#ifndef SCI_Classlib_FastHashTable_h
#define SCI_Classlib_FastHashTable_h 1

#include <Classlib/Assert.h>
#include <Tester/RigorousTest.h>

#ifdef __GNUG__
#pragma interface
#endif

template<class Key> class FastHashTableIter;

template<class Key> class FastHashTable {
    Key** table;
    int hash_size;
    int nelems;
    
    void rehash(int newsize);
    friend class FastHashTableIter<Key>;
public:
    FastHashTable();
    FastHashTable(const FastHashTable<Key>&);
    ~FastHashTable();
    // Inserts the key into the FastHash table
    inline void insert(Key* key);

    // conditionaly inserts the key into the FastHash table
    // returns a 0 if it was already inserted, 1 if otherwise
    inline int cond_insert(Key* key); 

    // Looks up key in the FastHashtable.  Returns 0 if not found.
    // Returns 1, and places the data item in data if it is found.
    // If more than one of "key" exist, it is undefined which it
    // will return.
    inline int lookup(const Key* key, Key*& data);

    // Removes all items with key "key" from the FastHash table.
    // Returns the number actually removed
    inline int remove(const Key* key);

    // Empties the FastHash table
    void remove_all();

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


// conditionaly Insert an item into a FastHashTable
// 0 if it wasn't inserted, 1 if it was

template<class Key>
int FastHashTable<Key>::cond_insert(Key* k)
{
    nelems++;
    if(3*nelems >= 2*hash_size){
        /* Build a new table... */
        rehash(2*hash_size+1);
    }

    int h=k->hash%hash_size;
    if (table) {
      for(Key* p=table[h];p!=0;p=p->next){
        if ((*p) == (*k)) {
	  --nelems; // take it out...
          return 0;  // a match was found
	}
      }
    }
    Key* bin=table[h];
    k->next=bin;
    table[h]=k;
    return 1;  // it was inserted...
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
//		if(count > 3)
//		    cerr << "Count=" << count << endl;
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

#endif
