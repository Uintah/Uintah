
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
#include <Tester/RigorousTest.h>

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


#include <iostream.h>
#include <Classlib/String.h>

struct intFastHashTable {
    intFastHashTable* next;
    int hash;
    int key;
    int data;
    intFastHashTable(int key, int data);
    int operator==(const intFastHashTable& t) const {
	return (key == t.key);
    }
};

intFastHashTable::intFastHashTable(int key, int data)
    :key(key),data(data)
{
    hash = key;
}






struct TestFastHashTable {
    TestFastHashTable* next;
    int hash;
    clString key;
    int data;
    TestFastHashTable(const clString& key, int data);
    int operator==(const TestFastHashTable& t) const {
	return (key == t.key);
    }
};

TestFastHashTable::TestFastHashTable(const clString& key, int data)
:key(key),data(data)
{
    hash = key.hash(1000);

}

void FastHashTable<int>::test_rigorous(RigorousTest* __test){
    
    
    FastHashTable<TestFastHashTable> table;
    table.insert(new TestFastHashTable("one",1));
    table.insert(new TestFastHashTable("two",2));
    table.insert(new TestFastHashTable("three",3));
    table.insert(new TestFastHashTable("four",4));
    table.insert(new TestFastHashTable("five",5));
    table.insert(new TestFastHashTable("six",6));
    table.insert(new TestFastHashTable("seven",7));
    table.insert(new TestFastHashTable("eight",8));
    table.insert(new TestFastHashTable("nine",9));
    table.insert(new TestFastHashTable("ten",10));
    
    TestFastHashTable* key=new TestFastHashTable("one", 2);
    TestFastHashTable* ret;
    TEST(table.lookup(key,ret));
    TEST(ret->data==1);
    TEST(ret->key=="one");
    delete key;

    key = new TestFastHashTable("two",0);
    TEST(table.lookup(key,ret));
    TEST(ret->data==2);
    TEST(ret->key=="two");
    delete key;

    key = new TestFastHashTable("three",0);
    TEST(table.lookup(key,ret));
    TEST(ret->data==3);
    TEST(ret->key=="three");
    delete key;

    key = new TestFastHashTable("four",0);
    TEST(table.lookup(key,ret));
    TEST(ret->data==4);
    TEST(ret->key=="four");
    delete key;

    key = new TestFastHashTable("five",5);
    TEST(table.lookup(key,ret));
    TEST(ret->data==5);
    TEST(ret->key=="five");
    delete key;

    key = new TestFastHashTable("six",6);
    TEST(table.lookup(key,ret));
    TEST(ret->data==6);
    TEST(ret->key=="six");
    delete key;

    key = new TestFastHashTable("seven",7);
    TEST(table.lookup(key,ret));
    TEST(ret->data==7);
    TEST(ret->key=="seven");
    delete key;

    key = new TestFastHashTable("eight",8);
    TEST(table.lookup(key,ret));
    TEST(ret->data==8);
    TEST(ret->key=="eight");
    delete key;

    key = new TestFastHashTable("nine",9);
    TEST(table.lookup(key,ret));
    TEST(ret->data==9);
    TEST(ret->key=="nine");
    delete key;

    key = new TestFastHashTable("ten",10);
    TEST(table.lookup(key,ret));
    TEST(ret->data==10);
    TEST(ret->key=="ten");
    delete key;

    FastHashTable<TestFastHashTable> table2=table;
    
    key = new TestFastHashTable("one",2);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==1);
    TEST(ret->key=="one");

    key = new TestFastHashTable("two",0);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==2);
    TEST(ret->key=="two");
    delete key;

    key = new TestFastHashTable("three",0);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==3);
    TEST(ret->key=="three");
    delete key;

    key = new TestFastHashTable("four",0);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==4);
    TEST(ret->key=="four");
    delete key;

    key = new TestFastHashTable("five",5);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==5);
    TEST(ret->key=="five");
    delete key;

    key = new TestFastHashTable("six",6);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==6);
    TEST(ret->key=="six");
    delete key;

    key = new TestFastHashTable("seven",7);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==7);
    TEST(ret->key=="seven");
    delete key;

    key = new TestFastHashTable("eight",8);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==8);
    TEST(ret->key=="eight");
    delete key;

    key = new TestFastHashTable("nine",9);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==9);
    TEST(ret->key=="nine");
    delete key;

    key = new TestFastHashTable("ten",10);
    TEST(table2.lookup(key,ret));
    TEST(ret->data==10);
    TEST(ret->key=="ten");
    delete key;

    TEST(table2.size()!=0);
    table2.remove_all();
    TEST(table2.size()==0);


    key = new TestFastHashTable("one",1);
    TEST(!table2.lookup(key,ret));
    
    key = new TestFastHashTable("two",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("three",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("four",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("five",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("six",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("seven",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("eight",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("nine",1);
    TEST(!table2.lookup(key,ret));

    key = new TestFastHashTable("ten",1);
    TEST(!table2.lookup(key,ret));


    intFastHashTable* intkey;
    intFastHashTable* intret;
    FastHashTable<intFastHashTable> int_table;
    TEST(int_table.size()==0);

    int y;
    int y2;
    int size=0;
    for(int x=1;x<=1000;x++){
	y=x+11;
	y2=x+2;
	int_table.insert(new intFastHashTable(x,y));
	int_table.insert(new intFastHashTable(x,y2));
	size+=2;
	TEST(int_table.size()==size);
    }
    
    int d;
    int c;
    for(x=1;x<=1000;x++){
	c=x+2;
	d=x+11;
	intkey = new intFastHashTable(x,1);
	TEST(int_table.lookup(intkey,intret));
	TEST(intret->key==x);
	TEST(intret->data==c||d);
	delete intkey;
    }


    TEST(int_table.size()==size);

    for(x=1;x<=1000;x++){
	intkey=new intFastHashTable(x,1);
	TEST(int_table.remove(intkey)==2);
	size-=2;
	TEST(int_table.size()==size);
	delete intkey;
    }

    for(x=1;x<1000;x++){
	intkey=new intFastHashTable(x,1);
	TEST(!int_table.lookup(intkey,intret));
	delete intkey;
    }
    
    TEST(int_table.size()==0);
    
    //Iterator Testing
    
    FastHashTable<intFastHashTable> table3;
    TEST(table3.size()==0);

    
    for(x=0;x<=1000;x++){
	intkey= new intFastHashTable(x,1);
	table3.insert(new intFastHashTable(x,x));
	TEST(table3.lookup(intkey,intret));
	TEST(intret->data==intret->key);
	delete intkey;
    }

    FastHashTableIter<intFastHashTable> iter(&table3);

    int count=0;
    for(iter.first();iter.ok();++iter){
	intkey=iter.get_key();
	table3.lookup(intkey,intret);
	TEST(intret->key==intret->data);
	++count;
    }

    TEST(count==table3.size());


    FastHashTableIter<TestFastHashTable> iter2(&table);
    count = 0;

    for(iter2.first();iter2.ok();++iter2){
	++count;
    }

    TEST(table.size()==count);
}























