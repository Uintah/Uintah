
#include <Classlib/FastHashTable.h>
#include <Tester/RigorousTest.h>
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
