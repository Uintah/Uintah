

#include <Classlib/HashTable.h>
#include <Tester/RigorousTest.h>
#include <iostream.h>
#include <Classlib/String.h>


void HashTable<char*, int>::test_rigorous(RigorousTest* __test)
{
    HashTable<clString,int> table;

    TEST(table.size()==0);
    table.insert("one",1);
    TEST(table.size()==1);
    table.insert("two",2);
    TEST(table.size()==2);
    table.insert("three",3);
    TEST(table.size()==3);
    table.insert("four",4);
    TEST(table.size()==4);
    table.insert("five",5);
    TEST(table.size()==5);
    table.insert("six",6);
    TEST(table.size()==6);
    table.insert("seven",7);
    TEST(table.size()==7);
    table.insert("eight",8);
    TEST(table.size()==8);
    table.insert("nine",9);
    TEST(table.size()==9);
    table.insert("ten",10);
    TEST(table.size()==10);

    
    
    int i = 0;
    TEST(table.lookup("one",i)&&i==1);
    TEST(table.lookup("two",i)&&i==2);
    TEST(table.lookup("three",i)&&i==3);
    TEST(table.lookup("four",i)&&i==4);
    TEST(table.lookup("five",i)&&i==5);
    TEST(table.lookup("six",i)&&i==6);
    TEST(table.lookup("seven",i)&&i==7);
    TEST(table.lookup("eight",i)&&i==8);
    TEST(table.lookup("nine",i)&&i==9);
    TEST(table.lookup("ten",i)&&i==10);


    /*
    HashTable<clString,int>table2 = table;  //Error occours here
    
    i = 0;
    TEST(table2.lookup("one",i)&&i==1);
    TEST(table2.lookup("two",i)&&i==2);
    TEST(table2.lookup("three",i)&&i==3);
    TEST(table2.lookup("four",i)&&i==4);
    TEST(table2.lookup("five",i)&&i==5);
    TEST(table2.lookup("six",i)&&i==6);
    TEST(table2.lookup("seven",i)&&i==7);
    TEST(table2.lookup("eight",i)&&i==8);
    TEST(table2.lookup("nine",i)&&i==9);
    TEST(table2.lookup("ten",i)&&i==10); 

    TEST(table2.size()!=0);
    table2.remove_all();
    
    
    TEST(!table2.lookup("one",i));
    TEST(!table2.lookup("two",i));
    TEST(!table2.lookup("three",i));
    TEST(!table2.lookup("four",i));
    TEST(!table2.lookup("five",i));
    TEST(!table2.lookup("six",i));
    TEST(!table2.lookup("seven",i));
    TEST(!table2.lookup("eight",i));
    TEST(!table2.lookup("nine",i));
    TEST(!table2.lookup("ten",i));

    TEST(table2.size()==0);
    
    

    
    HashTable<int,int> int_table;
    TEST(int_table.size()==0);
    
    int val1[1001];
    int size=0;
    for(int x=1;x<=1000;x++){
	val1[x]=x+11;
	int_table.insert(x,val1[x]);
	size=x;
	TEST(int_table.size()==size);
    }

    int val2[1001];
    for(x=1;x<=1000;x++){
	val2[x]=x+21;
	int_table.insert(x,val2[x]);
	size++;
	TEST(int_table.size()==size);
    }
	
   

    for(x=1;x<=1000;x++){
	int_table.lookup(x,i);
	TEST((i==val1[x])||(i==val2[x]));
    }

    TEST(int_table.size()==size);
    
    for(x=1;x<=1000;x++){
	TEST(int_table.remove(x)==2);
	size-=2;
	TEST(int_table.size()==size);
    }
    
    int v=0;
    
    for(x=1;x<1000;x++)
	TEST(!int_table.lookup(x,v));
    
   TEST(int_table.size()==0);

    
   //Iterator Testing

   HashTable<int,int> table3;
   TEST(table3.size()==0);

   int y=0;
   x=0;
   for(i=1;x<=1000;x++){
       table3.insert(x,x);
       TEST((table3.lookup(x,y)&&y==x));
   }

   HashTableIter<int,int> iter(&table3);

   int count=0;
   for(iter.first();iter.ok();++iter){
       TEST(iter.get_key()==iter.get_data());
       count++;
   }

   TEST(count==table3.size());


   HashTableIter<clString,int> iter2(&table);

   count=0;
   for (iter2.first();iter2.ok();++iter2)
   {
       count++;
   }
   
   TEST(table.size()==count);
   */       
}
