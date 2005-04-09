
#include <Classlib/Array1.h>
#include <Tester/RigorousTest.h>
#include <Classlib/String.h>
#include <iostream.h>

void Array1<float>::test_rigorous(RigorousTest* __test)
{
    Array1<int> my_array;
    TEST(my_array.size()==0);

    for(int i=0;i<1000;i++){
	my_array.grow(1);
	TEST(my_array.size()==i+1);
    }

    for(i=0;i<1000;i++){
	my_array[i]=i*1000;
	TEST(my_array.size()==1000);
    }

    for(i=0;i<1000;i++){
	TEST(my_array[i]==i*1000);
	TEST(my_array.size()==1000);
    }

    Array1<int> my_array2(10);
    TEST(my_array2.size()==10);

    for(i=0;i<10;i++){
	my_array2[i] = i*10;
    }

    for(i=0;i<10;i++){
	TEST(my_array2[i]==i*10);
	TEST(my_array2.size()==10);
    }

    Array1<int> x_array;

    for(i=0;i<1000;i++){
	x_array.setsize(i);
	TEST(x_array.size()==i);
    }

    x_array.setsize(1000);
    TEST(x_array.size()==1000);

    x_array.initialize(11);
    

    for(i=0;i<1000;i++){
	TEST(x_array[i]==11);
    }

    ///
    Array1<clString> string_array;

    
    for(i=0;i<1000;i++){
	string_array.setsize(i);
	TEST(string_array.size()==i);
    }

    string_array.remove_all();
    string_array.grow(10000);

    for(i=0;i<2996;i+=3){
    	string_array[i] = "hi ";
    	string_array[i+1] = "there";
    	string_array[i+2] = "hi there";
     }

    for(i=0;i<2996;i+=3){
    	TEST (string_array[i]=="hi ");
    	TEST (string_array[i+1]=="there");
    	TEST (string_array[i+2]=="hi there");
    }

    string_array.remove_all();
    TEST(string_array.size()==0);


    Array1<clString> string_array2;

    for(i=0;i<1000;i++){
	string_array2.grow(1);
	TEST(string_array2.size()==(i+1));   
    }

    for(i=999;i>=0;i--){
	string_array2.remove(i);
	TEST(string_array2.size()==i);
    }
    
    string_array2.remove_all();
    TEST(string_array2.size()==0);

    for(i=0;i<100;i++){
    	string_array2.add("hi there");
    }

    for(i=0;i<100;i++){
    	TEST(string_array2[i]=="hi there");
    }
    

    string_array2.initialize("hello");
    
    for(i=0;i<100;i++){
	TEST(string_array2[i]=="hello");
    }

}
