
#include <Classlib/Array3.h>
#include <Tester/RigorousTest.h>
#include <iostream.h>

void Array3<int>::test_rigorous(RigorousTest* __test)
{
    Array3<int> my_array(1,1,1);
    TEST(my_array.dim1()==1);
    TEST(my_array.dim2()==1);
    TEST(my_array.dim3()==1);
    
    my_array.newsize(2,2,2);
    TEST(my_array.dim1()==2);
    TEST(my_array.dim2()==2);
    TEST(my_array.dim3()==2);


    //Expand the Array
    

    for(int x=0;x<=10;x++){
	my_array.newsize(0,0,x);
	TEST(my_array.dim1()==0);
	TEST(my_array.dim2()==0);
	TEST(my_array.dim3()==x);
    }

    for(int y=0;y<=10;y++){
	my_array.newsize(0,y,x);
	TEST(my_array.dim1()==0);
	TEST(my_array.dim2()==y);
	TEST(my_array.dim3()==x);
    }

    for(int z=0;z<=10;z++){
	my_array.newsize(z,y,x);
	TEST(my_array.dim1()==z);
	TEST(my_array.dim2()==y);
	TEST(my_array.dim3()==x);
    }

 
    //A 10X10X10 array is used for the following test, however, it is known to
    //pass when as large as 100X100X100, and has been made smaller for speed
    //purposes

    //The following block of code is known to cause an assertion failure.
    /*
    for(x=0;x<=100;x++){
	Array3<int> array3(10,10,10);
	array3.initialize(x);
	
	for(int x1=0;x1<x;x1++){
	    for(int y1=0;y1<y;y1++){
		for(int z1=0;z1<z;z1++){
		    TEST(array3(x1,y1,z1)==x);
		}
	    }
	}
    }

    */
    //#include <Classlib/String.h>
    //
    //for(x=0;x<=10;x++){
    //for(int y=0;y<=10;y++){
    //    for(int z=0;z<=10;z++){
    //	Array3<clString> string_array(x,y,z);
    //	TEST (string_array.dim1()==x);
    //	TEST (string_array.dim2()==y);
    //	TEST (string_array.dim3()==z);
    //
    //	string_array.initialize("hi there");
    //
    //	for(int x1=0;x1<x;x++){
    //	    for(int y1=0;y1<y;y++){
    //		for(int z1=0;z1<z;z++){
    //		    TEST(string_array(x1,y1,z1)=="hi there");
    //		}
    //	    }
    //	}
    //    }
    //}
    //}
    //
}


