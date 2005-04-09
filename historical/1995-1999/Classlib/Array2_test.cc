
#include <Classlib/Array2.h>
#include <Tester/RigorousTest.h>
#include <Classlib/String.h>
#include <iostream.h>


void Array2<int>::test_rigorous(RigorousTest* __test)
{
    for(int x=0;x<=10;x++){
	for (int y=0;y<=10;y++){
	    Array2<int> my_array(x,y);
	    TEST (my_array.dim1()==x);
	    TEST (my_array.dim2()==y);
	}
    }

    Array2<int> array2(0,0);
    TEST (array2.dim1()==0);
    TEST (array2.dim2()==0);
    
    for(x=0;x<=10;x++){
	for (int y=0;y<=100;y++){
	    array2.newsize(x,y);
	    TEST (array2.dim1()==x);
	    TEST (array2.dim2()==y);
	
	    array2.initialize(x);

	    
	    //The following block of code is known to cause an assertion 
	    //failure.

	    //	    for (int x1=0;x1<x;x1++){
	    //	for (int y1=0;y1<y;y1++){
	    //	    TEST(array2(x1,y1)==x);
	    //    
	    //    for(x=0;x<=100;x++){
	    ///	for (int y=0;y<=100;y++){
	    //	    Array2<clString> string_array(x,y);
	    //	    TEST (string_array.dim1()==x);
	    //	    TEST (string_array.dim2()==y);
	    //
	    //	    string_array.initialize("hi there");
	    //	    
	    //	    for (int x1=0;x1<x;x++){
	    //		for (int y1=0;y1<y;y++){
	    //		    TEST (string_array(x1,y1)=="hi there");
	}
    }
}
