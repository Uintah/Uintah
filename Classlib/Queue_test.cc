
#include <Classlib/Queue.h>
#include <Tester/RigorousTest.h>

void Queue<int>::test_rigorous(RigorousTest* __test)
{
    //Test the queue when holding ints
    Queue x;
    
    int cnt=0;

    int z;
    for (z=1;z<=1000;z++)
    {
	x.append(z);
	++cnt;
	TEST(x.length()==z);
	TEST(x.is_empty()==0);
    }
    
    

    for (z=1;z<=1000;z++)
    {
	TEST(x.is_empty()==0);
	TEST(x.pop()==z);
	--cnt;
	TEST(x.length()==cnt);
    }

    TEST(x.is_empty()==1);
    TEST(x.length()==0);

    
    //Test the queue when holding floats

    Queue<float> f;

    TEST(f.is_empty()==1);

    cnt=0;

    float fcnt;
    for(fcnt=1.1;fcnt<=1000.1;fcnt+=1.0)
    {
	f.append(fcnt);
	TEST(f.is_empty()==0);
	++cnt;
	TEST(f.length()==cnt);
    }

    for(fcnt=1.1;fcnt<=1000.1;fcnt+=1.0)
    {
	TEST(f.pop()==fcnt);
	--cnt;
	TEST(f.length()==cnt);
    }

    TEST(f.length()==0);
    TEST(f.is_empty()==1);


   

    //Test the queue with char* variables
    Queue<char*> cq;
    
    TEST(cq.length()==0);
    TEST(cq.is_empty()==1);
    
    cq.append("Hello");
    
    TEST(cq.length()==1);
    TEST(cq.is_empty()==0);

    cq.append("There");

    TEST(cq.length()==2);
    TEST(cq.is_empty()==0);

    TEST(cq.pop()=="Hello");
    
    TEST(cq.length()==1);
    TEST(cq.is_empty()==0);

    TEST(cq.pop()=="There");
     
    TEST(cq.length()==0);
    TEST(cq.is_empty()==1);


}
