
#ifndef SCI_Classlib_PQueue_h
#define SCI_Classlib_PQueue_h 1
#include <Tester/RigorousTest.h>

/* Provides a priority queue of a fixed maximum size N that holds integers
   in the range 1..N (inclusive) that are weighted with positive integers. */

class PQueue {


public:
  
    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);

  /* Constructs an empty priority queue of maximum size N. */
    
  PQueue (unsigned N);

  ~PQueue ();                               // Destructor
  PQueue (const PQueue &pq);                // Copy constructor
  PQueue &operator= (const PQueue &pq);     // Assignment operator

  /* Returns non-zero if the PQueue is empty, zero otherwise. */

  int isEmpty ();


  /* "data" is assumed to be an integer between 1 and N (inclusive), and
     "weight" is assumed to be a positive double.  There are three
     possibilities:

     (1) If "data" is not already in the PQueue, insert it and its weight
     and return 1.

     (2) If "data" is already in the PQueue with a weight that is larger
     than "weight", replace that weight with "weight" and return 1.

     (3) if "data" is there with weight < "weight" replace and return 1

     (4) Otherwise, simply return 0. */

  int replace (int data, double weight);


  /* Removes and returns the data element in the PQueue with the smallest
     associated weight.  If the PQueue is empty, returns 0. */

  int remove ();
  
  void print(); // prints the heap...

  // gets rid of element i...

  int nuke(int i) { if (replace(i,-0.5)) { return (remove()==i); } return 0;};

  int size() { return count; };
private:

  int N;
  int count;  // number of elements in queue

  int  *heap,*pos;
  double *weight;

  void  upheap ( int k );
  void  downheap ( int k );
};

#endif









