/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


#ifndef SCI_Containers_PQueue_h
#define SCI_Containers_PQueue_h 1

#include <Core/share/share.h>

namespace SCIRun {

class RigorousTest;

/* Provides a priority queue of a fixed maximum size N that holds integers
   in the range 1..N (inclusive) that are weighted with positive integers. */

class SCICORESHARE PQueue {


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

} // End namespace SCIRun

#endif
