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


/*
   PQueue.cc

*/

#include <Core/Containers/PQueue.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Tester/RigorousTest.h>

#include <iostream>
using std::cerr;
#include <stdlib.h>

namespace SCIRun {

PQueue :: PQueue ( unsigned n )
{
  N = n;
  int i;
    
  heap = scinew int[ N + 1 ];
  weight = scinew double[ N + 1];
  pos = scinew int [N  + 1 ];

  for ( i = 0; i <= N; i++ )
      weight[i] = -1;
  
  count = 0;
}

PQueue :: ~PQueue ( )
{
  delete [] heap;
  delete [] weight;
  delete [] pos;
}

PQueue ::  PQueue ( const PQueue &pq )
{
  N = pq.N;
  count = pq.count;
  int i;
  
  heap = scinew int[ N + 1 ];
  weight = scinew double[ N + 1 ];
  pos = scinew int[ N + 1 ];
  
  for ( i = 0; i <= count; i++ )
      heap[i] = pq.heap[i];
  for ( i = 0; i <= N; i++ )
  {
       weight[i] = pq.weight[i];
       pos[i] = pq.pos[i];
  }
}

PQueue &  PQueue :: operator = ( const PQueue &pq )
{
  N = pq.N;
  count = pq.count;
  
  delete [] heap;
  delete [] weight;
  delete [] pos;
  int i;
  
  heap = scinew int[ N + 1 ];
  weight = scinew double[ N + 1 ];
  pos = scinew int[ N + 1 ];
  
  for ( i = 0; i <= count; i++ )
      heap[i] = pq.heap[i];
  for ( i = 0; i <= N; i++ )
  {
      weight[i] = pq.weight[i];
      pos[i] = pq.pos[i];
  }

  return *this;
}


int  PQueue :: isEmpty ( )
{
  return ( count == 0 );
}

void  PQueue :: upheap ( int k )
{
  double  v    = weight[ heap[k] ];
  int     data = heap[k];
  
  heap[0] = 0;
  weight[0] = -2;

  while ( weight[ heap[ k / 2 ] ] >= v )
  {
        heap[k] = heap[ k / 2 ];
        pos[ heap[k] ] = k;
        k = k / 2;
  }
  heap[k] = data;
  pos[ data ] = k;
}  


int  PQueue :: replace ( int d, double w )
{
  if ( weight[ d] != -1 && weight[d] <= w ) {
    //      return 0;
    weight[ d ] = w;
    downheap( pos[d] );
    return 1;
  }
  
  if ( weight[ d ] == -1 )
  {
     weight[ d] = w;
     count ++;
     heap[ count ] = d;
     upheap( count );
  }
  else
  {
     weight[d] = w;
     upheap( pos[d] );
  }

  return 1;
}

void  PQueue :: downheap ( int k )
{
  int j;

  double v = weight[ heap[k] ];
  int    data = heap[k];
  
  while ( k <= count / 2 )
  {
     j = k + k;
     if ( j < count && weight[ heap[j] ] > weight[ heap[ j + 1 ] ] )
        j ++ ;
     if ( v <= weight[ heap[j] ] )
        break;
     heap[k] = heap[j];
     pos[ heap[k] ] = k;
     k = j;
   }
   heap[k] = data;
   pos[data] = k;
}

int  PQueue :: remove ( )
{
  if ( count == 0 )
     return 0;
  
  int data = heap[1];

  heap[1] = heap[count];
  pos[ heap[1] ] = 1;
  count -- ;

  downheap( 1 );

  weight[data] = -1;  // it is off the queue...

  return data;
}

void PQueue::print() 
{
  int size=1;
  int cpos=1;
  while(cpos <= count) {
    int onthis=size;
    while(onthis-- && (cpos <= count)) {
      cerr << " " << weight[ heap[cpos] ] << " ";
      cpos++;
    }
    size  = size << 1;
    cerr << "\n";
  }
  
}

#if 0
// build a heap of 50 elements...
const int num_vals=16;

void main(int argc, char* argv)
{
  double val[num_vals+1];
  int i;

  PQueue the_queue(num_vals);

  for(i=1;i<=num_vals;i++) {
    val[i] = drand48()*100; // get a [0,1] value
    if (the_queue.replace(i,val[i])) {
      cerr << "worked...\n";
    } else {
      cerr << "huh?\n";
    }
  }

  // now take random ones and change the values

  the_queue.print();

  int k;
  for(k=0;k<200;k++) {
    int pos = drand48()*(num_vals-1)+1;
    double nval;
    if (drand48() > 0.5) { // make it bigger...
      nval = val[pos]+drand48()*20;
    } else { // make it smaller...
      nval = val[pos]*drand48();
    }
    the_queue.replace(pos,nval);

//    cerr << "Changed " << pos << " to " << nval << " from " << val[pos] << "\n";
    val[pos] = nval;
//    the_queue.print();
  }
  
  // try and delete an element
  the_queue.print();

  for (k=0;k<5;k++) {

    int pos = drand48()*(num_vals-1)+1;
    
    cerr << "Deleting " << pos << " -> " << val[pos] << "\n";
    
    the_queue.replace(pos,-0.5);
    if (the_queue.remove() != pos) {
      cerr << "Couldn't pop!\n";
    }
    the_queue.print();
  }

  // now remove them

  int done=1;
  while(done) {
    done = the_queue.remove();
    if (done)
      cerr << done << " -> " << val[done] << " \n";
  }

  

}
#endif

void PQueue::test_rigorous(RigorousTest* __test)
{
    int c;
    int size;
	int i;

    PQueue q(100);
    TEST(q.isEmpty());
    TEST(q.size()==0);
    size=q.size();
    for(i=1;i<=100;i++){
	q.replace(i,i);
	++size;
	TEST(q.size()==size);
	TEST(!q.isEmpty());
    }

    
    for(i=1;i<=100;i++){
	TEST(q.remove()==i);
	--size;
	TEST(q.size()==size);
    }
    
    c=100;
    for(i=1;i<=100;i++){
	q.replace(i,c);
	++size;
	--c;
	TEST(q.size()==size);
    }

    for(i=100;i>=1;i--){
	TEST(q.remove()==i);
	size--;
	TEST(q.size()==size);
    }

    for(i=1;i<=100;i++)
	q.replace(i,i);
    
    size=100;
    for(i=100;i>50;i--){
	q.nuke(i);
	size--;
	TEST(q.size()==size);
    }

    for(i=1;i<=100;i++){
	if(i>50)
	    TEST(q.remove()==0);
	else
	    TEST(q.remove()!=0);
    }
    
    PQueue a(1000);
    
    size=0;
    for(i=1;i<=1000;i++){
	a.replace(i,i);
	++size;
	TEST(a.size()==size);
    }
    
    PQueue b = a;
    
    for(i=1;i<=1000;i++)
	TEST(b.remove()==i);

    TEST(b.isEmpty());

    for(i=1;i<=1000;i++)
	TEST(b.replace(i,i));

    c=1000;
    for(i=1;i<=1000;i++){
	TEST(b.replace(i,c));
	--c;
    }

    for(i=1000;i>=1;i--)
	TEST(b.remove()==i);

    
    TEST(a.size()==1000);

    for(i=1;i<=1000;i+=2)
	a.nuke(i);
    
    c=0;
    for(i=1;i<=1000;i++){
	c+=2;
	if(i<=500)
	    TEST(a.remove()==c);
	else
	    TEST(a.remove()==0);

    }

}

} // End namespace SCIRun

