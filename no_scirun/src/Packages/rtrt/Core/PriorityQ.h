/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef PRIORITYQ_H
#define PRIORITYQ_H 1

#include <cstdio>
#include <cstdlib>

//namespace rtrt { 

template<class T, int heapsize>
class PriorityQ {
    T* heap;
    char* dataspace[sizeof(T)*heapsize];
    int nheap;
    void heapify(int n, int i);
    void overflow();
public:
    PriorityQ();
    ~PriorityQ();
    void insert(const T&);
    T pop();
    inline int length() const {
	return nheap;
    }
};

template<class T, int heapsize>
PriorityQ<T, heapsize>::PriorityQ()
{
    nheap=0;
    heap=(T*)dataspace;
}

template<class T, int heapsize>
PriorityQ<T, heapsize>::~PriorityQ()
{
}

template<class T, int heapsize>
void PriorityQ<T, heapsize>::heapify(int n, int i)
{
    int l=2*i+1;
    int r=l+1;
    int largest=i;
    if(l<n && heap[l].pri() > heap[i].pri())
	largest=l;
    if(r<n && heap[r].pri() > heap[largest].pri())
	largest=r;
    if(largest != i){
	T tmp=heap[i];
	heap[i]=heap[largest];
	heap[largest]=tmp;
	heapify(n, largest);
    }
}

template<class T, int heapsize>
void PriorityQ<T, heapsize>::overflow()
{
  fprintf(stderr, "PriorityQ overflow!\n");
  fflush(stderr);
  abort();
}

template<class T, int heapsize>
void PriorityQ<T, heapsize>::insert(const T& newnode)
{
    nheap++;
    if(nheap >= heapsize)
	overflow();
    int i=nheap-1;
    int parent=(i+1)/2-1;
    while(i && heap[parent].pri() < newnode.pri()){
	heap[i]=heap[parent];
	i=parent;
	parent=(i+1)/2-1;
    }
    heap[i]=newnode;
}

template<class T, int heapsize>
T PriorityQ<T, heapsize>::pop()
{
    T val=heap[0];
    nheap--;
    heap[0]=heap[nheap];
    heapify(nheap, 0);
    return val;
}

//} // end namespace rtrt

#endif
