
#ifndef PRIORITYQ_H
#define PRIORITYQ_H 1

#include <iostream>
#include <stdlib.h>

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
    cerr << "PriorityQ overflow!\n";
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
