#ifndef GFXTOOLS_HEAP_INCLUDED // -*- C++ -*-
#define GFXTOOLS_HEAP_INCLUDED

#include <gfx/tools/Array.h>

#define NOT_IN_HEAP -47

//
//
// This file extracted from Terra
//
//

class Heapable
{
private:
    int token;

public:
    Heapable() { notInHeap(); }

    inline int isInHeap() { return token!=NOT_IN_HEAP; }
    inline void notInHeap() { token = NOT_IN_HEAP; }
    inline int getHeapPos() { return token; }
    inline void setHeapPos(int t) { token=t; }
};


class heap_node {
public:
    float import;
    Heapable *obj;

    heap_node() { obj=NULL; import=0.0; }
    heap_node(Heapable *t, float i=0.0) { obj=t; import=i; }
    heap_node(const heap_node& h) { import=h.import; obj=h.obj; }
};



class Heap : public array<heap_node> {

    //
    // The actual size of the heap.  array::length()
    // simply returns the amount of allocated space
    int size;

    void swap(int i, int j);

    int parent(int i) { return (i-1)/2; }
    int left(int i) { return 2*i+1; }
    int right(int i) { return 2*i+2; }

    void upheap(int i);
    void downheap(int i);

public:

    Heap() { size=0; }
    Heap(int s) : array<heap_node>(s) { size=0; }


    void insert(Heapable *, float);
    void update(Heapable *, float);

    heap_node *extract();
    heap_node *top() { return size<1 ? (heap_node *)NULL : &ref(0); }
    heap_node *kill(int i);
};



// GFXTOOLS_HEAP_INCLUDED
#endif
