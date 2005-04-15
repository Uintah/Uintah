#include <iostream.h>
#include <gfx/std.h>
#include <gfx/tools/Heap.h>


void Heap::swap(int i,int j)
{
    heap_node tmp = ref(i);

    ref(i) = ref(j);
    ref(j) = tmp;

    ref(i).obj->setHeapPos(i);
    ref(j).obj->setHeapPos(j);
}

void Heap::upheap(int i)
{
    if( i==0 ) return;

    if( ref(i).import > ref(parent(i)).import ) {
        swap(i,parent(i));
        upheap(parent(i));
    }
}

void Heap::downheap(int i)
{
    if (i>=size) return;        // perhaps just extracted the last

    int largest = i,
        l = left(i),
        r = right(i);

    if( l<size && ref(l).import > ref(largest).import ) largest = l;
    if( r<size && ref(r).import > ref(largest).import ) largest = r;

    if( largest != i ) {
        swap(i,largest);
        downheap(largest);
    }
}



void Heap::insert(Heapable *t,float v)
{
    if( size == maxLength() )
    {
	cerr << "NOTE: Growing heap from " << size << " to " << 2*size << endl;
	resize(2*size);
    }

    int i = size++;

    ref(i).obj = t;
    ref(i).import = v;

    ref(i).obj->setHeapPos(i);

    upheap(i);
}

void Heap::update(Heapable *t,float v)
{
    int i = t->getHeapPos();

    if( i >= size )
    {
	cerr << "WARNING: Attempting to update past end of heap!" << endl;
	return;
    }
    else if( i == NOT_IN_HEAP )
    {
	cerr << "WARNING: Attempting to update object not in heap!" << endl;
	return;
    }

    float old=ref(i).import;
    ref(i).import = v;

    if( v<old )
        downheap(i);
    else
        upheap(i);
}



heap_node *Heap::extract()
{
    if( size<1 ) return 0;

    swap(0,size-1);
    size--;

    downheap(0);

    ref(size).obj->notInHeap();

    return &ref(size);
}

heap_node *Heap::kill(int i)
{
    if( i>=size )
	cerr << "WARNING: Attempt to delete invalid heap node." << endl;

    swap(i, size-1);
    size--;
    ref(size).obj->notInHeap();

    if( ref(i).import < ref(size).import )
	downheap(i);
    else
	upheap(i);


    return &ref(size);
}
