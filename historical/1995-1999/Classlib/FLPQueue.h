
/*
 *  FLPQueue.h: A fixed length priority queue
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   September 1998
 *
 *  Copyright (C) 1998 SCI Group
 */


#ifndef SCI_Classlib_FLPQueue_h
#define SCI_Classlib_FLPQueue_h 1

#include <Classlib/Persistent.h>

class Piostream;

template<class T> class FLPQueue;

template<class T> class FLPQueueNode {
  T item;
  double w;
  FLPQueueNode* next;
  FLPQueueNode* prev;
  inline FLPQueueNode(const T& item, FLPQueueNode* next, FLPQueueNode* prev, double w) : item(item), next(next), prev(prev), w(w){}
  inline FLPQueueNode(const T& item, double w): item(item), next(0), prev(0), w(w){}
  friend class FLPQueue<T>;
  friend void Pio(Piostream&, FLPQueueNode<T>&);
  friend void Pio(Piostream&, FLPQueue<T>&);
};

template<class T> class FLPQueue {
    FLPQueueNode<T>* head;
    FLPQueueNode<T>* tail;
    int _length;
    int _size;
public:
    FLPQueue(int size);
    ~FLPQueue();
    T pop(double &w);
    int is_empty();
    int length();
    void sanity_check();
    // insert retuns true if something was bumped from the q, and false if not
    int insert(const T& item, double weight, int& caused_bump, T& bumped);
    void update_weight(const T&, double weight);
    friend void Pio(Piostream&, FLPQueue<T>&);
};

#endif
