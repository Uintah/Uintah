/*
 * General Data structure for augmented edges
 * used for all simplification methods...
 * Peter-Pike Sloan
 */


#ifndef _AUGEDGE_H_
#define _AUGEDGE_H_ 1

#include <Geometry/Point.h>
#include <Malloc/Allocator.h>


// this stuff augments the data strucutres - so you can pull
// out boundaries, hash edges, stuff like that

// Edge + other stuff for simplifying meshes...

struct AugEdge{  
  int hash;             // for hash table...
  AugEdge *next;        // for hash table
  int n[2];             // nodes that contain this edge
  
  double weight;        // potential error for this collapse...
  
  int id; // where it is in the edge list...

  enum edge_info {
    // information about collapse - only makes sense if valid!
    node_0_collapse=1, // set if collapse is to node0
    node_1_collapse=(1<<1), // set if collapse is to node1
    node_mid_collapse=(1<<2), // set if collapse is to midle - or somewhere else
    node_mask=(1+2+4),         // mask for node stuff...
    
    // information about how this has been computed
    
    full_test=(1<<3),      // done the works on this puppy
    point_3_test=(1<<4),   // did no,n1,middle only - no optimization
    only_last=(1<<5),     // only did what previously worked
    // mask for test stuff...
    test_mask=((1<<3) + (1<<4) + (1<<5)),
    
    // this is the counter for when to only use last guy

    pulled_from_queue=(1<<6), // was on pqueue, needs to be redone...

    // this flag tells if the edge is valid or not

    valid_collapse=(1<<7),

    // edges that suck maybe should not be encoded as much...

    edge_suck_mask=((1<<8) +(1<<9) +(1<<10) +(1<<11)),

    // this is for non-manifold edges...
    // lookup in another hash table if you have too...

    bdry_edge=(1<<12)
  };

  int  flags;             // contains above flags and stuff...

  inline void IncEdgeSuck();

  // seriosly think about only having some info for a subset of the
  // edges...
  
  Point p;                // best collapse point
  double v;               // best scalar value - if neccesary...
  
  inline AugEdge():next(0),flags(0) {;};
  inline AugEdge(int n0, int n1, int f=-1);
  inline int operator==(const AugEdge& e) const;

  int f0,f1;

  // for fast allocator...
  void* operator new(size_t);
  void operator delete(void*, size_t);
};


// - inline functions

AugEdge::AugEdge(int n0, int n1, int f)
  :next(0),flags(0),f0(f),f1(-1) 
{
  n[0] = n0;
  n[1] = n1;
  if (n1>n0) { n[0] = n1;n[1] = n0;}
  hash = (n[0]*7+5)^(n[1]*5+3);
}

int AugEdge::operator==(const AugEdge& e) const 
{
  return (n[0]==e.n[0] && n[1] == e.n[1]);
}

void AugEdge::IncEdgeSuck() { 
  if ((flags&edge_suck_mask) != edge_suck_mask) {
    int val = ((flags&edge_suck_mask)>>8)+1; // bump it over...
    
    flags = (flags&(~edge_suck_mask))|(val<<8);
  }
}

#endif
