#include "SimpObjNM2d.h"
#include <iostream.h>

//
// This code isn't finished either
// it is supposed to do the non-manifold
// simplification stuff.

SimpObj2dNM::SimpObj2dNM()
{

}

void SimpObj2dNM::AddEdge(int v0, int v1, int f)
{
  AugEdge *tedge = new AugEdge(vs0,vs1,f);
  
  if (!edges.cond_insert(tedge)) {
    AugEdge *res;
    // look up this guy...

    edges.lookup(tedge,res);
    delete tedge; // free the memmory...

    // see if it is just becoming non-manifold...
    if ((res->f1 != -1) && !(res->flags&AugEdge::bdry_edge)) {
      res->flags |= AugEdge::bdry_edge;
      
      res->f0 = -1; // another thing to signify this...

      // now you have to set these points...
      if (!isPtNM_[v0]) {
	isPtNM_[v0] = nmPoint_.size();
	AugNMPoint me;
	me.pid = v0;
	me.conPts.add(v1); // do faces later...
	nmPoint_.add(me);
      } else { // already in here, make sure v1 is in conPts...
	int found=0;
	for(int i = 0;i<nmPoint_[isPtNM_[v0]].conPts.size();i++) {
	  if (nmPoint_[isPtNM_[v0]].conPts[i] == v1)
	    found =1;
	}

	if (!found) {
	  nmPoint_[isPtNM_[v0]].conPts[i].add(v1);
	}
      }
      if (!isPtNM_[v1]) {
	isPtNM_[v1] = nmPoint_.size();
	AugNMPoint me;
	me.pid = v1;
	me.conPts.add(v0); // do faces later...
	nmPoint_.add(me);
      } else { // already in here, make sure v1 is in conPts...
	int found=0;
	for(int i = 0;i<nmPoint_[isPtNM_[v1]].conPts.size();i++) {
	  if (nmPoint_[isPtNM_[v1]].conPts[i] == v0)
	    found =1;
	}

	if (!found) {
	  nmPoint_[isPtNM_[v1]].conPts[i].add(v0);
	}
      }
      return 1;
    } else {
      res->f1 = f; // we are the second face...
    }
  }
  return 0; // maybe later for this lucky bloke...
}

int FindSListIntersect(Array1<int> &la, Array1<int> &lb, int me)
{
  int ca=0,cb=0;

  int mtch=-1;
  while((ca < la.size()) && (cb < lb.size())) {
    if ((la[ca] == lb[cb]) && (la[ca] != me)) {
      if (mtch != -1) {
	mtch = la[ca];
      } else {
	// this is a third match
	return -2;
      }

    if (la[ca] < lb[cb]) {
      ca++; // go to next index
    } else {
      cb++; // b was smaller - so increment it...
    }
  }
  
  return -1;
}

void SimpObj2dNM::Init(TriSurface *ts)
{
  mesh2d.Init(ts,0); // no connectivity...

  edges.remove_all(); // blow away all of the edges...

  isPtNM_.resize(mesh2d.points.size());
  isPtNM_.initialize(-1); // clear'em all out...

  nmPoint_.resize(0);

  for(int i=0;i<mesh2d.topology.size();i++) {
    int vs[3] = {mesh2d.topology[i].v[0],
		 mesh2d.topology[i].v[1],
		 mesh2d.topology[i].v[2]};
    AddEdge(vs[0],vs[1],i);
    AddEdge(vs[0],vs[2],i);
    AddEdge(vs[1],vs[2],i);

    // also build up node neighbors...
    for(int j=0;j<3;j++) {
      mesh2d.nodeFaces[mesh2d.topology[i].v[j]].add(i);
      mesh2d.nodeFace[mesh2d.topology[i].v[j]] = i; // good for normal nodes...
    }
  }

  // one more pass - build the "face" neighbors
  int vremap[5] = {0,1,2,0,1};
  for(i=0;i<mesh2d.topology.size();i++) {
    for(int j=0;j<3;j++) {
      int nbr;
      // see if it has 1 intersection - manifold...
      nbr = FindSListIntersect(mesh2d.nodeFaces[topology[i].v[vremap[j+1]]],
			       mesh2d.nodeFaces[topology[i].v[vremap[j+2]]],
			       i);
      mesh2d.topology[i].f[j] = nbr; // -2 if it is NM
    }
  }

  // now you have to go through all of the
  // non-manifold points...

  for(i=0;i<nmPoint_.size();i++) {
    int me = nmPoint_[i].pid;
    Array1< int > ffaces; // final faces...

    Array1< int > wfaces = mesh2d.nodeFaces[me];
    while(wfaces.size()) {
      ffaces.add(wfaces[0]);
      mesh2d.nodeFace[me] = wfaces[0];
      Array1< int > ring;
      Array1< int > wc;

      mesh2d.GetFaceRing(me,ring);
     
      for(int j=0;j<wfaces.size();j++) {
	int good=1;
	for(int k=0;k<ring.size();k++) {
	  if (ring[k] == wfaces[j]) {
	    good=0; // this guy got pulled...
	    k = ring.size() + 1;
	  }
	}

	if (good)
	  wc.add(wfaces[j]);
      }

      wfaces = wc; // what is left...
    }
    nmPoint_[i].conFcs = ffaces;
    if (ffaces.size() <= 2) {
      cerr << "Woah - not enough faces for NM point!\n";
    }
  }
}

MeshEdgePieceNM2d::MeshEdgePieceNM2d()
{

}

int MeshEdgePieceNM2d::Init(SimpObj2d* s)
{
  owner = (GenSimpObj*) s;
  pieces.resize(0);
}

int MeshEdgePieceNM2d::SetEdge(int eid, int top_check)
{
  edge = eid;

  pieces.resize(0);

  if (!owner->edgelst[eid]) // no edge here...
    return 0;

  A = owner->edgelst[eid]->n[0];
  B = owner->edgelst[eid]->n[1];

  SimpObjNM2d *O = (SimpObj2d*) owner;
  
  // now look at these pts to see what we need
  // to do...

  flags |= (isPtNM_[A] == -1)?0:A_NONMANIFOLD;
  flags |= (isPtNM_[B] == -1)?0:B_NONMANIFOLD;

  if (isPtNM_[A] == -1 &&
      isPtNM_[B] == -1) { // manifold edge...
    flags = BOTH_MANIFOLD;
    int rval = pieces[0].SetEdge(eid,top_check);
    return rval;
  }

  if (edgelst[eid].flags&AugEdge::bdry_edge) { // both NM
    // see if they are junction points
    int Ai = nmPoint_[A];
    int Bi = nmPoint_[B];

    /* if ( */
  } else { // only 1 node is NM

  }
}
