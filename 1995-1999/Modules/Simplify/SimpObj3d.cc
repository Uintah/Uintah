/*
 * 3D simplification code
 * Peter-Pike Sloan
 */

#include "SimpObj3d.h"

// these requires extra stuff in Mesh.h and Mesh.cc
// in particular, code to pull neighborhoods from a
// node
//
// it isn't totaly working, but I checked it in anyways
// in case somebody else want's to work on it.
//
// the Mesh.cc and Mesh.h in here are the versions I used
// for this stuff.  They also have lighter weight elements
// and nodes.  The handle stuff is really stupid for something
// as simple as a node...

#if 0

#include <iostream.h>

void SimpObj3d::Init(Mesh *ts)
{

}

// computes the quadrics for this object

void SimpObj3d::ComputeQuadrics()
{

}

MeshEdgePiece3d::MeshEdgePiece3d()
{
  // do nothing...
}

void MeshEdgePiece3d::Init(SimpObj3d *sobj)
{
  owner = (GenSimpObj*) sobj;
}

// this is a utility function
// it builds the 3 rings and does some topology
// checking

int MeshEdgePiece3d::TryAdd(Node *nd,               // node to add
			  int me, int other,      // indeces for edge
			  Array1< Element3dAB >& ring,
			  int do_Cring,          
			  int top_check)
{
  // 1st grab the element ring

  Array1<int> &nd_elems = scratch1;

  nd_elems.remove_all(); // clear it out...

  ++curGen;
  nd->GetElemsNT(nd_elems,mesh,me,
		 -1,-1,curGen,elemGen);

  for(int i=0;i<nd_elems.size();i++) {
    int me_match=-1,other_match=-1;

    Element *test = mesh->elems[nd_elems[i]];

    if (!test) {
      cerr << nd_elems[i] << " Is already deleted!\n";
    }

    int othernodes[4] = {-1,-1,-1,-1};
    int nother=0; 

    for(int j=0;j<4;j++) {
      if (test->n[j] == me) {
	me_match = j;
      } else if (test->n[j] == other) {
	other_match=j;
      } else {
	othernodes[nother++] = j;
      }
    }

#ifdef PETE_DEBUG
    if (me_match == -1) {
      cerr << "Woah - node isn't in own ring!\n";
      return 0;
    }
#endif

    if (nother == 3) { // it goes in the main ring...
      int myid = ring.size();
      ring.grow(1);
      ring[myid].Init(mesh,nd_elems[i],me_match);

      if (top_check) {
	// NEED CODE
      }
	
    } else if (nother == 2) {
#ifdef PETE_DEBUG
      if (other_match == -1) {
	cerr << "Nother = 2 and no match for other vertex!\n";
	return 0;
      }
#endif
      if (do_Cring) {
	int myid = Cring.size();
	Cring.grow(1);

	Cring[myid].Init(me_match,other_match,nd_elems[i],
			 othernodes[0],othernodes[1]);
      }
    }
#ifdef PETE_DEBUG
    else {
      cerr << nother << " Problem in element ring!\n";
      return 0;
    }
#endif

  }

  return 1;
}


// This is one of the work horse functions
// it sets up the local data if a given
// edge is to be collapsed

int MeshEdgePiece3d::SetEdge(int eid, int top_check)
{
  Node *nA = mesh->nodes[A];
  Node *nB = mesh->nodes[B];

  if (!nA || !nB) {
    cerr << "Woah - edge has null node!\n";
    return 0;
  }
  
  edge = eid;

  if (!owner->edgelst[eid]) // no edge - major problem...
    return 0;

  A = owner->edgelst[eid]->n[0];
  B = owner->edgelst[eid]->n[1];

  SimpObj3d *O = (SimpObj3d*) owner;

  curQ = O->quadrics_[A] + O->quadrics_[B];
  pA = mesh->nodes[A]->p;
  pB = mesh->nodes[B]->p;

  // clean up everything...

  Aring.remove_all();
  Bring.remove_all();
  Cring.remove_all();

  bdryFaces.remove_all(); // clear this out
  
  // now that you have this info, build the rings
  // if top_check is on, you might return something

  if (!TryAdd(nA,A,B,Aring,
	      1, // just build C ring the first time...
	      top_check)) {
    return 0;
  }

  if (!TryAdd(nB,B,A,Bring,
	      0, // just build C ring the first time...
	      top_check)) {
    return 0;
  }

  return 1; // aparently it's ok

}

// this updates everything and starts building
// the MergeTree as things are flushed down the Q

/*
 * All nodes much change the A/B reference to the new vertex
 * All "old" edges must be deleted and replaced by
 * edges using the new vertex.
 */
int MeshEdgePiece3d::TryFlush(int do_checks)
{

  // 1st compute the new stuff...

  mVertex v;

  v.pt = owner->edgelst[edge]->p;
  v.s = A;
  v.u = B;

  SimpObj3d *O = (SimpObj3d*) owner;

  v.nv = mesh->nodes.size(); // add one...

  mergeInfoV mi; // fill this in later...

  Array1<int> &asEdges = scratch1;

  curGen++;

  mesh->nodes[A]->GetNodesNT(asEdges,mesh,A,
			     -1,-1,
			     curGen,
			     nodeGen,
			     elemGen);

  for(int i=0;i<asEdges.size();i++) {
    if (asEdges[i] != B) {
      TryReplaceEdge(asEdges[i],A,v.nv); 
    }
  }

  // now do B...

  curGen++;

  mesh->nodes[B]->GetNodesNT(asEdges,mesh,B,
			     -1,-1,
			     curGen,
			     nodeGen,
			     elemGen);

  for(i=0;i<asEdges.size();i++) {
    if (asEdges[i] != A) {
      TryReplaceEdge(asEdges[i],B,v.nv); 
    }
  }

  // now patch up the topology of the C ring...

  const int nmap[] = {0,1,2,3,0,1,2,3}; // cyclic map...

  for(i=0;i<Cring.size();i++) {
    int fopA,fopB;

    Element *work = mesh->elems[Cring[i].eid];

#ifdef PETE_DEBUG
    if (!work) {
      cerr << "Woah - bad element in C ring!\n";
      return 0;
    }

#endif    

    fopA = work->faces[Cring[i].Ai];
    fopB = work->faces[Cring[i].Ai];

    if (fopA != -1) { // try and fix it!
      Element *opp = mesh->elems[fopA];
      for(int j=0;j<4;j++) {
	if (opp->faces[j] == Cring[i].eid) {
	  opp->faces[j] = fopB;
	}
      }
    }

    if (fopB != -1) { // try and fix it!
      Element *opp = mesh->elems[fopB];
      for(int j=0;j<4;j++) {
	if (opp->faces[j] == Cring[i].eid) {
	  opp->faces[j] = fopA;
	}
      }
    }
    
    delete mesh->elems[Cring[i].eid];  // free it up...
    mesh->elems[Cring[i].eid] = 0;

    // keep the history - this is the merge tree...

    v.ei.add(Cring[i].eid); // add this to the list...
    mi.eA.add(fopA);
    mi.eB.add(fopB);
  }
  
  for(i=0;i<Bring.size();i++) {
#ifdef PETE_DEBUG
    if (!mesh->elems[Bring[i].eid]) {
      cerr << "Empty element in the Bring!\n";
    } else {
      if (mesh->elems[Bring[i].eid]->n[Bring[i].nd] != B) {
	cerr << "Bad node reference in Bring!\n";
      }
    }
#endif
    mesh->elems[Bring[i].eid]->n[Bring[i].nd] = v.nv; // patch it up...

    // also fix up the elems array if neccesary...

    // just assign the nodes...
    mesh->nodes[nmap[Bring[i].nd+1]]->elem = Bring[i].eid;
    mesh->nodes[nmap[Bring[i].nd+2]]->elem = Bring[i].eid;
    mesh->nodes[nmap[Bring[i].nd+3]]->elem = Bring[i].eid;

  }

  for(i=0;i<Aring.size();i++) {
#ifdef PETE_DEBUG
    if (!mesh->elems[Aring[i].eid]) {
      cerr << "Empty element in the Aring!\n";
    } else {
      if (mesh->elems[Aring[i].eid]->n[Aring[i].nd] != A) {
	cerr << "Bad node reference in Aring!\n";
      }
    }
#endif
    mesh->elems[Aring[i].eid]->n[Aring[i].nd] = v.nv; // patch it up...

    mesh->nodes[nmap[Aring[i].nd+1]]->elem = Aring[i].eid;
    mesh->nodes[nmap[Aring[i].nd+2]]->elem = Aring[i].eid;
    mesh->nodes[nmap[Aring[i].nd+3]]->elem = Aring[i].eid;

  }
  return 1;
}

void Element3dAB::Init(Mesh* mesh, int node, int elem)
{
  eid = elem;

  Element *me = mesh->elems[eid];
  Point p1(mesh->nodes[me->n[0]]->p);
  Point p2(mesh->nodes[me->n[1]]->p);
  Point p3(mesh->nodes[me->n[2]]->p);
  Point p4(mesh->nodes[me->n[3]]->p);

  nd = node;

  double x1=p1.x();
  double y1=p1.y();
  double z1=p1.z();
  double x2=p2.x();
  double y2=p2.y();
  double z2=p2.z();
  double x3=p3.x();
  double y3=p3.y();
  double z3=p3.z();
  double x4=p4.x();
  double y4=p4.y();
  double z4=p4.z();
  double a1,a2,a3,a4;
  
  switch(nd) {
  case 0:
    a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    C = a1/6.0; // constant part is always just the node you are...
  
    X = (-y3*z4+y4*z3+y2*z4-y4*z2-y2*z3+y3*z2)/6.0;
    Y = (x3*z4+x4*z2-x4*z3+z3*x2-z4*x2-x3*z2)/6.0;
    Z = (-x3*y4+x4*y3-x4*y2+x2*y4-x2*y3+x3*y2)/6.0;
    break;
  case 1:
    a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
  
    C = a2/6.0; // constant part is always just the node you are...
  
    X = (y3*z4-y4*z3+y4*z1-y1*z4-y3*z1+y1*z3)/6.0;
    Y = (z4*x1-x4*z1-z3*x1-x3*z4+x4*z3+x3*z1)/6.0;
    Z = (-x1*y4+x4*y1+x1*y3+x3*y4-x4*y3-x3*y1)/6.0;
    break;
  case 2:
    a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
  
    C = a3/6.0; // constant part is always just the node you are...
  
    X = (y4*z2-y2*z4-y4*z1+y1*z4-y1*z2+y2*z1)/6.0;
    Y = (z4*x2-z4*x1+z2*x1+x4*z1-x2*z1-x4*z2)/6.0;
    Z = (-x2*y4+x1*y4-x1*y2-x4*y1+x2*y1+x4*y2)/6.0;
    break;
  case 3:
    a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
  
    C = a4/6.0; // constant part is always just the node you are...
  
    X = (y2*z3-y3*z2-y1*z3+y3*z1+y1*z2-y2*z1)/6.0;
    Y = (-x3*z1-z2*x1-z3*x2+x2*z1+x3*z2+z3*x1)/6.0;
    Z = (x3*y1+x1*y2+x2*y3-x2*y1-x3*y2-x1*y3)/6.0;
    break;
  }  
}

void Element3dC::Init(int nodeA, int nodeB, int elem, int opp0, int opp1)
{
  eid = elem;

  Ai = nodeA;
  Bi = nodeB;

  Oppi[0] = opp0;
  Oppi[1] = opp1;
}

int MeshEdgePiece3d::ValidPoint(Point &p)
{
  for(int i=0;i<Bring.size();i++) {
    if (Bring[i].IsOk(p) < 1e-9) // this is the volume...
      return 0;
  }

  for(i=0;i<Aring.size();i++) {
    if (Aring[i].IsOk(p) < 1e-9) // this is the volume...
      return 0;
  }

  return 1; // space wasn't inverted!
}

#endif
