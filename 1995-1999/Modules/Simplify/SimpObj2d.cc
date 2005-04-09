
#include "SimpObj2d.h"
#include <iostream.h>

SimpObj2d::SimpObj2d()
{
  // do nothing in the constructor...
}

void SimpObj2d::Init(TriSurface *ts)
{
  mesh2d.Init(ts);
  // everything should be set now
  // just build all of the edges...

  edges.remove_all();

  for(int i=0;i<mesh2d.topology.size();i++) {
    int vs[3] = {mesh2d.topology[i].v[0],
		 mesh2d.topology[i].v[1],
		 mesh2d.topology[i].v[2]};
    AugEdge *tedge = new AugEdge(vs[0],vs[1]);
    if (edges.cond_insert(tedge)) {
      // that should be it...
      tedge = new AugEdge(vs[0],vs[2]); // create this
    } else {
      *tedge = AugEdge(vs[0],vs[2]); // just copy it over...
    }

    if (edges.cond_insert(tedge)) {
      // that should be it...
      tedge = new AugEdge(vs[1],vs[2]); // create this
    } else {
      *tedge = AugEdge(vs[1],vs[2]); // just copy it over...
    }
    
    if (!edges.cond_insert(tedge)) {
      // already there, so delete...
      delete tedge;
    }
  }

  // now that you have the hash table,
  // just build the edge list...

  BuildEdgeList();
  
  mpieces.resize(numProc);

  for(i=0;i<mpieces.size();i++) {
    mpieces[i] = scinew MeshEdgePiece2d();
    ((MeshEdgePiece2d*) mpieces[i])->Init(this);
  }
  
  origNodes = mesh2d.points.size();
  origElems = mesh2d.topology.size();

//  faceFlags_ = scinew BitArray1(origElems,1);
}

// 1st you have to figure out how to renumber things,
// then you have to do it and send the surface...

void SimpObj2d::DumpSurface(TriSurface *ts)
{
  recomputeIndex(origNodes,origElems);

  // now just run through all of the faces,
  // adding them as you go

  ts->points.resize(0);
  ts->elements.resize(0);

  // run though the elements...
#if 0
  int maxv=0;

  for(int i=0;i<mesh2d.topology.size();i++) {
    if (mesh2d.topology[i].flags) {
      int v0,v1,v2;
      v0 = vRemap_[mesh2d.topology[i].v[0]];
      v1 = vRemap_[mesh2d.topology[i].v[1]];
      v2 = vRemap_[mesh2d.topology[i].v[2]];
      ts->elements.add(new TSElement(v0,v1,v2));
      maxv = (v0>maxv)?v0:maxv;
      maxv = (v1>maxv)?v1:maxv;
      maxv = (v2>maxv)?v2:maxv;
    }
  }

  if (maxv >= (origNodes - vHiearchy_.size())) {
    cerr << maxv << " Bad vertex index!\n";
  }

  ts->points.resize(maxv+1);

  for(i=0;i<ts->points.size();i++) {
    ts->points[i] = mesh2d.points[vRemap_[i]];
  }
#else

  // just do a blind copy...

  for(int i=0;i<mesh2d.points.size();i++) {
    ts->points.add(mesh2d.points[i]);
  }

  // now do all of the "valid" elements...

  for(i=0;i<mesh2d.topology.size();i++) {
    if (mesh2d.topology[i].flags) {
      int v0,v1,v2;
      v0 = mesh2d.topology[i].v[0];
      v1 = mesh2d.topology[i].v[1];
      v2 = mesh2d.topology[i].v[2];
      ts->elements.add(new TSElement(v0,v1,v2));
    }
  }

#endif
}

void SimpObj2d::ComputeQuadrics()
{
  // treat boundary edges as line segments for now...

  quadrics_.resize(mesh2d.points.size());

  for(int i=0;i<quadrics_.size();i++)
    quadrics_[i].zero();

  for(i=0;i<mesh2d.topology.size();i++) {
    Quadric3d cQ;

    Vector v1 = mesh2d.points[mesh2d.topology[i].v[1]] -
      mesh2d.points[mesh2d.topology[i].v[0]];
    Vector v2 = mesh2d.points[mesh2d.topology[i].v[2]] -
      mesh2d.points[mesh2d.topology[i].v[0]];

    Vector nrm = Cross(v1,v2);
    nrm.normalize();

    cQ.DoSym(nrm.x(),nrm.y(),nrm.z(),
	     -Dot(nrm,mesh2d.points[mesh2d.topology[i].v[0]]));

    for(int j=0;j<3;j++) { // add this quadric to all verts...
      quadrics_[mesh2d.topology[i].v[j]] += cQ;

      if (mesh2d.topology[i].f[j] == -1) { // add boundary constraint...
	int i0 = mesh2d.topology[i].v[(j+1)%3];
	int i1 = mesh2d.topology[i].v[(j+2)%3];

	Point p0 = mesh2d.points[i0];
	Point p1 = mesh2d.points[i1];
			
	Quadric3d tQ;
	tQ.CreateLine(p0,p1);

	quadrics_[i0] += tQ;
	quadrics_[i1] += tQ;
      }

    }
  }

}

MeshEdgePiece2d::MeshEdgePiece2d()
{
  // just do nothing...
}

void MeshEdgePiece2d::Init(SimpObj2d *sobj)
{
  owner = (GenSimpObj*) sobj;
}

/*
 * Check for "pinching" - multiple bdry edges that aren't
 * connected (> 1 connected component)  They also have to
 * be connected to one of the "special" vertices...
 * if any face has a bdry (connected to either edge) and
 * the collapsing edge is not on the bdry, consider it illegal
 */

int MeshEdgePiece2d::SetEdge(int eid, int top_check)
{
  edge = eid;

  if (!owner->edgelst[eid]) // no edge here...
    return 0;

  A = owner->edgelst[eid]->n[0];
  B = owner->edgelst[eid]->n[1];

  SimpObj2d *O = (SimpObj2d*) owner;

  curQ = O->quadrics_[A] + O->quadrics_[B];
  pA = O->mesh2d.points[A];
  pB = O->mesh2d.points[B];

  Array1<int> fa;
  Array1<int> fb;

  elems.resize(0); // start over...

  O->mesh2d.GetFaceRing(A,fa);
  O->mesh2d.GetFaceRing(B,fb);

  toDie[0] = toDie[1] = -1;

  int nbdryA=0,nbdryB=0;

  // now run through these faces, doing the correct thing...
  int sharedI=0;
  for(int i=0;i<fa.size();i++) {
    int ci = fa[i];
    int isCface=0;
    for(int j=0;j<3;j++) {
      if (O->mesh2d.topology[ci].v[j] == B) { // shared face!
	if (sharedI >= 2) {
	  cerr << ci << " Woah - overflow!\n";
	  for(int kk=0;kk<fa.size();kk++) {
	    cerr << kk << " : Screwed Face: " << fa[kk] << endl;
	  }
	  for(kk=0;kk<2;kk++) {
	    cerr << "In Q: " << toDie[kk] << endl;
	  }
	} else {
	  toDie[sharedI++] = ci;
	  isCface = 1;
	}
      } else if (O->mesh2d.topology[ci].v[j] != A) {
	// check for "bdry" edges...
	if (O->mesh2d.topology[ci].f[j] == -1)
	  nbdryA++;
      }
    }

    if (!isCface) { // stick it in the other ring...
      Element2dAB ce;
      ce.Init(&(O->mesh2d),A,ci);
      elems.add(ce); // add it to the list...
    }
  }

  for(i=0;i<fb.size();i++) {
    int ci = fb[i];
    int isCface=0;
    for(int j=0;j<3;j++) {
      if (O->mesh2d.topology[ci].v[j] == A) { // shared face!
	// already stuck in the toDie ring by above...
	// toDie[sharedI++] = ci;
	isCface = 1;
      } else if (O->mesh2d.topology[ci].v[j] != B) {
	// check for "bdry" edges...
	if (O->mesh2d.topology[ci].f[j] == -1)
	  nbdryB++;
      }
    }
    if (!isCface) { // stick it in the other ring...
      Element2dAB ce;
      ce.Init(&(O->mesh2d),B,ci);
      elems.add(ce); // add it to the list...
    }
  }

  if (sharedI != 2) { // we have a boundary...
    if (sharedI != 1) {
      cerr << sharedI << " Woah - no neighbors????\n";
      return 0;
    } else {
      toDie[1] = -1; // nothing there...
    }
  } else {
    // check for "pinching" effect...
#if 0
    if (nbdryB || nbdryA) {
      cerr << "Pinch?\n";
      return 0;
    }
#endif
  }

  if (top_check) {
    // if any of the vertices that aren't on
    // the "toDie" faces are connected to A and B
    // we have a problem - run through A looking for B

    // first build the "opp" vertices...
    
    int oppV[2] = {-1,-1};

    for(i=0;i<3;i++) {
      if (toDie[0] != -1)
	if (O->mesh2d.topology[toDie[0]].v[i] != A &&
	    O->mesh2d.topology[toDie[0]].v[i] != B)
	  oppV[0] = O->mesh2d.topology[toDie[0]].v[i];

      if (toDie[1] != -1)
	if (O->mesh2d.topology[toDie[1]].v[i] != A &&
	    O->mesh2d.topology[toDie[1]].v[i] != B)
	  oppV[1] = O->mesh2d.topology[toDie[1]].v[i];
    }
    
    for(i=0;i<fa.size();i++) {
      int cf = fa[i];
      if ((cf != toDie[0]) && (cf != toDie[1])) { // don't look at these
	for(int j=0;j<3;j++) {
	  if ((O->mesh2d.topology[cf].v[j] != A) &&
	      (O->mesh2d.topology[cf].v[j] != oppV[0]) &&
	      (O->mesh2d.topology[cf].v[j] != oppV[1])) {
	    AugEdge teste(B,O->mesh2d.topology[cf].v[j]),*res;

	    if (owner->edges.lookup(&teste,res)) {
	      //cerr << "Failed topology!\n";
	      return 0; // failed topology!
	    }
	    
	  }
	}
      }
    }
  }

  if (elems.size())
    return 1;
  else
    return 0; // you must have some elements.
}

// this assumes is valid, has to patch
// up the surface and do all of that stuff...

int MeshEdgePiece2d::TryFlush(int do_checks)
{

  if (do_checks) { // check the topology...
  }

  // you have to save of the neccesary stuff...

  mVertex v;

  v.pt = owner->edgelst[edge]->p;
  v.s = A;
  v.u = B;
  
  SimpObj2d *O = (SimpObj2d*) owner;
  v.nv = O->mesh2d.points.size();

  // this has to be added to the mesh structure as well...
  O->mesh2d.nodeFace.add(elems[0].eid); // isn't destroyed...
  O->mesh2d.points.add(v.pt); 

  O->mesh2d.pt_flags.add(1); 

  mergeInfo mi;

  if (toDie[0] != -1)
    v.ei.add(toDie[0]);
  
  if (toDie[1] != -1)
    v.ei.add(toDie[1]);

  int oppV[2] = {-1,-1}; // opposite vertices

  for(int i=0;i<2;i++) {
    if (toDie[i] != -1) {
      int ci = toDie[i];
      for(int j=0;j<3;j++) {
	if (O->mesh2d.topology[ci].v[j] == A) {
	  mi.fA[i] = O->mesh2d.topology[ci].f[j];
	} else if (O->mesh2d.topology[ci].v[j] == B) {
	  mi.fB[i] = O->mesh2d.topology[ci].f[j];
	} else { // the other vertex - for new edges...
	  oppV[i] = O->mesh2d.topology[ci].v[j];
#if 0
	  // these get patched up later...
	  if (O->mesh2d.nodeFace[oppV[i]] == toDie[0] ||
	      O->mesh2d.nodeFace[oppV[i]] == toDie[1]) {
	    // replace it with something better...
	    O->mesh2d.nodeFace[oppV[i]] = elems[0].eid;
	  }
#endif
	}
      }
    }
  }

  // that is the neccesary neighborhood stencil...
  // for merge trees - up to 4 triangles...

  // do a pass to fix the connectivity info...

  for(i=0;i<2;i++) {
    if (toDie[i] != -1) {
      int ci=toDie[i];

      if (mi.fA[i] != -1) { // patch it...
	int didSwap=0;
	for(int j=0;j<3;j++) {
	  if (O->mesh2d.topology[mi.fA[i]].f[j] == ci) { // swap it...
	    O->mesh2d.topology[mi.fA[i]].f[j] = mi.fB[i];
	    didSwap++;
	  }
	}
	if (didSwap != 1) {
	  cerr << didSwap << " Couldn't fix mesh!\n";
	}
      }

      if (mi.fB[i] != -1) { // patch it...
	int didSwap=0;
	for(int j=0;j<3;j++) {
	  if (O->mesh2d.topology[mi.fB[i]].f[j] == ci) { // swap it...
	    O->mesh2d.topology[mi.fB[i]].f[j] = mi.fA[i];
	    didSwap++;
	  }
	}
	if (didSwap != 1) {
	  cerr << didSwap << " Couldn't fix mesh!\n";
	}

      }

    }
  }

  // now you need to fix vertex indeces, delete up to 4 edges
  // and recompute the rest, with 2 of the 4 removed ones being
  // reused...

  // first do the edges we know about...

  for(i=0;i<2;i++) {
    if (toDie[i] != -1) {
      int ci = toDie[i];

      if (!TryZapEdge(B,oppV[i])) {
	cerr << "Problem - couldn't zap integral edge!\n";
	return 0;
      }	

      if (!TryReplaceEdge(oppV[i],A,v.nv)) {
	cerr << "Problem - couldn't replace integral edge! B\n";
	return 0;
      }

    }
  }

  // the loop through all of the connected faces...
  // remove/replace edges and vertices...

  for(i=0;i < elems.size(); i++) {
    int ci = elems[i].eid;

    for(int j=0;j<3;j++) {
      if (O->mesh2d.topology[ci].v[j] == A) {
	O->mesh2d.topology[ci].v[j] = v.nv; // update the vertex...
	int ov0,ov1;
	ov0 = O->mesh2d.topology[ci].v[(j+1)%3];
	ov1 = O->mesh2d.topology[ci].v[(j+2)%3];
	
	if (!UnFlagEdge(ov0,ov1)) {
	  cerr << "Couldn't unflag a edge!\n";
	  return 0;
	}
	
	// now just try and replace the other edges...
	
	TryReplaceEdge(ov0,A,v.nv);
	TryReplaceEdge(ov1,A,v.nv);
      } else if (O->mesh2d.topology[ci].v[j] == B) {
	O->mesh2d.topology[ci].v[j] = v.nv; // update the vertex...
	int ov0,ov1;
	ov0 = O->mesh2d.topology[ci].v[(j+1)%3];
	ov1 = O->mesh2d.topology[ci].v[(j+2)%3];
	
	if (!UnFlagEdge(ov0,ov1)) {
	  cerr << "Couldn't unflag a edge B!\n";
	  return 0;
	}
	
	// now just try and replace the other edges...
	
	TryReplaceEdge(ov0,B,v.nv);
	TryReplaceEdge(ov1,B,v.nv);
      } else if (O->mesh2d.nodeFace[O->mesh2d.topology[ci].v[j]] == toDie[0] ||
		 O->mesh2d.nodeFace[O->mesh2d.topology[ci].v[j]] == toDie[1]) {
	// replace it with something better...
	O->mesh2d.nodeFace[O->mesh2d.topology[ci].v[j]] = ci;
      }
    }
  }

  // everything is done now, so nuke this edge
  // and update everything else...

  if (!TryZapEdge(A,B)) {
    cerr << "Couldn't kill myself!\n";
    return 0;
  }

  owner->vHiearchy_.add(v); // add this guy...
  O->mergeInfo_.add(mi);    // and this guy...

  // flag the vertices as "dead"

  O->mesh2d.pt_flags[A] = 0;
  O->mesh2d.pt_flags[B] = 0;

  // flag the faces as "dead"
  int nret=0;
  if (toDie[0] != -1) {
    O->mesh2d.topology[toDie[0]].flags = 0;//faceFlags_.clear(toDie[0]);
    nret++;
  }

  if (toDie[1] != -1) {
    O->mesh2d.topology[toDie[1]].flags = 0;//faceFlags_.clear(toDie[1]);
    nret++;
  }

  return nret;
}

// this is a dot product, so it is a COS...
// 45 degrees for now...

const double NORMAL_THRESH = 0.707;

int MeshEdgePiece2d::ValidPoint(Point &p)
{
  for(int i=0;i<elems.size();i++) {
    if (elems[i].IsOk(p) < NORMAL_THRESH)
      return 0;
  }
  return 1; // everybody passed!
}

void Element2dAB::Init(Mesh2d *msh, int node, int elem)
{
  Point A,B,C;
  eid = elem;

  MeshTop2d &me = msh->topology[elem];

  int ci=0;
  int have_valid=0;
  int verts[5] = {me.v[0],me.v[1],me.v[2],me.v[0],me.v[1]}; // cyclic...

  for(;ci<3;ci++) {
    if (verts[ci] == node) {
      have_valid=1;
      C = msh->points[node];
      A = msh->points[verts[ci+1]];
      B = msh->points[verts[ci+2]];
      ci = 4;
    }
  }

  if (!have_valid) {
    cerr << "Invalid Element - node isn't here!\n";
    return;
  }

  BA = B-A;
  No = Cross(BA,(C-A));
  No.normalize();

  Cv = Cross(B.vector(),A.vector());
#if 0
  Cx = BA.z()*A.y() - BA.y()*A.z();
  Cy = BA.x()*A.z() - BA.z()*A.x();
  Cz = BA.y()*A.x() - BA.x()*A.y();
#else
  
#endif

  pA = A;
  pB = B;

}

double Element2dAB::IsOk(Point &p)
{
#if 0
  Vector PA(p.z()*BA.y() - p.y()*BA.z() + Cx,
	    p.x()*BA.z() - p.z()*BA.x() + Cy,
	    p.y()*BA.x() - p.x()*BA.y() + Cz);

  return Dot(No,Cross(BA,PA).normal()); // cos theta...
#else
#if 0
  Vector rv = Cross(BA,p.vector()) - Cv;
#else
  Vector rv = Cross(BA,p-pA);
#endif
  if (rv.length2() < 1e-10)
    return 0; // it sucks...
  return Dot(No,rv.normal());
#endif
}

// quadrics can be initialized by somebody else...

QuadricError2d::QuadricError2d(SimpObj2d *obj)
:GenSimpFunct((GenSimpObj *) obj),quadrics_(0)
{
  
}

int QuadricError2d::Error(Point &p, double &, double &err, 
			  GenMeshEdgePiece *mp)
{
  MeshEdgePiece2d *MP = (MeshEdgePiece2d*) mp; // cast it down...

  if (MP->ValidPoint(p)) {
    err = MP->curQ.MulPt(p) + 1e-6; // add small delta...
    if (err < 0) {
      cerr << err << " : " << p.x() << " " << p.y() << " " << p.z() << " Negative Error!\n";
      for(int i=0;i<10;i++)
	cerr << MP->curQ.vals[i] << " ";
      cerr << endl;
    }
    return 1;
  }
  return 0;
}

int QuadricError2d::MinError(Point &p, double &v, double &err, 
			     GenMeshEdgePiece *mp)
{
  MeshEdgePiece2d *MP = (MeshEdgePiece2d*) mp; // cast it down...

  static int ntries=0;
  static int nfail=0;

  ntries++;

  if (MP->curQ.FindMin(p)) { // we found a minimum!
    if (MP->ValidPoint(p)) { // it is valid!
      err = MP->curQ.MulPt(p);
      return 1;
    }
  }
  int which; // we don't care...
  int rval = TrySimp(p,v,err,mp,which);

  if (!rval)
      nfail++;
#if 0  
  if ((ntries/100)*100 == ntries)
    cerr << ntries << " : " << nfail << endl;
#endif

  return rval;
}

void QuadricError2d::PostCollapseStuff(GenMeshEdgePiece *mp)
{
  MeshEdgePiece2d *MP = (MeshEdgePiece2d*) mp; // cast it down...

  SimpObj2d *O = (SimpObj2d*) owner;
  // add the quadrics...
  
  MP->curQ = O->quadrics_[MP->A] + O->quadrics_[MP->B];


  O->quadrics_.add(MP->curQ); // vertex was added, now add quadric...
}
