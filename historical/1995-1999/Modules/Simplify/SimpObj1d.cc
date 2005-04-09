/*
 * This code is for simplifying 1d "curves" - streamlines...
 * Peter-Pike Sloan
 */

#include "SimpObj1d.h"
#include <iostream.h>

SimpObj1d::SimpObj1d()
{
  // do nothing???
}

void SimpObj1d::Init(Array1<Point> *pts,
		     Array1<float> *sv)
{
  if (pts->size() < 2) {
    cerr << "Empty Point list!\n";
    return;
  }
  segs_.resize(pts->size()-1);

  pts_ = *pts;

  startSize_ = pts_.size();

  // now you have to build all of the edges...

  edges.remove_all();

  edgelst.resize(segs_.size());

  for(int i=0;i<segs_.size();i++) {
    segs_[i].A = i;
    segs_[i].B = i+1;
    segs_[i].Ac = i-1;
    segs_[i].Bc = i+1;

    edgelst[i] = new AugEdge(i,i+1);

//    edges.cond_insert(new AugEdge(i,i+1));
  }
  segs_[i-1].Bc = -1; // no neighbor for this one...

  // you also have to compute the thread local stuff...

  mpieces.resize(numProc);
  
  for(i=0;i<mpieces.size();i++) {
    mpieces[i] = scinew MeshEdgePiece1d();
    ((MeshEdgePiece1d*) mpieces[i])->Init(this);
  }
  
}

void SimpObj1d::ComputeQuadrics(int planeEnds)
{
  // first compute the quadrics for all of the
  // intermediate pts...

  quadrics_.resize(pts_.size());

  // 1st zero out all of the quadrics...
  for(int i=0;i<pts_.size();i++) {
    quadrics_[i].zero();
  }

  Quadric3d cQ;
  // add this segment to both it's points...
  for(i=1;i<pts_.size();i++) {
    cQ.CreateLine(pts_[i-1],pts_[i]);
    quadrics_[i-1] += cQ;
    quadrics_[i] += cQ;
  }
#if 1
  if (planeEnds) {
    Vector v = pts_[1] - pts_[0];
    v.normalize(); 

    cQ.DoSym(v.x(),v.y(),v.z(),-Dot(v,pts_[0]));
    cQ.Scale(100); // don't move off the planes...
    quadrics_[0] += cQ;

    v = pts_[pts_.size()-1] - pts_[pts_.size()-2];
    v.normalize();

    cQ.DoSym(v.x(),v.y(),v.z(),-Dot(v,pts_[pts_.size()-1]));
    quadrics_[pts_.size()-1] += cQ;
  }
#endif
}

MeshEdgePiece1d::MeshEdgePiece1d()
{

}

void MeshEdgePiece1d::Init(SimpObj1d *o)
{
  owner = o;
}

int MeshEdgePiece1d::SetEdge(int eid, int top_check)
{
  edge = eid;
  A = owner->edgelst[eid]->n[0];
  B = owner->edgelst[eid]->n[1];

  SimpObj1d *O = (SimpObj1d*) owner;

  curQ = O->quadrics_[A] + O->quadrics_[B];
  pA = O->pts_[A];
  pB = O->pts_[B];

  return 1; // just return when you are done...
}

// just bold this guy, using the computed value...
// create a new vertex and update the records of
// the owner...

int MeshEdgePiece1d::TryFlush(int do_checks)
{	
  SimpObj1d *O = (SimpObj1d*) owner;

  // assume that this new vertex is just
  // the last vertex on the chain...

  mVertex v;

  v.pt = owner->edgelst[edge]->p;
  v.s = A;
  v.u = B;

#if 0
  cerr << "Doing a collpse :";
  cerr << O->pts_[A] << " : ";
  cerr << O->pts_[B] << " -> " << v.pt << endl;
#endif

  v.nv = O->pts_.size();

  O->pts_.add(v.pt); // add this point...

  // you don't even use the hash table for this one,
  // just use the edgelst itself...

  Seg1d seg = O->segs_[edge];

  // patch up all of the references...
  if (seg.Ac != -1) { // these are always sequential...
    O->segs_[seg.Ac].B = v.nv;
    O->segs_[seg.Ac].Bc = seg.Bc;
    O->oEdges_.add(seg.Ac); // it needs to be recomputed
  }
  if (seg.Bc != -1) {
    O->segs_[seg.Bc].A = v.nv;
    O->segs_[seg.Bc].Ac = seg.Ac;
    O->oEdges_.add(seg.Bc); // it needs to be recomputed
  }
  
  delete owner->edgelst[edge];
  owner->edgelst[edge] = 0; // zero out this guy...

  return 1;
}	

// well, this is a polyline, so everything is ok...

int MeshEdgePiece1d::ValidPoint(Point &p)
{
  return 1;
}

// quadrics can be initialized by somebody else...

QuadricError1d::QuadricError1d(SimpObj1d *obj)
:GenSimpFunct((GenSimpObj *) obj),quadrics_(0)
{
  
}

int QuadricError1d::Error(Point &p, double &, double &err, 
			  GenMeshEdgePiece *mp)
{
  MeshEdgePiece1d *MP = (MeshEdgePiece1d*) mp; // cast it down...

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

int QuadricError1d::MinError(Point &p, double &v, double &err, 
			     GenMeshEdgePiece *mp)
{
  MeshEdgePiece1d *MP = (MeshEdgePiece1d*) mp; // cast it down...

  if (MP->curQ.FindMin(p)) { // we found a minimum!
    if (MP->ValidPoint(p)) { // it is valid!
      err = MP->curQ.MulPt(p);
      return 1;
    }
  }
  int which; // we don't care...
  return TrySimp(p,v,err,mp,which);
}

void QuadricError1d::PostCollapseStuff(GenMeshEdgePiece *mp)
{
  MeshEdgePiece1d *MP = (MeshEdgePiece1d*) mp; // cast it down...

  SimpObj1d *O = (SimpObj1d*) owner;
  // add the quadrics...
  
  MP->curQ = O->quadrics_[MP->A] + O->quadrics_[MP->B];


  O->quadrics_.add(MP->curQ); // vertex was added, now add quadric...
}
