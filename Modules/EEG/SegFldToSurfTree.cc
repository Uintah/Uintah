/*
 *  SegFldToSurfTree.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Dataflow/Module.h>
#include <Datatypes/SegFldPort.h>
#include <Datatypes/SegFld.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/SurfTree.h>
#include <Malloc/Allocator.h>

class SegFldToSurfTree : public Module {
    SegFldIPort* infield;
    SurfaceOPort* outsurf;
public:
    SegFldToSurfTree(const clString& id);
    SegFldToSurfTree(const SegFldToSurfTree&, int deep);
    virtual ~SegFldToSurfTree();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SegFldToSurfTree(const clString& id)
{
    return new SegFldToSurfTree(id);
}
}

SegFldToSurfTree::SegFldToSurfTree(const clString& id)
: Module("SegFldToSurfTree", id, Filter)
{
    infield=new SegFldIPort(this, "Field", SegFldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    outsurf=new SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(outsurf);
}

SegFldToSurfTree::SegFldToSurfTree(const SegFldToSurfTree& copy, int deep)
: Module(copy, deep)
{
}

SegFldToSurfTree::~SegFldToSurfTree()
{
}

Module* SegFldToSurfTree::clone(int deep)
{
    return new SegFldToSurfTree(*this, deep);
}

// each component will correspond to one surface -- the outer surface
// which bounds it.
// starting from air(0), flood fill to find all components which touch air.
// as each component is found, we mark it as visited, and push the bounding
// voxel location and direction onto 
// from all of those 
void fldToTree(SegFld &field, SurfTree &surf) {
    Array1<Array1<int> > touches;
    touches.resize(field.comps.size());
    Array1<int> is(3), js(3), ks(3);
    is[0]=0; is[1]=1; is[2]=1;
    int i,j,k,ii,jj,kk;

    // find out which components "touch" each other
    for (i=1; i<field.nx; i++, is[0]++, is[1]++, is[2]++) {
	js[0]=1; js[1]=0; js[2]=1;
	for (j=1; j<field.ny; j++, js[0]++, js[1]++, js[2]++) {
	    ks[0]=1; ks[1]=1; ks[2]=0;
	    for (k=1; k<field.nz; k++, ks[0]++, ks[1]++, ks[2]++) {
//		tripleInt idx(i,j,k);
		int comp=field.grid(i,j,k);
		for (int nbr=0; nbr<3; nbr++) {
		    ii=is[nbr]; jj=js[nbr]; kk=ks[nbr];
		    int bcomp=field.grid(ii,jj,kk);
		    if (bcomp != comp) {
			int found=0;
			for (int f=0; f<touches[comp].size() && !found; f++)
			    if (touches[comp][f] == bcomp) found=1;
			if (!found) {
			    touches[comp].add(bcomp);
			    touches[bcomp].add(comp);
			}
		    }
		}
	    }	
	}
	
    }
    cerr << "\n";
    for (i=0; i<touches.size(); i++) {
	if (touches[i].size() != 0) {
	    cerr << "Component "<<i<<" touches: ";
	    for (int j=0; j<touches[i].size(); j++)
		cerr << touches[i][j]<<" ";
	    cerr << "\n";
	}
    }

    Array1<int> visited(field.comps.size());
    Array1<int> queued(field.comps.size());
    surf.surfI.resize(field.comps.size());
    for (i=0; i<field.comps.size(); i++) surf.surfI[i].outer=-1;
    visited.initialize(0);
    queued.initialize(0);
    for (i=0; i<touches.size(); i++) if (touches[i].size() == 0) visited[i]=1;
    Queue<int> q;
    Queue<int> kids;

    int air=field.grid(0,0,0);
    // visit 0 -- push all of its nbrs
    visited[air]=1;
    queued[air]=1;


    // we'll enqueue each component index, along with which
    // component is outside of it.

    for (i=0; i<touches[air].size(); i++) {
	q.append(air);
	q.append(touches[air][i]);
	visited[touches[air][i]]=queued[touches[air][i]]=1;
    }

    // go in one "level" at a time -- push everyone you touch into the
    // "kids" queue.  when there isn't anyone else, move everyone from
    // that queue into the main one.  continue till there's no one left.

    while (!q.is_empty()) {
	while (!q.is_empty()) {
	    int outer=q.pop();
	    int inner=q.pop();
	    cerr << "outer="<<outer<<"   inner="<<inner<<"\n";
	    surf.surfI[inner].outer=outer;
	    for (i=0; i<touches[inner].size(); i++) {
		// if another component it touches has been visited
		int nbr=touches[inner][i];
		if (!visited[nbr]) {
		    if (queued[nbr]) {	// noone should have queued -- must be
			                // crack-connected
			cerr << "why am i here??\n";
			q.append(outer);
			q.append(nbr);
			visited[nbr]=1;
		    } else {
			queued[nbr]=1;
			kids.append(inner);
			kids.append(nbr);
		    }
		}
	    }
	}
	while (!kids.is_empty()) {
	    int outer=kids.pop();
	    int inner=kids.pop();
	    visited[inner]=1;
	    q.append(outer);
	    q.append(inner);
	}
    }
    // make sure everything's cool
    cerr << "Outer boundaries...\n";
    for (i=0; i<surf.surfI.size(); i++) {
	if (surf.surfI[i].outer != -1) {
	    surf.surfI[surf.surfI[i].outer].inner.add(i);
	    cerr << "  Surface "<<i<<" is bounded by surface "<<surf.surfI[i].outer<<"\n";
	}
    }

    cerr << "Inner boundaries...\n";
    for (i=0; i<surf.surfI.size(); i++) {
	if (surf.surfI[i].inner.size() != 0) {
	    cerr << "  Surface "<<i<<" bounds surfaces: ";
	    for (int j=0; j<surf.surfI[i].inner.size(); j++)
		cerr << surf.surfI[i].inner[j] << " ";
	    cerr << "\n";
	}
    }

    int p1, p2, p3, p4, p5, p6, p7, bcomp;
    int e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12;
    Array1<int> edges(3);
    Array1<int> orient(3);
    HashTable<int,int>* hash = new HashTable<int,int>;
    HashTable<int,int>* ehash = new HashTable<int,int>;
    Point min, max;
    field.get_bounds(min, max);
    Vector v((max-min)*.5);
    v.x(v.x()/(field.nx-1));
    v.y(v.y()/(field.ny-1));
    v.z(v.z()/(field.nz-1));


#if 0
    Array3<int> puncture;
    punctures.resize(field.nx, field.ny, field,nz);
    punctures.initialize(0);


    // first, go through and find nodes that are puncture points or which
    // lie along a diagonal crack between two voxels.  these would be the 
    // following pairs:  	(0,7), (1,6), (2,5), (3,4) - punctures,
    //      (0,3), (0,5), (0,6), (1,2), (1,4), (1,7),
    //	    (2,4), (2,7), (3,5), (3,6), (4,7), (5,6) - cracks

    // if those are the only voxels containing a certain component, split
    // the node into two nodes, and for the elements surrounding one of the
    // voxels, change them to point to the other (new) nodes.
    
    // for now, we'll just handle punctures...
    // we have to fix the node info every time we do a split!
    // PROBLEM: we make new triangles/tetras when we do a split -- gotta
    //   think about this more...
    
    Array1<int> octCells[8];
    Array1<int> same;
    for (i=1; i<field.nx; i++)
	for (j=1; j<field.ny; j++)
	    for (k=1; k<field.nz; k++) {
		octCells[0]=field.grid(i,j,k);
		octCells[1]=field.grid(i+1,j,k);
		octCells[2]=field.grid(i,j+1,k);
		octCells[3]=field.grid(i+1,j+1,k);
		octCells[4]=field.grid(i,j,k+1);
		octCells[5]=field.grid(i+1,j,k+1);
		octCells[6]=field.grid(i,j+1,k+1);
		octCells[7]=field.grid(i+1,j+1,k+1);
		same.inititalize(0);
		if (octCells[0] == octCells[1]) same[0]=same[1]=1;
		if (octCells[0] == octCells[2]) same[0]=same[2]=1;
		if (octCells[0] == octCells[4]) same[0]=same[4]=1;
		if (octCells[1] == octCells[3]) same[1]=same[3]=1;
		if (octCells[1] == octCells[5]) same[1]=same[5]=1;
		if (octCells[2] == octCells[3]) same[2]=same[3]=1;
		if (octCells[2] == octCells[6]) same[2]=same[6]=1;
		if (octCells[3] == octCells[7]) same[3]=same[7]=1;
		if (octCells[4] == octCells[5]) same[4]=same[5]=1;
		if (octCells[4] == octCells[6]) same[4]=same[6]=1;
		if (octCells[5] == octCells[7]) same[5]=same[7]=1;
		if (octCells[6] == octCells[7]) same[6]=same[7]=1;
		int count=0;
		for (int cc=0; cc<8; cc++) if (!same[i]) cc++;
		if (cc>=2) {
		    if (!same[0] && !same[7] && (octCells[0] == octCells[7]));
		    else if (!same[1] && !same[6] && 
			     (octCells[1] == octCells[6]));
		    else if (!same[2] && !same[5] && 
			     (octCells[2] == octCells[5]));
		    else if (!same[3] && !same[4] && 
			     (octCells[3] == octCells[4]));
		}
	    }
#endif


    // for each cell, look at the negative neighbors (down, back, left)
    // if they're of a different material, build the triangles
    // for each triangle, hash the nodes -- if they don't exist, add em
    // we also need to build the edges -- we'll hash these too.

    for (i=1; i<field.nx; i++)
	for (j=1; j<field.ny; j++)
	    for (k=1; k<field.nz; k++) {
		int comp=field.grid(i,j,k);
		int pidx;
		int eidx;
		bcomp=field.grid(i-1,j,k);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.nodes.size());
			p3=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj++; pidx+=1<<10;
		    if (!hash->lookup(pidx, p1)) {
			hash->insert(pidx, surf.nodes.size());
			p1=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk++; pidx++;
		    if (!hash->lookup(pidx, p2)) {
			hash->insert(pidx, surf.nodes.size());
			p2=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj--; pidx-=1<<10;
		    if (!hash->lookup(pidx, p4)) {
			hash->insert(pidx, surf.nodes.size());
			p4=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk--;

		    eidx=(ii<<21)+(jj<<12)+(kk<<3);	// edge 1 in dir 1

		    eidx+=0;
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e1)) {
			ehash->insert(eidx, surf.edges.size());
			e1=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p1));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 1 in dir 1 ("<<p3<<","<<p1<<")\n";
		    }
		    eidx-=0;

		    eidx+=3;	// edge 3 in dir 4
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e3)) {
			ehash->insert(eidx, surf.edges.size());
			e3=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p2));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 3 in dir 4 ("<<p3<<","<<p2<<")\n";
		    }
		    eidx-=3;

		    eidx+=1;	// edge 5 in dir 2
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e5)) {
			ehash->insert(eidx, surf.edges.size());
			e5=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p4));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 5 in dir 2 ("<<p3<<","<<p4<<")\n";
		    }
		    eidx-=1;

		    eidx+=((1<<12)+1); // j++ -> edge 2 in dir 2
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e2)) {
			ehash->insert(eidx, surf.edges.size());
			e2=surf.edges.size();
			surf.edges.add(new TSEdge(p1, p2));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 2 in dir 2 ("<<p1<<","<<p2<<")\n";
		    }
		    eidx-=((1<<12)+1);

		    eidx+=(1<<3);  // k++ -> edge 4 in dir 1
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e4)) {
			ehash->insert(eidx, surf.edges.size());
			e4=surf.edges.size();
			surf.edges.add(new TSEdge(p4, p2));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 4 in dir 1 ("<<p4<<","<<p2<<")\n";
		    }
		    eidx-=(1<<3);

		    int iii=surf.faces.size();

		    surf.edgeI[e2].faces.add(iii);
		    surf.edgeI[e3].faces.add(iii);
		    surf.edgeI[e1].faces.add(iii);
		    surf.faces.add(new TSElement(p3, p1, p2));
		    edges[0]=e2; edges[1]=e3; edges[2]=e1;
		    orient[0]=1; orient[1]=0; orient[2]=1;
		    surf.faceI.resize(surf.faceI.size()+2);
		    surf.faceI[iii].edges = edges; 
		    surf.faceI[iii].edgeOrient = orient;
		    surf.edgeI[e5].faces.add(iii+1);
		    surf.edgeI[e3].faces.add(iii+1);
		    surf.edgeI[e4].faces.add(iii+1);
		    surf.faces.add(new TSElement(p2, p4, p3));
		    edges[0]=e5; edges[1]=e3; edges[2]=e4;
		    orient[0]=0; orient[1]=1; orient[2]=0;
		    surf.faceI[iii+1].edges = edges; 
		    surf.faceI[iii+1].edgeOrient = orient;
		    
//		    if (surf.outer[comp]!=bcomp) {
			surf.surfI[bcomp].faces.add(iii);
			surf.surfI[bcomp].faces.add(iii+1);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
			surf.surfI[bcomp].faceOrient.add(1);
			surf.surfI[bcomp].faceOrient.add(1);
//		    }
//		    if (surf.outer[bcomp]!=comp) {
			surf.surfI[comp].faces.add(iii);
			surf.surfI[comp].faces.add(iii+1);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
			surf.surfI[comp].faceOrient.add(0);
			surf.surfI[comp].faceOrient.add(0);
//		    }
		}
		bcomp=field.grid(i,j-1,k);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.nodes.size());
			p3=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii++; pidx+=1<<20;
		    if (!hash->lookup(pidx, p5)) {
			hash->insert(pidx, surf.nodes.size());
			p5=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk++; pidx++;
		    if (!hash->lookup(pidx, p6)) {
			hash->insert(pidx, surf.nodes.size());
			p6=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii--; pidx-=1<<20;
		    if (!hash->lookup(pidx, p4)) {
			hash->insert(pidx, surf.nodes.size());
			p4=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk--;

		    eidx=(ii<<21)+(jj<<12)+(kk<<3);

		    eidx+=1;	// edge 5 in dir 2
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e5)) {
			ehash->insert(eidx, surf.edges.size());
			e5=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p4));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 5 in dir 2 ("<<p3<<","<<p4<<")\n";
		    }
		    eidx-=1;

		    eidx+=4;	// edge 7 in dir 5
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e7)) {
			ehash->insert(eidx, surf.edges.size());
			e7=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p6));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 7 in dir 5 ("<<p3<<","<<p6<<")\n";
		    }
		    eidx-=4;

		    eidx+=2;	// edge 6 in dir 3
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e6)) {
			ehash->insert(eidx, surf.edges.size());
			e6=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p5));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 6 in dir 3 ("<<p3<<","<<p5<<")\n";
		    }		    
		    eidx-=2;

		    eidx+=((1<<3)+2);	// k++ -> edge 9 in dir 3
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e9)) {
			ehash->insert(eidx, surf.edges.size());
			e9=surf.edges.size();
			surf.edges.add(new TSEdge(p4, p6));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 9 in dir 3 ("<<p4<<","<<p6<<")\n";
		    }
		    eidx-=((1<<3)+2);
		    
		    eidx+=((1<<21)+1);  // i++ -> edge 8 in dir 2
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e8)) {
			ehash->insert(eidx, surf.edges.size());
			e8=surf.edges.size();
			surf.edges.add(new TSEdge(p5, p6));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 8 in dir 2 ("<<p5<<","<<p6<<")\n";
		    }
		    eidx-=((1<<21)+1);
			
		    int iii=surf.faces.size();

		    surf.edgeI[e6].faces.add(iii);
		    surf.edgeI[e7].faces.add(iii);
		    surf.edgeI[e8].faces.add(iii);
		    surf.faces.add(new TSElement(p6, p5, p3));
		    edges[0]=e6; edges[1]=e7; edges[2]=e8;
		    orient[0]=0; orient[1]=1; orient[2]=0;
		    surf.faceI.resize(surf.faceI.size()+2);
		    surf.faceI[iii].edges = edges; 
		    surf.faceI[iii].edgeOrient = orient;
		    surf.edgeI[e9].faces.add(iii+1);
		    surf.edgeI[e7].faces.add(iii+1);
		    surf.edgeI[e5].faces.add(iii+1);
		    surf.faces.add(new TSElement(p3, p4, p6));
		    edges[0]=e9; edges[1]=e7; edges[2]=e5;
		    orient[0]=1; orient[1]=0; orient[2]=1;
		    surf.faceI[iii+1].edges = edges; 
		    surf.faceI[iii+1].edgeOrient = orient;

//		    if (surf.outer[comp]!=bcomp) {
			surf.surfI[bcomp].faces.add(iii);
			surf.surfI[bcomp].faces.add(iii+1);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
			surf.surfI[bcomp].faceOrient.add(1);
			surf.surfI[bcomp].faceOrient.add(1);
//		    }
//		    if (surf.outer[bcomp]!=comp) {
			surf.surfI[comp].faces.add(iii);
			surf.surfI[comp].faces.add(iii+1);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
			surf.surfI[comp].faceOrient.add(0);
			surf.surfI[comp].faceOrient.add(0);
//		    }
		}
		bcomp=field.grid(i,j,k-1);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.nodes.size());
			p3=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii++; pidx+=1<<20;
		    if (!hash->lookup(pidx, p5)) {
			hash->insert(pidx, surf.nodes.size());
			p5=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj++; pidx+=1<<10;
		    if (!hash->lookup(pidx, p7)) {
			hash->insert(pidx, surf.nodes.size());
			p7=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii--; pidx-=1<<20;
		    if (!hash->lookup(pidx, p1)) {
			hash->insert(pidx, surf.nodes.size());
			p1=surf.nodes.size();
			surf.nodes.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj--;

		    eidx=(ii<<21)+(jj<<12)+(kk<<3);	// edge 1 in dir 1

		    eidx+=0;
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e1)) {
			ehash->insert(eidx, surf.edges.size());
			e1=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p1));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 1 in dir 1 ("<<p3<<","<<p1<<")\n";
		    }
		    eidx-=0;

		    eidx+=5;	// edge 11 in dir 6
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e11)) {
			ehash->insert(eidx, surf.edges.size());
			e11=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p7));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 11 in dir 6 ("<<p3<<","<<p7<<")\n";
		    }
		    eidx-=5;

		    eidx+=2;	// edge 6 in dir 3
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e6)) {
			ehash->insert(eidx, surf.edges.size());
			e6=surf.edges.size();
			surf.edges.add(new TSEdge(p3, p5));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 6 in dir 3 ("<<p3<<","<<p5<<")\n";
		    }		    
		    eidx-=2;

		    eidx+=((1<<12)+2);	// j++ -> edge 12 in dir 3
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e12)) {
			ehash->insert(eidx, surf.edges.size());
			e12=surf.edges.size();
			surf.edges.add(new TSEdge(p1, p7));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 12 in dir 3 ("<<p1<<","<<p7<<")\n";
		    }
		    eidx-=((1<<12)+2);

		    eidx+=(1<<21);  	// i++ -> edge 10 in dir 1
//		    cerr << "eidx="<<eidx<<"\n";
		    if (!ehash->lookup(eidx, e10)) {
			ehash->insert(eidx, surf.edges.size());
			e10=surf.edges.size();
			surf.edges.add(new TSEdge(p5, p7));
			surf.edgeI.resize(surf.edgeI.size()+1);
//			cerr << "Adding edge 10 in dir 1 ("<<p5<<","<<p7<<")\n";
		    }
		    eidx-=(1<<21);

		    int iii=surf.faces.size();

		    surf.edgeI[e10].faces.add(iii);
		    surf.edgeI[e11].faces.add(iii);
		    surf.edgeI[e6].faces.add(iii);
		    surf.faces.add(new TSElement(p3, p5, p7));
		    edges[0]=e10; edges[1]=e11; edges[2]=e6;
		    orient[0]=1; orient[1]=0; orient[2]=1;
		    surf.faceI.resize(surf.faceI.size()+2);
		    surf.faceI[iii].edges = edges; 
		    surf.faceI[iii].edgeOrient = orient;
		    surf.edgeI[e1].faces.add(iii+1);
		    surf.edgeI[e11].faces.add(iii+1);
		    surf.edgeI[e12].faces.add(iii+1);
		    surf.faces.add(new TSElement(p7, p1, p3));
		    edges[0]=e1; edges[1]=e11; edges[2]=e12;
		    orient[0]=0; orient[1]=1; orient[2]=0;
		    surf.faceI[iii+1].edges = edges; 
		    surf.faceI[iii+1].edgeOrient = orient;

//		    if (surf.outer[comp]!=bcomp) {
			surf.surfI[bcomp].faces.add(iii);
			surf.surfI[bcomp].faces.add(iii+1);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
//			surf.surfI[bcomp].faceOrient.add(comp>bcomp);
			surf.surfI[bcomp].faceOrient.add(1);
			surf.surfI[bcomp].faceOrient.add(1);
//		    }
//		    if (surf.outer[bcomp]!=comp) {
			surf.surfI[comp].faces.add(iii);
			surf.surfI[comp].faces.add(iii+1);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
//			surf.surfI[comp].faceOrient.add(bcomp>comp);
			surf.surfI[comp].faceOrient.add(0);
			surf.surfI[comp].faceOrient.add(0);
//		    }
		}
	    }

    int bigGreyIdx=-1;
    int bigGreySize=-1;
    int bigWhiteIdx=-1;
    int bigWhiteSize=-1;
    for (i=0; i<field.comps.size(); i++) {
	if (field.comps[i]) {
	    int thisType=field.get_type(field.comps[i]);
	    int thisSize=field.get_size(field.comps[i]);
	    surf.surfI[i].matl=thisType;
	    if (thisType==4 && thisSize>bigGreySize) {
		bigGreyIdx=i;
		bigGreySize=thisSize;
	    } else if (thisType==5 && thisSize>bigWhiteSize) {
		bigWhiteIdx=i;
		bigWhiteSize=thisSize;
	    }
	}
	else surf.surfI[i].matl=-1;
	cerr << "surf.matl["<<i<<"]="<<surf.surfI[i].matl<<"\n";
    }
    if (surf.surfI.size()>1 && surf.surfI[1].outer == 0)
        surf.surfI[1].name="scalpAll";
    surf.surfI[0].name="scalp";
    if (bigGreyIdx != -1) {
        surf.surfI[bigGreyIdx].name="cortex";
	cerr << "**** Biggest grey matter (material 4) is region "<<bigGreyIdx<<"\n";

	if (surf.surfI[bigGreyIdx].inner.size()) {
	    cerr << "**** WARNING: this region contains inner regions!\n";
	}
    } else {
	cerr << "No grey matter (material 4) found.\n";
    }
    if (bigWhiteIdx != -1) {
	cerr << "**** Biggest white matter (material 5) is region "<<bigWhiteIdx<<"\n";
	if (surf.surfI[bigWhiteIdx].inner.size()) {
	    cerr << "**** WARNING: this region contains inner regions!\n";
	}
    } else {
	cerr << "No white matter (material 5) found.\n";
    }
    cerr << "BUILDING NODE INFORMATION...\n";
    surf.bldNodeInfo();
    cerr << "DONE!\n";


}


void SegFldToSurfTree::execute()
{
    SegFldHandle isf;
    if(!infield->get(isf))
	return;
    if(!isf.get_rep()) return;
    SurfTree* st=scinew SurfTree();
    fldToTree(*(isf.get_rep()), *st);
    outsurf->send(SurfaceHandle(st));
}
