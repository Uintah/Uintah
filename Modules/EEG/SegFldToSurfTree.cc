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
};

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
    for (i=1; i<field.nx; i++, is[0]++, is[1]++, is[2]++) {
	js[0]=1; js[1]=0; js[2]=1;
	for (j=1; j<field.ny; j++, js[0]++, js[1]++, js[2]++) {
	    ks[0]=1; ks[1]=1; ks[2]=0;
	    for (k=1; k<field.nz; k++, ks[0]++, ks[1]++, ks[2]++) {
		tripleInt idx(i,j,k);
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
	    cerr << "Component "<<i<<" is bounded by: ";
	    for (int j=0; j<touches[i].size(); j++)
		cerr << touches[i][j]<<" ";
	    cerr << "\n";
	}
    }

    Array1<int> visited(field.comps.size());
    Array1<int> queued(field.comps.size());
    surf.inner.resize(field.comps.size());
    surf.outer.resize(field.comps.size());
    surf.outer.initialize(-1);
    visited.initialize(0);
    visited.initialize(0);
    for (i=0; i<touches.size(); i++) if (touches[i].size() == 0) visited[i]=1;
    Queue<int> q;
    Queue<int> kids;

    int air=field.grid(0,0,0);
    // visit 0 -- push all of its nbrs
    visited[air]=1;
    queued[air]=1;

    for (i=0; i<touches[air].size(); i++) {
	q.append(air);
	q.append(touches[air][i]);
	visited[touches[air][i]]=queued[touches[air][i]]=1;
    }

    while (!q.is_empty()) {
	while (!q.is_empty()) {
	    int outer=q.pop();
	    int inner=q.pop();
	    surf.outer[inner]=outer;
	    for (i=0; i<touches[inner].size(); i++) {
		// if another node it touches has been visited
		int nbr=touches[inner][i];
		if (!visited[nbr]) {
		    if (queued[nbr]) {	// noone should have queued -- must be
			                // crack-connected
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
    for (i=0; i<surf.outer.size(); i++) {
	if (surf.outer[i] != -1) {
	    surf.inner[surf.outer[i]].add(i);
	    cerr << "  Surface "<<i<<" is bounded by surface "<<surf.outer[i]<<"\n";
	}
    }

    cerr << "Inner boundaries...\n";
    for (i=0; i<surf.inner.size(); i++) {
	if (surf.inner[i].size() != 0) {
	    cerr << "  Surface "<<i<<" bounds surfaces: ";
	    for (int j=0; j<surf.inner[i].size(); j++)
		cerr << surf.inner[i][j] << " ";
	    cerr << "\n";
	}
    }

    surf.surfEls.resize(field.comps.size());
    int p1, p2, p3, p4, p5, p6, p7, bcomp;
    HashTable<int,int>* hash = new HashTable<int,int>;
    Point min, max;
    field.get_bounds(min, max);
    Vector v((max-min)*.5);
    v.x(v.x()/(field.nx-1));
    v.y(v.y()/(field.ny-1));
    v.z(v.z()/(field.nz-1));
    for (i=1; i<field.nx; i++)
	for (j=1; j<field.ny; j++)
	    for (k=1; k<field.nz; k++) {
		int comp=field.grid(i,j,k);
		int pidx;
		bcomp=field.grid(i-1,j,k);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.points.size());
			p3=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj++; pidx+=1<<10;
		    if (!hash->lookup(pidx, p1)) {
			hash->insert(pidx, surf.points.size());
			p1=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk++; pidx++;
		    if (!hash->lookup(pidx, p2)) {
			hash->insert(pidx, surf.points.size());
			p2=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj--; pidx-=1<<10;
		    if (!hash->lookup(pidx, p4)) {
			hash->insert(pidx, surf.points.size());
			p4=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    int iii=surf.elements.size();
		    surf.elements.add(new TSElement(p3, p1, p2));
		    surf.elements.add(new TSElement(p2, p4, p3));
		    if (surf.outer[comp]!=bcomp) {
			surf.surfEls[bcomp].add(iii);
			surf.surfEls[bcomp].add(iii+1);
		    }
		    if (surf.outer[bcomp]!=comp) {
			surf.surfEls[comp].add(iii);
			surf.surfEls[comp].add(iii+1);
		    }
		}
		bcomp=field.grid(i,j-1,k);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.points.size());
			p3=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii++; pidx+=1<<20;
		    if (!hash->lookup(pidx, p5)) {
			hash->insert(pidx, surf.points.size());
			p5=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    kk++; pidx++;
		    if (!hash->lookup(pidx, p6)) {
			hash->insert(pidx, surf.points.size());
			p6=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii--; pidx-=1<<20;
		    if (!hash->lookup(pidx, p4)) {
			hash->insert(pidx, surf.points.size());
			p4=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    int iii=surf.elements.size();
		    surf.elements.add(new TSElement(p3, p5, p6));
		    surf.elements.add(new TSElement(p6, p4, p3));
		    if (surf.outer[comp]!=bcomp) {
			surf.surfEls[bcomp].add(iii);
			surf.surfEls[bcomp].add(iii+1);
		    }
		    if (surf.outer[bcomp]!=comp) {
			surf.surfEls[comp].add(iii);
			surf.surfEls[comp].add(iii+1);
		    }
		}
		bcomp=field.grid(i,j,k-1);
		if (bcomp != comp) {
		    ii=i-1; jj=j-1; kk=k-1;
		    pidx=(ii<<20)+(jj<<10)+kk;
		    if (!hash->lookup(pidx, p3)) {
			hash->insert(pidx, surf.points.size());
			p3=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii++; pidx+=1<<20;
		    if (!hash->lookup(pidx, p5)) {
			hash->insert(pidx, surf.points.size());
			p5=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    jj++; pidx+=1<<10;
		    if (!hash->lookup(pidx, p7)) {
			hash->insert(pidx, surf.points.size());
			p7=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    ii--; pidx-=1<<20;
		    if (!hash->lookup(pidx, p1)) {
			hash->insert(pidx, surf.points.size());
			p1=surf.points.size();
			surf.points.add(field.get_point(ii,jj,kk)+v);
		    }
		    int iii=surf.elements.size();
		    surf.elements.add(new TSElement(p3, p5, p7));
		    surf.elements.add(new TSElement(p7, p1, p3));
		    if (surf.outer[comp]!=bcomp) {
			surf.surfEls[bcomp].add(iii);
			surf.surfEls[bcomp].add(iii+1);
		    }
		    if (surf.outer[bcomp]!=comp) {
			surf.surfEls[comp].add(iii);
			surf.surfEls[comp].add(iii+1);
		    }
		}
	    }
    surf.matl.resize(field.comps.size());
    int bigGreyIdx=-1;
    int bigGreySize=-1;
    int bigWhiteIdx=-1;
    int bigWhiteSize=-1;
    for (i=0; i<field.comps.size(); i++) {
	if (field.comps[i]) {
	    int thisType=field.get_type(field.comps[i]);
	    int thisSize=field.get_size(field.comps[i]);
	    surf.matl[i]=thisType;
	    if (thisType==4 && thisSize>bigGreySize) {
		bigGreyIdx=i;
		bigGreySize=thisSize;
	    } else if (thisType==5 && thisSize>bigWhiteSize) {
		bigWhiteIdx=i;
		bigWhiteSize=thisSize;
	    }
	}
	else surf.matl[i]=-1;
	cerr << "surf.matl["<<i<<"]="<<surf.matl[i]<<"\n";
    }
    if (bigGreyIdx != -1) {
	cerr << "**** Biggest grey matter (material 4) is region "<<bigGreyIdx<<"\n";
	if (surf.inner[bigGreyIdx].size()) {
	    cerr << "**** WARNING: this region contains inner regions!\n";
	}
    } else {
	cerr << "No grey matter (material 4) found.\n";
    }
    if (bigWhiteIdx != -1) {
	cerr << "**** Biggest white matter (material 5) is region "<<bigWhiteIdx<<"\n";
	if (surf.inner[bigWhiteIdx].size()) {
	    cerr << "**** WARNING: this region contains inner regions!\n";
	}
    } else {
	cerr << "No white matter (material 5) found.\n";
    }
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
