//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/SegFldPort.h>
#include <DaveW/Datatypes/General/SegFld.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Containers/Queue.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Tester/RigorousTest.h>

#include <map.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class SegFldToSurfTree : public Module {
    SegFldIPort* infield;
    SurfaceOPort* outsurf;
public:

    SegFldToSurfTree(const clString& id);
    virtual ~SegFldToSurfTree();
    virtual void execute();
};

extern "C" Module* make_SegFldToSurfTree(const clString& id)
{
    return new SegFldToSurfTree(id);
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

SegFldToSurfTree::~SegFldToSurfTree()
{
}


// for a cube of 8 cells, determine the dominant type and see if it is
// split into two separated groups.
// if so, determine the list of faces that bound each group.
// for each face in each group, determine which p index touches the
// center node of the 8 cells, and add it to the list for that group

int bldPMask(Array1<int>& cells, Array1<int>& thin) {
    int pMask=0;
    
    // figure out the dominant type of the cells
    Array1<int> valTypes;
    Array1<int> valCnt;
    int i,j;
    for (i=0; i<8; i++) {
	for (j=0; j<valTypes.size(); j++)
	    if (cells[i] == valTypes[j]) {valCnt[j]=valCnt[j]+1; break;}
	if (j==valTypes.size()) {valTypes.add(cells[i]); valCnt.add(1);}
    }

    // if there aren't 2 types, return 0
    if (valTypes.size()<2) return 0;

    int dType=valTypes[0];
    int dNum=valCnt[0];
    for (i=1; i<valTypes.size(); i++) 
	if (valCnt[i]>dNum) {
	    dNum=valCnt[i];
	    dType=valTypes[i];
	}

    // if dominant type has less than 4 cells or more than 6 cells, return 0
    if (dNum<4 || dNum>6) return 0;

    if (dNum==4 && valTypes.size()==2)
	if (thin[valTypes[0]] > thin[valTypes[1]]) dType=valTypes[0];
	else dType=valTypes[1];

    int dMask=0;
    for (i=0; i<8; i++)
	if (cells[i] == dType) dMask |= (1<<i);

//    cerr << "dMask="<<dMask<<"\n";
//    cerr << "dType="<<dType<<"\n";

    // build mask of dominant type
    if ((dMask & 0xc3) == 0xc3) { // 11000011
	if (cells[2]!=dType) pMask |= (1<<3 | 1<<1); // 0/2 2/(3,6)
	if (cells[3]!=dType) pMask |= (1<<5 | 1<<1 | 1<<7); // 1/3 2/3 3/7
	return pMask;
    }
    if ((dMask & 0x3c) == 0x3c) { // 00111100
	if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
	if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
	return pMask;
    }
    if ((dMask & 0xa5) == 0xa5) { // 10100101
	if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
	if (cells[3]!=dType) pMask |= (1<<5 | 1<<1 | 1<<7); // 1/3  2/3  3/7
	return pMask;
    }
    if ((dMask & 0x5a) == 0x5a) { // 01011010
	if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
	if (cells[2]!=dType) pMask |= (1<<3 | 1<<1); // 0/2 2/(3,6)
	return pMask;
    }
    if ((dMask & 0x99) == 0x99) { // 10011001
	if (cells[1]!=dType) pMask |= (1<<3 | 1<<5); // 0/1 1/(3,5)
	if (cells[5]!=dType) pMask |= (1<<5 | 1<<4 | 1<<6); // 1/5 4/5 5/7
	return pMask;
    }
    if ((dMask & 0x66) == 0x66) { // 01100110
	if (cells[0]!=dType) pMask |= (1<<3); // 0/(1,2,4)
	if (cells[4]!=dType) pMask |= (1<<3 | 1<<4); // 0/4 4/(5,6)
	return pMask;
    }
    return 0;
}

void bldSplits(ScalarFieldRGchar* ch,
	       Array1<tripleInt>& splits,
	       Array1<int>& splT, Array1<int>& thin) {
    int i,j,k;
    Array1<int> cells(8);

    for (i=1; i<ch->nx; i++)
	for (j=1; j<ch->ny; j++)
	    for (k=1; k<ch->nz; k++) {
		cells[0]=ch->grid(i,j,k);
		cells[1]=ch->grid(i-1,j,k);
		cells[2]=ch->grid(i,j-1,k);
		cells[3]=ch->grid(i-1,j-1,k);
		cells[4]=ch->grid(i,j,k-1);
		cells[5]=ch->grid(i-1,j,k-1);
		cells[6]=ch->grid(i,j-1,k-1);
		cells[7]=ch->grid(i-1,j-1,k-1);
		int pMask=bldPMask(cells, thin);
		if (pMask) {
		    splits.add(tripleInt(i,j,k));
		    splT.add(pMask);
		}
	    }
#if 0
    cerr << "Here are the splits:\n";
    for (i=0; i<splits.size(); i++)
	cerr << "   node = ("<<splits[i].x <<", "<<splits[i].y<<", "<<splits[i].z<<"), type = "<<(int)splT[i]<<"\n";
    cerr << "DONE!\n";
#endif
}

int getNode(int ii, int jj, int kk, int n, const Vector &v, 
	    SegFld& field, const Array1<int>& splT, 
	    const Array1<int>& splI, map<int,int>* hash,
	    SurfTree& surf) {
    int h;
    int pidx=(ii<<20)+(jj<<10)+kk;
    map<int,int>::iterator iter = hash->find(pidx);
    if (iter == hash->end()) {
	(*hash)[pidx] = surf.nodes.size();
	h = surf.nodes.size();
	surf.nodes.add(field.get_point(ii,jj,kk)+v);
    } else {
	h = (*iter).second;
	if (h<0) {
	    h = -h-1;
	    if (splT[h] & (1<<n)) 
		h = splI[h];
	    else
		h = splI[h]+1;
	}
    }
    //cerr << "Getting node ("<<ii<<","<<jj<<","<<kk<<") n="<<n<<" h="<<h<<"\n";
    return h;
}

int getEdge(int ii, int jj, int kk, int o, SurfTree& surf, int p1, int p2,
	    map<int,int>* ehash) {
    int h;
    int eidx=(ii<<21)+(jj<<12)+(kk<<3)+o;
    map<int,int>::iterator iter = ehash->find(eidx);
    if (iter == ehash->end()) {
	(*ehash)[eidx] = surf.edges.size();
	h=surf.edges.size();
	surf.edges.add(new TSEdge(p1, p2));
	surf.edgeI.resize(surf.edgeI.size()+1);
    }
    else {
	h = (*iter).second;
    }
    return h;
}

void setFaces(int e1, int e2, int e4, int e5, int e3, int p3, int p1, int p2,
	      int p4, SurfTree& surf, int comp, int bcomp) {
//    cerr << "Blding faces from nodes: "<<p1<<" "<<p2<<" "<<p3<<" "<<p4<<"\n";



    int iii=surf.faces.size();
    Array1<int> edges(3);
    Array1<int> orient(3);
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
    
    surf.faceI[iii].surfIdx.add(bcomp);
    surf.faceI[iii].surfOrient.add(1);
    surf.faceI[iii+1].surfIdx.add(bcomp);
    surf.faceI[iii+1].surfOrient.add(1);
    
    surf.faceI[iii].surfIdx.add(comp);
    surf.faceI[iii].surfOrient.add(0);
    surf.faceI[iii+1].surfIdx.add(comp);
    surf.faceI[iii+1].surfOrient.add(0);		    
    
    surf.surfI[bcomp].faces.add(iii);
    surf.surfI[bcomp].faces.add(iii+1);
    surf.surfI[bcomp].faceOrient.add(1);
    surf.surfI[bcomp].faceOrient.add(1);
    surf.surfI[comp].faces.add(iii);
    surf.surfI[comp].faces.add(iii+1);
    surf.surfI[comp].faceOrient.add(0);
    surf.surfI[comp].faceOrient.add(0);
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
#if 0
    cerr << "\n";
    for (i=0; i<touches.size(); i++) {
	if (touches[i].size() != 0) {
	    cerr << "Component "<<i<<" touches: ";
	    for (int j=0; j<touches[i].size(); j++)
		cerr << touches[i][j]<<" ";
	    cerr << "\n";
	}
    }
#endif

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
//	    cerr << "outer="<<outer<<"   inner="<<inner<<"\n";
	    surf.surfI[inner].outer=outer;
	    for (i=0; i<touches[inner].size(); i++) {
		// if another component it touches has been visited
		int nbr=touches[inner][i];
		if (!visited[nbr]) {
		    if (queued[nbr]) {	// noone should have queued -- must be
			                // crack-connected
//			cerr << "why am i here??\n";
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

//    cerr << "Outer boundaries...\n";
    for (i=0; i<surf.surfI.size(); i++) {
	if (surf.surfI[i].outer != -1) {
	    surf.surfI[surf.surfI[i].outer].inner.add(i);
//	    cerr << "  Surface "<<i<<" is bounded by surface "<<surf.surfI[i].outer<<"\n";
	}
    }

#if 0
    cerr << "Inner boundaries...\n";
    for (i=0; i<surf.surfI.size(); i++) {
	if (surf.surfI[i].inner.size() != 0) {
	    cerr << "  Surface "<<i<<" bounds surfaces: ";
	    for (int j=0; j<surf.surfI[i].inner.size(); j++)
		cerr << surf.surfI[i].inner[j] << " ";
	    cerr << "\n";
	}
    }
#endif

    int p1, p2, p3, p4, p5, p6, p7, bcomp;
    int e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12;
    map<int,int>* hash = new map<int,int>;
    map<int,int>* ehash = new map<int,int>;
    Point min, max;
    field.get_bounds(min, max);
    Vector v((max-min)*.5);
    v.x(v.x()/(field.nx-1));
    v.y(v.y()/(field.ny-1));
    v.z(v.z()/(field.nz-1));

    Array1<tripleInt> splits;
    Array1<int> splT;
    bldSplits(field.getTypeFld(), splits, splT, field.thin);

    Array1<int> splI;
    // first add in all of the nodes that need to be split
    for (i=0; i<splits.size(); i++) {
	tripleInt idx=splits[i];
	int ii=idx.x-1;
	int jj=idx.y-1;
	int kk=idx.z-1;
	int pidx=(ii<<20)+(jj<<10)+kk;
	if (hash->find(pidx) == hash->end()) {
	    (*hash)[pidx] = -(i+1);
	} 
	// else cerr << "SegFldToSurfTree - why would I be here??\n";
	splI.add(surf.nodes.size());
	surf.nodes.add(field.get_point(ii,jj,kk)+v);
	surf.nodes.add(field.get_point(ii,jj,kk)+v);
    }


    // for each cell, look at the negative neighbors (down, back, left)
    // if they're of a different material, build the triangles
    // for each triangle, hash the nodes -- if they don't exist, add em
    // we also need to build the edges -- we'll hash these too.

    for (i=1; i<field.nx; i++)
	for (j=1; j<field.ny; j++)
	    for (k=1; k<field.nz; k++) {
		int comp=field.grid(i,j,k);
		bcomp=field.grid(i-1,j,k);
		if (bcomp != comp) {
		    p3=getNode(i-1,j-1,k-1,3,v,field,splT,splI,hash,surf);
		    p1=getNode(i-1,j  ,k-1,1,v,field,splT,splI,hash,surf);
		    p2=getNode(i-1,j  ,k  ,2,v,field,splT,splI,hash,surf);
		    p4=getNode(i-1,j-1,k  ,4,v,field,splT,splI,hash,surf);

		    e1=getEdge(i-1,j-1,k-1,0,surf,p3,p1,ehash);
		    e3=getEdge(i-1,j-1,k-1,3,surf,p3,p2,ehash);
		    e5=getEdge(i-1,j-1,k-1,1,surf,p3,p4,ehash);
		    e2=getEdge(i-1,j,  k-1,1,surf,p1,p2,ehash);
		    e4=getEdge(i-1,j-1,k  ,0,surf,p4,p2,ehash);

		    setFaces(e1,e2,e4,e5,e3,p3,p1,p2,p4,surf,comp,bcomp);
		}
		bcomp=field.grid(i,j-1,k);
		if (bcomp != comp) {
		    p3=getNode(i-1,j-1,k-1,3,v,field,splT,splI,hash,surf);
		    p5=getNode(i  ,j-1,k-1,5,v,field,splT,splI,hash,surf);
		    p6=getNode(i  ,j-1,k  ,6,v,field,splT,splI,hash,surf);
		    p4=getNode(i-1,j-1,k  ,4,v,field,splT,splI,hash,surf);

		    e5=getEdge(i-1,j-1,k-1,1,surf,p3,p4,ehash);
		    e7=getEdge(i-1,j-1,k-1,4,surf,p3,p6,ehash);
		    e6=getEdge(i-1,j-1,k-1,2,surf,p3,p5,ehash);
		    e9=getEdge(i-1,j-1,k  ,2,surf,p4,p6,ehash);
		    e8=getEdge(i  ,j-1,k-1,1,surf,p5,p6,ehash);

		    setFaces(e5,e9,e8,e6,e7,p3,p4,p6,p5,surf,comp,bcomp);
		}
		bcomp=field.grid(i,j,k-1);
		if (bcomp != comp) {
		    p3=getNode(i-1,j-1,k-1,3,v,field,splT,splI,hash,surf);
		    p5=getNode(i  ,j-1,k-1,5,v,field,splT,splI,hash,surf);
		    p7=getNode(i  ,j  ,k-1,7,v,field,splT,splI,hash,surf);
		    p1=getNode(i-1,j  ,k-1,1,v,field,splT,splI,hash,surf);

		    e1 =getEdge(i-1,j-1,k-1,0,surf,p3,p1,ehash);
		    e11=getEdge(i-1,j-1,k-1,5,surf,p3,p7,ehash);
		    e6 =getEdge(i-1,j-1,k-1,2,surf,p3,p5,ehash);
		    e12=getEdge(i-1,j  ,k-1,2,surf,p1,p7,ehash);
		    e10=getEdge(i  ,j-1,k-1,0,surf,p5,p7,ehash);

		    setFaces(e6,e10,e12,e1,e11,p3,p5,p7,p1,surf,comp,bcomp);
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

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.7  2000/03/17 18:44:21  dahart
// Replaced all instances of HashTable<class X, class Y> with the STL
// map<class X, class Y>.  Removed all includes of HashTable.h
//
// Revision 1.6  2000/03/17 09:25:35  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.5  2000/03/04 00:16:34  dmw
// update some DaveW stuff
//
// Revision 1.4  1999/10/07 02:06:28  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:24  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:39  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:02  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:14  dmw
// Added and updated DaveW Datatypes/Modules
//
//
