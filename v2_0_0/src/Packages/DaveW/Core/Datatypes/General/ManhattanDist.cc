
/*
 *  ManhattanDist.cc:  For coregistering a set of points to a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/ManhattanDist.h>
#include <Core/Containers/String.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

#include <queue>
using std::queue;

#include <iostream>
using std::cerr;

namespace DaveW {
using namespace SCIRun;

static Persistent* make_ManhattanDist()
{
    return scinew ManhattanDist;
}

PersistentTypeID ManhattanDist::type_id("ManhattanDist", "ScalarFieldRGint", make_ManhattanDist);

ManhattanDist::ManhattanDist(const Array1<Point>&Pts, int n, int init,
			     double minX, double minY, double minZ,
			     double maxX, double maxY, double maxZ)
: ScalarFieldRGint(0, 0, 0), pts(Pts), fullyInitialized(init)
{

    Point min(minX, minY, minZ);
    Point max(maxX, maxY, maxZ);
    Vector v(max-min);
    double l=Max(v.x(), v.y(), v.z());
    double dd=l/(n-1)+.00001;
    nx=v.x()/dd+1; ny=v.y()/dd+1; nz=v.z()/dd+1;
    max.x(min.x()+dd*nx); max.y(min.y()+dd*ny); max.z(min.z()+dd*nz);
    set_bounds(min,max);
    closestNodeIdx.resize(nx,ny,nz);
    grid.resize(nx,ny,nz);
    grid.initialize(-1);
    if (init) {
	cerr << "starting full initialize of ManhattanDist...\n";
	initialize();
	cerr << "done with full initialize of ManhattanDist!\n";
    } else {
	cerr << "starting partial initialize of ManhattanDist...\n";
	partial_initialize();
	cerr << "done with partial initialize of ManhattanDist!\n";
    }
}

ManhattanDist::ManhattanDist(const ManhattanDist& copy)
: ScalarFieldRGint(copy), pts(copy.pts), closestNodeIdx(copy.closestNodeIdx),
fullyInitialized(copy.fullyInitialized)
{
}

ManhattanDist::ManhattanDist()
  : ScalarFieldRGint(0, 0, 0)
{
}

ManhattanDist::~ManhattanDist()
{
}

// just initialize the cells containing surface pts to be dist 0...
// we'll compute and store the others as we need them!

void ManhattanDist::partial_initialize() {
    if (pts.size()==0) return;
    for (int a=0; a<pts.size(); a++) {
	int x,y,z;
	locate(pts[a], x, y, z);
	closestNodeIdx(x,y,z).add(a);
	grid(x,y,z)=0;
    }
}

// grab any point, and compute the Manhattan distances through the whole
// gridDist to that point.
// then, for each other point, update the distances that are wrong.

void ManhattanDist::initialize() {
    fullyInitialized=1;
    if (pts.size()==0) return;
    int i,j,k;
    locate(pts[0], i, j, k);
    int x,y,z;
    int dist;
    int sign=-1;
    for (x=0, dist=i, sign=-1; x<nx; x++, dist+=sign) {
	if (dist==0) sign=1; 
	for (y=0; y<ny; y++) {
	    for (z=0; z<nz; z++) {
		closestNodeIdx(x,y,z).add(0);
		grid(x,y,z)=dist;
	    }
	}
    }
    for (y=0, dist=j, sign=-1; y<ny; y++, dist+=sign) {
	if (dist==0) sign=1; 
	for (x=0; x<nx; x++) {
	    for (z=0; z<nz; z++) {
		grid(x,y,z)+=dist;
	    }
	}
    }
    for (z=0, dist=k, sign=-1; z<nz; z++, dist+=sign) {
	if (dist==0) sign=1; 
	for (x=0; x<nx; x++) {
	    for (y=0; y<ny; y++) {
		grid(x,y,z)+=dist;
	    }
	}
    }
cerr << "   ... phase 1 complete!\n";
    Array3<char> visited(nx,ny,nz);
    queue<int> q;
    for (int p=1; p<pts.size(); p++) {
	visited.initialize(0);	
	locate(pts[p], i, j, k);
	closestNodeIdx(i,j,k).add(p);
	grid(i,j,k)=0;
	q.push(i); q.push(j); q.push(k);
	visited(i,j,k)=1;
	while (!q.empty()) {
	    const int a = q.front(); q.pop();
	    const int b = q.front(); q.pop();
	    const int c = q.front(); q.pop();
	    int dist=grid(a,b,c);
//	    int idx=closestNodeIdx(a,b,c);
	    int a1, b1, c1;	
	    for (a1=a-1; a1<=a+1; a1++)
		for (b1=b-1; b1<=b+1; b1++)
		    for (c1=c-1; c1<=c+1; c1++) {
			if (a1>=0 && a1<nx && b1>=0 && b1<ny && c1>=0 && 
			    c1<nz && !visited(a1,b1,c1)) {
			    if (dist+1 <= grid(a1,b1,c1)) {
				if (dist+1 < grid(a1,b1,c1)) {
				    grid(a1,b1,c1)=dist+1;
				    closestNodeIdx(a1,b1,c1).remove_all();
				}
				closestNodeIdx(a1,b1,c1).add(p);
				visited(a1,b1,c1)=1;
				q.push(a1); q.push(b1); q.push(c1);
			    }
			}
		    }
	}
    }
}

// build a queue with all of the neighbors, until we touch a cell with
// a grid() distance of 0.  save all of the cells on paths back to the
// original cell.

void ManhattanDist::computeCellDistance(int i, int j, int k) {
    if (closestNodeIdx(i,j,k).size()) return;

    // build the distance grid until we find a surface cell

    Array3<int> cellDist(nx,ny,nz);
    cellDist.initialize(-1);
    queue<int> q;
    q.push(i); q.push(j); q.push(k);
    cellDist(i,j,k)=0;
    int done=0;
    while(!q.empty() && !done) {
        const int a = q.front(); q.pop();
        const int b = q.front(); q.pop();
        const int c = q.front(); q.pop();
        if (grid(a,b,c)==0) {    // we're done!  just update the path...
            done=1;
        } else {
            int dist=cellDist(a,b,c);
	    int a1, b1, c1;	
	    for (a1=a-1; a1<=a+1; a1++)
		for (b1=b-1; b1<=b+1; b1++)
		    for (c1=c-1; c1<=c+1; c1++) {
			if (a1>=0 && a1<nx && b1>=0 && b1<ny && c1>=0 && 
			    c1<nz && cellDist(a1,b1,c1)==-1) {
                            cellDist(a1,b1,c1)=dist+1;
			    q.push(a1); q.push(b1); q.push(c1);
			}
		    }
        }
    }
    if (!done) {
        cerr << "Error: q empty, but no surface cells found!\n";
	return;
    } 
    int sa=a;
    int sb=b;
    int sc=c;
    int gd=cellDist(sa,sb,sc);

    // sa, sb, sc is the closest surface node.  update cells along the path
    
    queue<int> q1, q2;
    queue<int> *oldQ, *newQ, *tempQ;
    oldQ = &q1; newQ = &q2;
    oldQ->push(sa); oldQ->push(sb); oldQ->push(sc);
    int cd=gd+1;
    done=0;
    while (cd>0) {
	cd--;
	while(!oldQ->empty()) {
            a = oldQ->front(); oldQ->pop();
	    b = oldQ->front(); oldQ->pop();
	    c = oldQ->front(); oldQ->pop();
            cellDist(a,b,c)=-1;
            closestNodeIdx(a,b,c).add(sa);
	    closestNodeIdx(a,b,c).add(sb);
	    closestNodeIdx(a,b,c).add(sc);
            if ((closestNodeIdx(a,b,c).size()!=3) && (grid(a,b,c)!=gd-cd))
                cerr << "Uhoh -- old dist="<<grid(a,b,c)<<" new dist="<<gd-cd<<"\n";
	    grid(a,b,c)=gd-cd;
            if (a==i && b==j && c==k) {
		done=1;
	    } else {
		if (cd==0) cerr << "Shouldn't be here!!!!\n";
		int a1, b1, c1;	
		for (a1=a-1; a1<=a+1; a1++)
		    for (b1=b-1; b1<=b+1; b1++)
			for (c1=c-1; c1<=c+1; c1++) {
			    if (a1>=0 && a1<nx && b1>=0 && b1<ny && c1>=0 && 
				c1<nz && cellDist(a1,b1,c1)==cd-1) {
				newQ->push(a1); newQ->push(b1); newQ->push(c1);
				cellDist(a1,b1,c1)=-1;
			    }
			}
	    }
        }
        tempQ=oldQ; oldQ=newQ; newQ=tempQ;
    }
}

int ManhattanDist::distFast(const Point& p) {
    int i, j, k;
    locate(p, i, j, k);
    if (i>=grid.dim1()) i=grid.dim1()-1;
    if (i<0) i=0;
    if (j>=grid.dim2()) j=grid.dim2()-1;
    if (j<0) j=0;
    if (k>=grid.dim3()) k=grid.dim3()-1;
    if (k<0) k=0;
    if (!closestNodeIdx(i,j,k).size())
        computeCellDistance(i,j,k);
    return grid(i,j,k);
}

double ManhattanDist::dist2(const Point& p) {
    int q;
    return dist2(p,q);
}

double ManhattanDist::dist2(const Point& p, int &idx) {
    int i, j, k;
    locate(p, i, j, k);
    if (i>=grid.dim1()) i=grid.dim1()-1;
    if (i<0) i=0;
    if (j>=grid.dim2()) j=grid.dim2()-1;
    if (j<0) j=0;
    if (k>=grid.dim3()) k=grid.dim3()-1;
    if (k<0) k=0;
    if (!closestNodeIdx(i,j,k).size())
        computeCellDistance(i,j,k);
    Array1<int> nds(closestNodeIdx(i,j,k));
    double dist=Vector(p-pts[nds[0]]).length2();
    idx=nds[0];
    for (int a=1; a<nds.size(); a++) {
	double tmp=Vector(p-pts[nds[a]]).length2();
	if (tmp<dist) {
	    dist=tmp;
	    idx=nds[a];
	}
    }
    return dist;
}

double ManhattanDist::dist(const Point& p) {
    int q;
    return dist(p,q);
}

double ManhattanDist::dist(const Point& p, int &idx) {
    int i, j, k;
    locate(p, i, j, k);
    if (i>=grid.dim1()) i=grid.dim1()-1;
    if (i<0) i=0;
    if (j>=grid.dim2()) j=grid.dim2()-1;
    if (j<0) j=0;
    if (k>=grid.dim3()) k=grid.dim3()-1;
    if (k<0) k=0;
    if (!closestNodeIdx(i,j,k).size())
        computeCellDistance(i,j,k);
    Array1<int> nds(closestNodeIdx(i,j,k));
    double dist=Vector(p-pts[nds[0]]).length();
    idx=nds[0];
    for (int a=1; a<nds.size(); a++) {
	double tmp=Vector(p-pts[nds[a]]).length();
	if (tmp<dist) {
	    cerr << "(was "<<idx<<" now "<<nds[a]<<")";
	    dist=tmp;
	    idx=nds[a];
	}
    }
    return dist;
}
    
#define ManhattanDist_VERSION 1

void ManhattanDist::io(Piostream& stream) {

    int version=stream.begin_class("ManhattanDist", ManhattanDist_VERSION);
    if (version == 1) {
        ScalarFieldRGint::io(stream);
	Pio(stream, closestNodeIdx);
	Pio(stream, pts);
	Pio(stream, fullyInitialized);
    }
    stream.end_class();
}

ScalarField* ManhattanDist::clone()
{
    return scinew ManhattanDist(*this);
}
} // End namespace DaveW


