
/*
 *  SurfTree.cc: Tree of non-manifold bounding surfaces
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
  *
 *  Copyright (C) 1997 SCI Group
 */
#include <iostream.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/SurfTree.h>
#include <Geometry/BBox.h>
#include <Geometry/Grid.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <Malloc/Allocator.h>

static Persistent* make_SurfTree()
{
    return scinew SurfTree;
}

PersistentTypeID SurfTree::type_id("SurfTree", "Surface", make_SurfTree);

SurfTree::SurfTree(Representation r)
: Surface(r, 0)
{
}

SurfTree::SurfTree(const SurfTree& copy, Representation)
: Surface(copy)
{
    NOT_FINISHED("SurfTree::SurfTree");
}

SurfTree::~SurfTree() {
}	

int SurfTree::inside(const Point&)
{
    NOT_FINISHED("SurfTree::inside");
    return 1;
}

void SurfTree::construct_grid() {
    NOT_FINISHED("SurfTree::construct_grid()");
    return;
}

void SurfTree::construct_grid(int, int, int, const Point &, double) {
    NOT_FINISHED("SurfTree::construct_grid");
    return;
}

void SurfTree::construct_hash(int, int, const Point &, double) {
    NOT_FINISHED("SurfTree::construct_hash");
    return;
}

void order (Array1<int>& a) {
    int swap=1;
    int tmp;
    while (swap) {
	swap=0;
	for (int i=0; i<a.size()-1; i++)
	    if (a[i]>a[i+1]) {
		tmp=a[i];
		a[i]=a[i+1];
		a[i+1]=tmp;
		swap=1;
	    }
    }
}

inline int getIdx(const Array1<int>& a, int size) {
    int val=0;
    for (int k=a.size()-1; k>=0; k--)
	val = val*size + a[k];
    return val;
}

void getList(Array1<int>& surfList, int i, int size) {
    int valid=1;
//    cerr << "starting list...\n";
    while (valid) {
	surfList.add(i%size);
//	cerr << "just added "<<i%size<<" to the list.\n";
	if (i >= size) valid=1; else valid=0;
	i /= size;
    }
 //   cerr << "done with list.\n";
}

// call this before outputting persistently.  that way we have the Arrays
// for VTK Decimage algorithm.
void SurfTree::SurfsToTypes() {
    int i,j;

    cerr << "We have "<<elements.size()<<" elements.\n";
    // make a list of what surfaces each elements is attached to
    Array1<Array1<int> > elemMembership(elements.size());
//    cerr << "Membership.size()="<<elemMembership.size()<<"\n";
    for (i=0; i<surfEls.size(); i++)
	for (j=0; j<surfEls[i].size(); j++)
	    elemMembership[surfEls[i][j]].add(i);
    int maxMembership=1;

    // sort all of the lists from above, and find the maximum number of
    // surfaces any element belongs to
    for (i=0; i<elemMembership.size(); i++)
	if (elemMembership[i].size() > 1) {
	    if (elemMembership[i].size() > maxMembership)
		maxMembership=elemMembership[i].size();
	    order(elemMembership[i]);
	}
    int sz=pow(surfEls.size(), maxMembership);
    cerr << "allocating "<<maxMembership<<" levels with "<< surfEls.size()<< " types (total="<<sz<<").\n";

    // allocate all combinations of the maximum number of surfaces
    // from the lists of which surfaces each element belong to,
    // construct a list of elements which belong for each surface
    // combination

    Array1 <Array1<int> > tmpTypeMembers(pow(surfEls.size(), maxMembership));
    cerr << "Membership.size()="<<elemMembership.size()<<"\n";
    for (i=0; i<elemMembership.size(); i++) {
//	cerr << "  **** LOOKING IT UP!\n";
	int idx=getIdx(elemMembership[i], surfEls.size());
//	cerr << "this elements has index: "<<idx<<"\n";
	tmpTypeMembers[idx].add(i);
    }
    typeSurfs.resize(0);
    typeIds.resize(elements.size());
    for (i=0; i<tmpTypeMembers.size(); i++) {
	// if there are any elements of this combination type...
	if (tmpTypeMembers[i].size()) {
	    // find out what surfaces there were	
//	    cerr << "found "<<tmpTypeMembers[i].size()<<" elements of type "<<i<<"\n";
	    Array1<int> surfList;
	    getList(surfList, i, surfEls.size());
	    int currSize=typeSurfs.size();
	    typeSurfs.resize(currSize+1);
	    typeSurfs[currSize].resize(0);
//	    cerr << "here's the array: ";
	    for (j=0; j<tmpTypeMembers[i].size(); j++) {
//		cerr << tmpTypeMembers[i][j]<<" ";
		typeIds[tmpTypeMembers[i][j]]=currSize;
	    }
//	    cerr << "\n";
//	    cerr << "copying array ";
//	    cerr << "starting to add elements...";
	    for (j=0; j<surfList.size(); j++) {
		typeSurfs[currSize].add(surfList[j]);
//		cerr << ".";
	    }
//	    cerr << "   done!\n";
	}
    }
    cerr << "done with SurfsToTypes!!\n";
}

// call this after VTK Decimate has changed the Elements and points -- need
// to rebuild typeSurfs information
void SurfTree::TypesToSurfs() {
    int i,j;
//    cerr << "building surfs from types...\n";
    for (i=0; i<surfEls.size(); i++) {
	surfEls[i].resize(0);
    }
//    cerr << "typeSurfs.size() = "<<typeSurfs.size()<<"\n";
//    cerr << "surfEls.size() = "<<surfEls.size()<<"\n";
    for (i=0; i<typeIds.size(); i++) {
//	cerr << "working on typeSurfs["<<typeIds[i]<<"\n";
	for (j=0; j<typeSurfs[typeIds[i]].size(); j++) {
//	    cerr << "adding "<<i<<" to surfEls["<<typeSurfs[typeIds[i]][j]<<"]\n";
	    surfEls[typeSurfs[typeIds[i]][j]].add(i);
	}
    }
}

#define SurfTree_VERSION 2

void SurfTree::io(Piostream& stream) {
    int version=stream.begin_class("SurfTree", SurfTree_VERSION);
    Surface::io(stream);		    
    if (version >= 2) {
	if (stream.writing() && !surfNames.size()) 
	    surfNames.resize(surfEls.size());
	Pio(stream, surfNames);		    
	Pio(stream, bcIdx);
	Pio(stream, bcVal);
    }
    Pio(stream, surfEls);
    Pio(stream, elements);
    Pio(stream, points);
    Pio(stream, matl);
    Pio(stream, outer);
    Pio(stream, inner);
    Pio(stream, typeSurfs);
    Pio(stream, typeIds);
    stream.end_class();
}

Surface* SurfTree::clone()
{
    return scinew SurfTree(*this);
}

void SurfTree::get_surfnodes(Array1<NodeHandle> &n)
{
    for (int i=0; i<points.size(); i++) {
	n.add(new Node(points[i]));
    }
}

void SurfTree::get_surfnodes(Array1<NodeHandle>&n, clString name) {
    for (int s=0; s<surfNames.size(); s++)
	if (surfNames[s] == name) break;
    if (s == surfNames.size()) {
	cerr << "ERROR: Coudln't find surface: "<<name()<<"\n";
	return;
    }

    // allocate all of the Nodes -- make the ones from other surfaces void
    Array1<int> member(points.size());
    member.initialize(0);
    for (int i=0; i<surfEls[s].size(); i++) {
	TSElement *e = elements[surfEls[s][i]];
	member[e->i1]=1; member[e->i2]=1; member[e->i3]=1;
    }

    for (i=0; i<points.size(); i++) {
	if (member[i]) n.add(new Node(points[i]));
	else n.add((Node*)0);
    }
}

GeomObj* SurfTree::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("SurfTree::get_obj");
    return 0;
}
