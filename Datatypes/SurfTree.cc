
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

using sci::NodeHandle;
using sci::Node;

static Persistent* make_SurfTree()
{
    return scinew SurfTree;
}

PersistentTypeID SurfTree::type_id("SurfTree", "Surface", make_SurfTree);

SurfTree::SurfTree(Representation r)
: Surface(r, 0), haveNodeInfo(0), valid_bboxes(0)
{
}

SurfTree::SurfTree(const SurfTree& copy, Representation)
: Surface(copy), haveNodeInfo(0), valid_bboxes(0)
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

void SurfTree::printNbrInfo() {
    if (!haveNodeInfo) {
	cerr << "No nbr info yet!\n";
	return;
    }
    for (int i=0; i<nodeNbrs.size(); i++) {
	cerr << "("<<i<<") "<< points[i]<<" nbrs:";
	for (int j=0; j<nodeNbrs[i].size(); j++) {
	    cerr << " "<<points[nodeNbrs[i][j]];
	}
	cerr << "\n";
    }
}

// map will be the mapping from a tree idx to a tri index --
// imap will be a mapping from a tri index to a tree index.
int SurfTree::extractTriSurface(TriSurface* ts, Array1<int>& map, 
				Array1<int>& imap, int comp) {
    map.resize(0);
    imap.resize(0);
    if (comp>surfNames.size()) {
	cerr << "Error: bad surface idx "<<comp<<"\n";
	ts=0;
	return 0;
    }

    map.resize(points.size());
    map.initialize(-1);
    cerr << "Extracting component #"<<comp<<" with "<<surfEls[comp].size()<<" elements...\n";
    for (int i=0; i<surfEls[comp].size(); i++) {
	map[elements[surfEls[comp][i]]->i1]=
	map[elements[surfEls[comp][i]]->i2]=
	map[elements[surfEls[comp][i]]->i3]=1;
    }

    ts->elements.resize(surfEls[comp].size());
    ts->points.resize(0);

    int currIdx=0;
    for (i=0; i<map.size(); i++) {
	if (map[i] != -1) {
	    imap.add(i);
	    map[i]=currIdx;
	    ts->points.add(points[i]);
	    currIdx++;
	}
    }

    for (i=0; i<surfEls[comp].size(); i++) {
//	cerr << "surfOrient["<<comp<<"]["<<i<<"]="<<surfOrient[comp][i]<<"\n";
	TSElement *e=elements[surfEls[comp][i]];
	if (surfOrient.size()>comp && surfOrient[comp].size()>i && 
	    !surfOrient[comp][i])
	    ts->elements[i]=new TSElement(map[e->i1], map[e->i3], map[e->i2]);
	else
	    ts->elements[i]=new TSElement(map[e->i1], map[e->i2], map[e->i3]);
    }

    ts->name = surfNames[comp];
    for (i=0; i<bcIdx.size(); i++) {
	if (map[bcIdx[i]] != -1) {
	    ts->bcIdx.add(map[bcIdx[i]]);
	    ts->bcVal.add(bcVal[i]);
	}
    }

    cerr << "surface "<<ts->name<<" has "<<ts->points.size()<<" points, "<<ts->elements.size()<<" elements and "<<ts->bcVal.size()<<" known vals.\n";

    return 1;
}

void SurfTree::bldNormals() {

    // go through each surface.  for each one, look at each element.
    // compute the normal of the element and add it to the normal of each
    // of its nodes.

    if (haveNormals) return;
    if (!haveNodeInfo) bldNodeInfo();
    nodeNormals.resize(surfEls.size());

    for (int i=0; i<surfEls.size(); i++) {
	nodeNormals[i].resize(points.size());
	nodeNormals[i].initialize(Vector(0,0,0));
    }

    for (i=0; i<surfEls.size(); i++) {
	for (int j=0; j<surfEls[i].size(); j++) {
	    int sign=1;
	    if (surfOrient.size()>i && surfOrient[i].size()>j && 
		!surfOrient[i][j]) sign=-1;
	    TSElement *e=elements[surfEls[i][j]];
	    Vector v(Cross((points[e->i1]-points[e->i2]), 
			   (points[e->i1]-points[e->i3]))*sign);
	    nodeNormals[i][e->i1]+=v;
	    nodeNormals[i][e->i2]+=v;
	    nodeNormals[i][e->i3]+=v;
	}
    }

    // gotta go through and normalize all the normals

    for (i=0; i<nodeSurfs.size(); i++) {	// for each node
	for (int j=0; j<nodeSurfs[i].size(); j++) {  // for each surf it's on
	    if (nodeNormals[j][i].length2())
		nodeNormals[j][i].normalize();
	}
    }
}

void SurfTree::bldNodeInfo() {
    if (haveNodeInfo) return;
    haveNodeInfo=1;

#if 0
    for (int a=0; a<surfEls.size(); a++) {
	Array1<Array1<int> > nodeE;
	Array1<Array1<int> > nodeN;
	nodeN.resize(points.size());
	nodeE.resize(points.size());
	for (int b=0; b<surfEls[a].size(); b+=2) {
	    int s,t,u,v,w,x;
	    TSElement *e1, *e2;
	    w=surfEls[a][b];
	    e1=elements[w];
	    x=surfEls[a][b+1];
	    e2=elements[x];
	    s=e1->i1;
	    t=e1->i2;
	    u=e1->i3;
	    v=e2->i2;
	    int kk;
	    int found1, found2, found3;
	    found1=found2=found3=0;
	    for (kk=0; kk<nodeN[s].size(); kk++)
		if (nodeN[s][kk] == t) found1=1; 
		else if (nodeN[s][kk] == u) found2=1;
		else if (nodeN[s][kk] == v) found3=1;
	    if (!found1) nodeN[s].add(t);
	    if (!found2) nodeN[s].add(u);
	    if (!found3) nodeN[s].add(v);

	    found1=found2=found3=0;
	    for (kk=0; kk<nodeN[t].size(); kk++)
		if (nodeN[t][kk] == s) found1=1; 
		else if (nodeN[t][kk] == u) found2=1;
		else if (nodeN[t][kk] == v) found3=1;
	    if (!found1) nodeN[t].add(s);
	    if (!found2) nodeN[t].add(u);
	    if (!found3) nodeN[t].add(v);

	    found1=found2=found3=0;
	    for (kk=0; kk<nodeN[u].size(); kk++)
		if (nodeN[u][kk] == s) found1=1; 
		else if (nodeN[u][kk] == t) found2=1;
		else if (nodeN[u][kk] == v) found3=1;
	    if (!found1) nodeN[u].add(s);
	    if (!found2) nodeN[u].add(t);
	    if (!found3) nodeN[u].add(v);

	    found1=found2=found3=0;
	    for (kk=0; kk<nodeN[v].size(); kk++)
		if (nodeN[v][kk] == s) found1=1; 
		else if (nodeN[v][kk] == t) found2=1;
		else if (nodeN[v][kk] == u) found3=1;
	    if (!found1) nodeN[v].add(s);
	    if (!found2) nodeN[v].add(t);
	    if (!found3) nodeN[v].add(u);

	    found1=found2=0;
	    for (kk=0; kk<nodeE[s].size(); kk++)
		if (nodeE[s][kk] == w) found1=1; 
		else if (nodeE[s][kk] == x) found2=1;
	    if (!found1) nodeE[s].add(w);
	    if (!found2) nodeE[s].add(x);

	    found1=0;
	    for (kk=0; kk<nodeE[t].size(); kk++)
		if (nodeE[t][kk] == w) found1=1; 
	    if (!found1) nodeE[t].add(w);

	    found1=found2;
	    for (kk=0; kk<nodeE[u].size(); kk++)
		if (nodeE[u][kk] == w) found1=1; 
		else if (nodeE[u][kk] == x) found2=1;
	    if (!found1) nodeE[u].add(w);
	    if (!found2) nodeE[u].add(x);

	    found1=0;
	    for (kk=0; kk<nodeE[v].size(); kk++)
		if (nodeE[v][kk] == x) found1=1; 
	    if (!found1) nodeE[v].add(x);
 	}
	Array1<int> punctures;
	Array1<int> cracks;
	Array1<int> elevens;
	Array1<int> thirteens;
	Array1<int> fifteens;
	int flag=0;
	for (b=0; b<points.size(); b++) {
	    if (nodeN[b].size()>10) {
		flag=1;
		if (nodeN[b].size() == 12) punctures.add(b);
		else if (nodeN[b].size() == 11) elevens.add(b);
		else if (nodeN[b].size() == 14) cracks.add(b);
		else if (nodeN[b].size() == 13) thirteens.add(b);
		else if (nodeN[b].size() == 15) fifteens.add(b);
	    }
	}
	if (flag)
	    cerr << "Surface "<<a<<": found "<<elevens.size()<<" (11's), "<<punctures.size()<<" (12's), "<<thirteens.size()<<" (13's), "<<cracks.size()<<" (14's), and "<<fifteens.size()<<" (15's).\n";

	if (fifteens.size()) {
	}

    }

    // go through all the elements for each surface -- bld nbr info
    // if a node is connected to more than 4 nbrs, it's a join point
    // find the subsets of nbrs who don't have each other as nbrs --
    // add a copy of this point at the end of the node list.  now,
    // bld new elements for one of the sets, with the old node idx
    // replaced with the new idx (the last one in the node list).

#endif

    nodeSurfs.resize(points.size());
    nodeElems.resize(points.size());
    nodeNbrs.resize(points.size());
    for (int i=0; i<points.size(); i++) {
	nodeSurfs[i].resize(0);
	nodeElems[i].resize(0);
	nodeNbrs[i].resize(0);
    }

    TSElement *e;
    int i1, i2, i3;

    for (i=0; i<surfEls.size(); i++) {
	for (int j=0; j<surfEls[i].size(); j++) {
	    int elemIdx=surfEls[i][j];
	    e=elements[elemIdx];
	    i1=e->i1;
	    i2=e->i2;
	    i3=e->i3;
	    int found;
	    int k;
	    for (found=0, k=0; k<nodeSurfs[i1].size() && !found; k++)
		if (nodeSurfs[i1][k] == i) found=1;
	    if (!found) nodeSurfs[i1].add(i);
	    for (found=0, k=0; k<nodeSurfs[i2].size() && !found; k++)
		if (nodeSurfs[i2][k] == i) found=1;
	    if (!found) nodeSurfs[i2].add(i);
	    for (found=0, k=0; k<nodeSurfs[i3].size() && !found; k++)
		if (nodeSurfs[i3][k] == i) found=1;
	    if (!found) nodeSurfs[i3].add(i);
	    
	    for (found=0, k=0; k<nodeElems[i1].size() && !found; k++)
		if (nodeElems[i1][k] == elemIdx) found=1;
	    if (!found) nodeElems[i1].add(elemIdx);
	    for (found=0, k=0; k<nodeElems[i2].size() && !found; k++)
		if (nodeElems[i2][k] == elemIdx) found=1;
	    if (!found) nodeElems[i2].add(elemIdx);
	    for (found=0, k=0; k<nodeElems[i3].size() && !found; k++)
		if (nodeElems[i3][k] == elemIdx) found=1;
	    if (!found) nodeElems[i3].add(elemIdx);

	    for (found=0, k=0; k<nodeNbrs[i1].size() && !found; k++)
		if (nodeNbrs[i1][k] == i2) found=1;
	    if (!found) { 
		nodeNbrs[i1].add(i2);
		nodeNbrs[i2].add(i1);
	    }
	    for (found=0, k=0; k<nodeNbrs[i2].size() && !found; k++)
		if (nodeNbrs[i2][k] == i3) found=1;
	    if (!found) { 
		nodeNbrs[i2].add(i3);
		nodeNbrs[i3].add(i2);
	    }
	    for (found=0, k=0; k<nodeNbrs[i1].size() && !found; k++)
		if (nodeNbrs[i1][k] == i3) found=1;
	    if (!found) { 
		nodeNbrs[i1].add(i3);
		nodeNbrs[i3].add(i1);
	    }
	}
    }
    int tmp;
    for (i=0; i<nodeNbrs.size(); i++) {
	if (nodeNbrs[i].size()) {
	    int swapped=1;
	    while (swapped) {
		swapped=0;
		for (int j=0; j<nodeNbrs[i].size()-1; j++) {
		    if (nodeNbrs[i][j]>nodeNbrs[i][j+1]) {
			tmp=nodeNbrs[i][j];
			nodeNbrs[i][j]=nodeNbrs[i][j+1];
			nodeNbrs[i][j+1]=tmp;
			swapped=1;
		    }
		}
	    }
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
	surfOrient[i].resize(0);
    }
//    cerr << "typeSurfs.size() = "<<typeSurfs.size()<<"\n";
//    cerr << "surfEls.size() = "<<surfEls.size()<<"\n";
    for (i=0; i<typeIds.size(); i++) {
//	cerr << "working on typeSurfs["<<typeIds[i]<<"\n";
	for (j=0; j<typeSurfs[typeIds[i]].size(); j++) {
//	    cerr << "adding "<<i<<" to surfEls["<<typeSurfs[typeIds[i]][j]<<"]\n";
	    surfEls[typeSurfs[typeIds[i]][j]].add(i);
	    surfOrient[typeSurfs[typeIds[i]][j]].add(1);
	}
    }
}

void SurfTree::compute_bboxes() {
    valid_bboxes=1;
    bldNodeInfo();
    bboxes.resize(surfEls.size());
    for (int i=0; i<nodeSurfs.size(); i++)
	for (int j=0; j<nodeSurfs[i].size(); j++)
	    bboxes[nodeSurfs[i][j]].extend(points[i]);
}

void orderNormal(int i[], const Vector& v) {
    if (fabs(v.x())>fabs(v.y())) {
        if (fabs(v.y())>fabs(v.z())) {  // x y z
            i[0]=0; i[1]=1; i[2]=2;
        } else if (fabs(v.z())>fabs(v.x())) {   // z x y
            i[0]=2; i[1]=0; i[2]=1;
        } else {                        // x z y
            i[0]=0; i[1]=2; i[2]=1;
        }
    } else {
        if (fabs(v.x())>fabs(v.z())) {  // y x z
            i[0]=1; i[1]=0; i[2]=2;
        } else if (fabs(v.z())>fabs(v.y())) {   // z y x
            i[0]=2; i[1]=1; i[2]=0;
        } else {                        // y z x
            i[0]=1; i[1]=2; i[2]=0;
        }
    }
}       

// go through the elements in component comp and see if any of the triangles
// are closer then we've seen so far.
// have_hit indicates if we have a closest point,
// if so, compBest, elemBest and distBest have the information about that hit

void SurfTree::distance(const Point &p, int &have_hit, double &distBest, 
			int &compBest, int &elemBest, int comp) {
    
    double P[3], t, alpha, beta;
    double u0,u1,u2,v0,v1,v2;
    int i[3];
    double V[3][3];
    int inter;

    Vector dir(1,0,0);	// might want to randomize this?
    for (int ii=0; ii<surfEls[comp].size(); ii++) {
	TSElement* e=elements[surfEls[comp][ii]];
	Point p1(points[e->i1]);
	Point p2, p3;

	// orient the triangle correctly

	if (surfOrient[comp][ii]) {p2=points[e->i2]; p3=points[e->i3];}
	else {p2=points[e->i3]; p3=points[e->i2];}

	Vector n(Cross(p2-p1, p3-p1));
	n.normalize();
	
	double dis=-Dot(n,p1);
	t=-(dis+Dot(n,p))/Dot(n,dir);
	if (t<0) continue;
	if (have_hit && t>distBest) continue;

	V[0][0]=p1.x();
	V[0][1]=p1.y();
	V[0][2]=p1.z();
    
	V[1][0]=p2.x();
	V[1][1]=p2.y();
	V[1][2]=p2.z();

	V[2][0]=p3.x();
	V[2][1]=p3.y();
	V[2][2]=p3.z();

	orderNormal(i,n);

	P[0]= p.x()+dir.x()*t;
	P[1]= p.y()+dir.y()*t;
	P[2]= p.z()+dir.z()*t;

	u0=P[i[1]]-V[0][i[1]];
	v0=P[i[2]]-V[0][i[2]];
	inter=0;
	u1=V[1][i[1]]-V[0][i[1]];
	v1=V[1][i[2]]-V[0][i[2]];
	u2=V[2][i[1]]-V[0][i[1]];
	v2=V[2][i[2]]-V[0][i[2]];
	if (u1==0) {
	    beta=u0/u2;
	    if ((beta >= 0.) && (beta <= 1.)) {
		alpha = (v0-beta*v2)/v1;
		if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
	    }       
	} else {
	    beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
	    if ((beta >= 0.)&&(beta<=1.)) {
		alpha=(u0-beta*u2)/u1;
		if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
	    }
	}
	if (!inter) continue;
	if (t>0 && (!have_hit || t<distBest)) {
	    have_hit=1; compBest=comp; elemBest=ii; distBest=t;
	}
    }
}

int SurfTree::inside(const Point &p, int &component) {
    if (!valid_bboxes)
	compute_bboxes();

    Array1<int> candidate;

    for (int i=0; i<bboxes.size(); i++)
	if (bboxes[i].inside(p)) candidate.add(i);

    int have_hit=0;
    int compBest=0;
    int elemBest=0;	// we don't use this for inside()
    double distBest;
    for (i=0; i<candidate.size(); i++) {
	distance(p, have_hit, distBest, compBest, elemBest, candidate[i]);
    }
    return have_hit;
}
    
#define SurfTree_VERSION 3

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
    if (version >= 3) {
	Pio(stream, surfOrient);
	Pio(stream, haveNodeInfo);
	if (haveNodeInfo) {
	    Pio(stream, nodeSurfs);
	    Pio(stream, nodeElems);
	    Pio(stream, nodeNbrs);
	}
    } else {
	if (stream.reading()) {
	    surfOrient.resize(surfEls.size());
	}
	for (int i=0; i<surfEls.size(); i++) {
	    surfOrient[i].resize(surfEls[i].size());
	    surfOrient[i].initialize(1);
	}
	haveNodeInfo=0;
    }
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

void SurfTree::set_surfnodes(const Array1<NodeHandle> &n) {
    NOT_FINISHED("SurfTree::set_surfnodes");
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

void SurfTree::set_surfnodes(const Array1<NodeHandle>&n, clString name) {
    NOT_FINISHED("SurfTree::set_surfnodes");
}

GeomObj* SurfTree::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("SurfTree::get_obj");
    return 0;
}
