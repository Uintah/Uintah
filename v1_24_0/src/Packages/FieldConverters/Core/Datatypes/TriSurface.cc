/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  TriSurfFieldace.cc: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifdef _WIN32
#pragma warning(disable:4291) // quiet the visual C++ compiler
#endif

#include <FieldConverters/Core/Datatypes/TriSurfFieldace.h>

#include <Core/Util/Assert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/TrivialAllocator.h>
#include <FieldConverters/Core/Datatypes/SurfTree.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
using std::cerr;

#include <queue>
using std::queue;

#ifdef _WIN32
#include <stdlib.h>
#define drand48() rand()
#endif

namespace FieldConverters {

static TrivialAllocator TSElement_alloc(sizeof(TSElement));

void* TSElement::operator new(size_t)
{
    return TSElement_alloc.alloc();
}

void TSElement::operator delete(void* rp, size_t)
{
    TSElement_alloc.free(rp);
}

static TrivialAllocator TSEdge_alloc(sizeof(TSEdge));

void* TSEdge::operator new(size_t)
{
    return TSEdge_alloc.alloc();
}

void TSEdge::operator delete(void* rp, size_t)
{
    TSEdge_alloc.free(rp);
}

static Persistent* make_TriSurfFieldace()
{
    return scinew TriSurfFieldace;
}

PersistentTypeID TriSurfFieldace::type_id("TriSurfFieldace", "Surface", make_TriSurfFieldace);

TriSurfFieldace::TriSurfFieldace(Representation r)
  : Surface(r, 0),
    valType(NodeType),
    normType(NrmlsNone),
    haveNodeInfo(0),
    empty_index(-1),
    directed(1)
{
}

TriSurfFieldace::TriSurfFieldace(const TriSurfFieldace& copy, Representation)
  : Surface(copy)
{
    points=copy.points;
    elements=Array1<TSElement*>(copy.elements.size());
    for (int i=0; i<elements.size(); i++)
	elements[i]=new TSElement(*(copy.elements[i]));
//    elements=copy.elements;
    bcIdx=copy.bcIdx;
    bcVal=copy.bcVal;
    haveNodeInfo=copy.haveNodeInfo;
    nodeNbrs=copy.nodeNbrs;
    normType=copy.normType;
    valType=copy.valType;
    normals=copy.normals;
}

TriSurfFieldace& TriSurfFieldace::operator=(const TriSurfFieldace& t)
{
    points=t.points;
    elements=Array1<TSElement*>(t.elements.size());
    for (int i=0; i<t.elements.size(); i++)
	elements[i]=new TSElement(*(t.elements[i]));
    bcIdx=t.bcIdx;
    bcVal=t.bcVal;
    haveNodeInfo=t.haveNodeInfo;
    nodeNbrs=t.nodeNbrs;
    normType=t.normType;
    valType=t.valType;
    normals=t.normals;
    return *this;
}

TriSurfFieldace::~TriSurfFieldace() {
}

int TriSurfFieldace::inside(const Point&)
{
    NOT_FINISHED("TriSurfFieldace::inside");
    return 1;
}

void TriSurfFieldace::order_faces() {
    if (elements.size() == 0) 
	directed=1;
    else {
	ASSERTFAIL("Can't order faces yet!");
    }
}

void TriSurfFieldace::add_point(const Point& p) {
    points.add(p);
}

void TriSurfFieldace::buildNormals(NormalsType nt) {

  normals.resize(0);
  normType=NrmlsNone;

    // build per vertex, per point or per element normals

    // point->point     (return)                1
    // point->element   x-product		
    // point->vertex    copy			3
    // point->none      (throw away)		2

    // element->point	average			
    // element->element (return)		1
    // element->vertex  copy			4
    // element->none    (throw away)		2
    
    // vertex->point    average			
    // vertex->element  x-product		
    // vertex->vertex   (return)		1
    // vertex->none	(throw away)		2
    
    // none->point      x-products		
    // none->element    x-products		
    // none->vertex     average of x-products	
    // none->none       (return)		1

    if (normType==nt) return;						// 1
    if (nt==NrmlsNone) {normals.resize(0); normType=nt; return;}	// 2
    if (normType==PointType && nt==VertexType) {			// 3
	// we want normals at the vertices, we have them at each point...
	Array1<Vector> old(normals);
	normals.resize(elements.size()*3);
	for (int i=0; i<elements.size(); i++) {
	    TSElement *e=elements[i];
	    normals[i*3]=old[e->i1];
	    normals[i*3+1]=old[e->i2];
	    normals[i*3+2]=old[e->i3];
	}
	normType=VertexType;
	return;
    }
    if (normType==ElementType && nt==VertexType) {			// 4
	// we want normals at the vertices, we have them at the elements...
	normals.resize(elements.size()*3);
	for (int i=normals.size()/3; i>=0; i--) {
	    normals[i]=normals[i/3];
	}
	normType=VertexType;
    }

    // store/compute the vertex normals in tmp, then we'll compute from those
    Array1<Vector> tmp(elements.size()*3);
    if (normType==VertexType && nt==PointType) {
	tmp=normals;
    } else if (normType==ElementType && nt==PointType) {
	for (int i=0; i<elements.size()*3; i++)
	    tmp[i*3]=tmp[i*3+1]=tmp[i*3+2]=normals[i];
    } else {
	// for the rest of these we'll build a temp array of x-products at the
	// elements
	for (int i=0; i<elements.size(); i++) {
	    TSElement *e=elements[i];
	    Vector v(Cross((points[e->i1]-points[e->i2]), 
			   (points[e->i1]-points[e->i3])));
	    v.normalize();
	    tmp[i*3]=tmp[i*3+1]=tmp[i*3+2]=v;
	}
    }

    // now, for those that want them at the vertices, we copy;
    // for those that want them at the elements, we grab one;
    // and for those that want them at the points, we average.
    if (nt==VertexType) {
	normals=tmp;
    } else if (nt==ElementType) {
	normals.resize(elements.size());
	for (int i=0; i<elements.size(); i++)
	    normals[i]=tmp[i*3];
    } else {
	normals.resize(points.size());
	normals.initialize(Vector(0,0,0));

	int i;
//	for (i=0; i<tmp.size(); i++)
//	    cerr << "tmp["<<i<<"]="<<tmp[i]<<"\n";
	for (i=0; i<elements.size(); i++) {
	    TSElement *e=elements[i];
	    normals[e->i1]+=tmp[i*3];
	    normals[e->i2]+=tmp[i*3+1];
	    normals[e->i3]+=tmp[i*3+2];
	}
	for (i=0; i<points.size(); i++) {
	    if (normals[i].length2()) normals[i].normalize();
	    else {
		cerr << "Error -- normalizing a zero vector to (1,0,0)\n";
		normals[i].x(1);
	    }
	}
    }
//    int i;
//    for (i=0; i<points.size(); i++)
//	cerr << "p["<<i<<"]="<<points[i]<<"\n";
//    for (i=0; i<normals.size(); i++)
//	cerr << "n["<<i<<"]="<<normals[i]<<"\n";
    normType=nt;
}

void TriSurfFieldace::buildNodeInfo() {
    if (haveNodeInfo) return;
    haveNodeInfo=1;
    nodeNbrs.resize(points.size());
    nodeElems.resize(points.size());

    int i;
    for (i=0; i<points.size(); i++) {
	nodeElems[i].resize(0);
	nodeNbrs[i].resize(0);
    }

    TSElement *e;
    int i1, i2, i3;

    for (int elemIdx=0; elemIdx<elements.size(); elemIdx++) {
	e=elements[elemIdx];
	i1=e->i1;
	i2=e->i2;
	i3=e->i3;
	int found;
	int k;
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

int TriSurfFieldace::get_closest_vertex_id(const Point &p1, const Point &p2,
				      const Point &p3) {
    if (grid==0) {
	ASSERTFAIL("Can't run TriSurfFieldace::get_closest_vertex_id() w/o a grid\n");
    }
    int i[3], j[3], k[3];	// grid element indices containing these points
    int maxi, maxj, maxk, mini, minj, mink;
    grid->get_element(p1, &(i[0]), &(j[0]), &(k[0]));
    grid->get_element(p2, &(i[1]), &(j[1]), &(k[1]));
    grid->get_element(p3, &(i[2]), &(j[2]), &(k[2]));

    maxi=Max(i[0],i[1],i[2]); mini=Min(i[0],i[1],i[2]);
    maxj=Max(j[0],j[1],j[2]); minj=Min(j[0],j[1],j[2]);
    maxk=Max(k[0],k[1],k[2]); mink=Min(k[0],k[1],k[2]);

    int rad=Max(maxi-mini,maxj-minj,maxk-mink)/2;
    int ci=(maxi+mini)/2; int cj=(maxj+minj)/2; int ck=(maxk+mink)/2;

//    BBox bb;
//    bb.extend(p1); bb.extend(p2); bb.extend(p3);

    // We make a temporary surface which just contains this one triangle,
    // so we can use the existing Surface->distance code to find the closest
    // vertex to this triangle.

    TriSurfFieldace* surf=scinew TriSurfFieldace;
    surf->construct_grid(grid->dim1(), grid->dim2(), grid->dim3(), 
			 grid->get_min(), grid->get_spacing());
    surf->add_point(p1);
    surf->add_point(p2);
    surf->add_point(p3);
    surf->add_triangle(0,1,2);

    Array1<int> cu;
    for(;;) {
	double dist=0;
	int vid=-1;
	grid->get_cubes_at_distance(rad,ci,cj,ck, cu);
	for (int i=0; i<cu.size(); i+=3) {
	    Array1<int> &el=*grid->get_members(cu[i], cu[i+1], cu[i+2]);
	    if (&el) {
		for (int j=0; j<el.size(); j++) {
		    Array1<int> res;
		    double tdist=surf->distance(points[elements[el[j]]->i1],
						res);
		    if (vid==-1 || (Abs(tdist) < Abs(dist))) {
			vid=elements[el[j]]->i1;
			dist=tdist;
		    }
		}
	    }
	}
	if (vid != -1) return vid;
	rad++;
    }
}

int TriSurfFieldace::find_or_add(const Point &p) {
  if (pntHash==0) {
    points.add(p);
    return(points.size()-1);
  }
  //int oldId;
  MapIntInt::iterator oldId;
  int val=(Round((p.z()-hash_min.z())/resolution)*hash_y+
    Round((p.y()-hash_min.y())/resolution))*hash_x+
    Round((p.x()-hash_min.x())/resolution);
  //if (pntHash->lookup(val, oldId)) {
  oldId = pntHash->find(val);
  if (oldId != pntHash->end()) {
    return (*oldId).second;
  } else {
    //pntHash->insert(val, points.size());
    (*pntHash)[val] = points.size();
    points.add(p);	
    return(points.size()-1);
  }
}

int TriSurfFieldace::cautious_add_triangle(const Point &p1, const Point &p2, 
				       const Point &p3, int cw) {
    directed&=cw;
    if (grid==0) {
	ASSERTFAIL("Can't run TriSurfFieldace::cautious_add_triangle w/o a grid\n");
    }
    int i1=find_or_add(p1);
    int i2=find_or_add(p2);
    int i3=find_or_add(p3);
    return (add_triangle(i1,i2,i3,cw));
}

int TriSurfFieldace::add_triangle(int i1, int i2, int i3, int cw) {
    directed&=cw;
    int temp;
    if (i1==i2 || i1==i3)	// don't even add degenerate triangles
	return -1;
    if (empty_index == -1) {
	elements.add(new TSElement(i1, i2, i3));
	temp = elements.size()-1;
    } else {
	elements[empty_index]->i1=i1;
	elements[empty_index]->i2=i2;
	elements[empty_index]->i3=i3;
	temp=empty_index;
	empty_index=-1;
    }

    //if we have a grid add this triangle to it
    if (grid) grid->add_triangle(temp, points[i1], points[i2], points[i3]);

    return temp;
}

void TriSurfFieldace::separate(int idx, TriSurfFieldace* conn, TriSurfFieldace* d_conn, int updateConnIndices, int updateDConnIndices) {
    if (idx<0 || idx>points.size()) {
	cerr << "Trisurface:separate() failed -- index out of range";
	return;
    }

    // brute force -- every node has its neighbor nodes in this list... twice!
    Array1<Array1<int> > nbr(points.size());
    Array1<int> newLocation(points.size());
    Array1<int> d_newLocation(points.size());
    Array1<int> visited(points.size());

    int i;
    for (i=0; i<points.size(); i++) {
	newLocation[i]=d_newLocation[i]=-1;
	visited[i]=0;
    }
    for (i=0; i<points.size(); i++) {
	nbr[i].resize(0);
    }
    for (i=0; i<elements.size(); i++) {
	nbr[elements[i]->i1].add(elements[i]->i2);	
	nbr[elements[i]->i1].add(elements[i]->i3);	
	nbr[elements[i]->i2].add(elements[i]->i1);	
	nbr[elements[i]->i2].add(elements[i]->i3);	
	nbr[elements[i]->i3].add(elements[i]->i1);	
	nbr[elements[i]->i3].add(elements[i]->i2);	
    }
    queue<int> q;
    q.push(idx);
    visited[idx]=1;
    while(!q.empty()) {
	// enqueue non-visited neighbors
	int c=q.front();
	q.pop();
	for (int j=0; j<nbr[c].size(); j++) {
	    if (!visited[nbr[c][j]]) {
		q.push(nbr[c][j]);
		visited[nbr[c][j]] = 1;
	    }
	}
    }

    if (updateConnIndices) conn->points.resize(0); else conn->points=points; 
    if (updateDConnIndices)d_conn->points.resize(0);else d_conn->points=points;

    int tmp;
    if (updateConnIndices || updateDConnIndices) {
	for (i=0; i<visited.size(); i++) {
	    if (visited[i]) {
		if (updateConnIndices) {
		    tmp = conn->points.size();
		    conn->points.add(points[i]);	
		    newLocation[i] = tmp;
		}
	    } else {
		if (updateDConnIndices) {
		    tmp = d_conn->points.size();		
		    d_conn->points.add(points[i]);
		    d_newLocation[i] = tmp;
		}
	    }
	}
    }
// GOOD CODE FOR STEVE'S TEST FUNCTIONS!
//    for (i=0; i<points.size(); i++) {
//	if (!visited[i]) {
//	    if (newLocation[i] != -1) {
//		cerr << "Error: " << i << " was not visited, but its mapped to "<<newLocation[i]<<"\n";
//	    }
//	} else {
//	    if (d_newLocation[i] != -1) {
//		cerr << "Error: " << i << " was visited, but its d_mapped to "<<d_newLocation[i]<<"\n";
//	    }
//	}
//   }
    for (i=0; i<elements.size(); i++)
	if (visited[elements[i]->i1]) {
	    if (updateConnIndices)
		conn->elements.add(new 
				   TSElement(newLocation[elements[i]->i1],
					     newLocation[elements[i]->i2],
					     newLocation[elements[i]->i3]));
	    else
		conn->elements.add(new TSElement(elements[i]->i1,
						 elements[i]->i2,
						 elements[i]->i3));
	} else {
	    if (updateDConnIndices)
		d_conn->elements.add(new 
			     TSElement(d_newLocation[elements[i]->i1],
				       d_newLocation[elements[i]->i2],
				       d_newLocation[elements[i]->i3]));
	    else
		d_conn->elements.add(new 
				     TSElement(elements[i]->i1,
					       elements[i]->i2,
					       elements[i]->i3));
	}
}		    

void TriSurfFieldace::remove_empty_index() {
    if (empty_index!=-1) {
	elements.remove(empty_index);
	empty_index=-1;
    }
}

void TriSurfFieldace::construct_grid(int xdim, int ydim, int zdim, 
				const Point &min, double spacing) {
    remove_empty_index();
    if (grid) delete grid;
    grid = scinew Grid(xdim, ydim, zdim, min, spacing);
    for (int i=0; i<elements.size(); i++)
	grid->add_triangle(i, points[elements[i]->i1],
			   points[elements[i]->i2], points[elements[i]->i3]);
}

void TriSurfFieldace::construct_hash(int xdim, int ydim, const Point &p, double res) {
    xdim=(int)(xdim/res);
    ydim=(int)(ydim/res);
    hash_x = xdim;
    hash_y = ydim;
    hash_min = p;
    resolution = res;
    if (pntHash) {
	delete pntHash;
    }
    //pntHash = scinew HashTable<int, int>;
    pntHash = scinew MapIntInt;
    for (int i=0; i<points.size(); i++) {
	int val=(Round((points[i].z()-p.z())/res)*ydim+
		 Round((points[i].y()-p.y())/res))*xdim+
		     Round((points[i].x()-p.x())/res);
	//pntHash->insert(val, i);
	(*pntHash)[val] = i;
    }
}

// Method to find the distance from a point to the surface.  the algorithm
// goes like this:
// find which "cube" of the mesh the point is in; look for nearest neighbors;
// determine if it's closest to a vertex, edge, or face, and store
// the type in "type" and the info (i.e. edge #, vertex #, triangle #, etc)
// in "res" (result); finally, return the distance.
// the information in res will be stored thus...
// if the thing we're closest to is a:
// 	face   -- [0]=triangle index
//	edge   -- [0]=triangle[1] index
//	  	  [1]=triangle[1] edge #
//		  [2]=triangle[2] index
//		  [3]=triangle[2] edge #
//	vertex -- [0]=triangle[1] index
//		  [1]=triangle[1] vertex #
//		   ...

double TriSurfFieldace::distance(const Point &p,Array1<int> &res, Point *pp) {

    if (grid==0) {
	ASSERTFAIL("Can't run TriSurfFieldace::distance w/o a grid\n");
    }
    Array1<int>* elem;
    Array1<int> tri;
    int i, j, k, imax, jmax, kmax;
    
    double dmin;
    double sp=grid->get_spacing();
    grid->get_element(p, &i, &j, &k, &dmin);
    grid->size(&imax, &jmax, &kmax);
    imax--; jmax--; kmax--;
    int dist=0;
    int done=0;
    double Dist=1000000;
    Array1<int> info;
    int type;
    int max_dist=Max(imax,jmax,kmax);

    Array1<int> candid;
    while (!done) {
	while (!tri.size()) {
	    grid->get_cubes_within_distance(dist, i, j, k, candid);
	    for(int index=0; index<candid.size(); index+=3) {
		elem=grid->get_members(candid[index], candid[index+1], 
				       candid[index+2]);
		if (elem) {
		    for (int a=0; a<elem->size(); a++) {
			int duplicate;
			int b;
			for (duplicate=0, b=0; b<tri.size(); b++)
			    if (tri[b]==(*elem)[a]) duplicate=1;
			if (!duplicate) tri.add((*elem)[a]);
		    }
		}
	    }
	    dist++;
	    ASSERT(dist<=max_dist+2);
	}
	// now tri holds the indices of the triangles we're closest to

	Point ptemp;
	
	for (int index=0; index<tri.size(); index++) {
	    double d=distance(p, tri[index], &type, &ptemp);
	    if (Abs(d-Dist)<.00001) {
		if (type==0) {
		    info.remove_all();
		    Dist=d;
		    if (pp!=0) (*pp)=ptemp;
		    info.add(tri[index]);
		} else {
		    if (res.size() != 1) {
			info.add(tri[index]);
			info.add((type-1)%3);
			if (pp!=0) (*pp)=ptemp;
		    }
		}
	    } else if (Abs(d)<Abs(Dist)) {
		info.remove_all();
		Dist=d;
		if (pp!=0) *pp=ptemp;
		info.add(tri[index]);
		if (type>0) info.add((type-1)%3);
	    }
	}

	tri.remove_all();

	// if our closest point is INSIDE of the squares we looked at...
	if (Abs(Dist)<(dmin+(dist-1)*sp)) 
	    done=1;		// ... we're done
	else
	    info.remove_all();
    }
	
    res=info;
    return Dist;
}


// This is basically the ray/triangle interesect code from the ray-tacing
// chapter in Graphics Gems I.  Much of the c-code comes from page 735 --
// thanks to Didier Badouel.  Other parts, and a discussion of the algorithm
// were presented on pages 390-393.

// We return the signed distance from the point to the triangle "el",  
// and put the type of intersection (face, edge or vertex) into the
// variable type.  If we're closest to...
//	the face, then *type=0
//	an edge,  then *type=1+vertex# (that we're furthest from)
//      a vertex, then *type=4+vertex# (that we're closest to)

double TriSurfFieldace::distance(const Point &p, int el, int *type, Point *pp) {
    Point a(points[elements[el]->i1]);	//load the vertices of this element...
    Point b(points[elements[el]->i2]);  //... into a, b and c
    Point c(points[elements[el]->i3]);
    double V[3][3];	// our array of vertices
    V[0][0]=a.x(); V[0][1]=a.y(); V[0][2]=a.z();
    V[1][0]=b.x(); V[1][1]=b.y(); V[1][2]=b.z();
    V[2][0]=c.x(); V[2][1]=c.y(); V[2][2]=c.z();

    Vector e(a-b);
    Vector f(a-c);
    Vector N(Cross(e,f));
    N.normalize();
    double d=-(a.x()*N.x()+a.y()*N.y()+a.z()*N.z());
    double t=-(d+Dot(N, Vector(p-Point(0,0,0))));
    int sign=Sign(t);
    Point Pp(p+N*t);

    double P[3]; // our point on the plane
    P[0]=Pp.x(); P[1]=Pp.y(); P[2]=Pp.z();

    int i[3];	// order the normal components backwards by magnitude
    if (Abs(N.x()) > Abs(N.y())) {
	if (Abs(N.x()) > Abs(N.z())) {
	    i[0]=0; i[1]=1; i[2]=2;
	} else {
	    i[0]=2; i[1]=0; i[2]=1;
	}
    } else {
	if (Abs(N.y()) > Abs(N.z())) {
	    i[0]=1; i[1]=0; i[2]=2;
	} else {
	    i[0]=2; i[1]=0; i[2]=1;
	}
    }

    int I=2;	// because we only have a triangle
    double u0=P[i[1]]-V[0][i[1]];
    double v0=P[i[2]]-V[0][i[2]];
    double u1=V[I-1][i[1]]-V[0][i[1]];
    double v1=V[I-1][i[2]]-V[0][i[2]];
    double u2=V[I][i[1]]-V[0][i[1]];
    double v2=V[I][i[2]]-V[0][i[2]];

    double alpha, beta;
    int inter=0;

    if (Abs(u1)<.0001) {
        beta=u0/u2;
        if ((beta >= -0.0001) && (beta <= 1.0001)) {
            alpha = (v0-beta*v2)/v1;
            if ((inter=((alpha>=-0.0001) && ((alpha+beta)<=1.0001)))) {
		if (pp!=0) {
		    (*pp)=Pp;
		}
	    }
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= -0.000001)&&(beta<=1.0001)) {            
            alpha=(u0-beta*u2)/u1;
            if ((inter=((alpha>=-0.0001) && ((alpha+beta)<=1.0001)))) {
		if (pp!=0) {
		    (*pp)=Pp;
		}
	    }
        }
    }
    *type = 0;
    if (inter) return t;

    // we know the point is outside of the triangle (i.e. the distance is
    // *not* simply the distance along the normal from the point to the
    // plane).  so, we find out which of the six areas it's in by checking
    // which lines it's "inside" of, and which ones it's "outside" of.

    // first find the slopes and the intercepts of the lines.
    // since i[1] and i[2] are the projected components, we'll just use those
    //	   as x and y.
    
    // now form our projected vectors for edges, such that a known inside
    //     point is in fact inside.

    // Note: this could be optimized to take advantage of the surface being
    //     directed.  Wouldn't need to check signs, etc.

    double A[3][3], B[3][3], C[3][3];
    //double mid[2];
    //mid[0]=(V[0][i[1]]+V[1][i[1]]+V[2][i[1]])/3;
    //mid[1]=(V[0][i[2]]+V[1][i[2]]+V[2][i[2]])/3;

    int X;
    for (X=0; X<3; X++) {
	// handle vertical lines
	if (V[(X+2)%3][i[1]]==V[(X+1)%3][i[1]]) { 		// x=const.
	    B[X][2]=0;
	    // primary edge
	    A[X][2]=1; C[X][2]=-V[(X+2)%3][i[1]];
	    // perpindicular segments
	    A[X][1]=A[X][0]=0; B[X][0]=1; B[X][1]=-1;
	    C[X][0]=-V[(X+1)%3][i[2]]; C[X][1]=V[(X+2)%3][i[2]];
	} else {						// not vertical
	    B[X][2]=1; A[X][2]=-((V[(X+2)%3][i[2]]-V[(X+1)%3][i[2]])/
				 (V[(X+2)%3][i[1]]-V[(X+1)%3][i[1]]));
	    C[X][2]=-B[X][2]*V[(X+2)%3][i[2]]-A[X][2]*V[(X+2)%3][i[1]];
	    if (A[X][2]==0) {
		A[X][0]=-1; A[X][1]=1;
		B[X][0]=B[X][1]=0;
	    } else {		 
		B[X][0]=-1; B[X][1]=1;
		A[X][0]=1/A[X][2];
		A[X][1]=-A[X][0];
	    }
	    C[X][0]=-B[X][0]*V[(X+1)%3][i[2]]-A[X][0]*V[(X+1)%3][i[1]];
	    C[X][1]=-B[X][1]*V[(X+2)%3][i[2]]-A[X][1]*V[(X+2)%3][i[1]];
	}
    }
    // now make sure we have all of the signs right!
    for (X=0; X<3; X++){
	for (int j=0; j<3; j++) 
	    if (A[X][j]*V[(X+2-j)%3][i[1]]+B[X][j]*V[(X+2-j)%3][i[2]]+C[X][j] 
		< 0) {
		A[X][j]*=-1; B[X][j]*=-1; C[X][j]*=-1;
	    }
    }
    
    // we'll use the variable out to tell us if we're "outside" of that
    // edge.  
    int out[3][3];
    for (X=0; X<3; X++){
	for (int j=0; j<3; j++)
	    out[X][j]=(A[X][j]*P[i[1]]+B[X][j]*P[i[2]]+C[X][j] < 0); 
    }

    if (out[2][0] && out[1][1]) { 
	*type=1; 
	if (pp) (*pp)=a;
	return (sign*(p-a).length()); 
    }
    if (out[0][0] && out[2][1]) { 
	*type=2; 
	if (pp) (*pp)=b;
	return (sign*(p-b).length()); 
    }
    if (out[1][0] && out[0][1]) { 
	*type=3; 
	if (pp) (*pp)=c;
	return (sign*(p-c).length()); 
    }

    ASSERT(out[0][2] || out[1][2] || out[2][2]);
    double theDist=-100;
    for (X=0; X<3; X++)
	if (out[X][2]) {
	    // take twice the area of the triangle formed between the two
	    // end vertices and this point, then divide by the length
	    // of the edge...this gives the distance.  The area is twice
	    // the length of the cross-product of two edges.
	    Vector v1(V[(X+1)%3][0]-p.x(),
		      V[(X+1)%3][1]-p.y(),
		      V[(X+1)%3][2]-p.z());
	    Vector v2(V[(X+1)%3][0]-V[(X+2)%3][0],
		      V[(X+1)%3][1]-V[(X+2)%3][1],
		      V[(X+1)%3][2]-V[(X+2)%3][2]);
	    double dtemp=Sqrt(Cross(v1, v2).length2()/v2.length2());
	    if (dtemp>theDist) {
		theDist=dtemp;
		*type=X+4;
		if (pp)
		    (*pp)=AffineCombination(Point(V[(X+1)%3][0],
						  V[(X+1)%3][1],
						  V[(X+1)%3][2]), .5,
					    Point(V[(X+2)%3][0],
						  V[(X+2)%3][1],
						  V[(X+2)%3][2]), .5);
	    }
	}
    return sign*theDist;
}

static void orderNormal(int i[], const Vector& v) {
    if (Abs(v.x())>Abs(v.y())) {
        if (Abs(v.y())>Abs(v.z())) {  // x y z
            i[0]=0; i[1]=1; i[2]=2;
        } else if (Abs(v.z())>Abs(v.x())) {   // z x y
            i[0]=2; i[1]=0; i[2]=1;
        } else {                        // x z y
            i[0]=0; i[1]=2; i[2]=1;
        }
    } else {
        if (Abs(v.x())>Abs(v.z())) {  // y x z
            i[0]=1; i[1]=0; i[2]=2;
        } else if (Abs(v.z())>Abs(v.y())) {   // z y x
            i[0]=2; i[1]=1; i[2]=0;
        } else {                        // y z x
            i[0]=1; i[1]=2; i[2]=0;
        }
    }
}       
    
int TriSurfFieldace::intersect(const Point& origin, const Vector& dir, double &d, int &v, int face)
{
    double P[3], t, alpha, beta;
    double u0,u1,u2,v0,v1,v2;
    int i[3];
    double V[3][3];
    int inter;

    TSElement* e=elements[face];
    Point p1(points[e->i1]);
    Point p2(points[e->i2]);
    Point p3(points[e->i3]);

    Vector n(Cross(p2-p1, p3-p1));
    n.normalize();
    
    double dis=-Dot(n,p1);
    t=-(dis+Dot(n,origin))/Dot(n,dir);
    if (t<0) return 0;
    if (d!=-1 && t>d) return 0;

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

    P[0]= origin.x()+dir.x()*t;
    P[1]= origin.y()+dir.y()*t;
    P[2]= origin.z()+dir.z()*t;

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
    if (!inter) return 0;

    if (alpha<beta && alpha<(1-(alpha+beta))) v=e->i1;
    else if (beta<alpha && beta<(1-(alpha+beta))) v=e->i2;
    else v=e->i3;
    
    d=t;
    return (1);
}

void TriSurfFieldace::remove_triangle(int i) {
    // if there hasn't been a triangle added since the last one was deleted
    // then we need to start deleting.  Otherwise, we're probably merging
    // contours, so just setting the empty_index is fine.

    // if we have a grid remove this triangle from it
    if (grid) 
	grid->remove_triangle(i,points[elements[i]->i1],
			      points[elements[i]->i2],points[elements[i]->i3]);

    // if we don't have an empty index lying around...
    if (empty_index != -1) {
	elements.remove(i);
	elements.remove(empty_index);
	empty_index=-1;
    // else make this the empty index -- hopefully someone will fill it
    } else {
	empty_index=i;
    }
}

#define TRISURFACE_VERSION 5

void TriSurfFieldace::io(Piostream& stream) {
    remove_empty_index();
    int version=stream.begin_class("TriSurfFieldace", TRISURFACE_VERSION);
    Surface::io(stream);
    if (version >= 2) {
      SCIRun::Pio(stream, bcIdx);
      SCIRun::Pio(stream, bcVal);
    }
    if (version >= 5) {
      int *flag=(int*)&valType;
      SCIRun::Pio(stream, *flag);
    } else valType=NodeType;

    if (version >= 4) {
	int* flag=(int*)&normType;
	SCIRun::Pio(stream, *flag);
	if (normType != NrmlsNone) 
	  SCIRun::Pio(stream, normals);
    } else if (version >= 3) {
	int haveNormals;
	SCIRun::Pio(stream, haveNormals);
	if (haveNormals) {
	    normType=VertexType;
	    SCIRun::Pio(stream, normals);
	} else {
	    normType=NrmlsNone;
	}
    }
    SCIRun::Pio(stream, points);
    SCIRun::Pio(stream, elements);
    stream.end_class();
}

SurfTree* TriSurfFieldace::toSurfTree() {
    SurfTree* st=new SurfTree;

    st->surfI.resize(1);
    st->faces=elements;
    st->nodes=points;
    for (int i=0; i<elements.size(); i++) {
	st->surfI[0].faces.add(i);
	st->surfI[0].faceOrient.add(i);
    }
    st->surfI[0].name=name;
    st->surfI[0].inner.add(0);
    st->surfI[0].outer=0;
    st->surfI[0].matl=0;
    st->typ=SurfTree::NodeValuesSome;
    st->data=bcVal;
    st->idx=bcIdx;
    return st;
}

Surface* TriSurfFieldace::clone()
{
    return scinew TriSurfFieldace(*this);
}

void TriSurfFieldace::construct_grid()
{
    BBox bbox;
    for(int i=0;i<points.size();i++){
	bbox.extend(points[i]);
    }
    Vector d(bbox.diagonal());
    double volume=d.x()*d.y()*d.z();
    // Make it about 10000 elements...
    int ne=10000;
    double spacing=Cbrt(volume/ne);
    int nx=RoundUp(d.x()/spacing);
    int ny=RoundUp(d.y()/spacing);
    int nz=RoundUp(d.z()/spacing);
    construct_grid(nx, ny, nz, bbox.min(), spacing);
}

#if 0
void TriSurfFieldace::get_surfnodes(Array1<NodeHandle> &n)
{
    for (int i=0; i<points.size(); i++) {
	n.add(new Node(points[i]));
    }
}

void TriSurfFieldace::set_surfnodes(const Array1<NodeHandle> &n)
{
    if (n.size() != points.size()) {
	cerr << "TriSurfFieldace::set_surfnodes - wrong sized array!\n";
	return;
    }
    for (int i=0; i<points.size(); i++)
	points[i]=n[i]->p;
}
#endif

GeomObj* TriSurfFieldace::get_obj(const ColorMapHandle&)
{
    NOT_FINISHED("TriSurfFieldace::get_obj");
    return 0;
}

void TriSurfFieldace::compute_samples(int nsamp)
{
    samples.remove_all();
    weights.remove_all();

    samples.resize(nsamp);
    weights.resize(elements.size());

    for(int i=0;i<elements.size();i++) {
	if (elements[i]) {
	    TSElement* me = elements[i];
	    Vector v1(points[me->i2]-points[me->i1]);
	    Vector v2(points[me->i3]-points[me->i1]);

	    weights[i] = Cross(v1,v2).length()*0.5;
	} else {
	    weights[i] = 0.0;
	}
    }
}

Point RandomPoint(Point& p1, Point& p2, Point& p3)
{
    double alpha,beta;

    alpha = sqrt(drand48());
    beta = drand48();

    return AffineCombination(p1,1-alpha,
			     p2,alpha-alpha*beta,
			     p3,alpha*beta);
}

void TriSurfFieldace::distribute_samples()
{
  double total_importance =0.0;
  Array1<double> psum(weights.size());
  
  int i;
  for(i=0;i<elements.size();i++) {
    if (elements[i]) {
      total_importance += weights[i];
      psum[i] = total_importance;
    } else {
      psum[i] = -1;  // bad, so it will just skip over this...
    }
  }

  // now just jump into the prefix sum table...
  // this is a bit faster, especialy initialy...

  int pi=0;
  int nsamp = samples.size();
  double factor = 1.0/(nsamp-1)*total_importance;

  for(i=0;i<nsamp;i++) {
    double val = (i*factor);
    while ( (pi < weights.size()) && 
           (psum[pi] < val))
      pi++;
    if (pi == weights.size()) {
      cerr << "Over flow!\n";
    } else {
	samples[i] = RandomPoint(points[elements[pi]->i1],
				 points[elements[pi]->i2],
				 points[elements[pi]->i3]);
    }
  }
}

} // end namespace FieldConverters;

namespace SCIRun {
void Pio(Piostream& stream, FieldConverters::TSElement*& data)
{
    if(stream.reading())
	data=new FieldConverters::TSElement(0,0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->i1);
    Pio(stream, data->i2);
    Pio(stream, data->i3);
    stream.end_cheap_delim();
}

void Pio(Piostream& stream, FieldConverters::TSEdge*& data)
{
    if(stream.reading())
	data=new FieldConverters::TSEdge(0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->i1);
    Pio(stream, data->i2);
    stream.end_cheap_delim();
}

} // End namespace SCIRun

