
/*
 *  TriSurface.cc: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */
#include <iostream.h>
#include <Datatypes/TriSurface.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Geometry/Grid.h>
#include <Math/MiscMath.h>
#include <Geometry/BBox.h>
#include <Classlib/TrivialAllocator.h>

static TrivialAllocator TSElement_alloc(sizeof(TSElement));

void* TSElement::operator new(size_t)
{
    return TSElement_alloc.alloc();
}

void TSElement::operator delete(void* rp, size_t)
{
    TSElement_alloc.free(rp);
}

static Persistent* make_TriSurface()
{
    return new TriSurface;
}

PersistentTypeID TriSurface::type_id("TriSurface", "Surface", make_TriSurface);

TriSurface::TriSurface()
: Surface(TriSurf, 1), empty_index(-1), directed(1)
{
}

TriSurface::TriSurface(const TriSurface& copy)
: Surface(copy)
{
    NOT_FINISHED("TriSurface::TriSurface");
}

TriSurface::~TriSurface() {
}

int TriSurface::inside(const Point&)
{
    NOT_FINISHED("TriSurface::inside");
    return 1;
}

void TriSurface::order_faces() {
    if (elements.size() == 0) 
	directed=1;
    else {
	ASSERTL1(0 && "Can't order faces yet!");
    }
}

void TriSurface::add_point(const Point& p) {
    points.add(p);
}

int TriSurface::get_closest_vertex_id(const Point &p1, const Point &p2,
				      const Point &p3) {
    if (grid==0) {
	ASSERT(!"Can't run TriSurface::get_closest_vertex_id() w/o a grid\n");
    }
    int i[3], j[3], k[3];
    int maxi, maxj, maxk, mini, minj, mink;
    grid->get_element(p1, &(i[0]), &(j[0]), &(k[0]));
    grid->get_element(p2, &(i[1]), &(j[1]), &(k[1]));
    grid->get_element(p3, &(i[2]), &(j[2]), &(k[2]));

    maxi=Max(i[0],i[1],i[2]); mini=Min(i[0],i[1],i[2]);
    maxj=Max(j[0],j[1],j[2]); minj=Min(j[0],j[1],j[2]);
    maxk=Max(k[0],k[1],k[2]); mink=Min(k[0],k[1],k[2]);

    int rad=Max(maxi-mini,maxj-minj,maxk-mink)/2;
    int ci=(maxi+mini)/2; int cj=(maxj+minj)/2; int ck=(maxk+mink)/2;

    BBox bb;
    bb.extend(p1); bb.extend(p2); bb.extend(p3);

    TriSurface* surf=new TriSurface;
    surf->construct_grid(grid->dim1(), grid->dim2(), grid->dim3(), 
			 grid->get_min(), grid->get_spacing());
    surf->add_point(p1);
    surf->add_point(p2);
    surf->add_point(p3);
    surf->add_triangle(0,1,2);

    Array1<int> cu;
    while (1) {
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

int TriSurface::find_or_add(const Point &p) {
    if (pntHash==0) {
	points.add(p);
	return(points.size()-1);
    }
    int oldId;
    int val=(Round((p.z()-hash_min.z())/resolution)*hash_y+
	     Round((p.y()-hash_min.y())/resolution))*hash_x+
		 Round((p.x()-hash_min.x())/resolution);
    if (pntHash->lookup(val, oldId)) {
	return oldId;
    } else {
	pntHash->insert(val, points.size());
	points.add(p);	
	return(points.size()-1);
    }
}

int TriSurface::cautious_add_triangle(const Point &p1, const Point &p2, 
				       const Point &p3, int cw) {
    directed&=cw;
    if (grid==0) {
	ASSERT("Can't run TriSurface::cautious_add_triangle w/o a grid\n");
    }
    int i1=find_or_add(p1);
    int i2=find_or_add(p2);
    int i3=find_or_add(p3);
    return (add_triangle(i1,i2,i3,cw));
}

int TriSurface::add_triangle(int i1, int i2, int i3, int cw) {
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

void TriSurface::remove_empty_index() {
    if (empty_index!=-1) {
	elements.remove(empty_index);
	empty_index=-1;
    }
}

void TriSurface::construct_grid(int xdim, int ydim, int zdim, 
				const Point &min, double spacing) {
    remove_empty_index();
    if (grid) delete grid;
    grid = new Grid(xdim, ydim, zdim, min, spacing);
    for (int i=0; i<elements.size(); i++)
	grid->add_triangle(i, points[elements[i]->i1],
			   points[elements[i]->i2], points[elements[i]->i3]);
}

void TriSurface::construct_hash(int xdim, int ydim, const Point &p, double res) {
    xdim/=res;
    ydim/=res;
    hash_x = xdim;
    hash_y = ydim;
    hash_min = p;
    resolution = res;
    if (pntHash) {
	delete pntHash;
    }
    pntHash = new HashTable<int, int>;
    for (int i=0; i<points.size(); i++) {
	int val=(Round((points[i].z()-p.z())/res)*ydim+
		 Round((points[i].y()-p.y())/res))*xdim+
		     Round((points[i].x()-p.x())/res);
	pntHash->insert(val, i);
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

double TriSurface::distance(const Point &p,Array1<int> &res, Point *pp) {

    if (grid==0) {
	ASSERT("Can't run TriSurface::distance w/o a grid\n");
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
			for (int duplicate=0, b=0; b<tri.size(); b++)
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

double TriSurface::distance(const Point &p, int el, int *type, Point *pp) {
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
            if (inter=((alpha>=-0.0001) && ((alpha+beta)<=1.0001))) {
		if (pp!=0) {
		    (*pp)=Pp;
		}
	    }
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= -0.000001)&&(beta<=1.0001)) {            
            alpha=(u0-beta*u2)/u1;
            if (inter=((alpha>=-0.0001) && ((alpha+beta)<=1.0001))) {
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
    double mid[2];
    mid[0]=(V[0][i[1]]+V[1][i[1]]+V[2][i[1]])/3;
    mid[1]=(V[0][i[2]]+V[1][i[2]]+V[2][i[2]])/3;

    for (int X=0; X<3; X++) {
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
    
void TriSurface::remove_triangle(int i) {
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

#define TRISURFACE_VERSION 1

void TriSurface::io(Piostream& stream) {
    remove_empty_index();
    /*int version=*/stream.begin_class("TriSurface", TRISURFACE_VERSION);
    Surface::io(stream);
    Pio(stream, points);
    Pio(stream, elements);
    stream.end_class();
}

void Pio(Piostream& stream, TSElement*& data)
{
    if(stream.reading())
	data=new TSElement(0,0,0);
    stream.begin_cheap_delim();
    Pio(stream, data->i1);
    Pio(stream, data->i2);
    Pio(stream, data->i3);
    stream.end_cheap_delim();
}

Surface* TriSurface::clone()
{
    return new TriSurface(*this);
}

void TriSurface::construct_grid()
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

void TriSurface::get_surfpoints(Array1<Point>&)
{
    NOT_FINISHED("TriSurface::get_surfpoints");
}
