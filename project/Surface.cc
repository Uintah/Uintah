/*
 *  Surface.cc: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#define SCI_ASSERTION_LEVEL 3
#include <Classlib/Array1.h>
#undef SCI_ASSERTION_LEVEL
#define SCI_ASSERTION_LEVEL 2
#undef ASSERTL3
#define ASSERTL3(condition)
#include <Surface.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>
#include <Classlib/Assert.h>

PersistentTypeID Surface::typeid("Surface", "Datatype", 0);
static Persistent* make_TriSurface()
{
    return new TriSurface;
}
PersistentTypeID TriSurface::typeid("TriSurface", "Surface", make_TriSurface);

Surface::Surface() {
}

Surface::~Surface() {
}

Surface::Surface(const Surface& copy) {
}

#define SURFACE_VERSION 1

void Surface::io(Piostream& stream) {
    int version=stream.begin_class("Surface", SURFACE_VERSION);
    Pio(stream, name);
    stream.end_class();
}

TriSurface::TriSurface() :empty_index(-1), ordered_cw(0) {
}

TriSurface::TriSurface(const TriSurface& t) {
}

TriSurface::~TriSurface() {
}

int TriSurface::inside(const Point& p) {
    return 1;
}

ObjGroup* TriSurface::getGeomFromSurface() {
    ObjGroup* group = new ObjGroup;
    for (int i=0; i<elements.size(); i++) {
	group->add(new Triangle(points[elements[i]->i1], 
				points[elements[i]->i2],
				points[elements[i]->i3]));
    }
    return group;
}

void TriSurface::add_point(const Point& p) {
    points.add(p);
}

int TriSurface::add_triangle(int i1, int i2, int i3) {
    int temp;
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
    if (temp==22229)
	temp=22229;
    return temp;
}

double TriSurface::distance(const Point &p, int el, int *type) {
    Point a(points[elements[el]->i1]);
    Point b(points[elements[el]->i2]);
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

    int inside=0;	// so far we're not inside
    double alpha, beta;
    int inter=0;

    if (u1==0) {
        beta=u0/u2;
        if ((beta >= 0.) && (beta <= 1.)) {
            alpha = (v0-beta*v2)/v1;
            inter=((alpha>=0.) && ((alpha+beta)<=1.));
        }       
    } else {
        beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
        if ((beta >= 0.)&&(beta<=1.)) {            
            alpha=(u0-beta*u2)/u1;
            inter=((alpha>=0.) && ((alpha+beta)<=1.));
        }
    }
    *type = 0;
    if (inter) return Abs(t);

    // we know the point is outside of the triangle (i.e. the distance is
    // *not* simply the distance along the normal from the point to the
    // plane).  so, we find out which of the six areas it's in by checking
    // which lines it's "inside" of, and which ones it's "outside" of.

    // first find the slopes and the intercepts of the lines.
    // since i[1] and i[2] are the projected components, we'll just use those
    //	   as x and y.
    
    // now form our projected vectors for edges, such that a known inside
    //     point is in fact inside.

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
    for (X=0; X<3; X++)
	for (int j=0; j<3; j++) 
	    if (A[X][j]*V[(X+2-j)%3][i[1]]+B[X][j]*V[(X+2-j)%3][i[2]]+C[X][j] 
		< 0) {
		A[X][j]*=-1; B[X][j]*=-1; C[X][j]*=-1;
	    }
    
    // we'll use the variable out to tell us if we're "outside" of that
    // edge.  
    int out[3][3];
    for (X=0; X<3; X++)
	for (int j=0; j<3; j++)
	    out[X][j]=(A[X][j]*P[i[1]]+B[X][j]*P[i[2]]+C[X][j] < 0); 

    if (out[2][0] && out[1][1]) { *type=1; return ((p-a).length()); }
    if (out[0][0] && out[2][1]) { *type=2; return ((p-b).length()); }
    if (out[1][0] && out[0][1]) { *type=3; return ((p-c).length()); }

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
	    double dtemp=Abs(Cross(v1, v2).length()/v2.length());
	    if (dtemp>theDist) {
		theDist=dtemp;
		*type=X+4;
	    }
	}
    return theDist;
}
    
void TriSurface::remove_triangle(int i) {
    // if there hasn't been a triangle added since the last one was deleted
    // then we need to start deleting.  Otherwise, we're probably merging
    // contours, so just setting the empty_index is fine.
    if (empty_index != -1) {
	ASSERT(!"Shouldn't be here either!");
	elements.remove(i);
	elements.remove(empty_index);
	empty_index=-1;
    } else {
	empty_index=i;
    }
}

#define TRISURFACE_VERSION 1

void TriSurface::io(Piostream& stream) {
    int version=stream.begin_class("TriSurface", TRISURFACE_VERSION);
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
