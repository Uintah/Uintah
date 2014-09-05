#ifndef GFXGEOM_3D_INCLUDED // -*- C++ -*-
#define GFXGEOM_3D_INCLUDED

/*
 *  3D.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <gfx/math/Vec3.h>
#include <gfx/math/Vec4.h>
#include <gfx/tools/Array.h>

//
// Generally useful geometric functions
//
extern Vec3 randomPoint(const Vec3&, const Vec3&);  // on segment
extern Vec3 randomPoint(const Vec3&, const Vec3&, const Vec3&); // in triangle

extern real triangleArea(const Vec3&, const Vec3&, const Vec3&);
extern real triangleCompactness(const Vec3&, const Vec3&, const Vec3&);

class Bounds
{
public:

    Vec3 min, max;
    Vec3 center;
    real radius;
    unsigned int points;

    Bounds() { reset(); }

    void reset();
    void addPoint(const Vec3&);
    void complete();
};

class Plane
{
    //
    // A plane is defined by the equation:  n*p + d = 0
    Vec3 n;
    real d;

public:

    Plane() : n(0,0,1) { d=0; } // -- this will define the XY plane
    Plane(const Vec3& p, const Vec3& q, const Vec3& r) { calcFrom(p,q,r); }
    Plane(const array<Vec3>& verts) { calcFrom(verts); }
    Plane(const Plane& p) { n=p.n; d=p.d; }

    void calcFrom(const Vec3& p, const Vec3& q, const Vec3& r);
    void calcFrom(const array<Vec3>&);

    bool isValid() const { return n[X]!=0.0 || n[Y]!=0.0 || n[Z]!= 0.0; }
    void markInvalid() { n[X] = n[Y] = n[Z] = 0.0; }

    real distTo(const Vec3& p) const { return n*p + d; }
    const Vec3& normal() const { return n; }

    void coeffs(real *a, real *b, real *c, real *dd) const {
        *a=n[X]; *b=n[Y]; *c=n[Z]; *dd=d;
    }
    Vec4 coeffs() const { return Vec4(n,d); }
};


//
// A triangular face in 3D (ie. a 2-simplex in E3)
//
class Face3
{
protected:
    Plane P;

private:
    void recalcPlane() { P.calcFrom(vertexPos(0),vertexPos(1),vertexPos(2)); }
    void recalcPlane(const Vec3& a,const Vec3& b,const Vec3& c)
    { P.calcFrom(a,b,c); }

public:
    Face3(const Vec3& a,const Vec3& b,const Vec3& c)
	: P(a,b,c)
    { }

    //
    // Basic primitive operations on faces
    virtual const Vec3& vertexPos(int i) const = 0;
    virtual void vertexPos(int i, const Vec3&) = 0; 
    const Plane& plane() { if(!P.isValid()) recalcPlane();   return P;}
    void invalidatePlane() { P.markInvalid(); }

    real distTo(const Vec3& p) const;
    real area();
};

//
// $Log$
// Revision 1.1  1999/07/27 16:58:03  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:13  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//


#endif // GFXGEOM_3D_INCLUDED
