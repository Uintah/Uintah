
/*
 *  BasicSurfaces.h: Cylinders and stuff
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_CoreDatatypes_BasicSurfaces_h
#define SCI_CoreDatatypes_BasicSurfaces_h 1

#include <CoreDatatypes/Surface.h>
#include <CoreDatatypes/Mesh.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Containers/Array1.h>

namespace SCICore {
namespace CoreDatatypes {

using Containers::Array1;
using Geometry::Point;
using Geometry::Vector;

class CylinderSurface : public Surface {
    Point p1;
    Point p2;
    double radius;
    int nu;
    int nv;
    int ndiscu;

    Vector u;
    Vector v;

    Vector axis;
    double rad2;
    double height;
    void add_node(Array1<NodeHandle>& nodes,
		  char* id, const Point& p, double r, double rn,
		  double theta, double h, double hn);
public:
    CylinderSurface(const Point& p1, const Point& p2, double radius,
		    int nu, int nv, int ndiscu);
    CylinderSurface(const CylinderSurface&);
    virtual ~CylinderSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    virtual GeomObj* get_obj(const ColorMapHandle&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class PointSurface : public Surface {
    Point pos;
    void add_node(Array1<NodeHandle>& nodes,
		  char* id, const Point& p);
public:
    PointSurface(const Point& pos);
    PointSurface(const PointSurface&);
    virtual ~PointSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    virtual GeomObj* get_obj(const ColorMapHandle&);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SphereSurface : public Surface {
    Point cen;
    Vector pole;
    double radius;
    int nu;
    int nv;

    Vector u;
    Vector v;

    double rad2;
    void add_node(Array1<NodeHandle>& nodes,
		  char* id, const Point& p, double r,
		  double theta, double phi);
public:
    SphereSurface(const Point& cen, double radius, const Vector& pole,
		    int nu, int nv);
    SphereSurface(const SphereSurface&);
    virtual ~SphereSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    virtual GeomObj* get_obj(const ColorMapHandle&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class PointsSurface : public Surface {
public:
    Array1<Point> pos;
    Array1<double> val;
public:
    PointsSurface();
    PointsSurface(const Array1<Point>& pos, const Array1<double>& val);
    PointsSurface(const PointsSurface&);
    virtual ~PointsSurface();
    virtual Surface* clone();
    virtual int inside(const Point& p);
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();

    virtual GeomObj* get_obj(const ColorMapHandle&);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:18  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:45  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:35  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:26  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:33  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif /* SCI_CoreDatatypes_BasicSurfaces_h */

