
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

#ifndef SCI_Datatypes_BasicSurfaces_h
#define SCI_Datatypes_BasicSurfaces_h 1

#include <SCICore/share/share.h>

#include <SCICore/Datatypes/Surface.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Containers/Array1.h>

namespace SCICore {
namespace Datatypes {

using Containers::Array1;
using Geometry::Point;
using Geometry::Vector;

class SCICORESHARE CylinderSurface : public Surface {
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

class SCICORESHARE PointSurface : public Surface {
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

class SCICORESHARE SphereSurface : public Surface {
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

class SCICORESHARE PointsSurface : public Surface {
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

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:30  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:43  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:33  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

#endif /* SCI_Datatypes_BasicSurfaces_h */

