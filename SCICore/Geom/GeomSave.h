
/*
 *  GeomSave.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef GeomSave_h
#define GeomSave_h 1

#include <Geometry/Vector.h>
#include <Geometry/Point.h>

#ifdef KCC
#include <iosfwd.h>  // Forward declarations for KCC C++ I/O routines
#else
class ostream;
#endif

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

struct GeomSave {
    int nindent;
    void indent();
    void unindent();
    void indent(ostream&);

    // For VRML.
    void start_sep(ostream&);
    void end_sep(ostream&);
    void start_tsep(ostream&);
    void end_tsep(ostream&);
    void orient(ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    // For RIB.
    void start_attr(ostream&);
    void end_attr(ostream&);
    void start_trn(ostream&);
    void end_trn(ostream&);
    void rib_orient(ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    void translate(ostream&, const Point& p);
    void rotateup(ostream&, const Vector& up, const Vector& new_up);
    void start_node(ostream&, char*);
    void end_node(ostream&);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:43  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:07  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:01  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif
