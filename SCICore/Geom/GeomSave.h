
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

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <iosfwd>

namespace SCICore {
namespace GeomSpace {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

struct GeomSave {
    int nindent;
    void indent();
    void unindent();
    void indent(std::ostream&);

    // For VRML.
    void start_sep(std::ostream&);
    void end_sep(std::ostream&);
    void start_tsep(std::ostream&);
    void end_tsep(std::ostream&);
    void orient(std::ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    // For RIB.
    void start_attr(std::ostream&);
    void end_attr(std::ostream&);
    void start_trn(std::ostream&);
    void end_trn(std::ostream&);
    void rib_orient(std::ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    void translate(std::ostream&, const Point& p);
    void rotateup(std::ostream&, const Vector& up, const Vector& new_up);
    void start_node(std::ostream&, char*);
    void end_node(std::ostream&);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/10/07 02:07:44  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/08/17 06:39:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
