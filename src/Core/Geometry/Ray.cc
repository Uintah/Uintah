//static char *id="@(#) $Id$";

/*
 *  Ray.cc:  The Ray datatype
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Geometry/Ray.h>

namespace SCICore {
namespace Geometry {

Ray::Ray(const Point& o, const Vector& d)
: o(o), d(d)
{
}

Ray::Ray(const Ray& copy)
: o(copy.o), d(copy.d)
{
}

Ray::~Ray()
{
}

Ray& Ray::operator=(const Ray& copy)
{
    o=copy.o;
    d=copy.d;
    return *this;
}

Point Ray::origin() const
{
    return o;
}

Vector Ray::direction() const
{
    return d;
}

void Ray::direction(const Vector& newdir)
{
    d=newdir;
}

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:56  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//
