
/*
 *  ScalarTriSurface.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_ScalarTriSurface_h
#define SCI_DaveW_Datatypes_ScalarTriSurface_h 1

#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <stdlib.h> // For size_t

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::Geometry::Point;

using namespace SCICore::Datatypes;

class ScalarTriSurface : public TriSurface {
public:
    Array1<double> data;
public:
    ScalarTriSurface();
    ScalarTriSurface(const ScalarTriSurface& copy);
    ScalarTriSurface(const TriSurface& ts, const Array1<double>& d);
    ScalarTriSurface(const TriSurface& ts);
    virtual ~ScalarTriSurface();
    virtual Surface* clone();
    virtual GeomObj* get_obj(const ColorMapHandle&);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:47:34  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:00  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:04  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
