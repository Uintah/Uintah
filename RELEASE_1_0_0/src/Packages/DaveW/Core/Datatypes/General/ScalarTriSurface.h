
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

#ifndef SCI_Packages_DaveW_Datatypes_ScalarTriSurface_h
#define SCI_Packages_DaveW_Datatypes_ScalarTriSurface_h 1

#include <Core/Datatypes/TriSurface.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <stdlib.h> // For size_t

namespace DaveW {
using namespace SCIRun;


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
} // End namespace DaveW



#endif
