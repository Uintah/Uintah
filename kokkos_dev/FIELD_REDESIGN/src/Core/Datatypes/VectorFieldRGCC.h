
/*
 *  VectorFieldRGCC.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_VectorFieldRGCC_h
#define SCI_project_VectorFieldRGCC_h 1

#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>

namespace SCICore {
namespace Datatypes {

using SCICore::Containers::Array1;
using SCICore::Containers::Array3;

class SCICORESHARE VectorFieldRGCC : public VectorFieldRG {
public:
    Point get_point(int, int, int);
    virtual void locate(const Point&, int&, int&, int&);

    VectorFieldRGCC();
    virtual ~VectorFieldRGCC();
    virtual VectorField* clone();

    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);

    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

#endif
