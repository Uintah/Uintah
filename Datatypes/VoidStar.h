
/*
 *  VoidStar.h: Just has a rep member -- other trivial classes can inherit
 *		from this, rather than having a full-blown datatype and data-
 *		port for every little thing that comes along...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_Datatypes_VoidStar_h
#define SCI_Datatypes_VoidStar_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/String.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Datatypes/Datatype.h>
#include <Multitask/ITC.h>

class VoidStar;
typedef LockingHandle<VoidStar> VoidStarHandle;
class VoidStar : public Datatype {
protected:
    VoidStar();
public:
    VoidStar(const VoidStar& copy);
    virtual ~VoidStar();
    virtual VoidStar* clone()=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
