
/*
 *  PIDLObject: SCIRun wrapper for a PIDL Object pointer
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Uintah_Datatypes_Particles_PIDLObject_h
#define Uintah_Datatypes_Particles_PIDLObject_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Component/PIDL/Object.h>

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using namespace SCICore::Datatypes;

class PIDLObject;
typedef LockingHandle<PIDLObject> PIDLObjectHandle;

class PIDLObject : public Datatype {
    Component::PIDL::Object obj;
public:
    PIDLObject(const Component::PIDL::Object& obj);
    virtual ~PIDLObject();
    PIDLObject(const PIDLObject&);

    inline Component::PIDL::Object getObject() {
	return obj;
    }

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/10/07 02:08:24  sparker
// use standard iostreams and complex type
//
//

#endif
