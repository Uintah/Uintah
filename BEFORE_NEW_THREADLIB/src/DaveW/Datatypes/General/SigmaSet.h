
/*
 *  SigmaSet.h: Set of sigmas (e.g. conductivies) for finite-elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_SigmaSet_h
#define SCI_DaveW_Datatypes_SigmaSet_h 1

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/Datatype.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Containers::Array2;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;

using namespace SCICore::Datatypes;

class SigmaSet;
typedef LockingHandle<SigmaSet> SigmaSetHandle;

class SigmaSet : public Datatype {
public:
    Array1<clString> names;
    Array2<double> vals;
    SigmaSet();
    SigmaSet(const SigmaSet&);
    SigmaSet(int nsigs, int vals_per_sig);
    virtual SigmaSet* clone();
    virtual ~SigmaSet();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.2  1999/08/25 03:47:35  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/23 02:53:01  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:06  dmw
// Added and updated DaveW Datatypes/Modules
//
//

#endif
