
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

#ifndef SCI_Packages_DaveW_Datatypes_SigmaSet_h
#define SCI_Packages_DaveW_Datatypes_SigmaSet_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/Datatype.h>

namespace DaveW {
using namespace SCIRun;


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
} // End namespace DaveW



#endif
