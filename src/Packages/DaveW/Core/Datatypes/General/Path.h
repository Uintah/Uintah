/*
 *  Path.h: Camera path
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_Path_h
#define SCI_DaveW_Datatypes_Path_h 1

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

class Path;
typedef LockingHandle<Path> PathHandle;

class Path : public Datatype {
public:
    Array1<clString> names;
    Array2<double> vals;
    Path();
    Path(const Path&);
    Path(int nsigs, int vals_per_sig);
    virtual Path* clone();
    virtual ~Path();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//

#endif
