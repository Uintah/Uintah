//static char *id="@(#) $Id$";

/*
 *  Path.h: Set of sigmas (e.g. conductivies) for finite-elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/Path.h>
#include <SCICore/Malloc/Allocator.h>

namespace DaveW {
namespace Datatypes {

static Persistent* make_Path()
{
    return scinew Path;
}

PersistentTypeID Path::type_id("Path", "Datatype", make_Path);

Path::Path()
{
}

Path::Path(const Path& copy)
: names(copy.names), vals(copy.vals)
{
}

Path::Path(int nsigs, int vals_per_sig)
: names(nsigs), vals(nsigs, vals_per_sig)
{
}

Path::~Path()
{
}

Path* Path::clone()
{
    return scinew Path(*this);
}

#define Path_VERSION 1

void Path::io(Piostream& stream)
{
    using SCICore::Containers::Pio;

    stream.begin_class("Path", Path_VERSION);
    Pio(stream, names);
    Pio(stream, vals);
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//
