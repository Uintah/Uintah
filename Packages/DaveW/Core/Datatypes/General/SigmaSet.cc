//static char *id="@(#) $Id$";

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

#include <DaveW/Datatypes/General/SigmaSet.h>
#include <SCICore/Malloc/Allocator.h>

#include <iostream.h>
#include <fstream.h>

namespace DaveW {
namespace Datatypes {

static Persistent* make_SigmaSet()
{
    return scinew SigmaSet;
}

PersistentTypeID SigmaSet::type_id("SigmaSet", "Datatype", make_SigmaSet);

SigmaSet::SigmaSet()
{
}

SigmaSet::SigmaSet(const SigmaSet& copy)
: names(copy.names), vals(copy.vals)
{
}

SigmaSet::SigmaSet(int nsigs, int vals_per_sig)
: names(nsigs), vals(nsigs, vals_per_sig)
{
}

SigmaSet::~SigmaSet()
{
}

SigmaSet* SigmaSet::clone()
{
    return scinew SigmaSet(*this);
}

#define SIGMASET_VERSION 1

void SigmaSet::io(Piostream& stream)
{
    using SCICore::Containers::Pio;

    stream.begin_class("SigmaSet", SIGMASET_VERSION);
    Pio(stream, names);
    Pio(stream, vals);
    stream.end_class();
}

} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/08/23 02:53:01  dmw
// Dave's Datatypes
//
// Revision 1.1  1999/05/03 04:52:06  dmw
// Added and updated DaveW Datatypes/Modules
//
//
