
/*
 *  Datatype.cc: The Datatype Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Thread/AtomicCounter.h>

namespace SCIRun {

static AtomicCounter current_generation("Datatypes generation counter", 1);

Datatype::Datatype()
: lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    generation=current_generation++;
}

Datatype::Datatype(const Datatype&)
    : lock("Datatype ref_cnt lock")
{
    ref_cnt=0;
    generation=current_generation++;
}

Datatype& Datatype::operator=(const Datatype&)
{
    // XXX:
    // Should probably throw an exception if ref_cnt is > 0 or
    // something.
    generation=current_generation++;
    return *this;
}

Datatype::~Datatype()
{
}

} // End namespace SCIRun

