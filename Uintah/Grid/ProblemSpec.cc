/* REFERENCED */
static char *id="@(#) $Id$";

#include "ProblemSpec.h"

#include <iostream>

using std::cerr;

namespace Uintah {
namespace Grid {

ProblemSpec::ProblemSpec()
{
}

ProblemSpec::~ProblemSpec()
{
}

double ProblemSpec::getStartTime() const
{
    return 0;
}

double ProblemSpec::getMaximumTime() const
{
    //    return 0.00002;
    //return 1.e-5;
    return MAXTIME;
}

const TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
