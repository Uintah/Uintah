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

const TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/21 18:52:11  sparker
// Prototyped header file for new problem spec functionality
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
