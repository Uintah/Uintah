
#include <Uintah/Grid/Variable.h>

using namespace Uintah;

Variable::Variable()
{
   foreign=false;
}

Variable::~Variable()
{
}

void Variable::setForeign()
{
   foreign=true;
}

//
// $Log$
// Revision 1.1.4.1  2000/10/19 05:18:04  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.2  2000/10/13 20:46:11  sparker
// Added the notion of a "foreign" variable, to assist in cleaning
//  them out of the data warehouse at the end of a timestep
//
// Revision 1.1  2000/07/27 22:39:51  sparker
// Implemented MPIScheduler
// Added associated support
//
//

