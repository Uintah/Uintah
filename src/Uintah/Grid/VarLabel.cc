
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/Patch.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

VarLabel::VarLabel(const std::string& name, const TypeDescription* td,
		   VarType vartype)
   : d_name(name), d_td(td), d_vartype(vartype)
{
}

bool
VarLabel::Compare::operator()(const VarLabel* v1,
			      const VarLabel* v2) const
{
   if(v1 == v2)
      return false;
   return v1->getName() < v2->getName();
}

string
VarLabel::getFullName(int matlIndex, const Patch* patch) const
{
   ostringstream out;
   out << d_name << "(matl=" << matlIndex;
   if(patch)
      out << ", patch=" << patch->getID();
   else
      out << ", no patch";
   out << ")";
   return out.str();
}

ostream & 
operator<<( ostream & out, const Uintah::VarLabel & vl )
{
  out << vl.getName();
  return out;
}


//
// $Log$
// Revision 1.8  2000/09/25 18:12:20  sparker
// do not use covariant return types due to problems with g++
// other linux/g++ fixes
//
// Revision 1.7  2000/08/23 22:36:50  dav
// added output operator
//
// Revision 1.6  2000/07/27 22:39:51  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.5  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.4  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.3  2000/04/28 07:35:37  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.2  2000/04/26 06:49:00  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

