
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;
using namespace SCICore::Exceptions;

map<string, VarLabel*> VarLabel::allLabels;

VarLabel::VarLabel(const std::string& name, const TypeDescription* td,
		   VarType vartype)
   : d_name(name), d_td(td), d_vartype(vartype),
     d_allowMultipleComputes(false)
{
   map<string, VarLabel*>::value_type mappair(name, this);
   if (allLabels.insert(mappair).second == false) {
      // two labels with the same name -- make sure they are the same type
      VarLabel* dup = allLabels[name];
      if (d_td != dup->d_td || d_vartype != dup->d_vartype)
	 throw InternalError(string("VarLabel with same name exists, '")
			     + name + "', but with different type");
   }
}

VarLabel::~VarLabel()
{
   allLabels.erase(d_name);
}

VarLabel* VarLabel::find(string name)
{
   map<string, VarLabel*>::iterator found = allLabels.find(name);
   if (found == allLabels.end())
      return NULL;
   else
      return (*found).second;
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

void VarLabel::allowMultipleComputes()
{
   if (!d_td->isReductionVariable())
      throw InternalError(string("Only reduction variables may allow multiple computes.\n'" + d_name + "' is not a reduction variable."));
   d_allowMultipleComputes = true;
}

ostream & 
operator<<( ostream & out, const Uintah::VarLabel & vl )
{
  out << vl.getName();
  return out;
}


//
// $Log$
// Revision 1.15  2001/01/05 21:52:01  witzel
// One more try -- I should really compile before I commit something
//
// Revision 1.14  2001/01/05 20:14:05  witzel
// Oops, d_td is a pointer
//
// Revision 1.13  2001/01/05 20:09:29  witzel
// Only let reduction VarLabel's allow multiple computes.
//
// Revision 1.12  2001/01/04 22:32:34  witzel
// Added allowMultipleComputes flag to allow one to indicate that a
// VarLabel may be computed multiple times in a taskgraph without conflict
// (i.e. can be done with reduction variables like delT).
//
// Revision 1.11  2000/12/23 00:35:23  witzel
// Added a static member variable to VarLabel that maps VarLabel names to
// the appropriate VarLabel* for all VarLabel's in existent, and added
// VarLabel::find which uses this map.
//
// Revision 1.10  2000/12/19 16:55:39  jas
// Added implementation of getFullName.
//
// Revision 1.9  2000/12/10 09:06:18  sparker
// Merge from csafe_risky1
//
// Revision 1.8.4.1  2000/10/10 05:28:08  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
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

