
#include <Uintah/Grid/VarLabel.h>

using namespace Uintah;

VarLabel::VarLabel(const std::string& name, const TypeDescription* td,
		   VarType vartype)
   : d_name(name), d_td(td), d_vartype(vartype)
{
}



//
// $Log$
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

