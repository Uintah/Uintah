
#include <Uintah/Grid/VarLabel.h>

using namespace Uintah::Grid;

VarLabel::VarLabel(const std::string& name, const TypeDescription* td)
   : d_name(name), d_td(td)
{
}

//
// $Log$
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

