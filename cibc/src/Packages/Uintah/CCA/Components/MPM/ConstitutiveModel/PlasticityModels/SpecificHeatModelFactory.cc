#include "SpecificHeatModelFactory.h"
#include "ConstantCp.h"
#include "CopperCp.h"
#include "SteelCp.h"
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

/// Create an instance of a specific heat model
SpecificHeatModel* SpecificHeatModelFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP child = ps->findBlock("specific_heat_model");
  if(!child) {
    ostringstream desc;
    desc << "**Error in Input UPS File: " 
	 << "MPM:SpecificHeatModel:  "
	 << "No specific_heat_model tag found in input file." << endl;
    throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
  string mat_type;
  if(!child->getAttribute("type", mat_type)) {
    ostringstream desc;
    desc << "**Error in Input UPS File: " 
	 << "MPM:SpecificHeatModel:  "
	 << "No specific_heat_model type tag found in input file. " << endl 
	 << "Types include constant_Cp, copper_Cp, and steel_Cp." << endl;
    throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
   
  if (mat_type == "constant_Cp")
    return(scinew ConstantCp(child));
  else if (mat_type == "copper_Cp")
    return(scinew CopperCp(child));
  else if (mat_type == "steel_Cp")
    return(scinew SteelCp(child));
  else {
    ostringstream desc;
    desc << "**Error in Input UPS File: " 
	 << "MPM:SpecificHeatModel:  "
	 << "Incorrect specific_heat_model type (" << mat_type 
	 << ") found in input file. " << endl 
	 << "Correct type tags include constant_Cp, copper_Cp, and steel_Cp." 
	 << endl;
    throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
}

SpecificHeatModel* 
SpecificHeatModelFactory::createCopy(const SpecificHeatModel* smm)
{
  if (dynamic_cast<const ConstantCp*>(smm))
    return(scinew ConstantCp(dynamic_cast<const ConstantCp*>(smm)));
  else if (dynamic_cast<const CopperCp*>(smm))
    return(scinew CopperCp(dynamic_cast<const CopperCp*>(smm)));
  else if (dynamic_cast<const SteelCp*>(smm))
    return(scinew SteelCp(dynamic_cast<const SteelCp*>(smm)));
  else {
    ostringstream desc;
    desc << "**Error in Material Copying: " 
	 << "MPM:SpecificHeatModel:  "
         << "Cannot create copy of unknown specific heat model"
	 << endl;
    throw ProblemSetupException(desc.str(), __FILE__, __LINE__);
  }
}
