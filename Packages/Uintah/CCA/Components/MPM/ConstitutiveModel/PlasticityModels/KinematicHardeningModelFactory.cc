#include "KinematicHardeningModelFactory.h"                                             
#include "NoKinematicHardening.h"
#include "PragerKinematicHardening.h"
#include "ArmstrongFrederickKinematicHardening.h"
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

KinematicHardeningModel* KinematicHardeningModelFactory::create(ProblemSpecP& ps)
{
   ProblemSpecP child = ps->findBlock("kinematic_hardening_model");
   if(!child) {
      cerr << "**WARNING** Creating default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening());
      //throw ProblemSetupException("Cannot find kinematic hardening model tag", __FILE__, __LINE__);
   }

   string mat_type;
   if(!child->getAttribute("type", mat_type))
      throw ProblemSetupException("No type for kinematic hardening model", __FILE__, __LINE__);

   if (mat_type == "none")
      return(scinew NoKinematicHardening(child));
   else if (mat_type == "prager_hardening")
      return(scinew PragerKinematicHardening(child));
   else if (mat_type == "armstrong_frederick_hardening")
      return(scinew ArmstrongFrederickKinematicHardening(child));
   else {
      cerr << "**WARNING** Creating default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening(child));
      //throw ProblemSetupException("Unknown KinematicHardening Model ("+mat_type+")", __FILE__, __LINE__);
   }
}

KinematicHardeningModel* 
KinematicHardeningModelFactory::createCopy(const KinematicHardeningModel* pm)
{
   if (dynamic_cast<const NoKinematicHardening*>(pm))
      return(scinew NoKinematicHardening(dynamic_cast<const 
                                        NoKinematicHardening*>(pm)));

   else if (dynamic_cast<const PragerKinematicHardening*>(pm))
      return(scinew PragerKinematicHardening(dynamic_cast<const 
                                       PragerKinematicHardening*>(pm)));

   else if (dynamic_cast<const ArmstrongFrederickKinematicHardening*>(pm))
      return(scinew ArmstrongFrederickKinematicHardening(dynamic_cast<const ArmstrongFrederickKinematicHardening*>(pm)));

   else {
      cerr << "**WARNING** Creating copy of default (no kinematic hardening) model" << endl;
      return(scinew NoKinematicHardening(dynamic_cast<const 
                                        NoKinematicHardening*>(pm)));
      //throw ProblemSetupException("Cannot create copy of unknown kinematic_hardening model", __FILE__, __LINE__);
   }
}

