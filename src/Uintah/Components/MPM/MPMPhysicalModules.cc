//Physical Models Interested:
#include <Uintah/Components/MPM/Contact/ContactFactory.h>
#include <Uintah/Components/MPM/HeatConduction/HeatConductionFactory.h>
#include <Uintah/Components/MPM/Fracture/FractureFactory.h>
#include <Uintah/Components/MPM/ThermalContact/ThermalContactFactory.h>

/* REFERENCED */
static char *id="@(#) $Id$";

#include "MPMPhysicalModules.h"

using namespace Uintah::MPM;

HeatConduction*  MPMPhysicalModules::heatConductionModel;
Contact*         MPMPhysicalModules::contactModel;
ThermalContact*  MPMPhysicalModules::thermalContactModel;

void MPMPhysicalModules::build(const ProblemSpecP& prob_spec,
                              SimulationStateP& sharedState)
{
   //solid mechanical contact
   contactModel = ContactFactory::create(prob_spec,sharedState);

   //solid heat conduction
   heatConductionModel = HeatConductionFactory::create(prob_spec,sharedState);

   //solid thermal contact
   thermalContactModel = ThermalContactFactory::create(prob_spec,
     sharedState);
}


//
// $Log$
// Revision 1.3  2000/09/05 05:12:14  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.2  2000/06/22 22:59:38  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.1  2000/06/22 21:22:07  tan
// MPMPhysicalModules class is created to handle all the physical modules
// in MPM, currently those physical submodules include HeatConduction,
// Fracture, Contact, and ThermalContact.
//
