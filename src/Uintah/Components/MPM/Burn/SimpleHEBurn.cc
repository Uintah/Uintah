/* REFERENCED */
static char *id="@(#) $Id$";

// SimpleHEBurn.cc
//
// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "SimpleHEBurn.h"
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SimulationStateP.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah::MPM;

SimpleHEBurn::SimpleHEBurn(ProblemSpecP& ps)
{
  // Constructor

  ps->require("a",a);
  ps->require("b",b); 
  d_burnable = true;
  
  std::cerr << "a = " << a << " b = " << b << std::endl;

}

SimpleHEBurn::~SimpleHEBurn()
{
  // Destructor

}

void SimpleHEBurn::addCheckIfComputesAndRequires(Task* task,
                                                 const MPMMaterial* matl,
                                                 const Patch* patch,
                                                 DataWarehouseP& old_dw,
                                                 DataWarehouseP& new_dw) const
{

}

void SimpleHEBurn::checkIfIgnited(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouseP& old_dw,
				  DataWarehouseP& new_dw)
{

}
 
void SimpleHEBurn::computeMassRate()
{
}
 
void SimpleHEBurn::updatedParticleMassAndVolume()
{
}                                               

// $Log$
// Revision 1.2  2000/06/06 18:04:02  guilkey
// Added more stuff for the burn models.  Much to do still.
//
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//
