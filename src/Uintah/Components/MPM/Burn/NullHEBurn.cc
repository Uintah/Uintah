/* REFERENCED */
static char *id="@(#) $Id$";

// NullHEBurn.cc
//
// One of the derived HEBurn classes.  This particular
// class is used when no burn is desired.  

#include "NullHEBurn.h"
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

NullHEBurn::NullHEBurn(ProblemSpecP& ps)
{
  // Constructor
 
  d_burnable = false;

}

NullHEBurn::~NullHEBurn()
{
  // Destructor

}

void NullHEBurn::checkIfIgnited()
{
}

void NullHEBurn::computeMassRate()
{
}

void NullHEBurn::updatedParticleMassAndVolume() 
{
}

// $Log$
// Revision 1.1  2000/06/02 22:48:26  jas
// Added infrastructure for Burn models.
//
