/*
 * Copyright © 2026 by Geocosm LLC                                   
 */


// NullDissolution.cc
// One of the derived Dissolution classes.  This particular
// class is used when no dissolution is desired.
#include <CCA/Components/MPM/Materials/Dissolution/NullDissolution.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
using namespace Uintah;

NullDissolution::NullDissolution(const ProcessorGroup* myworld,
                         MaterialManagerP& d_sS,
                         MPMLabel* Mlb, MPMFlags* flag)
  : Dissolution(myworld, Mlb, 0, flag)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;
}

NullDissolution::~NullDissolution()
{
}

void NullDissolution::outputProblemSpec(ProblemSpecP& ps)
{
//  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
//  dissolution_ps->appendElement("type","null");
//  d_matls.outputProblemSpec(dissolution_ps);
}


void NullDissolution::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* /*old_dw*/,
                                              DataWarehouse* new_dw)
{
}

void NullDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                    const PatchSet* patches,
                                                    const MaterialSet* ms) 
{
  Task * t = scinew Task("NullDissolution::computeMassBurnFraction", this, 
                         &NullDissolution::computeMassBurnFraction);
  
  sched->addTask(t, patches, ms);
}
