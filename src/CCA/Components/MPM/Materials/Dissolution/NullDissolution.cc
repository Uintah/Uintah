/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
                         MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, 0)
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
