/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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


#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/DamageModel.h>

using namespace Uintah;
static DebugStream dbg("DamageModel", false);
//______________________________________________________________________
//      TO DO
//______________________________________________________________________
//

DamageModel::DamageModel()
{
  d_lb = scinew MPMLabel();
}

DamageModel::~DamageModel()
{
  delete d_lb;
}

//______________________________________________________________________
//
void DamageModel::addComputesAndRequires(Task* task,
                                         const MPMMaterial* matl)
{
}


//______________________________________________________________________
//
void DamageModel::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
}


//______________________________________________________________________
//
void
DamageModel::carryForward(const PatchSubset* patches,
                          const MPMMaterial* matl,
                          DataWarehouse*     old_dw,
                          DataWarehouse*     new_dw)
{
  // do nothing
}

//______________________________________________________________________
//
void 
DamageModel::addInitialComputesAndRequires(Task* task,
                                           const MPMMaterial* matl )
{
}
//______________________________________________________________________
//
void 
DamageModel::initializeLabels(const Patch*       patch,
                              const MPMMaterial* matl,
                              DataWarehouse*     new_dw)
{
}

//______________________________________________________________________
//
void 
DamageModel::computeSomething( ParticleSubset    * pset,
                               const MPMMaterial * matl,
                               const Patch       * patch,    
                               DataWarehouse     * old_dw,
                               DataWarehouse     * new_dw )
{
}
