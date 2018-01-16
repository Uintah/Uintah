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

#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

using namespace Uintah;
using namespace std;

DebugStream dbgExch("EXCHANGEMODELS", false);

//______________________________________________________________________
//
ExchangeModel::ExchangeModel(const ProblemSpecP     & prob_spec,
                             const SimulationStateP & sharedState )
{
  d_sharedState = sharedState;
  d_numMatls    = sharedState->getNumMatls();
}

ExchangeModel::~ExchangeModel()
{
}


#if 0

//______________________________________________________________________
//
void ExchangeModel::scheduleComputeSurfaceNormal( SchedulerP& sched,
                                                  const PatchSet       * patches,
                                                  const MaterialSubset * press_matl,
                                                  const MaterialSet    * mpm_matls)
{
  std::string name = "ExchangeModel::ComputeSurfaceNormalValues";

  Task* t = scinew Task( name, this, &ExchangeModel::ComputeSurfaceNormalValues);

  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gac  = Ghost::AroundCells;
  t->requires( Task::NewDW, MIlb->gMassLabel,       mpm_matls->getUnion,  gac, 1);
  t->requires( Task::OldDW, MIlb->NC_CCweightLabel, press_matl, gac, 1);

  t->computes( d_surfaceNormLabel );

  sched->addTask(t, patches, mpm_matls);
}
//______________________________________________________________________
//
void ExchangeModel::ComputeSurfaceNormalValues( const ProcessorGroup*,
                                                const PatchSubset* patches,
                                                const MaterialSubset* mpm_matls,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw )
{
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing ExchangeModel::ComputeSurfaceNormalValues" );

    Ghost::GhostType gac = Ghost::AroundCells;
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    
    std::vector<CCVariable<Vector> > surfaceNorm(numMPMMatls);
    constNCVariable<double> NC_CCweight;
    constNCVariable<double> NCsolidMass;

    old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel, 0, patch, gac,1);
    Vector dx = patch->dCell();

    for(int m=0; m<mpm_matls->size(); m++){
      Material* matl = d_sharedState->getMaterial(m);
      int dwindex = matl->getDWIndex();

      new_dw->allocateAndPut(surfaceNorm[m], d_surfaceNormLabel, dwindex, patch);
      surfaceNorm[m].initialize( Vector(0,0,0) );

      new_dw->get(NCsolidMass, MIlb->gMassLabel, dwindex, patch, gac,1);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;

        IntVector nodeIdx[8];
        patch->findNodesFromCell(*iter,nodeIdx);
        
        double MaxMass = d_SMALL_NUM;
        double MinMass = 1.0/d_SMALL_NUM;
        for (int nN=0; nN<8; nN++) {

          MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                     NCsolidMass[nodeIdx[nN]]);
          MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                     NCsolidMass[nodeIdx[nN]]);
        }

        if ((MaxMass-MinMass)/MaxMass == 1 && (MaxMass > d_SMALL_NUM)){
          double gradRhoX = 0.25 *
                 ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                   NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+         // xminus
                   NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                   NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                 -
                 ( NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                   NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+         // xplus
                   NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                   NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]]) ) / dx.x();
          double gradRhoY = 0.25 *
                 ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                   NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+         // yminus
                   NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                   NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                 -
                 ( NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                   NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+         // yplus
                   NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                   NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]]) ) / dx.y();
          double gradRhoZ = 0.25 *
                 ((NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                   NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+         // zminus
                   NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                   NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                 -
                  (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                   NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+         // zplus
                   NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                   NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]]) ) / dx.z();

          double absGradRho = sqrt(gradRhoX*gradRhoX +
                                   gradRhoY*gradRhoY +
                                   gradRhoZ*gradRhoZ );

          surfaceNorm[m][c] = Vector(gradRhoX/absGradRho,
                                     gradRhoY/absGradRho,
                                     gradRhoZ/absGradRho);

        }  // if a surface cell
      }  // cellIterator
    }  // for MPM matls
  } // patches
}

#endif
