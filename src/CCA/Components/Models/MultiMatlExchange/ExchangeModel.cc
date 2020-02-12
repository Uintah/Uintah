/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

#define d_TINY_RHO 1e-12

using namespace Uintah;
using namespace std;

DebugStream dbgExch("EXCHANGEMODELS", false);

//______________________________________________________________________
//
ExchangeModel::ExchangeModel(const ProblemSpecP     & prob_spec,
                             const MaterialManagerP & materialManager,
                             const bool with_mpm )
{
  d_matlManager = materialManager;
  d_numMatls    = materialManager->getNumMatls();
  d_zero_matl   = scinew MaterialSubset();
  d_zero_matl->add(0);
  d_zero_matl->addReference();
  
  d_with_mpm = with_mpm;
  Ilb = scinew ICELabel();
  
  if(with_mpm){
    Mlb = scinew MPMLabel();
  }
  
  
  d_surfaceNormLabel   = VarLabel::create("surfaceNorm",   CCVariable<Vector>::getTypeDescription());
  d_isSurfaceCellLabel = VarLabel::create("isSurfaceCell", CCVariable<int>::getTypeDescription());
}

//______________________________________________________________________
//
ExchangeModel::~ExchangeModel()
{
  VarLabel::destroy( d_surfaceNormLabel );
  VarLabel::destroy( d_isSurfaceCellLabel );
  
  if( d_zero_matl  && d_zero_matl->removeReference() ) {
    delete d_zero_matl;
  }
  
  delete Ilb;
  
  if( d_with_mpm ){
    delete Mlb;
  }
  
}


//______________________________________________________________________
//
void ExchangeModel::schedComputeSurfaceNormal( SchedulerP           & sched,        
                                               const PatchSet       * patches,
                                               const MaterialSubset * mpm_matls )      
{
  std::string name = "ExchangeModel::ComputeSurfaceNormal";

  Task* t = scinew Task( name, this, &ExchangeModel::ComputeSurfaceNormal);

  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gac  = Ghost::AroundCells;
  t->requires( Task::NewDW, Mlb->gMassLabel,       mpm_matls,   gac, 1 );
  t->requires( Task::OldDW, Mlb->NC_CCweightLabel, d_zero_matl, gac, 1 );

  t->computes( d_surfaceNormLabel,   mpm_matls );
  t->computes( d_isSurfaceCellLabel, d_zero_matl ); 
  
  const MaterialSet* mpm_matlset = d_matlManager->allMaterials( "MPM" );
  sched->addTask(t, patches, mpm_matlset );
}
//______________________________________________________________________
//
void ExchangeModel::ComputeSurfaceNormal( const ProcessorGroup*,
                                          const PatchSubset    * patches,           
                                          const MaterialSubset * mpm_matls,      
                                          DataWarehouse        * old_dw,                
                                          DataWarehouse        * new_dw )               
{
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing ExchangeModel::ComputeSurfaceNormal" );

    Ghost::GhostType gac = Ghost::AroundCells;
    
    constNCVariable<double> NC_CCweight;
    old_dw->get(NC_CCweight, Mlb->NC_CCweightLabel, 0, patch, gac,1);
    
    CCVariable<int> isSurfaceCell;
    new_dw->allocateAndPut(isSurfaceCell, d_isSurfaceCellLabel, 0, patch);
    isSurfaceCell.initialize( 0 );
    
    Vector dx = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();

    //__________________________________
    //    loop over MPM matls
    int numMPM_matls = d_matlManager->getNumMatls( "MPM" );
    
    for(int m=0; m<numMPM_matls; m++){
      MPMMaterial* matl = (MPMMaterial*) d_matlManager->getMaterial( "MPM", m);
      int dwindex = matl->getDWIndex();

      CCVariable<Vector> surfaceNorm;
      new_dw->allocateAndPut(surfaceNorm, d_surfaceNormLabel, dwindex, patch);
      surfaceNorm.initialize( Vector(0,0,0) );

      constNCVariable<double> NCsolidMass;
      new_dw->get(NCsolidMass, Mlb->gMassLabel, dwindex, patch, gac,1);

      //__________________________________
      //
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

        // Surface Cell
        if ((MaxMass-MinMass)/MaxMass == 1 && (MaxMass > (d_TINY_RHO * vol))){
        
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

          surfaceNorm[c] = Vector(gradRhoX/absGradRho,
                                  gradRhoY/absGradRho,
                                  gradRhoZ/absGradRho);
          isSurfaceCell[c] = 1;

        }  // if a surface cell
      }  // cellIterator
    }  // for MPM matls
  } // patches
}
