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

#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/Models/MultiMatlExchange/Scalar.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

#define d_MAX_MATLS 8

using namespace Uintah;
using namespace ExchangeModels;
using namespace std;
extern DebugStream dbgExch;
//______________________________________________________________________
//
ScalarExch::ScalarExch( const ProblemSpecP     & prob_spec,
                        const SimulationStateP & sharedState )
  : ExchangeModel( prob_spec, sharedState )
{
  d_exchCoeff = scinew ExchangeCoefficients();
}

ScalarExch::~ScalarExch()
{
  delete d_exchCoeff;
}

//______________________________________________________________________
//
void ScalarExch::problemSetup(const ProblemSpecP & prob_spec)
{
  // read in the exchange coefficients
  d_exchCoeff->problemSetup( prob_spec, d_numMatls );
}

//______________________________________________________________________
//
void ScalarExch::sched_AddExch_VelFC( SchedulerP            & sched,            
                                      const PatchSet        * patches,          
                                      const MaterialSubset  * ice_matls,        
                                      const MaterialSet     * all_matls,        
                                      customBC_globalVars   * BC_globalVars,    
                                      const bool recursion )
{
  int levelIndex = getLevel(patches)->getIndex();
  std::string name = "ScalarExch::sched_AddExch_VelFC";
  
  Task* task = scinew Task( name, this, &ScalarExch::addExch_VelFC, 
                            BC_globalVars, 
                            recursion);

  printSchedule( patches, dbgExch, name );
  
  if(recursion) {
    task->requires(Task::ParentOldDW, Ilb->delTLabel,getLevel(patches));
  } else {
    task->requires(Task::OldDW,       Ilb->delTLabel,getLevel(patches));
  }

  Ghost::GhostType  gac = Ghost::AroundCells;

  //__________________________________
  // define parent data warehouse
  // change the definition of parent(old/new)DW
  // when using semi-implicit pressure solve
  Task::WhichDW pNewDW = Task::NewDW;
  if(recursion) {
    pNewDW  = Task::ParentNewDW;
  }
  
  // All matls
  task->requires( pNewDW,      Ilb->sp_vol_CCLabel,  gac, 1);
  task->requires( pNewDW,      Ilb->vol_frac_CCLabel,gac, 1);
  task->requires( Task::NewDW, Ilb->uvel_FCLabel,    gac, 2);
  task->requires( Task::NewDW, Ilb->vvel_FCLabel,    gac, 2);
  task->requires( Task::NewDW, Ilb->wvel_FCLabel,    gac, 2);
  
  computesRequires_CustomBCs(task, "velFC_Exchange", Ilb, ice_matls,
                             BC_globalVars, recursion);

  task->computes( Ilb->sp_volX_FCLabel );
  task->computes( Ilb->sp_volY_FCLabel );
  task->computes( Ilb->sp_volZ_FCLabel );
  task->computes( Ilb->uvel_FCMELabel );
  task->computes( Ilb->vvel_FCMELabel );
  task->computes( Ilb->wvel_FCMELabel );

  sched->addTask(task, patches, all_matls);
}

/* _____________________________________________________________________
 Purpose~   Add the exchange contribution to vel_FC and compute
            sp_vol_FC for implicit Pressure solve
_____________________________________________________________________*/
template<class constSFC, class SFC>
void ScalarExch::vel_FC_exchange( CellIterator    iter,
                                  IntVector       adj_offset,
                                  int             numMatls,
                                  FastMatrix  &   K,
                                  double          delT,
                                  std::vector<constCCVariable<double> >& vol_frac_CC,
                                  std::vector<constCCVariable<double> >& sp_vol_CC,
                                  std::vector< constSFC>               & vel_FC,
                                  std::vector< SFC >                   & sp_vol_FC,
                                  std::vector< SFC >                   & vel_FCME)
{
  //__________________________________
  //          Single Material
  if (numMatls == 1){

    // put in tmp arrays for speed!
    constCCVariable<double>& sp_vol_tmp = sp_vol_CC[0];
    constSFC& vel_FC_tmp  = vel_FC[0];
    SFC& vel_FCME_tmp     = vel_FCME[0];
    SFC& sp_vol_FC_tmp    = sp_vol_FC[0];

    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adj = c + adj_offset;
      double sp_vol     = sp_vol_tmp[c];
      double sp_vol_adj = sp_vol_tmp[adj];
      double sp_volFC   = 2.0 * (sp_vol_adj * sp_vol)/
                                (sp_vol_adj + sp_vol);

      sp_vol_FC_tmp[c] = sp_volFC;
      vel_FCME_tmp[c] = vel_FC_tmp[c];
    }
  }
  else{         // Multi-material
    double b[d_MAX_MATLS];
    double b_sp_vol[d_MAX_MATLS];
    double vel[d_MAX_MATLS];
    double tmp[d_MAX_MATLS];
    FastMatrix a(numMatls, numMatls);

    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adj = c + adj_offset;

      //__________________________________
      //   Compute beta and off diagonal term of
      //   Matrix A, this includes b[m][m].
      //  You need to make sure that mom_exch_coeff[m][m] = 0

      // - Form diagonal terms of Matrix (A)
      //  - Form RHS (b)
      for(int m = 0; m < numMatls; m++)  {
        b_sp_vol[m] = 2.0 * (sp_vol_CC[m][adj] * sp_vol_CC[m][c])/
                            (sp_vol_CC[m][adj] + sp_vol_CC[m][c]);

        tmp[m] = -0.5 * delT * (vol_frac_CC[m][adj] + vol_frac_CC[m][c]);
        vel[m] = vel_FC[m][c];
      }

      for(int m = 0; m < numMatls; m++)  {
        double betasum = 1;
        double bsum    = 0;
        double bm      = b_sp_vol[m];
        double vm      = vel[m];

        for(int n = 0; n < numMatls; n++)  {
          double b = bm * tmp[n] * K(n,m);
          a(m,n)   = b;
          betasum -= b;
          bsum    -= b * (vel[n] - vm);
        }
        a(m,m) = betasum;
        b[m] = bsum;
      }

      //__________________________________
      //  - solve and backout velocities

      a.destructiveSolve(b, b_sp_vol);
      //  For implicit solve we need sp_vol_FC
      for(int m = 0; m < numMatls; m++) {
        vel_FCME[m][c]  = vel_FC[m][c] + b[m];
        sp_vol_FC[m][c] = b_sp_vol[m];        // only needed by implicit Pressure
      }
    }  // iterator
  }  // multiple materials
}

/*_____________________________________________________________________

 Purpose~
   This function adds the momentum exchange contribution to the
   existing face-centered velocities


                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_FC[1]  |
|                                               |   |           |
| -b21              (1+b21 + b23) -b32          |   |del_FC[2]  |
|                                               |   |           |
| -b31              -b32          (1+b31 + b32) |   |del_FC[2]  |

                        =

                        (B)
| b12( uvel_FC[2] - uvel_FC[1] ) + b13 ( uvel_FC[3] -uvel_FC[1])    |
|                                                                   |
| b21( uvel_FC[1] - uvel_FC[2] ) + b23 ( uvel_FC[3] -uvel_FC[2])    |
|                                                                   |
| b31( uvel_FC[1] - uvel_FC[3] ) + b32 ( uvel_FC[2] -uvel_FC[3])    |

 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 _____________________________________________________________________  */
void ScalarExch::addExch_VelFC( const ProcessorGroup * pg,
                                const PatchSubset    * patches,
                                const MaterialSubset * matls,
                                DataWarehouse        * old_dw,
                                DataWarehouse        * new_dw,
                                customBC_globalVars  * BC_globalVars,
                                const bool recursion )
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing ScalarExch::addExch_VelFC" );

    // change the definition of parent(old/new)DW
    // when implicit
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;
    if(recursion) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    }

    delt_vartype delT;
    pOldDW->get(delT, Ilb->delTLabel, level);

    std::vector< constCCVariable<double> >   sp_vol_CC(d_numMatls);
    std::vector< constCCVariable<double> >   vol_frac_CC(d_numMatls);
    std::vector< constSFCXVariable<double> > uvel_FC(d_numMatls);
    std::vector< constSFCYVariable<double> > vvel_FC(d_numMatls);
    std::vector< constSFCZVariable<double> > wvel_FC(d_numMatls);

    std::vector< SFCXVariable<double> > uvel_FCME(d_numMatls);
    std::vector< SFCYVariable<double> > vvel_FCME(d_numMatls);
    std::vector< SFCZVariable<double> > wvel_FCME(d_numMatls);
    
    std::vector< SFCXVariable<double> > sp_vol_XFC(d_numMatls);
    std::vector< SFCYVariable<double> > sp_vol_YFC(d_numMatls);
    std::vector< SFCZVariable<double> > sp_vol_ZFC(d_numMatls);

    // lowIndex is the same for all vel_FC
    IntVector lowIndex(patch->getExtraSFCXLowIndex());

    // Extract the momentum exchange coefficients
    FastMatrix K(d_numMatls, d_numMatls), junk(d_numMatls, d_numMatls);

    K.zero();
    
    d_exchCoeff->getConstantExchangeCoeff( K, junk);
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    for(int m = 0; m < d_numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      
      pNewDW->get( sp_vol_CC[m],    Ilb->sp_vol_CCLabel,  indx, patch,gac, 1 );
      pNewDW->get( vol_frac_CC[m],  Ilb->vol_frac_CCLabel,indx, patch,gac, 1 );
      new_dw->get( uvel_FC[m],      Ilb->uvel_FCLabel,    indx, patch,gac, 2 );
      new_dw->get( vvel_FC[m],      Ilb->vvel_FCLabel,    indx, patch,gac, 2 );
      new_dw->get( wvel_FC[m],      Ilb->wvel_FCLabel,    indx, patch,gac, 2 );

      new_dw->allocateAndPut( uvel_FCME[m],  Ilb->uvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( vvel_FCME[m],  Ilb->vvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( wvel_FCME[m],  Ilb->wvel_FCMELabel, indx, patch );

      new_dw->allocateAndPut( sp_vol_XFC[m], Ilb->sp_volX_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_YFC[m], Ilb->sp_volY_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_ZFC[m], Ilb->sp_volZ_FCLabel,indx, patch );

      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCXHighIndex());
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCYHighIndex());
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCZHighIndex());

      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex());
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex());
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex());
    }

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();

    //__________________________________
    //  tack on exchange contribution
    vel_FC_exchange<constSFCXVariable<double>, SFCXVariable<double> >
                    (XFC_iterator,
                    adj_offset[0],  d_numMatls,    K,
                    delT,           vol_frac_CC, sp_vol_CC,
                    uvel_FC,        sp_vol_XFC,  uvel_FCME);

    vel_FC_exchange<constSFCYVariable<double>, SFCYVariable<double> >
                    (YFC_iterator,
                    adj_offset[1],  d_numMatls,    K,
                    delT,           vol_frac_CC, sp_vol_CC,
                    vvel_FC,        sp_vol_YFC,  vvel_FCME);

    vel_FC_exchange<constSFCZVariable<double>, SFCZVariable<double> >
                        (ZFC_iterator,
                        adj_offset[2],  d_numMatls,    K,
                        delT,           vol_frac_CC, sp_vol_CC,
                        wvel_FC,        sp_vol_ZFC,  wvel_FCME);

    //________________________________
    //  Boundary Conditons
    for (int m = 0; m < d_numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
            
      customBC_localVars* BC_localVars = scinew customBC_localVars();
      BC_localVars->recursiveTask = recursion;

      preprocess_CustomBCs("velFC_Exchange",pOldDW, pNewDW, Ilb,  patch, indx,
                            BC_globalVars, BC_localVars);

      setBC<SFCXVariable<double> >(uvel_FCME[m], "Velocity", patch, indx,
                                    d_sharedState, BC_globalVars, BC_localVars);
      setBC<SFCYVariable<double> >(vvel_FCME[m], "Velocity", patch, indx,
                                    d_sharedState, BC_globalVars, BC_localVars);
      setBC<SFCZVariable<double> >(wvel_FCME[m], "Velocity", patch, indx,
                                    d_sharedState, BC_globalVars, BC_localVars);
      delete_CustomBCs( BC_globalVars, BC_localVars );
    }
  }  // patch loop
}
//______________________________________________________________________
//
void ScalarExch::sched_AddExch_Vel_Temp_CC( SchedulerP           & sched,
                                            const PatchSet       * patches,
                                            const MaterialSubset * ice_matls,
                                            const MaterialSubset * mpm_matls,
                                            const MaterialSubset * press_matl,
                                            const MaterialSet    * all_matls )
{
}
//______________________________________________________________________
//
void ScalarExch::addExch_Vel_Temp_CC( const ProcessorGroup * pg,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                      DataWarehouse        * old_dw,
                                      DataWarehouse        * new_dw)
{
}

