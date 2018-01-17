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

#include <CCA/Components/Models/MultiMatlExchange/Slip.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
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

#define MAX_MATLS 8

using namespace Uintah;
using namespace ExchangeModels;
using namespace std;

extern DebugStream dbgExch;
//______________________________________________________________________
//
SlipExch::SlipExch(const ProblemSpecP     & exch_ps,
                   const SimulationStateP & sharedState )
  : ExchangeModel( exch_ps, sharedState )
{
  proc0cout << "__________________________________\n";
  proc0cout << " Now creating the Slip Exchange model " << endl;

  d_exchCoeff = scinew ExchangeCoefficients();
  Ilb  = scinew ICELabel();
  Mlb  = scinew MPMLabel();

  d_vel_CCTransLabel  = VarLabel::create("vel_CCTransposed", CCVariable<Vector>::getTypeDescription());
  d_meanFreePathLabel = VarLabel::create("meanFreePath",     CCVariable<double>::getTypeDescription());
}

//______________________________________________________________________
//
SlipExch::~SlipExch()
{
  delete d_exchCoeff;
  delete Ilb;
  delete Mlb;
  VarLabel::destroy( d_vel_CCTransLabel );
  VarLabel::destroy( d_meanFreePathLabel );
}

//______________________________________________________________________
//
void SlipExch::problemSetup(const ProblemSpecP & exch_ps)
{

  // read in the exchange coefficients
  d_exchCoeff->problemSetup( exch_ps, d_numMatls );

  ProblemSpecP model_ps = exch_ps->findBlock( "Model" );
  model_ps->require("fluidMatlIndex",               d_fluidMatlIndx);
  model_ps->require("solidMatlIndex",               d_solidMatlIndx);
  model_ps->require("momentum_accommodation_coeff", d_momentum_accommodation_coeff);
  model_ps->require("thermal_accommodation_coeff",  d_thermal_accommodation_coeff);


  proc0cout << " fluidMatlIndex: " << d_fluidMatlIndx << " thermal_accommodation_coeff " << d_thermal_accommodation_coeff << endl;
//  d_exchCoeff->problemSetup(mat_ps, d_sharedState);
}

//______________________________________________________________________
//
void SlipExch::sched_AddExch_VelFC(SchedulerP           & sched,
                                   const PatchSet       * patches,
                                   const MaterialSubset * ice_matls,
                                   const MaterialSet    * all_matls,
                                   customBC_globalVars  * BC_globalVars,
                                   const bool recursion)
{

  //__________________________________
  // compute surface normal and isSurfaceCell
  schedComputeSurfaceNormal( sched, patches);

  //__________________________________
  //  compute Mean Free Path
  schedComputeMeanFreePath( sched, patches );


  //__________________________________
  //
  std::string tName = "SlipExch::sched_AddExch_VelFC";

  Task* t = scinew Task( tName, this, &SlipExch::addExch_VelFC,
                         BC_globalVars, recursion);

  printSchedule( patches, dbgExch, tName );

  if(recursion) {
    t->requires(Task::ParentOldDW, Ilb->delTLabel,getLevel(patches));
  } else {
    t->requires(Task::OldDW,       Ilb->delTLabel,getLevel(patches));
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  const MaterialSet* mpm_ms       = d_sharedState->allMPMMaterials();
  const MaterialSubset* mpm_matls = mpm_ms->getUnion();

  //__________________________________
  // define parent data warehouse
  // change the definition of parent(old/new)DW
  // when using semi-implicit pressure solve
  Task::WhichDW pNewDW = Task::NewDW;
  Task::WhichDW pOldDW = Task::OldDW;
  if(recursion) {
    pNewDW  = Task::ParentNewDW;
    pOldDW  = Task::ParentOldDW;
  }

  // All matls
  t->requires( pNewDW,      Ilb->rho_CCLabel,     gac, 1);
  t->requires( pNewDW,      Ilb->sp_vol_CCLabel,  gac, 1);
  t->requires( pNewDW,      Ilb->vol_frac_CCLabel,gac, 1);
  t->requires( Task::NewDW, Ilb->uvel_FCLabel,    gac, 2);
  t->requires( Task::NewDW, Ilb->vvel_FCLabel,    gac, 2);
  t->requires( Task::NewDW, Ilb->wvel_FCLabel,    gac, 2);
  t->requires( Task::NewDW, d_meanFreePathLabel,   ice_matls,  gac, 1);
  t->requires( Task::NewDW, d_surfaceNormLabel,    mpm_matls,  gac, 1);
  t->requires( Task::NewDW, d_isSurfaceCellLabel,  d_zero_matl,gac, 1);

  t->requires( pOldDW,      Ilb->vel_CCLabel,      ice_matls,  gac, 1);
  t->requires( pNewDW,      Ilb->vel_CCLabel,      mpm_matls,  gac, 1);
  

  computesRequires_CustomBCs(t, "velFC_Exchange", Ilb, ice_matls,
                             BC_globalVars, recursion);

  t->computes( Ilb->sp_volX_FCLabel );
  t->computes( Ilb->sp_volY_FCLabel );
  t->computes( Ilb->sp_volZ_FCLabel );
  t->computes( Ilb->uvel_FCMELabel );
  t->computes( Ilb->vvel_FCMELabel );
  t->computes( Ilb->wvel_FCMELabel );

  sched->addTask(t, patches, all_matls);
}

/* _____________________________________________________________________
 Purpose~   Add the exchange contribution to vel_FC and compute
            sp_vol_FC for implicit Pressure solve
_____________________________________________________________________*/
template<class constSFC, class SFC>
void SlipExch::vel_FC_exchange( CellIterator  iter,
                                const IntVector     adj_offset,
                                const int           pDir,
                                const FastMatrix  & K,
                                const double        delT,
                                std::vector<constCCVariable<double> >& vol_frac_CC,
                                std::vector<constCCVariable<double> >& sp_vol_CC,
                                std::vector<constCCVariable<double> >& rho_CC,
                                std::vector<CCVariable<Vector> >     & vel_CC_exch,
                                std::vector< constSFC>               & vel_FC,
                                std::vector< SFC >                   & sp_vol_FC,
                                std::vector< SFC >                   & vel_FCME)
{
  double tmp[MAX_MATLS];
  double b_sp_vol[MAX_MATLS];
  FastMatrix a(d_numMatls, d_numMatls);

  //__________________________________
  //  Specific Volume Exchange
  //  For implicit solve we need sp_vol_FC
  for(;!iter.done(); iter++){
    IntVector R = *iter;
    IntVector L = R + adj_offset;

    for(int m = 0; m < d_numMatls; m++) {
      b_sp_vol[m] = 2.0 * (sp_vol_CC[m][L] * sp_vol_CC[m][R])/
                          (sp_vol_CC[m][L] + sp_vol_CC[m][R]);

      tmp[m] = 0.5 * delT * (vol_frac_CC[m][L] + vol_frac_CC[m][R]);
    }

    for(int m = 0; m < d_numMatls; m++) {
      double adiag = 1.0;
      for(int n = 0; n < d_numMatls; n++) {
        a(m,n) = -b_sp_vol[m] * tmp[n] * K(m,n);
        adiag -= a(m,n);
      }
      a(m,m) = adiag;
    }

    a.destructiveSolve(b_sp_vol);

    for(int m = 0; m < d_numMatls; m++) {
      sp_vol_FC[m][R] = b_sp_vol[m];
    }
  }

  //__________________________________
  //  Face Centered Velocity Exchange
  for(int m = 0; m < d_numMatls; m++) {
    for(;!iter.done(); iter++){
      IntVector R = *iter;
      IntVector L = R + adj_offset;

      Vector vel_R = vel_CC_exch[m][R];
      Vector vel_L = vel_CC_exch[m][L];
      double rho_R = rho_CC[m][R];
      double rho_L = rho_CC[m][L];
      
#if 0 // Jennifer's original code:
      double sp_vol_R = sp_vol_CC[m][R];
      double sp_vol_L = sp_vol_CC[m][L];
        
      double exchange = (sp_vol_L * vel_R[pDir] + sp_vol_R * vel_L[pDir])/(sp_vol_L + sp_vol_R);  // This looks like it has a bug --Todd
#endif

      double exchange = ( rho_L * vel_L[pDir] + rho_R * vel_R[pDir])/( rho_L + rho_R );
      vel_FCME[m][R]  = vel_FC[m][R] - exchange;
    }
  } 
}
//______________________________________________________________________
//
void SlipExch::addExch_VelFC(const ProcessorGroup  * pg,
                             const PatchSubset     * patches,
                             const MaterialSubset  * matls,
                             DataWarehouse         * old_dw,
                             DataWarehouse         * new_dw,
                             customBC_globalVars   * BC_globalVars,
                             const bool recursion)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing SlipExch::addExch_VelFC" );

    // change the definition of parent(old/new)DW
    // if using semi-implicit pressure solve
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
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMPMMatls = d_sharedState->getNumMPMMatls();

    constCCVariable<int>    isSurfaceCell;
    constCCVariable<Vector> vel_CC;

    std::vector< constCCVariable<double> > sp_vol_CC(    d_numMatls  );
    std::vector< constCCVariable<double> > rho_CC(       d_numMatls  );
    std::vector< constCCVariable<double> > vol_frac_CC(  d_numMatls  );

    std::vector< constCCVariable<double> > meanFreePath( numICEMatls );
    std::vector< constCCVariable<Vector> > surfaceNorm(  numMPMMatls );

    std::vector< constSFCXVariable<double> > uvel_FC( d_numMatls );
    std::vector< constSFCYVariable<double> > vvel_FC( d_numMatls );
    std::vector< constSFCZVariable<double> > wvel_FC( d_numMatls );

    std::vector< CCVariable<Vector> > notUsed(   d_numMatls );
    std::vector< CCVariable<Vector> > vel_CC_exch( d_numMatls );
    std::vector< SFCXVariable<double> >uvel_FCME( d_numMatls ), sp_vol_XFC( d_numMatls );
    std::vector< SFCYVariable<double> >vvel_FCME( d_numMatls ), sp_vol_YFC( d_numMatls );
    std::vector< SFCZVariable<double> >wvel_FCME( d_numMatls ), sp_vol_ZFC( d_numMatls );

    // Extract the momentum exchange coefficients
    FastMatrix K(d_numMatls, d_numMatls), junk(d_numMatls, d_numMatls);
    K.zero();

    d_exchCoeff->getConstantExchangeCoeff( K, junk);

    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->get( isSurfaceCell, d_isSurfaceCellLabel, 0, patch, gac, 1);

    //__________________________________
    //  Multimaterial arrays
    for(int m = 0; m < d_numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      int indx = matl->getDWIndex();
      
      // retreive from dw
      pNewDW->get( sp_vol_CC[m],   Ilb->sp_vol_CCLabel,  indx, patch,gac, 1);
      pNewDW->get( vol_frac_CC[m], Ilb->vol_frac_CCLabel,indx, patch,gac, 1);
      pNewDW->get( rho_CC[m],      Ilb->rho_CCLabel,     indx, patch,gac, 1);

      new_dw->get( uvel_FC[m],     Ilb->uvel_FCLabel,    indx, patch,gac, 2);
      new_dw->get( vvel_FC[m],     Ilb->vvel_FCLabel,    indx, patch,gac, 2);
      new_dw->get( wvel_FC[m],     Ilb->wvel_FCLabel,    indx, patch,gac, 2);

      if(mpm_matl) {
        pNewDW->get( vel_CC,         Ilb->vel_CCLabel,   indx, patch,gac, 1);
        new_dw->get( surfaceNorm[m], d_surfaceNormLabel, indx, patch,gac, 1);
      }

      if(ice_matl) {
        pOldDW->get( vel_CC,          Ilb->vel_CCLabel,    indx, patch,gac, 1);
        new_dw->get( meanFreePath[m], d_meanFreePathLabel, indx, patch,gac, 1);
      }

      // allocate
      new_dw->allocateTemporary( notUsed[m],     patch );
      new_dw->allocateTemporary( vel_CC_exch[m], patch );
      new_dw->allocateAndPut( uvel_FCME[m], Ilb->uvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( vvel_FCME[m], Ilb->vvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( wvel_FCME[m], Ilb->wvel_FCMELabel, indx, patch );

      new_dw->allocateAndPut( sp_vol_XFC[m],Ilb->sp_volX_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_YFC[m],Ilb->sp_volY_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_ZFC[m],Ilb->sp_volZ_FCLabel,indx, patch );

      // initialize
      vel_CC_exch[m].copyData( vel_CC );

      // lowIndex is the same for all face centered vars
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCXHighIndex() );
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCYHighIndex() );
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCZHighIndex() );

      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex() );
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex() );
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex() );
    }

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    CellIterator EC_iterator  = patch->getExtraCellIterator();
    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();

    //__________________________________
    //  compute CC exchange
    vel_CC_exchange( EC_iterator,  patch, K, delT,
                    isSurfaceCell,surfaceNorm, vol_frac_CC, sp_vol_CC, 
                    meanFreePath, notUsed, vel_CC_exch);


    //__________________________________
    //  tack on exchange contribution to FC velocities
    vel_FC_exchange<constSFCXVariable<double>, SFCXVariable<double> >
                    (XFC_iterator,
                    adj_offset[0],  0,           K,
                    delT,           vol_frac_CC, sp_vol_CC,  rho_CC,
                    vel_CC_exch,    uvel_FC,     sp_vol_XFC, uvel_FCME);

    vel_FC_exchange<constSFCYVariable<double>, SFCYVariable<double> >
                    (YFC_iterator,
                    adj_offset[1],  1,           K,
                    delT,           vol_frac_CC, sp_vol_CC,  rho_CC,
                    vel_CC_exch,    vvel_FC,     sp_vol_YFC, vvel_FCME);

    vel_FC_exchange<constSFCZVariable<double>, SFCZVariable<double> >
                    (ZFC_iterator,
                    adj_offset[2],  2,           K,
                    delT,           vol_frac_CC, sp_vol_CC,  rho_CC,
                    vel_CC_exch,    wvel_FC,     sp_vol_ZFC, wvel_FCME);

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
//  This method computes the cell centered exchange
void SlipExch::vel_CC_exchange( CellIterator  iter,
                                const Patch  * patch,
                                FastMatrix   & k_org,
                                const double   delT,
                                constCCVariable<int> & isSurfaceCell,
                                std::vector< constCCVariable<Vector> > & surfaceNorm,
                                std::vector< constCCVariable<double> > & vol_frac_CC,
                                std::vector< constCCVariable<double> > & sp_vol_CC,
                                std::vector< constCCVariable<double> > & meanFreePath,
                                std::vector< CCVariable<Vector> >      & vel_T_CC,
                                std::vector< CCVariable<Vector> >      & vel_CC)
{
  double b[MAX_MATLS];
  FastMatrix Q(3,3);
  FastMatrix K(d_numMatls, d_numMatls);
  FastMatrix Kslip(d_numMatls, d_numMatls);
  FastMatrix a(d_numMatls, d_numMatls);
  Vector vel_T[MAX_MATLS];                    // Transposed velocity

  // for readability
  int gm = d_fluidMatlIndx;
  int sm = d_solidMatlIndx;
  
  Vector dx = patch->dCell();

  for(;!iter.done(); iter++){
    IntVector c = *iter;

    Q.identity();         // Transformation matrix is initialized
    Kslip.copy(k_org);

    //__________________________________
    //  If cell is a surface cell modify exchange coefficients
    if( isSurfaceCell[c] ){
      
      // This should work for more that one solid.  It's hard wired for 2 matls-- Todd
      
      computeSurfaceRotationMatrix(Q, surfaceNorm[sm][c]); // Makes Q at each cell c

      double A_V = 1.0/( dx.x()*fabs(Q(1,0)) +
                         dx.y()*fabs(Q(1,1)) +
                         dx.z()*fabs(Q(1,2)) );

      double av     = d_momentum_accommodation_coeff;
      double Beta_v = (2 - av)/av;

      Kslip(gm,sm) = A_V / (Beta_v * meanFreePath[gm][c] * vol_frac_CC[sm][c]); // DOUBLE CHECK the material index of meanFreePath  -Todd

      if(Kslip(gm,sm) > k_org(gm,sm)) {
        Kslip(gm,sm) = k_org(gm,sm);                                            // K > Kslip in reality, so if Kslip > K in computation, fix this.
      }
      Kslip(sm,gm) = Kslip(gm,sm);                                              // Make the inverse indices of the Kslip matrix equal to each other
    }  // if a surface cell

    a.zero();

    //---------- M O M E N T U M   E X C H A N G E
    for(int i = 0; i < 3; i++) {

      if(i != 1) {                          // if the principle direction is NOT the normal to the surface
        K.copy(Kslip);                      // WARNING:  I don't understand this!    --Todd
      }

      //__________________________________
      //  coordinate Transformation
      for(int m = 0; m < d_numMatls; m++) {
        vel_T[m][i] = 0;

        for(int j = 0; j < 3; j++) {
          vel_T[m][i] += Q(i,j) * vel_CC[m][c][j];
        }
      }

      //__________________________________
      //  compute exchange using transposed velocity
      for(int m = 0; m < d_numMatls; m++) {
        double adiag = 1.0;
        b[m] = 0.0;

        for(int n = 0; n < d_numMatls; n++) {
          a(m,n) = - delT * vol_frac_CC[m][c] * sp_vol_CC[m][c] * K(m,n);  // double check equation --Todd
          adiag -= a(m,n);
          b[m]  -= a(m,n) * (vel_T[n][i] - vel_T[m][i]);
        }
        a(m,m) = adiag;
      }

      a.destructiveSolve(b);

      for(int m = 0; m < d_numMatls; m++) {
        vel_T[m][i] = b[m];
      }
    } // loop over directions

    //__________________________________
    //  coordinate transformation
    for(int m = 0; m < d_numMatls; m++) {
      vel_T_CC[m][c] = vel_T[m];               // for visualization

      Vector vel_exch(0);

      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          vel_exch[i] += Q(j,i) * vel_T[m][j]; // Use the transpose of Q to back out the velocity in the cartesian system
        }
      }
      vel_CC[m][c] += vel_exch;
    }
  }
}






//______________________________________________________________________
//
void SlipExch::sched_AddExch_Vel_Temp_CC(SchedulerP           & sched,
                                         const PatchSet       * patches,
                                         const MaterialSubset * ice_matls,
                                         const MaterialSubset * mpm_matls,
                                         const MaterialSet    * all_matls,
                                         customBC_globalVars  * BC_globalVars)
{


  //__________________________________
  //
  string name = "SlipExch::addExch_Vel_Temp_CC";
  Task* t = scinew Task(name, this, &SlipExch::addExch_Vel_Temp_CC, BC_globalVars);

  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gn  = Ghost::None;

  t->requires( Task::OldDW,  Ilb->delTLabel,getLevel(patches));
  t->requires( Task::NewDW,  d_surfaceNormLabel,    mpm_matls,  gn, 0 );
  t->requires( Task::NewDW, d_isSurfaceCellLabel,  d_zero_matl, gn, 0 );
                                // I C E
  t->requires( Task::OldDW,  Ilb->temp_CCLabel,       ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->specific_heatLabel, ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->gammaLabel,         ice_matls, gn );
  t->requires( Task::NewDW,  d_meanFreePathLabel,     ice_matls, gn );

                                // A L L  M A T L S
  t->requires( Task::NewDW,  Ilb->mass_L_CCLabel,    gn );
  t->requires( Task::NewDW,  Ilb->mom_L_CCLabel,     gn );
  t->requires( Task::NewDW,  Ilb->int_eng_L_CCLabel, gn );
  t->requires( Task::NewDW,  Ilb->sp_vol_CCLabel,    gn );
  t->requires( Task::NewDW,  Ilb->vol_frac_CCLabel,  gn );

  computesRequires_CustomBCs(t, "CC_Exchange", Ilb, ice_matls, BC_globalVars);

  t->computes( Ilb->Tdot_CCLabel );
  t->computes( Ilb->mom_L_ME_CCLabel );
  t->computes( Ilb->eng_L_ME_CCLabel );
  t->computes( d_vel_CCTransLabel );

  t->modifies( Ilb->temp_CCLabel, mpm_matls );
  t->modifies( Ilb->vel_CCLabel,  mpm_matls );

  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//
void SlipExch::addExch_Vel_Temp_CC(const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw,
                                   customBC_globalVars  * BC_globalVars)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, Ilb->timeStepLabel);
  bool isNotInitialTimeStep = (timeStep > 0);
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch,"Doing SlipExch::addExchangeToMomentumAndEnergy");

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMatls    = numMPMMatls + numICEMatls;
    Ghost::GhostType  gn = Ghost::None;

    delt_vartype delT;
    old_dw->get(delT, Ilb->delTLabel, level);

    // Create arrays for the grid data
    constCCVariable<int> isSurfaceCell;
    
    std::vector< CCVariable<double> >      cv( numMatls );
    std::vector< CCVariable<double> >      Temp_CC( numMatls );
    std::vector< constCCVariable<double> > gamma( numICEMatls );
    std::vector< constCCVariable<double> > vol_frac_CC( numMatls );
    std::vector< constCCVariable<double> > sp_vol_CC( numMatls );
    std::vector< constCCVariable<Vector> > mom_L( numMatls );
    std::vector< constCCVariable<double> > int_eng_L( numMatls );
    std::vector< constCCVariable<double> > mass_L( numMatls );
    std::vector< constCCVariable<double> > old_temp( numMatls );
    
    std::vector< constCCVariable<double> > meanFreePath( numICEMatls );  // This mean free path does not have viscosity in it, which is okay per how it is used in this code
    std::vector< constCCVariable<Vector> > surfaceNorm( numMPMMatls );
    
    // Create variables for the results
    std::vector< CCVariable<Vector> > mom_L_ME( numMatls );
    std::vector< CCVariable<Vector> > vel_CC( numMatls );
    std::vector< CCVariable<double> > int_eng_L_ME( numMatls );
    std::vector< CCVariable<double> > Tdot( numMatls );
    std::vector< CCVariable<Vector> > vel_T_CC( numMatls );


    new_dw->get( isSurfaceCell, d_isSurfaceCellLabel, 0, patch, gn, 0);
    
    for (int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);

      if(mpm_matl){                 // M P M
        CCVariable<double> oldTemp;
        new_dw->getCopy(       oldTemp,    Ilb->temp_CCLabel,indx, patch, gn,0 );
        new_dw->getModifiable( vel_CC[m],  Ilb->vel_CCLabel, indx, patch, gn,0 );
        new_dw->getModifiable( Temp_CC[m], Ilb->temp_CCLabel,indx, patch, gn,0 );
        old_temp[m] = oldTemp;
        cv[m].initialize(mpm_matl->getSpecificHeat());
      }

      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get( old_temp[m],     Ilb->temp_CCLabel,      indx, patch, gn, 0 );
        new_dw->get( cv_ice,          Ilb->specific_heatLabel,indx, patch, gn, 0 );
        new_dw->get( gamma[m],        Ilb->gammaLabel,        indx, patch, gn, 0 );
        new_dw->get( meanFreePath[m], d_meanFreePathLabel,    indx, patch, gn, 0 );

        new_dw->allocateTemporary( vel_CC[m],  patch );
        new_dw->allocateTemporary( Temp_CC[m], patch );
        cv[m].copyData(cv_ice);
      }                             // A L L  M A T L S

      new_dw->get( mass_L[m],        Ilb->mass_L_CCLabel,   indx, patch, gn, 0 );
      new_dw->get( sp_vol_CC[m],     Ilb->sp_vol_CCLabel,   indx, patch, gn, 0 );
      new_dw->get( mom_L[m],         Ilb->mom_L_CCLabel,    indx, patch, gn, 0 );
      new_dw->get( int_eng_L[m],     Ilb->int_eng_L_CCLabel,indx, patch, gn, 0 );
      new_dw->get( vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch, gn, 0 );

      new_dw->allocateAndPut( Tdot[m],         Ilb->Tdot_CCLabel,    indx, patch );   
      new_dw->allocateAndPut( mom_L_ME[m],     Ilb->mom_L_ME_CCLabel,indx, patch );   
      new_dw->allocateAndPut( int_eng_L_ME[m], Ilb->eng_L_ME_CCLabel,indx, patch );   
      new_dw->allocateAndPut( vel_T_CC[m],     d_vel_CCTransLabel,   indx, patch );   

      vel_T_CC[m].initialize(Vector(0,0,0));
    }

    // Convert momenta to velocities and internal energy to Temp
    for (int m = 0; m < numMatls; m++) {
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
      
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }

    // local variables
    double b[MAX_MATLS];
    FastMatrix Q(3,3);
    FastMatrix a(numMatls, numMatls);
    FastMatrix K(numMatls, numMatls);
    FastMatrix h(numMatls, numMatls);
    FastMatrix H(numMatls, numMatls);

    // Initialize
    K.zero();
    a.zero();
    h.zero();
    H.zero();

    d_exchCoeff->getConstantExchangeCoeff( K,h );

    //__________________________________
    //  compute CC velocity exchange
    CellIterator cell_iterator = patch->getCellIterator();
    
    vel_CC_exchange( cell_iterator,  patch, K, delT,
                    isSurfaceCell, surfaceNorm, vol_frac_CC, sp_vol_CC, 
                    meanFreePath,  vel_T_CC, vel_CC);


    //__________________________________
    //  E N E R G Y   E X C H A N G E
    
    int gm = d_fluidMatlIndx;
    int sm = d_solidMatlIndx;
        
    Vector dx = patch->dCell();   
    
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      H.copy(h);

      //__________________________________
      //  If cell is a surface cell modify exchange coefficients
      if ( isSurfaceCell[c] ){ 

        // This should work for more that one solid.  It's hard wired for 2 matls-- Todd

        computeSurfaceRotationMatrix(Q, surfaceNorm[sm][c]); // Makes Q at each cell c


	 double A_V = 1.0/( dx.x()*fabs(Q(1,0)) +
                           dx.y()*fabs(Q(1,1)) +
                           dx.z()*fabs(Q(1,2)) );

        // thermal
        double at     = d_thermal_accommodation_coeff;
        double Beta_t = ((2 - at)/at) * (2/(1 + gamma[gm][c])) * (1/cv[gm][c]);    // Does this assume an ideal gas????  -Todd

        H(gm,sm) = A_V / (Beta_t * meanFreePath[gm][c] * vol_frac_CC[sm][c]); // The viscosity does not appear here because it's taken out of lambda

	 if(H(gm,sm) > h(gm,sm)) {
	   H(gm,sm) = h(gm,sm);                                                // H > Hslip in reality, so if Hslip > H in computation, fix this.
	 }
	 H(sm,gm) = H(gm,sm);                                                  // Make inverse indices of the Kslip matrix equal to each other
      }  // if a surface cell


      for(int m = 0; m < numMatls; m++) {
        double adiag = 1.0;
        b[m] = 0.0;

        for(int n = 0; n < numMatls; n++) {
          a(m,n) = - delT * vol_frac_CC[m][c] * sp_vol_CC[m][c] * H(m,n) / cv[m][c];  // Here H has already been changed into Hslip, and Hslip is the same in every direction.
          adiag -= a(m,n);
          b[m]  -= a(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
        a(m,m) = adiag;
      }

      a.destructiveSolve(b);

      for(int m = 0; m < numMatls; m++) {
        Temp_CC[m][c] += b[m];
      }
    }  //end CellIterator loop

    //__________________________________
    //  Boundary Condition Code
    if( BC_globalVars->usingLodi || BC_globalVars->usingMicroSlipBCs){

      std::vector<CCVariable<double> > temp_CC_Xchange(numMatls);
      std::vector<CCVariable<Vector> > vel_CC_Xchange(numMatls);

      for (int m = 0; m < numMatls; m++) {
        Material* matl = d_sharedState->getMaterial(m);
        int indx = matl->getDWIndex();

        new_dw->allocateAndPut(temp_CC_Xchange[m], Ilb->temp_CC_XchangeLabel, indx, patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m],  Ilb->vel_CC_XchangeLabel,  indx, patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }

    //__________________________________
    //  Set boundary conditions
    for (int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();

      customBC_localVars* BC_localVars   = scinew customBC_localVars();
      preprocess_CustomBCs("CC_Exchange", old_dw, new_dw, Ilb, patch, indx, BC_globalVars, BC_localVars);

      setBC(vel_CC[m], "Velocity",   patch, d_sharedState, indx, new_dw,
                                                        BC_globalVars, BC_localVars, isNotInitialTimeStep);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_sharedState,
                                         indx, new_dw,  BC_globalVars, BC_localVars, isNotInitialTimeStep);
#if SET_CFI_BC
//      set_CFI_BC<Vector>(vel_CC[m],  patch);
//      set_CFI_BC<double>(Temp_CC[m], patch);
#endif
      delete_CustomBCs( BC_globalVars, BC_localVars );
    }

    //__________________________________
    // Convert vars. primitive-> flux
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;                                              // shouldn't the loop over matls be outside the cell iterator for speed. --Todd
      for (int m = 0; m < numMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c]*cv[m][c] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]           * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
      }
    }
  } //patches
}

//______________________________________________________________________
//
void SlipExch::schedComputeMeanFreePath(SchedulerP       & sched,
                                        const PatchSet   * patches)
{
  std::string tName =  "SlipExch::computeMeanFreePath";
  Task* t = scinew Task(tName, this, &SlipExch::computeMeanFreePath);

  printSchedule(patches, dbgExch, tName);

  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, Ilb->temp_CCLabel,       gn);
  t->requires(Task::OldDW, Ilb->sp_vol_CCLabel,     gn);
  t->requires(Task::NewDW, Ilb->gammaLabel,         gn);
  t->requires(Task::NewDW, Ilb->specific_heatLabel, gn);

  t->computes(d_meanFreePathLabel);

  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  sched->addTask(t, patches, ice_matls);
}

//______________________________________________________________________
//
void SlipExch::computeMeanFreePath(const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset *,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbgExch, "Doing SlipExch::computeMeanFreePath" );

    int numICEMatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType  gn = Ghost::None;

    for (int m = 0; m < numICEMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      constCCVariable<double> temp;
      constCCVariable<double> sp_vol;
      constCCVariable<double> gamma;
      constCCVariable<double> cv;
      CCVariable<double> meanFreePath;

      old_dw->get(temp,   Ilb->temp_CCLabel,      indx,patch, gn,0);
      old_dw->get(sp_vol, Ilb->sp_vol_CCLabel,    indx,patch, gn,0);
      new_dw->get(gamma,  Ilb->gammaLabel,        indx,patch, gn,0);
      new_dw->get(cv,     Ilb->specific_heatLabel,indx,patch, gn,0);

      new_dw->allocateAndPut(meanFreePath, d_meanFreePathLabel, indx, patch);
      meanFreePath.initialize(0.0);

      //This is really the mean free path divided by the dynamic viscosity
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        meanFreePath[c] = sp_vol[c] /sqrt(2 * cv[c] * (gamma[c] - 1) * temp[c] / M_PI);
      }
    }
  }
}

//______________________________________________________________________
//
void SlipExch::computeSurfaceRotationMatrix(FastMatrix   & Q,
                                            const Vector & surfaceNorm )
{
  Q.zero();
  
  if(fabs(surfaceNorm[1]) > 0.9999) {
    Q.identity();
  }else {                               //if coord.'s are not already aligned with surface
    Q(1,0) = surfaceNorm[0];
    Q(1,1) = surfaceNorm[1];
    Q(1,2) = surfaceNorm[2];

    double sqrtTerm = sqrt(1 - Q(1,1) * Q(1,1));
    double invTerm  = 1.0/sqrtTerm;

    Q(0,0) = Q(1,1) * Q(1,0) * invTerm;
    Q(0,1) =-sqrtTerm;
    Q(0,2) = Q(1,1) * Q(1,2) * invTerm;
    Q(2,0) =-Q(1,2) * invTerm;
    Q(2,1) = 0.0;
    Q(2,2) = Q(1,0) * invTerm;
  }
}

