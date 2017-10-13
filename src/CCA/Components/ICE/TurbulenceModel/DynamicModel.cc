/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/ICE/TurbulenceModel/DynamicModel.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Patch.h>

//#define DEBUGGING  // turn on computes for scratch* inside ofICE::scheduleViscousShearStress


using namespace Uintah;
using namespace std;

static DebugStream cout_doing("ICE_DOING_COUT", false);

DynamicModel::DynamicModel(ProblemSpecP& ps, SimulationStateP& sharedState)
  : Turbulence(ps, sharedState), d_smag() 
{ 
  //__________________________________ 
  //Filter_width=grid space(uniform) for implicit filter.
  //test_filter_width usually equal 2*filter_width

  ps->require("filter_width",d_filter_width);
//  ps->require("test_filter_width",d_test_filter_width);
  ps->require("model_constant", d_model_constant);

//  ps->require("turb_Pr",d_turbPr);

}

DynamicModel::~DynamicModel()
{
}

/* ---------------------------------------------------------------------
  Function~  computeTurbViscosity
  Purpose~ Calculate the turbulent viscosity
  -----------------------------------------------------------------------  */  
void DynamicModel::computeTurbViscosity(DataWarehouse* new_dw,
                                        const Patch* patch,
                                        const ICELabel* lb,
                                        constCCVariable<Vector>& vel_CC,
                                        constSFCXVariable<double>& uvel_FC,
                                        constSFCYVariable<double>& vvel_FC,
                                        constSFCZVariable<double>& wvel_FC,
                                        constCCVariable<double>& rho_CC,
                                        const int indx,
                                        SimulationStateP&  d_sharedState,
                                        CCVariable<double>& turb_viscosity)
{
  //-------- implicit filter, filter_width=(dx*dy*dz)**(1.0/3.0), 
  //--------don't use d_filter_width given in input file
  
  Vector dx = patch->dCell();
  filter_width = pow((dx.x()*dx.y()*dx.z()), 1.0/3.0);

  Ghost::GhostType  gac = Ghost::AroundCells;
  CCVariable<double> meanSIJ, term;

  new_dw->allocateTemporary(meanSIJ, patch, gac,2);  
  new_dw->allocateTemporary(term,    patch, gac,1);  
  term.initialize(0.0);
 
  computeSmagCoeff(new_dw, patch, lb, vel_CC, uvel_FC, vvel_FC, wvel_FC, 
                   indx, d_sharedState, term, meanSIJ);
  
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {
    IntVector c = *iter;
    turb_viscosity[c] = d_model_constant * rho_CC[c] * term[c] * meanSIJ[c];
  }
}
/* ---------------------------------------------------------------------
  Function~  applyFilter
  Purpose~ Calculate the filtered values
  -----------------------------------------------------------------------  */
template <class T, class V> 
  void DynamicModel::applyFilter(const Patch* patch, 
                                 T& var,
                                 CCVariable<V>& var_hat)
{ 
  int NGC =1;  // number of ghostCells
  V zero(0);
  
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) { 
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();
    
    var_hat[c] = zero;
    
    for (int kk = -1; kk <= 1; kk ++) {
      for (int jj = -1; jj <= 1; jj ++) {
        for (int ii = -1; ii <= 1; ii ++) {
          IntVector neighborCell = IntVector(i+ii,j+jj,k+kk);

          double temp = (1.0-0.5*abs(ii))*(1.0-0.5*abs(jj))*(1.0-0.5*abs(kk))/8;
          var_hat[c] += var[neighborCell]*temp;
        }
      }
    }   
//   At boundary, under developing
   }    
}

//______________________________________________________________________
// Explicit template instantiations:
template void DynamicModel::applyFilter< constCCVariable<Vector>, Vector >( const Patch* patch, constCCVariable<Vector>& var, CCVariable<Vector>& var_hat);
template void DynamicModel::applyFilter< CCVariable<double>, double >(      const Patch* patch, CCVariable<double>& var,      CCVariable<double>& var_hat);


/* ---------------------------------------------------------------------
  Function~  applyFilter
  Purpose~ Calculate the filtered values
  -----------------------------------------------------------------------  */
void DynamicModel::applyFilter(const Patch* patch, 
                               std::vector<CCVariable<double> >& var,
                               std::vector<CCVariable<double> >& var_hat)
{ 
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) { 
    IntVector c = *iter;

    int i = c.x();
    int j = c.y();
    int k = c.z();

    for (int comp = 0; comp < 6; comp++) {
      var_hat[comp][c]=0.0;   
    }     
               
    for (int kk = -1; kk <= 1; kk ++) {
      for (int jj = -1; jj <= 1; jj ++) {
        for (int ii = -1; ii <= 1; ii ++) {
          IntVector neighborCell = IntVector(i+ii,j+jj,k+kk);

          double temp = (1.0-0.5*abs(ii))*(1.0-0.5*abs(jj))*(1.0-0.5*abs(kk))/8;
          for (int comp = 0; comp < 6; comp++) {
             var_hat[comp][c] += var[comp][neighborCell]*temp;   
           }                 
         }
       }
     } 
  }//iter    
}

/* ---------------------------------------------------------------------
  Function~  ComputeSmagCoeff
  Purpose~ Calculate Cs
  -----------------------------------------------------------------------  */
void DynamicModel::computeSmagCoeff(DataWarehouse* new_dw,
                                    const Patch* patch,
                                    const ICELabel* lb,          // for debugging
                                    constCCVariable<Vector>& vel_CC,
                                    constSFCXVariable<double>& uvel_FC,
                                    constSFCYVariable<double>& vvel_FC,
                                    constSFCZVariable<double>& wvel_FC,
                                    const int indx,
                                    SimulationStateP&  d_sharedState,
                                    CCVariable<double>& term,
                                    CCVariable<double>& meanSIJ)
{
  double Cs, meanSIJ_hat;
  std::vector<CCVariable<double> > SIJ(6), SIJ_hat(6), LIJ(6), MIJ(6); 
  std::vector<CCVariable<double> > alpha(6), beta(6), beta_hat(6);
  CCVariable<Vector> vel_CC_hat;
  CCVariable<double> vel_prod, vel_prod_hat, LM, MM; 

  Ghost::GhostType  gac = Ghost::AroundCells;  
  for (int comp = 0; comp < 6; comp++) {
    new_dw->allocateTemporary(SIJ[comp],      patch, gac ,2);
    new_dw->allocateTemporary(LIJ[comp],      patch, gac ,1);
    new_dw->allocateTemporary(MIJ[comp],      patch, gac ,1);
    new_dw->allocateTemporary(alpha[comp],    patch, gac ,1);
    new_dw->allocateTemporary(beta[comp],     patch, gac ,2);
    new_dw->allocateTemporary(beta_hat[comp], patch, gac ,1);
    new_dw->allocateTemporary(SIJ_hat[comp],  patch, gac ,1);
    SIJ[comp].initialize(-9);
    beta[comp].initialize(-9);
    
    LIJ[comp].initialize(-9);
    MIJ[comp].initialize(-9);
  }
   
  new_dw->allocateTemporary(LM,           patch, gac ,1);
  new_dw->allocateTemporary(MM,           patch, gac ,1);
  new_dw->allocateTemporary(vel_prod,     patch, gac ,2);
  new_dw->allocateTemporary(vel_prod_hat, patch, gac ,1);
  new_dw->allocateTemporary(vel_CC_hat,   patch, gac ,1);
  
  /*===========DEBUGGING==========*/  
#ifdef DEBUGGING
  CCVariable<double> dbg0, dbg1, dbg2, dbg3, dbg4, dbg5;
  new_dw->allocateAndPut(dbg0,    lb->scratch0Label,    indx,patch);
  new_dw->allocateAndPut(dbg1,    lb->scratch0Label,    indx,patch);
  new_dw->allocateAndPut(dbg2,    lb->scratch0Label,    indx,patch);
  new_dw->allocateAndPut(dbg3,    lb->scratch0Label,    indx,patch);
  new_dw->allocateAndPut(dbg4,    lb->scratch0Label,    indx,patch);
  new_dw->allocateAndPut(dbg5,    lb->scratch0Label,    indx,patch);
#endif
  /*===========DEBUGGING==========*/ 
  
  //__________________________________
  //  compute mean strain rate
  d_smag.computeStrainRate(patch, uvel_FC, vvel_FC, wvel_FC, 
                           indx, d_sharedState, new_dw, SIJ);
 
  int NGC =2;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) { 
    IntVector c = *iter;

    meanSIJ[c] = 
        sqrt(2.0 * (SIJ[0][c]*SIJ[0][c] + SIJ[1][c]*SIJ[1][c] + SIJ[2][c]*SIJ[2][c] + 
             2.0 * (SIJ[3][c]*SIJ[3][c] + SIJ[4][c]*SIJ[4][c] + SIJ[5][c]*SIJ[5][c])));
         
    double A = 2.0 * filter_width * filter_width * meanSIJ[c];
    for (int comp = 0; comp < 6; comp ++ ) {
      beta[comp][c] = A * SIJ[comp][c];
    }
  }

  for (int comp = 0; comp < 6; comp++ ) {
    setZeroNeumannBC_CC( patch, beta[comp], NGC);
  } 

  applyFilter(patch, vel_CC,    vel_CC_hat);
  applyFilter(patch, SIJ,        SIJ_hat);  
  applyFilter(patch, beta,       beta_hat); 

  vector<IntVector> vel_prod_comp(6);    // ignore the z component
  vel_prod_comp[0] = IntVector(0,0,0);   // UUvel_CC      Using a map might be cleaner
  vel_prod_comp[1] = IntVector(1,1,0);   // VVvel_CC
  vel_prod_comp[2] = IntVector(2,2,0);   // WWvel_CC
  vel_prod_comp[3] = IntVector(0,1,0);   // UVvel_CC
  vel_prod_comp[4] = IntVector(0,2,0);   // UWvel_CC
  vel_prod_comp[5] = IntVector(1,2,0);   // VWvel_CC

  for( int i = 0; i < 6; i++ ){
    vel_prod.initialize(0.0);
    
    int comp0 = vel_prod_comp[i][0];
    int comp1 = vel_prod_comp[i][1];
      
    //__________________________________
    // compute velocity products and LIJ
    NGC = 2;
    for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++) { 
      IntVector c = *iter;
      vel_prod[c] = vel_CC[c][comp0] * vel_CC[c][comp1];
    }

    setZeroNeumannBC_CC( patch, vel_prod, NGC);

    applyFilter(patch, vel_prod,   vel_prod_hat); 

    NGC = 1;
    for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
      IntVector c = *iter;
      LIJ[i][c] = vel_prod_hat[c] - vel_CC_hat[c][comp0] * vel_CC_hat[c][comp1];
    }
  }
  
  //__________________________________
  //  Compute LM & MM
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
    IntVector c = *iter;    

    meanSIJ_hat = 
    sqrt(2.0 * (SIJ_hat[0][c]*SIJ_hat[0][c] + SIJ_hat[1][c]*SIJ_hat[1][c] + SIJ_hat[2][c]*SIJ_hat[2][c] + 
         2.0 * (SIJ_hat[3][c]*SIJ_hat[3][c] + SIJ_hat[4][c]*SIJ_hat[4][c] + SIJ_hat[5][c]*SIJ_hat[5][c])));

    //__________________________________
    //test filter width is assumed to be twice that of the basic filter 
    // instead of using d_test_filter_width given in input file
    double A  = 2.0 * (2*filter_width) * (2*filter_width) * meanSIJ_hat;
    
    for (int comp = 0; comp < 6; comp++ ) {
      alpha[comp][c] = A * SIJ_hat[comp][c];
      MIJ[comp][c]   = beta_hat[comp][c] - alpha[comp][c];
    }
    
    LM[c] = LIJ[0][c] * MIJ[0][c] + 
            LIJ[1][c] * MIJ[1][c] + 
            LIJ[2][c] * MIJ[2][c] +
       2 * (LIJ[3][c] * MIJ[3][c] + 
            LIJ[4][c] * MIJ[4][c] + 
            LIJ[5][c] * MIJ[5][c] );
                 
    MM[c] = MIJ[0][c] * MIJ[0][c] + 
            MIJ[1][c] * MIJ[1][c] + 
            MIJ[2][c] * MIJ[2][c] +
       2 * (MIJ[3][c] * MIJ[3][c] + 
            MIJ[4][c] * MIJ[4][c] + 
            MIJ[5][c] * MIJ[5][c] );
  }
  
/*==========DEBUGGING==========*/
#ifdef DEBUGGING
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {  
      IntVector c = *iter;

#if 0
      dbg0[c] = LIJ[0][c];
      dbg1[c] = LIJ[1][c];
      dbg2[c] = LIJ[2][c];
      dbg3[c] = LIJ[3][c];
      dbg4[c] = LIJ[4][c];
      dbg5[c] = LIJ[5][c];
#endif

      dbg0[c] = MIJ[0][c];
      dbg1[c] = MIJ[1][c];
      dbg2[c] = MIJ[2][c];
      dbg3[c] = MIJ[3][c];
      dbg4[c] = MIJ[4][c];
      dbg5[c] = MIJ[5][c];

#if 0
      dbg0[c] = SIJ[0][c];
      dbg1[c] = SIJ[1][c];
      dbg2[c] = SIJ[2][c];
      dbg3[c] = SIJ[3][c];
      dbg4[c] = SIJ[4][c];
      dbg5[c] = SIJ[5][c];
      
      dbg0[c] = beta[0][c];
      dbg1[c] = beta[1][c];
      dbg2[c] = beta[2][c];
      dbg3[c] = beta[3][c];
      dbg4[c] = beta[4][c];
      dbg5[c] = beta[5][c];
#endif
    } 
#endif
/*===========DEBUGGING==========*/  
      
  //__________________________________
  //calculate the local Smagorinsky coefficient
  //Cs is truncated to zero in case LM is negative 
  //to inhibit the potential for diverging solutions
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
    IntVector c = *iter;
    if (MM[c] < 1.0e-20) {
       term[c] = 0.0;
    } else {
      if(LM[c] <= 1.e-20){
        Cs = 0.0;
      } else {
        Cs = sqrt(LM[c] / MM[c]);
        if (Cs > 10.0){
          Cs = 10.0;
        }
      }
     term[c] = (Cs * filter_width) * (Cs * filter_width);
    } 
  }//iter
}
//______________________________________________________________________
//
void DynamicModel::scheduleComputeVariance(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* /*matls*/)
{
  if(filterScalars.size() > 0){
    for(int i=0;i<static_cast<int>(filterScalars.size());i++){
      FilterScalar* s = filterScalars[i];
      Task* task = scinew Task("DynamicModel::computeVariance",this, 
                               &DynamicModel::computeVariance, s);
                               
      task->requires(Task::OldDW, s->scalar, Ghost::AroundCells, 1);
      task->computes(s->scalarVariance);
      sched->addTask(task, patches, s->matl_set);
    }
  }
}
//______________________________________________________________________
//
void DynamicModel::computeVariance(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw,
                                   FilterScalar* s)
{
  cout_doing << "Doing computeVariance "<< "\t\t\t DynamicModel" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      constCCVariable<double> f;
      CCVariable<double> fvar;
      
      old_dw->get(f, s->scalar, matl, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(fvar, s->scalarVariance, matl, patch);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        const IntVector& c = *iter;

        double sum_f = 0;
        double sum_fsquared = 0;
        
        for (int kk = -1; kk <= 1; kk ++) {
          for (int jj = -1; jj <= 1; jj ++) {
            for (int ii = -1; ii <= 1; ii ++) {
            
              IntVector neighborCell = c+IntVector(ii,jj,kk);
              double weight = (1.0-0.5*abs(ii))*(1.0-0.5*abs(jj))*(1.0-0.5*abs(kk))/8;
              double value = f[neighborCell];
              sum_f += weight*value;
              sum_fsquared += weight*value*value;
            
            }
          }
        }
        fvar[c] = sum_fsquared - sum_f*sum_f;
      }
      
      // I'm not sure if this is correct --Todd
      setBC(fvar,s->scalarVariance->getName(),patch, d_sharedState, matl, new_dw);
    }
  }
}
