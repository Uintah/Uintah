#include <Packages/Uintah/CCA/Components/ICE/DynamicModel.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Patch.h>

using namespace Uintah;

DynamicModel::DynamicModel(ProblemSpecP& ps)
   : d_smag() 
{ 
  //__________________________________ 
  //Filter_width=grid space(uniform) for implicit filter.
  //test_filter_width usually equal 2*filter_width

  ps->require("filter_width",d_filter_width);
  ps->require("test_filter_width",d_test_filter_width);
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
                                        const CCVariable<Vector>& vel_CC,
                                        const SFCXVariable<double>& uvel_FC,
                                        const SFCYVariable<double>& vvel_FC,
                                        const SFCZVariable<double>& wvel_FC,
                                        const CCVariable<double>& rho_CC,
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
 
  computeSmagCoeff(new_dw, patch, vel_CC, uvel_FC, vvel_FC, wvel_FC, 
                   indx, d_sharedState, term, meanSIJ);

  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);   
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;    
    turb_viscosity[c] = rho_CC[c] * term[c] * meanSIJ[c];
   }
}
/* ---------------------------------------------------------------------
  Function~  applyFilter
  Purpose~ Calculate the filtered values
  -----------------------------------------------------------------------  */
template <class T> void DynamicModel::applyFilter(const Patch* patch, 
                                                  CCVariable<T>& var,
                                                  CCVariable<T>& var_hat)
{ 

  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);  
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {   
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();
   
/*    var_hat[c] = var[c]/8 +
            var[IntVector(i-1,j,  k  )]/16 + var[IntVector(i+1,j,  k  )]/16 +
            var[IntVector(i,  j-1,k  )]/16 + var[IntVector(i,  j+1,k  )]/16 +
            var[IntVector(i-1,j-1,k  )]/32 + var[IntVector(i+1,j-1,k  )]/32 +
            var[IntVector(i-1,j+1,k  )]/32 + var[IntVector(i+1,j+1,k  )]/32 +
            var[IntVector(i,  j,  k-1)]/16 +
            var[IntVector(i-1,j,  k-1)]/32 + var[IntVector(i+1,j,  k-1)]/32 +
            var[IntVector(i,  j-1,k-1)]/32 + var[IntVector(i,  j+1,k-1)]/32 +
            var[IntVector(i-1,j-1,k-1)]/64 + var[IntVector(i+1,j-1,k-1)]/64 +
            var[IntVector(i-1,j+1,k-1)]/64 + var[IntVector(i+1,j+1,k-1)]/64 +
            var[IntVector(i,  j,  k+1)]/16 +
            var[IntVector(i-1,j,  k+1)]/32 + var[IntVector(i+1,j,  k+1)]/32 +
            var[IntVector(i,  j-1,k+1)]/32 + var[IntVector(i,  j+1,k+1)]/32 +
            var[IntVector(i-1,j-1,k+1)]/64 + var[IntVector(i+1,j-1,k+1)]/64 +
            var[IntVector(i-1,j+1,k+1)]/64 + var[IntVector(i+1,j+1,k+1)]/64;*/

      for (int kk = -1; kk <= 1; kk ++) {
        for (int jj = -1; jj <= 1; jj ++) {
          for (int ii = -1; ii <= 1; ii ++) {
           IntVector neighbourCell = IntVector(i+ii,j+jj,k+kk);
           double temp = (1.0-0.5*abs(ii))*(1.0-0.5*abs(jj))*(1.0-0.5*abs(kk))/8;
           var_hat[c] += var[neighbourCell]*temp;
          }
        }
      }   

//   At boundary, under developing

          
       
   }    
}

/* ---------------------------------------------------------------------
  Function~  applyFilter
  Purpose~ Calculate the filtered values
  -----------------------------------------------------------------------  */
void DynamicModel::applyFilter(const Patch* patch, 
                               StaticArray<CCVariable<double> >& var,
                               StaticArray<CCVariable<double> >& var_hat)
{ 

  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);  
    
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {   
    IntVector c = *iter;

    int i = c.x();
    int j = c.y();
    int k = c.z();

  for (int comp = 0; comp <= 5; comp++) {
    var_hat[comp][c]=0.0;   
  }     
               
    for (int kk = -1; kk <= 1; kk ++) {
      for (int jj = -1; jj <= 1; jj ++) {
        for (int ii = -1; ii <= 1; ii ++) {
          IntVector neighbourCell = IntVector(i+ii,j+jj,k+kk);
          double temp = (1.0-0.5*abs(ii))*(1.0-0.5*abs(jj))*(1.0-0.5*abs(kk))/8;
          for (int comp = 0; comp <= 5; comp++) {
             var_hat[comp][c] += var[comp][neighbourCell]*temp;   
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
                                    const CCVariable<Vector>& vel_CC,
                                    const SFCXVariable<double>& uvel_FC,
                                    const SFCYVariable<double>& vvel_FC,
                                    const SFCZVariable<double>& wvel_FC,
                                    const int indx,
                                    SimulationStateP&  d_sharedState,
                                    CCVariable<double>& term,
                                    CCVariable<double>& meanSIJ)
{  
  double Cs, meanSIJ_hat;
  StaticArray<CCVariable<double> > SIJ(6), SIJ_hat(6), LIJ(6), MIJ(6); 
  StaticArray<CCVariable<double> > alpha(6), beta(6), beta_hat(6);
  CCVariable<Vector> vel_CC_tmp, vel_CC_hat;
  CCVariable<double> vel_prod, vel_prod_hat, LM, MM; 

  Ghost::GhostType  gac = Ghost::AroundCells;  
  for (int comp = 0; comp <= 5; comp++) {
    new_dw->allocateTemporary(SIJ[comp],      patch, gac ,2);
    new_dw->allocateTemporary(LIJ[comp],      patch, gac ,1);
    new_dw->allocateTemporary(MIJ[comp],      patch, gac ,1);
    new_dw->allocateTemporary(alpha[comp],    patch, gac ,1);
    new_dw->allocateTemporary(beta[comp],     patch, gac ,2);
    new_dw->allocateTemporary(beta_hat[comp], patch, gac ,1);
    new_dw->allocateTemporary(SIJ_hat[comp],  patch, gac ,1);
    SIJ[comp].initialize(0.0);
    beta[comp].initialize(0.0);    
  }  
  new_dw->allocateTemporary(LM,           patch, gac ,1);
  new_dw->allocateTemporary(MM,           patch, gac ,1);   
  new_dw->allocateTemporary(vel_prod,     patch, gac ,2);
  new_dw->allocateTemporary(vel_prod_hat, patch, gac ,1);
  new_dw->allocateTemporary(vel_CC_tmp,   patch, gac ,2);
  new_dw->allocateTemporary(vel_CC_hat,   patch, gac ,1);  

  vel_CC_tmp.initialize(Vector(0.0,0.0,0.0));
  vel_CC_hat.initialize(Vector(0.0,0.0,0.0));   
 
  d_smag.computeStrainRate(patch, uvel_FC, vvel_FC, wvel_FC, 
                           indx, d_sharedState, SIJ);

  CellIterator iter = patch->getCellIterator();
  CellIterator iterPlusGhost  = patch->addGhostCell_Iter(iter,1);
  CellIterator iterPlus2Ghost = patch->addGhostCell_Iter(iter,2);
    
  for(CellIterator iter = iterPlus2Ghost; !iter.done(); iter++) { 
   IntVector c = *iter;
   
   vel_CC_tmp[c] = vel_CC[c];
   meanSIJ[c] = 
    sqrt(2.0 * (SIJ[0][c]*SIJ[0][c] + SIJ[1][c]*SIJ[1][c] + SIJ[2][c]*SIJ[2][c] + 
         2.0 * (SIJ[3][c]*SIJ[3][c] + SIJ[4][c]*SIJ[4][c] + SIJ[5][c]*SIJ[5][c])));
         
    double A = 2.0 * filter_width * filter_width * meanSIJ[c];
    for (int comp = 0; comp < 6; comp ++ ) {
      beta[comp][c] = A * SIJ[comp][c];
    }
  }
   
  setBC(vel_CC_tmp,"Velocity", patch, d_sharedState, indx);

  for (int comp = 0; comp < 6; comp++ ) {
    setBC(beta[comp],"zeroNeumann",patch, d_sharedState, indx);
  } 

  applyFilter(patch, vel_CC_tmp, vel_CC_hat); //need vel_CC_tmp for the template function
  applyFilter(patch, SIJ,        SIJ_hat);  
  applyFilter(patch, beta,       beta_hat); 

  vector<IntVector> vel_prod_comp(6);  // ignore the z component
  vel_prod_comp[0] = IntVector(0,0,0);   // UUvel_CC
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
    // compute velocity products

    for(CellIterator iter = iterPlus2Ghost; !iter.done(); iter++) { 
     IntVector c = *iter;
     vel_prod[c] = vel_CC[c][comp0] * vel_CC[c][comp1];
    }
    
    setBC(vel_prod,"zeroNeumann",patch, d_sharedState, indx);
    
    vel_prod_hat.initialize(0.0);
    
    applyFilter(patch, vel_prod,   vel_prod_hat); 
      
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      IntVector c = *iter;
      LIJ[i][c] = vel_prod_hat[c] - vel_CC_hat[c][comp0] * vel_CC_hat[c][comp1];
    }
  }
  
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
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
                LIJ[5][c] * MIJ[5][c]);
                 
    MM[c] = MIJ[0][c] * MIJ[0][c] + 
                 MIJ[1][c] * MIJ[1][c] + 
                 MIJ[2][c] * MIJ[2][c] +
           2 * (MIJ[3][c] * MIJ[3][c] + 
                 MIJ[4][c] * MIJ[4][c] + 
                 MIJ[5][c] * MIJ[5][c] );
            
   }    
   //__________________________________
   //calculate the local Smagorinsky coefficient
   //Cs is truncated to zero in case LM is negative 
   //to inhibit the potential for diverging solutions
   for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
     IntVector c = *iter;
     if (MM[c] < 1.0e-20) {
        term[c] = 0.0;
      }        
     else{   
       Cs = sqrt(LM[c] / MM[c]);
       if (Cs < 0.0){       
         Cs = 0.0;
         }
       else if (Cs > 10.0){
         Cs = 10.0;
        }
       term[c] = (Cs * filter_width) * (Cs * filter_width);
     } 
   }//iter
}


