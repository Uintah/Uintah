/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#include <CCA/Components/ICE/CustomBCs/MMS_BCs.h>
#include <CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <CCA/Components/ICE/CustomBCs/LODI2.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/Stencil7.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/StaticArray.h>
#include <time.h>

#include <CCA/Components/ICE/uintahshare.h>
namespace Uintah {
 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);

  class DataWarehouse;

  void is_BC_specified(const ProblemSpecP& prob_spec, string variable);
  
  void BC_bulletproofing(const ProblemSpecP& prob_spec,SimulationStateP& sharedState );
  
  //__________________________________
  //  Temperature, pressure and other CCVariables
  UINTAHSHARE void setBC(CCVariable<double>& var,     
                      const std::string& type,
                      const CCVariable<double>&gamma,
                      const CCVariable<double>&cv, 
                      const Patch* patch,  
                      SimulationStateP& sharedState,
                      const int mat_id,
                      DataWarehouse* new_dw,
                      customBC_var_basket* C_BC_basket);
            
  UINTAHSHARE void setBC(CCVariable<double>& var,     
                      const std::string& type,     // stub function
                      const Patch* patch,  
                      SimulationStateP& sharedState,
                      const int mat_id,
                      DataWarehouse* new_dw); 
  //__________________________________
  //  P R E S S U R E        
  UINTAHSHARE void setBC(CCVariable<double>& press_CC,          
                      StaticArray<CCVariable<double> >& rho_micro,
                      StaticArray<constCCVariable<double> >& sp_vol,
                      const int surroundingMatl_indx,
                      const std::string& whichVar, 
                      const std::string& kind, 
                      const Patch* p, 
                      SimulationStateP& sharedState,
                      const int mat_id, 
                      DataWarehouse* new_dw,
                      customBC_var_basket* C_BC_basket);
             
  UINTAHSHARE void setBC(CCVariable<double>& press_CC,          
                      StaticArray<CCVariable<double> >& rho_micro,
                      StaticArray<constCCVariable<double> >& sp_vol,
                      const int surroundingMatl_indx,
                      const std::string& whichVar, 
                      const std::string& kind,       // stub function 
                      const Patch* p, 
                      SimulationStateP& sharedState,
                      const int mat_id, 
                      DataWarehouse* new_dw);
             
  //__________________________________
  //    V E C T O R   
  UINTAHSHARE void setBC(CCVariable<Vector>& variable,
                      const std::string& type,
                      const Patch* patch,
                      SimulationStateP& sharedState,
                      const int mat_id,
                      DataWarehouse* new_dw, 
                      customBC_var_basket* C_BC_basket);
             
  UINTAHSHARE void setBC(CCVariable<Vector>& variable,  // stub function
                      const std::string& type,
                      const Patch* patch,
                      SimulationStateP& sharedState,
                      const int mat_id,
                      DataWarehouse* new_dw);

  //__________________________________
  //    SPECIFC VOLUME
  UINTAHSHARE void setSpecificVolBC(CCVariable<double>& sp_vol,
                                 const string& kind,
                                 const bool isMassSp_vol,
                                 constCCVariable<double> rho_CC,
                                 constCCVariable<double> vol_frac,
                                 const Patch* patch,
                                 SimulationStateP& sharedState,
                                 const int mat_id);
  

  void set_imp_DelP_BC( CCVariable<double>& imp_delP, 
                        const Patch* patch,
                        const VarLabel* label,
                        DataWarehouse* new_dw);  
  
  
  void set_CFI_BC( CCVariable<double>& q_CC, const Patch* patch);
  
  
template<class T> 
  void setBC(T& variable, 
             const  string& kind,
             const string& comp,    
             const Patch* patch,    
             const int mat_id);
                        
template<class T>
 bool setNeumanDirichletBC( const Patch* patch,
                            const Patch::FaceType face,
                            CCVariable<T>& var,
                            Iterator& bound_ptr,
                            const string& bc_kind,
                            const T& value,
                            const Vector& cell_dx,
                            const int mat_id,
                            const int child);
                            
 template<class T>
 bool setNeumanDirichletBC_FC( const Patch* patch,
                               const Patch::FaceType face,
                               T& vel_FC,
                               Iterator& bound_ptr,
                               string& bc_kind,
                               double& value,
                               const Vector& cell_dx,
                               const IntVector& P_dir,
                               const string& whichVel);
  
  void ImplicitMatrixBC(CCVariable<Stencil7>& var, const Patch* patch);
 
/* --------------------------------------------------------------------- 
 Function~  getIteratorBCValueBCKind--
 Purpose~   does the actual work
 ---------------------------------------------------------------------  */
template <class T>
bool getIteratorBCValueBCKind( const Patch* patch, 
                               const Patch::FaceType face,
                               const int child,
                               const string& desc,
                               const int mat_id,
                               T& bc_value,
                               Iterator& bound_ptr,
                               string& bc_kind)
{ 
  //__________________________________
  //  find the iterator, BC value and BC kind
  Iterator nu;  // not used

  const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
						    desc, bound_ptr,
                                                    nu, child);

  const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
						       "Symmetric", bound_ptr, 
							nu, child);


  const BoundCond<T> *new_bcs =  dynamic_cast<const BoundCond<T> *>(bc);       

  bc_value=T(-9);
  bc_kind="NotSet";
  if (new_bcs != 0) {      // non-symmetric
    bc_value = new_bcs->getValue();
    bc_kind = new_bcs->getBCType__NEW();
  }        
  if (sym_bc != 0 && sym_bc->getBCType__NEW() == "symmetry") {  // symmetric
    bc_kind = "symmetric";
  }
  if (desc == "zeroNeumann" ){
    bc_kind = "zeroNeumann";
  }
  delete bc;
  delete sym_bc;

  // Did I find an iterator
  if( bc_kind == "NotSet" ){
    return false;
  }else{
    return true;
  }    
}
/* --------------------------------------------------------------------- 
 Function~  setNeumanDirichletBC--
 Purpose~   does the actual work of setting the BC for the simple BC
 ---------------------------------------------------------------------  */
 template<class T>
 bool setNeumanDirichletBC( const Patch* patch,
                            const Patch::FaceType face,
                            CCVariable<T>& var,
                            Iterator& bound_ptr,
                            string& bc_kind,
                            T& value,
                            const Vector& cell_dx,
                            const int mat_id,
                            const int child)
{
 IntVector oneCell = patch->faceDirection(face);
 IntVector dir= patch->getFaceAxes(face);
 double dx = cell_dx[dir[0]];

 bool IveSetBC = false;

 if (bc_kind == "Neumann" && value == T(0)) { 
   bc_kind = "zeroNeumann";  // for speed
 }
 //__________________________________        
 if (bc_kind == "Dirichlet") {    //   D I R I C H L E T 
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     var[*bound_ptr] = value;
   }
   IveSetBC = true;
 }
 //__________________________________
 // Random variations for density
 if (bc_kind == "Dirichlet_perturbed") {
   Iterator nu1,nu2;  // not used
   const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
						     "Density", 
                                                     nu1,nu2,child);


   const BoundCond<double> *density_bcs = 
     dynamic_cast<const BoundCond<double> *>(bc);
   
   double K=0.;
   if (density_bcs)
     K = density_bcs->getValue();


   // Seed the random number generator with the number of seconds since 
   // midnight Jan. 1, 1970.

   time_t seconds = time(NULL);
   srand(seconds);

   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     var[*bound_ptr] = value + K*((double(rand())/RAND_MAX)*2.- 1.)*value;
   }
   IveSetBC = true;
   delete bc;
 }

 if (bc_kind == "Neumann") {       //    N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell] - value * dx;
   }
   IveSetBC = true;
 }
 if (bc_kind == "zeroNeumann") {   //    Z E R O  N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell];
   }
   IveSetBC = true;
   value = T(0.0);   // so the debugging output is accurate
 }
 return IveSetBC;

}


/* --------------------------------------------------------------------- 
 Function~  setNeumanDirichletBC_FC--
 Purpose~   does the actual work of setting the BC for face-centered 
            velocities
 ---------------------------------------------------------------------  */
 template<class T>
 bool setNeumanDirichletBC_FC( const Patch* patch,
                               const Patch::FaceType face,
                               T& vel_FC,
                               Iterator& bound_ptr,
                               string& bc_kind,
                               double& value,
                               const Vector& cell_dx,
                               const IntVector& P_dir,
                               const string& whichVel)
{

  if(bc_kind == "Neumann" && value == 0.0){
    bc_kind = "zeroNeumann";   // speedup
  }

  bool IveSetBC = false;
  IntVector oneCell = patch->faceDirection(face);
  bool onMinusFace = false;
  //__________________________________
  // Dirichlet  -- can be set on any face
  if (bc_kind == "Dirichlet") {
    
    if ( (whichVel == "X_vel_FC" && face == Patch::xminus) || 
         (whichVel == "Y_vel_FC" && face == Patch::yminus) || 
         (whichVel == "Z_vel_FC" && face == Patch::zminus)){
      onMinusFace = true;
    }
    // on (x,y,z)minus faces move in one cell
    if( onMinusFace ) {
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        IntVector c = *bound_ptr - oneCell;
        vel_FC[c] = value;
      }
    }else {    // (xplus, yplus, zplus) faces
      for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
        IntVector c = *bound_ptr;
        vel_FC[c] = value;
 
      }
    }
    IveSetBC = true;
  }
  
  
//______________________________________________________________________
//  Theoretically, we don't need to set the transverse face centered velocities
//  boundary conditions.  This is from the days of doing corner coupling.  Not
//  positive that it won't create a problem. Leave this weed for while
//  01/31/07  
#if 0
  //__________________________________
  // Neumann
  // -- Only modify the velocities that are tangential to a face.
  //    find dx, sign on that face, and direction face is pointing  
  IntVector faceDir_tmp = patch->faceDirection(face);
  IntVector faceDir     = Abs(faceDir_tmp);
  IntVector dir = patch->getFaceAxes(face);
  double sign = faceDir_tmp[dir[0]];
  double dx   = cell_dx[dir[0]];

  if (bc_kind == "Neumann" && (faceDir != P_dir) ){
    IveSetBC = true;
    
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector adjCell = *iter - oneCell;
      vel_FC[*iter] = vel_FC[adjCell] + value*dx*sign;
    }  
  }
  //__________________________________
  //  zero Neumann
  // -- Only modify the velocities that are tangential to a face.
  if (bc_kind == "zeroNeumann" && (faceDir != P_dir) ){
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector adjCell = *iter - oneCell;
      vel_FC[*iter] = vel_FC[adjCell];
    }
    IveSetBC = true; 
    value = 0.0;  // so the debugging output is accurate 
  } 
#endif
  return IveSetBC; 
}

/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes capre of face centered velocities
 Note:      Neumann BC values are only set on the transverse faces, 
            The normal components are computed in 
            AddExchangeContributionToFCVel.
 ---------------------------------------------------------------------  */
 template<class T> 
void setBC(T& vel_FC, 
           const string& desc,
           const Patch* patch,    
           const int mat_id,
           SimulationStateP& sharedState,
           customBC_var_basket* custom_BC_basket)      
{
  BC_doing << "setBCFC (SFCVariable) "<< desc<< " mat_id = " << mat_id <<endl;
  Vector cell_dx = patch->dCell();
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;

  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    bool IveSetBC = false;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    for (int child = 0;  child < numChildren; child++) {

      Vector bc_value(-9,-9,-9);;
      string bc_kind = "NotSet";
      Iterator bound_ptr;
      bool foundIterator = 
        getIteratorBCValueBCKind<Vector>( patch, face, child, desc, mat_id,
					       bc_value, bound_ptr,bc_kind); 
                                       
      if(bc_kind == "LODI") {
        BC_dbg << "Face: "<<face<< " LODI bcs specified: do Nothing"<< endl; 
      }
      
      if (foundIterator && bc_kind != "LODI" ) {
        //__________________________________
        // Extract which SFC variable you're
        //  working on, the value and the principal
        //  direction
        double value=-9;
        IntVector P_dir(0,0,0);  // principal direction
        string whichVel = "";
        if (typeid(T) == typeid(SFCXVariable<double>)) {
          P_dir = IntVector(1,0,0);
          value = bc_value.x();
          whichVel = "X_vel_FC";
        }
        if (typeid(T) == typeid(SFCYVariable<double>)) {
          P_dir = IntVector(0,1,0);
          value = bc_value.y();
          whichVel = "Y_vel_FC";
        }
        if (typeid(T) == typeid(SFCZVariable<double>)) {
          P_dir = IntVector(0,0,1);
          value = bc_value.z();
          whichVel = "Z_vel_FC";
        }

        //__________________________________
        //  Symmetry boundary conditions
        //  -- faces not in the principal dir: vel[c] = vel[interior]
        //  -- faces in the principal dir:     vel[c] = 0
        IntVector faceDir = Abs(patch->faceDirection(face));
        if (bc_kind == "symmetric") {        
          // Other face direction
          string kind = "zeroNeumann";
          value = 0.0;
          IveSetBC= setNeumanDirichletBC_FC<T>( patch, face, vel_FC,
                               bound_ptr, kind, value, cell_dx, P_dir, whichVel);

          if(faceDir == P_dir ) {
            string kind = "Dirichlet";
            IveSetBC= setNeumanDirichletBC_FC<T>( patch, face, vel_FC,
                                 bound_ptr, kind, value, cell_dx, P_dir, whichVel);
          }
        }

        //__________________________________
        // Non Symmetric Boundary Conditions
        if (bc_kind != "symmetric") {  
          IveSetBC= setNeumanDirichletBC_FC<T>( patch, face, vel_FC,
                              bound_ptr, bc_kind, value, cell_dx, P_dir, whichVel); 
        }
        //__________________________________
        // Custom BCs
        if(bc_kind == "MMS_1"){
          IveSetBC= set_MMS_BCs_FC<T>(patch, face, vel_FC, bound_ptr, bc_kind,
                                      cell_dx, P_dir, whichVel, sharedState,
                                      custom_BC_basket->mms_var_basket,
                                      custom_BC_basket->mms_v);
        }
        //__________________________________
        // Custom BCs
        if(bc_kind == "Sine"){
          IveSetBC= set_Sine_BCs_FC<T>(patch, face, vel_FC, bound_ptr, bc_kind,
                                      cell_dx, P_dir, whichVel, sharedState,
                                      custom_BC_basket->sine_var_basket,
                                      custom_BC_basket->sine_v);
        }         
        
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<whichVel<< " Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< value
               <<"\t bound limits = " << bound_ptr.begin()<<" "<< (bound_ptr.end())


	        << endl;
        }              
      }  // Children loop
    }  // bcKind != notSet
  }  // face loop
}
/* --------------------------------------------------------------------- 
 Function~  set_CFI_BC--      
 Purpose~  set the boundary condition at the coarse fine interface.  Use
  A Taylor's series expansion using only fine level data
 ---------------------------------------------------------------------  */
template <class T>
void set_CFI_BC( CCVariable<T>& q_CC, const Patch* patch)        
{ 
  BC_doing << "set_CFI_BC "<< endl; 
  //__________________________________
  // On the fine levels at the coarse fine interface 
  BC_dbg << *patch << " ";
  patch->printPatchBCs(BC_dbg);

  if(patch->hasCoarseFaces() ){  
    BC_dbg << " BC at coarse/Fine interfaces " << endl;
    //__________________________________
    // Iterate over coarsefine interface faces
    vector<Patch::FaceType> cf;
    patch->getCoarseFaces(cf);
    vector<Patch::FaceType>::const_iterator iter;  
    for (iter  = cf.begin(); iter != cf.end(); ++iter){
      Patch::FaceType face = *iter;
      
      IntVector oneCell = patch->faceDirection(face);
      int p_dir = patch->getFaceAxes(face)[0];  //principal dir.
      Vector dx = patch->dCell();
      
      Patch::FaceIteratorType MEC = Patch::ExtraMinusEdgeCells;
      for(CellIterator itr = patch->getFaceIterator(face, MEC); !itr.done(); itr++){
        IntVector f_cell = *itr;
        IntVector f_adj  = f_cell  - oneCell;
        IntVector f_adj2 = f_cell  - IntVector(2,2,2)*oneCell;
        
        // backward differencing
        T grad_q = (q_CC[f_adj] - q_CC[f_adj2])/dx[p_dir];
        T q_new  =  q_CC[f_adj] + grad_q * dx[p_dir]; 
            
        T correction =  q_CC[f_cell] - q_new; 
        q_CC[f_cell] = q_new;
      }
    }  // face loop
  }  // patch has coarse fine interface 
}
} // End namespace Uintah
#endif
