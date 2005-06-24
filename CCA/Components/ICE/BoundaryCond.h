#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/MMS_BCs.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/microSlipBCs.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/NG_NozzleBCs.h>
#include <Packages/Uintah/CCA/Components/ICE/CustomBCs/LODI2.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/DensityBoundCond.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCond.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/StaticArray.h>
#include <time.h>

namespace Uintah {
 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);

  class DataWarehouse;

  void is_BC_specified(const ProblemSpecP& prob_spec, string variable);
  
  void BC_bulletproofing(const ProblemSpecP& prob_spec,SimulationStateP& sharedState );
  
  //__________________________________
  //  Temperature, pressure and other CCVariables
  void setBC(CCVariable<double>& var,     
            const std::string& type,
            const CCVariable<double>&gamma,
            const CCVariable<double>&cv, 
            const Patch* patch,  
            SimulationStateP& sharedState,
            const int mat_id,
            DataWarehouse* new_dw,
            customBC_var_basket* C_BC_basket);
            
  void setBC(CCVariable<double>& var,     
            const std::string& type,     // stub function
            const Patch* patch,  
            SimulationStateP& sharedState,
            const int mat_id,
            DataWarehouse* new_dw); 
  //__________________________________
  //  P R E S S U R E        
  void setBC(CCVariable<double>& press_CC,          
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
             
  void setBC(CCVariable<double>& press_CC,          
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
  void setBC(CCVariable<Vector>& variable,
             const std::string& type,
             const Patch* patch,
             SimulationStateP& sharedState,
             const int mat_id,
             DataWarehouse* new_dw, 
             customBC_var_basket* C_BC_basket);
             
   void setBC(CCVariable<Vector>& variable,  // stub function
             const std::string& type,
             const Patch* patch,
             SimulationStateP& sharedState,
             const int mat_id,
             DataWarehouse* new_dw);

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
                            const vector<IntVector> bound,
                            const string& bc_kind,
                            const T& value,
                            const Vector& cell_dx,
			       const int mat_id,
			       const int child);
  
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
                               vector<IntVector>& bound,
                               string& bc_kind)
{ 
  bound.push_back(IntVector(-9,-9,-9));
  //__________________________________
  //  find the iterator, BC value and BC kind
  vector<IntVector> nbound,sfx,sfy,sfz;
  const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
						    desc, bound,nbound,
						    sfx,sfy,sfz,child);

  const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
						       "Symmetric", bound, 
							nbound,sfx,sfy,
							sfz,child);

  const BoundCond<T> *new_bcs =  dynamic_cast<const BoundCond<T> *>(bc);       

  bc_value=T(-9);
  bc_kind="NotSet";
  if (new_bcs != 0) {      // non-symmetric
    bc_value = new_bcs->getValue();
    bc_kind = new_bcs->getKind();
  }        
  if (sym_bc != 0) {       // symmetric
    bc_kind = "symmetric";
  }
  if (desc == "zeroNeumann" ){
    bc_kind = "zeroNeumann";
  }
  delete bc;
  delete sym_bc;
  
  // Did I find an iterator
  if( bc_kind == "NotSet" || *bound.begin() == IntVector(-9,-9,-9)){
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
                            const vector<IntVector> bound,
                            string& bc_kind,
                            T& value,
                            const Vector& cell_dx,
			       const int mat_id,
			       const int child)
{
 vector<IntVector>::const_iterator iter;
 IntVector oneCell = patch->faceDirection(face);
 IntVector dir= patch->faceAxes(face);
 double dx = cell_dx[dir[0]];

 bool IveSetBC = false;

 if (bc_kind == "Neumann" && value == T(0)) { 
   bc_kind = "zeroNeumann";  // for speed
 }
 //__________________________________        
 if (bc_kind == "Dirichlet") {    //   D I R I C H L E T 
   for (iter = bound.begin(); iter != bound.end(); iter++) {
     var[*iter] = value;
   }
   IveSetBC = true;
 }
 //__________________________________
 // Random variations for density
 if (bc_kind == "Dirichlet_perturbed") {
   vector<IntVector> cbound,nbound,sfx,sfy,sfz;
   const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
						     "Density", cbound,nbound,
						     sfx,sfy,sfz,child);

   const BoundCond<double> *new_bcs = 
     dynamic_cast<const BoundCond<double> *>(bc);
   
   const DensityBoundCond *density_bcs =  
     dynamic_cast<const DensityBoundCond *>(new_bcs);   

   double K = density_bcs->getConstant();

   // Seed the random number generator with the number of seconds since 
   // midnight Jan. 1, 1970.

   time_t seconds = time(NULL);
   srand(seconds);

   for (iter = bound.begin(); iter != bound.end(); iter++) {
     var[*iter] = value + K*((double(rand())/RAND_MAX)*2.- 1.)*value;
   }
   IveSetBC = true;
   delete bc;
 }

 if (bc_kind == "Neumann") {       //    N E U M A N N
   for (iter=bound.begin(); iter != bound.end(); iter++) {
     IntVector adjCell = *iter - oneCell;
     var[*iter] = var[adjCell] - value * dx;
   }
   IveSetBC = true;
 }
 if (bc_kind == "zeroNeumann") {   //    Z E R O  N E U M A N N
   for (iter=bound.begin(); iter != bound.end(); iter++) {
     IntVector adjCell = *iter - oneCell;
     var[*iter] = var[adjCell];
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
                               const vector<IntVector> bound,
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
  vector<IntVector>::const_iterator iter;
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
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        IntVector c = *iter - oneCell;
        vel_FC[c] = value;
      }
    }else {    // (xplus, yplus, zplus) faces
      for (iter=bound.begin(); iter != bound.end(); iter++) {
        vel_FC[*iter] = value;
      }
    }
    IveSetBC = true;
  }
  //__________________________________
  // Neumann
  // -- Only modify the velocities that are tangential to a face.
  //    find dx, sign on that face, and direction face is pointing  
  IntVector faceDir_tmp = patch->faceDirection(face);
  IntVector faceDir     = Abs(faceDir_tmp);
  IntVector dir = patch->faceAxes(face);
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
  return IveSetBC; 
}

/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes care of face centered velocities
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
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    bool IveSetBC = false;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    for (int child = 0;  child < numChildren; child++) {

      Vector bc_value(-9,-9,-9);;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      bool foundIterator = 
        getIteratorBCValueBCKind<Vector>( patch, face, child, desc, mat_id,
					       bc_value, bound,bc_kind); 
                                       
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
                               bound, kind, value, cell_dx, P_dir, whichVel);

          if(faceDir == P_dir ) {
            string kind = "Dirichlet";
            IveSetBC= setNeumanDirichletBC_FC<T>( patch, face, vel_FC,
                                 bound, kind, value, cell_dx, P_dir, whichVel);
          }
        }

        //__________________________________
        // Non Symmetric Boundary Conditions
        if (bc_kind != "symmetric") {  
          IveSetBC= setNeumanDirichletBC_FC<T>( patch, face, vel_FC,
                              bound, bc_kind, value, cell_dx, P_dir, whichVel); 
        }
        //__________________________________
        // Custom BCs
        if (whichVel == "X_vel_FC" && 
            bc_kind == "Custom" &&  
            face == Patch::xminus) {
          setNGC_Nozzle_BC<T, double>(patch, face, vel_FC, "Vel_FC","FC", bound, 
                                   bc_kind,mat_id, child, sharedState, 
                                   custom_BC_basket->ng); 
        }
        if(bc_kind == "MMS_1"){
          IveSetBC= set_MMS_BCs_FC<T>(patch, face, vel_FC, bound, bc_kind,
                                      cell_dx, P_dir, whichVel, sharedState,
                                      custom_BC_basket->mms_var_basket,
                                      custom_BC_basket->mms_v);
        }        
        
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<whichVel<< " Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< value
               <<"\t bound limits = " <<*bound.begin()<<" "<< *(bound.end()-1)
	        << endl;
        }              
      }  // Children loop
    }  // bcKind != notSet
  }  // face loop
}
  
} // End namespace Uintah
#endif
