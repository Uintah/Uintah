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

static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream cout_BC_CC("ICE_BC_CC", false);
static DebugStream cout_BC_FC("ICE_BC_FC", false);

  class DataWarehouse;

  void is_BC_specified(const ProblemSpecP& prob_spec, string variable, const MaterialSubset* matls);
  
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

 int setSymmetryBC_CC( const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<Vector>& var_CC,               
                       Iterator& bound_ptr);

 template<class T>
 int setDirichletBC_FC( const Patch* patch,
                        const Patch::FaceType face,       
                        T& vel_FC,                        
                        Iterator& bound_ptr,                  
                        double& value,          
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
    bc_kind  = new_bcs->getBCType__NEW();
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
 Function~  setNeumanBC_CC--
 ---------------------------------------------------------------------  */
 template<class T>
 int setNeumannBC_CC( const Patch* patch,
                      const Patch::FaceType face,
                      CCVariable<T>& var,               
                      Iterator& bound_ptr,                 
                      T& value,                         
                      const Vector& cell_dx)                  
{
 IntVector oneCell = patch->faceDirection(face);
 IntVector dir= patch->getFaceAxes(face);
 double dx = cell_dx[dir[0]];

 int IveSetBC = 0;

 if (value == T(0)) {   //    Z E R O  N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell];
   }
   IveSetBC += 1;
 }else{                //    N E U M A N N
   for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
     IntVector adjCell = *bound_ptr - oneCell;
     var[*bound_ptr] = var[adjCell] - value * dx;
   }
   IveSetBC += 1;
 }
 return IveSetBC;
}

/* --------------------------------------------------------------------- 
 Function~  setDirichletBC_CC--
 ---------------------------------------------------------------------  */
 template<class T>
 int setDirichletBC_CC( CCVariable<T>& var,     
                        Iterator& bound_ptr,    
                        T& value) 
{
 for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
   var[*bound_ptr] = value;
 }
 int IveSetBC = 1;
 return IveSetBC;

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
 Function~  setDirichletBC_FC--
 Purpose~   does the actual work of setting the BC for face-centered 
            velocities
 ---------------------------------------------------------------------  */
 template<class T>
 int setDirichletBC_FC( const Patch* patch,
                        const Patch::FaceType face,       
                        T& vel_FC,                        
                        Iterator& bound_ptr,                 
                        double& value)           
{
  int IveSetBC = 0;
  IntVector oneCell(0,0,0);

  if ((face == Patch::xminus) ||     
      (face == Patch::yminus) ||     
      (face == Patch::zminus)){      
    oneCell = patch->faceDirection(face);                                        
  }                                                            

  // on (x,y,z)minus faces move inward one cell
  // on (x,y,z)plus faces oneCell == 0                                                             
  for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {  
    IntVector c = *bound_ptr - oneCell;                      
    vel_FC[c] = value;                                       
  }                                                          
  IveSetBC +=1;                                                
  return IveSetBC; 
}


/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes capre of face centered velocities
            The normal components are computed in  AddExchangeContributionToFCVel.
 ---------------------------------------------------------------------  */
 template<class T> 
void setBC(T& vel_FC, 
           const string& desc,
           const Patch* patch,    
           const int mat_id,
           SimulationStateP& sharedState,
           customBC_var_basket* custom_BC_basket)      
{
  cout_BC_FC << "setBCFC (SFCVariable) "<< desc<< " mat_id = " << mat_id <<endl;
  Vector cell_dx = patch->dCell();
  string whichVel = "unknown";  
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;

  patch->getBoundaryFaces(bf);

  for (iter  = bf.begin(); iter != bf.end(); ++iter){
    Patch::FaceType face = *iter;
    
    IntVector faceDir = Abs(patch->faceDirection(face));
    
    // SFC(X,Y,Z) Vars can only be set on (x,y,z)+ & (x,y,z)- faces
    if(  (faceDir.x() == 1 &&  (typeid(T) == typeid(SFCXVariable<double>)) ) ||
         (faceDir.y() == 1 &&  (typeid(T) == typeid(SFCYVariable<double>)) ) ||
         (faceDir.z() == 1 &&  (typeid(T) == typeid(SFCZVariable<double>)) ) ){
    
      int IveSetBC = 0;
      string bc_kind = "NotSet";
      
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
      for (int child = 0;  child < numChildren; child++) {

        Vector bc_value(-9,-9,-9);
        Iterator bound_ptr;
        bool foundIterator = 
          getIteratorBCValueBCKind<Vector>( patch, face, child, desc, mat_id,
					         bc_value, bound_ptr,bc_kind); 

        if (foundIterator && (bc_kind != "LODI" || bc_kind != "Neumann") ) {
          //__________________________________
          // Extract which SFC variable you're
          //  working on, the value 
          double value=-9;
          if (typeid(T) == typeid(SFCXVariable<double>)) {
            value = bc_value.x();
            whichVel = "X_vel_FC";
          }
          else if (typeid(T) == typeid(SFCYVariable<double>)) {
            value = bc_value.y();
            whichVel = "Y_vel_FC";
          }
          else if (typeid(T) == typeid(SFCZVariable<double>)) {
            value = bc_value.z();
            whichVel = "Z_vel_FC";
          }

          //__________________________________
          //  Neumann BCs
          //The normal components are computed in AddExchangeContributionToFCVel.
          if(bc_kind == "Neumann"){
            IveSetBC +=1;
          }
          
          //__________________________________
          //  Symmetry boundary conditions
          //  -- faces in the principal dir:     vel[c] = 0
          else if (bc_kind == "symmetric") { 
            value = 0.0;                                                                           
            IveSetBC += setDirichletBC_FC<T>( patch, face, vel_FC, bound_ptr, value);    
          }
          //__________________________________
          // Dirichlet
          else if (bc_kind == "Dirichlet") {  
            IveSetBC += setDirichletBC_FC<T>( patch, face, vel_FC, bound_ptr, value);
          }
          //__________________________________
          // Custom BCs
          else if(bc_kind == "MMS_1"){
            IveSetBC+= set_MMS_BCs_FC<T>(patch, face, vel_FC, bound_ptr,
                                        cell_dx, sharedState,
                                        custom_BC_basket->mms_var_basket,
                                        custom_BC_basket->mms_v);
          }
          //__________________________________
          // Custom BCs
          else if(bc_kind == "Sine"){
            IveSetBC+= set_Sine_BCs_FC<T>(patch, face, vel_FC, bound_ptr, sharedState,
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
      }
      cout_BC_FC << patch->getFaceName(face) << " \t " << whichVel << " \t" << bc_kind << " faceDir: " << faceDir << " numChildren: " << numChildren << " IveSetBC: " << IveSetBC << endl;
      //__________________________________
      //  bulletproofing
      if(IveSetBC != numChildren){
        ostringstream warn;
        warn << "ERROR ICE: Boundary conditions were not set for ("<< whichVel << ", " 
             << patch->getFaceName(face) << ", " << bc_kind  << " numChildren: " << numChildren 
             << " IveSetBC: " << IveSetBC << ") " << endl;
        throw InternalError(warn.str(), __FILE__, __LINE__);
      }
    }  // found iterator
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
  cout_BC_CC << "set_CFI_BC "<< endl; 
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
