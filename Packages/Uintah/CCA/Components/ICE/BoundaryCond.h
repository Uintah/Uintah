#ifndef Packages_Uintah_CCA_Components_Ice_BoundaryCond_h
#define Packages_Uintah_CCA_Components_Ice_BoundaryCond_h

#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Stencil7.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>
#include <Core/Containers/StaticArray.h>

/*`==========TESTING==========*/
#undef JET_BC    // needed if you want a jet for either LODI or ORG_BCS

#undef LODI_BCS  // note for LODI_BCs you also need ORG_BCS turned on

#undef ORG_BCS    // original setBC 

#define JOHNS_BC   // DEFAULT BOUNDARY CONDITIONS.
/*==========TESTING==========`*/



namespace Uintah {
 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);

  class DataWarehouse;
  
  //__________________________________
  // all the variables needed by LODI Bcs
  struct Lodi_vars{                
     Lodi_vars() : di(6) {}
     constCCVariable<double> rho_old;     
     constCCVariable<double> temp_old;    
     constCCVariable<Vector> vel_old;     
     CCVariable<double> press_tmp;        
     CCVariable<double> e;                
     CCVariable<Vector> nu;               
     StaticArray<CCVariable<Vector> > di; 
     double gamma;                        
  };


  void setHydrostaticPressureBC(CCVariable<double>& press,
                            Patch::FaceType face, Vector& gravity,
                            const CCVariable<double>& rho,
                            const Vector& dx,
                            IntVector offset = IntVector(0,0,0));

  void setBC(CCVariable<double>& variable,const std::string& type, 
            const Patch* p,  SimulationStateP& sharedState,
            const int mat_id);
  
  void setBC(CCVariable<double>& press_CC, const CCVariable<double>& rho,
             const std::string& whichVar, const std::string& type, 
             const Patch* p, SimulationStateP& sharedState,
             const int mat_id, DataWarehouse*);
  
  void setBC(CCVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);

/*`==========TESTING==========*/
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
                            const Vector& cell_dx);

bool are_We_Using_LODI_BC(const Patch* patch,
                          vector<bool>& which_face_LODI,
                          const int mat_id);

void setBCPress_LODI(CCVariable<double>& press_CC,
                     StaticArray<CCVariable<double> >& sp_vol_CC,
                     StaticArray<constCCVariable<double> >& Temp_CC,
                     StaticArray<CCVariable<double> >& f_theta,
                     const string& which_Var,
                     const string& kind, 
                     const Patch* patch,
                     SimulationStateP& sharedState, 
                     const int mat_id,
                     DataWarehouse* new_dw);

void setBCDensityLODI(CCVariable<double>& rho_CC,
                StaticArray<CCVariable<Vector> >& di,
                const CCVariable<Vector>& nu, 
                constCCVariable<double>& rho_tmp,
                const CCVariable<double>& p,
                constCCVariable<Vector>& vel,            
                const double delT,
                const Patch* patch, 
                const int mat_id); 
              
void setBCVelLODI(CCVariable<Vector>& vel_CC,
            StaticArray<CCVariable<Vector> >& di,
            const CCVariable<Vector>& nu,
            constCCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            constCCVariable<Vector>& vel,
            const double delT,
            const Patch* patch, 
            const int mat_id); 
           
              
 void setBCTempLODI(CCVariable<double>& temp_CC,
              StaticArray<CCVariable<Vector> >& di,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<Vector>& nu,
              constCCVariable<double>& rho_tmp,
              const CCVariable<double>& p,
              constCCVariable<Vector>& vel,
              const double delT,
              const double cv,
              const double gamma,
              const Patch* patch,
              const int mat_id);

void computeNu(CCVariable<Vector>& nu, 
               const vector<bool>& is_LODI_face,
               const CCVariable<double>& p, 
               const Patch* patch);  
              
void computeDi(StaticArray<CCVariable<Vector> >& d,
               const vector<bool>& is_LODI_face,
               constCCVariable<double>& rho_old,  
               const CCVariable<double>& press_tmp, 
               constCCVariable<Vector>& vel_old, 
               constCCVariable<double>& speedSound, 
               const Patch* patch,
               const int mat_id);
                     
// end of characteristic boundary condition
/*==========TESTING==========`*/
  template<class T> void Neuman_SFC(T& var, const Patch* patch,
                                Patch::FaceType face,
                                const double value, const Vector& dx,
                                IntVector offset = IntVector(0,0,0));


  void setBC(SFCXVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);

  void setBC(SFCYVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);  

  void setBC(SFCZVariable<double>& variable,const std::string& type,
             const std::string& comp, const Patch* p, const int mat_id);   
  
  void setBC(SFCXVariable<Vector>& variable,const std::string& type,
             const Patch* p, const int mat_id);
  
  void ImplicitMatrixBC(CCVariable<Stencil7>& var, const Patch* patch);
  
 
/* --------------------------------------------------------------------- 
 Function~  getIteratorBCValueBCKind--
 Purpose~   does the actual work
 ---------------------------------------------------------------------  */
 
template <class T>
void getIteratorBCValueBCKind( const Patch* patch, 
			       const Patch::FaceType face,
			       const int child,
			       const string& desc,
			       const int mat_id,
			       T& bc_value,
			       vector<IntVector>& bound,
			       string& bc_kind)
{ 
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
  if (sym_bc != 0)        
    bc_kind = "symmetric";
  
  delete bc;
  delete sym_bc;
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
                            const Vector& cell_dx)
{
 vector<IntVector>::const_iterator iter;
 IntVector oneCell = patch->faceDirection(face);
 IntVector dir= patch->faceAxes(face);
 double dx = cell_dx[dir[0]];

 bool IveSetBC = false;

 if (bc_kind == "Neumann" && value == T(0)) { 
   bc_kind = "zeroNeumann";  // for speed
 }

                                   //   C C _ D I R I C H L E T
 if (bc_kind == "Dirichlet") {     
   for (iter = bound.begin(); iter != bound.end(); iter++) {
     var[*iter] = value;
   }
   IveSetBC = true;
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
  // neumann
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
           const string&,    //--- not needed throw away when fully converted
           const Patch* patch,    
           const int mat_id)      
{
  BC_doing << "Johns setBCFC (SFCVariable) "<< desc<< " mat_id = " << mat_id 
	   <<endl;
  Vector cell_dx = patch->dCell();
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  // not the faces between neighboring patches.
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

      getIteratorBCValueBCKind<Vector>( patch, face, child, desc, mat_id,
					bc_value, bound,bc_kind); 


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
      //  debugging
      BC_dbg <<whichVel<< " Face: "<< face <<" I've set BC " << IveSetBC
             <<"\t child " << child  <<" NumChildren "<<numChildren 
             <<"\t BC kind "<< bc_kind <<" \tBC value "<< value
             <<"\t bound limits = " <<*bound.begin()<<" "<< *(bound.end()-1)
	      << endl;               
    }  // Children loop
  }  // face loop
}
  
} // End namespace Uintah
#endif
