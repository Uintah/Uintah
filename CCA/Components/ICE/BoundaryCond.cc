#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>

#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/fillFace.h>

#include <typeinfo>
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>

 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
 // Note:  cout_dbg doesn't work if the iterator bound is
 //        not defined
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);

using namespace Uintah;
namespace Uintah {

/* --------------------------------------------------------------------- 
 Function~  ImplicitMatrixBC--      
 Purpose~   Along each face of the domain set the stencil weight
 Naming convention
      +x -x +y -y +z -z
       e, w, n, s, t, b 
 
   A.p = beta[c] -
          (A.n + A.s + A.e + A.w + A.t + A.b);
   LHS       
   A.p*delP - (A.e*delP_e + A.w*delP_w + A.n*delP_n + A.s*delP_s 
             + A.t*delP_t + A.b*delP_b )
             
 Suppose the x- face has Press=Neumann BC, then you must add A.w to
 both A.p and set A.w = 0.  If the pressure is Dirichlet BC you leave A.p 
 alone and set A.w = 0;           
       
 ---------------------------------------------------------------------  */
void ImplicitMatrixBC( CCVariable<Stencil7>& A, 
                   const Patch* patch)        
{ 
  vector<Patch::FaceType>::const_iterator itr;
  for (itr  = patch->getBoundaryFaces()->begin(); 
       itr != patch->getBoundaryFaces()->end(); ++itr){
    Patch::FaceType face = *itr;
    
    int mat_id = 0; // hard coded for pressure
    
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);
    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      string bc_kind  = "NotSet";
      vector<IntVector> bound;  
      
      bool foundIterator =       
        getIteratorBCValueBCKind<double>( patch, face, child, "Pressure", 
                                         mat_id, bc_value, bound,bc_kind);
                                    
      // don't set BCs unless we've found the iterator                                   
      if (foundIterator) {
        //__________________________________
        //  Neumann or Dirichlet Press_BC;
        double one_or_zero = -999;
        if(bc_kind == "zeroNeumann" || bc_kind == "Neumann" ||
           bc_kind == "symmetric"){
          one_or_zero = 1.0;      // subtract from A.p
        }
        if(bc_kind == "Dirichlet" || bc_kind == "Custom" || 
           bc_kind == "MMS_1"){
          one_or_zero = 0.0;      // leave A.p Alone
        }                                 
        //__________________________________
        //  Set the BC  
        vector<IntVector>::const_iterator iter;

        switch (face) {
        case Patch::xplus:
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c(*iter - IntVector(1,0,0));
            A[c].p = A[c].p + one_or_zero * A[c].e;
            A[c].e = 0.0;
          }
          break;
        case Patch::xminus:
          for (iter=bound.begin(); iter != bound.end(); iter++) { 
            IntVector c(*iter + IntVector(1,0,0));
            A[c].p = A[c].p + one_or_zero * A[c].w;
            A[c].w = 0.0;
          }
          break;
        case Patch::yplus:
          for (iter=bound.begin(); iter != bound.end(); iter++) { 
            IntVector c(*iter - IntVector(0,1,0));
            A[c].p = A[c].p + one_or_zero * A[c].n;
            A[c].n = 0.0;
          }
          break;
        case Patch::yminus:
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c(*iter + IntVector(0,1,0)); 
            A[c].p = A[c].p + one_or_zero * A[c].s;
            A[c].s = 0.0;
          }
          break;
        case Patch::zplus:
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c(*iter - IntVector(0,0,1));
            A[c].p = A[c].p + one_or_zero * A[c].t;
            A[c].t = 0.0;
          }
          break;
        case Patch::zminus:
          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector c(*iter + IntVector(0,0,1));
            A[c].p = A[c].p + one_or_zero * A[c].b;
            A[c].b = 0.0;
          }
          break;
        case Patch::numFaces:
          break;
        case Patch::invalidFace:
          break; 
        }
        //__________________________________
        //  debugging
        #if 0
        cout <<"Face: "<< face << " one_or_zero " << one_or_zero
             <<"\t child " << child  <<" NumChildren "<<numChildren 
             <<"\t BC kind "<< bc_kind
             <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1)
	      << endl;
        #endif
      } // if (bc_kind !=notSet)
    } // child loop
  }  // face loop
}
/* --------------------------------------------------------------------- 
 Function~  get_rho_micro--
 Purpose~  This handles all the logic of getting rho_micro on the faces
    a) when using lodi bcs get rho_micro for all ice and mpm matls
    b) with gravity != 0 get rho_micro on P_dir faces, for all ICE matls
    c) during initialization only get rho_micro for ice_matls, you can't
       get rho_micro for mpm_matls
 ---------------------------------------------------------------------  */
void get_rho_micro(StaticArray<CCVariable<double> >& rho_micro,
                   StaticArray<CCVariable<double> >& rho_micro_tmp,
                   StaticArray<constCCVariable<double> >& sp_vol_CC,
                   const Patch* patch,
                   const string& which_Var,
                   SimulationStateP& sharedState,
                   DataWarehouse* new_dw,
                   customBC_var_basket* custom_BC_basket)
{
  BC_doing << "get_rho_micro: Which_var " << which_Var << endl;
  
  if( which_Var !="rho_micro" && which_Var !="sp_vol" ){
    throw InternalError("setBC (pressure): Invalid option for which_var");
  }
  
  Vector gravity = sharedState->getGravity(); 
  int timestep = sharedState->getCurrentTopLevelTimeStep();
  int numMatls  = sharedState->getNumICEMatls();
  
  if (timestep > 0 ) {
    numMatls += sharedState->getNumMPMMatls();
  }
      
  for (int m = 0; m < numMatls; m++) {
    new_dw->allocateTemporary(rho_micro[m],  patch);
  }
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if(is_LODI_face(patch, face, sharedState) || gravity.length() > 0) {
      
      //__________________________________
      // Create an iterator that iterates over the face
      // + 2 cells inward (hydrostatic press tweak).  
      // We don't need to hit every  cell on the patch. 
      CellIterator iter_tmp = patch->getFaceCellIterator(face, "plusEdgeCells");
      IntVector lo = iter_tmp.begin();
      IntVector hi = iter_tmp.end();
    
      int P_dir = patch->faceAxes(face)[0];  //principal dir.
      if(face==Patch::xminus || face==Patch::yminus || face==Patch::zminus){
        hi[P_dir] += 2;
      }
      if(face==Patch::xplus || face==Patch::yplus || face==Patch::zplus){
        lo[P_dir] -= 2;
      }
      CellIterator iterLimits(lo,hi);
      
      for (int m = 0; m < numMatls; m++) {
        if (which_Var == "rho_micro") { 
          for (CellIterator iter=iterLimits; !iter.done();iter++) {
            IntVector c = *iter;
            rho_micro[m][c] =  rho_micro_tmp[m][c];
          }
        }
        if (which_Var == "sp_vol") { 
          for (CellIterator iter=iterLimits; !iter.done();iter++) {
            IntVector c = *iter;
            rho_micro[m][c] =  1.0/sp_vol_CC[m][c];
          }
        }  // sp_vol
      }  // numMatls
    }  // LODI face or gravity != 0
  }  // face iter 
}

/* --------------------------------------------------------------------- 
 Function~  setBC-- (pressure)
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& press_CC,
           StaticArray<CCVariable<double> >& rho_micro_tmp,   //or placeHolder
           StaticArray<constCCVariable<double> >& sp_vol_CC,  //or placeHolder
           const int surroundingMatl_indx,
           const string& which_Var,
           const string& kind, 
           const Patch* patch,
           SimulationStateP& sharedState, 
           const int mat_id,
           DataWarehouse* new_dw,
           customBC_var_basket* custom_BC_basket)
{
  BC_doing << "setBC (press_CC) "<< kind <<" " << which_Var
           << " mat_id = " << mat_id << endl;

  int numMatls = sharedState->getNumMatls();
  int topLevelTimestep = sharedState->getCurrentTopLevelTimeStep();  
  Vector gravity = sharedState->getGravity();
  StaticArray<CCVariable<double> > rho_micro(numMatls);

  
  get_rho_micro(rho_micro, rho_micro_tmp, sp_vol_CC, 
                patch, which_Var, sharedState,  new_dw, custom_BC_basket);
                
  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase AND when
  //   lv->setLodiBcs = false              
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;

  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    bool is_lodi_pressBC = patch->haveBC(face,mat_id,"LODI","Pressure");
    int topLevelTimestep = sharedState->getCurrentTopLevelTimeStep();
    
    if(kind == "Pressure"      && is_lodi_pressBC 
       && topLevelTimestep > 0 && custom_BC_basket->setLodiBcs){
       FacePress_LODI(patch, press_CC, rho_micro, sharedState,face,
                      custom_BC_basket->lv);
    }
  }

  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  // Iterate over the faces encompassing the domain
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    bool IveSetBC = false;
   
    IntVector dir= patch->faceAxes(face);
    Vector cell_dx = patch->dCell();
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      
      bool foundIterator = 
        getIteratorBCValueBCKind<double>( patch, face, child, kind, mat_id,
					       bc_value, bound,bc_kind); 
                                   
      if(foundIterator && bc_kind != "LODI") {
        // define what a symmetric  pressure BC means
        if( bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }
        
        IveSetBC = setNeumanDirichletBC<double>(patch, face, press_CC,bound, 
						  bc_kind, bc_value, cell_dx,
						  mat_id,child);
                                          
        //__________________________________
        //  hardwiring for NGC nozzle simulation
        if (bc_kind == "Custom" && custom_BC_basket->setNGBcs) {
          setNGC_Nozzle_BC<CCVariable<double>,double>
          (patch, face, press_CC, "Pressure", "CC",
           bound, bc_kind,mat_id, child, sharedState,custom_BC_basket->ng);
        }
        //__________________________________
        //  method of manufactured solutions
        if (bc_kind == "MMS_1" && custom_BC_basket->set_MMS_BCs) {
          set_MMS_press_BC(patch, face, press_CC, bound,  bc_kind,
                           sharedState, 
                           custom_BC_basket->mms_var_basket,
                           custom_BC_basket->mms_v);
        }                    
                                            
        //__________________________________________________________
        // Tack on hydrostatic pressure correction after Dirichlet 
        // or Neumann BC have been applied.  Note, during the intializaton 
        //phase the hydrostatic pressure adjustment is  handled by a completely
        // separate task, therefore ignore it        
        // 
        // Hydrostatic pressure adjustment (HPA): 
        //   gravity*rho_micro*distance_from_ref_point.
        // R is BC location, L is adjacent to BC location
        //
        // Dirichlet BC: P_R= P_Dirichlet_R + HPA_R
        // Neumann BC:   P_R = P_neumann_R + HPA_R - HPA_L,
        // where HPA_R - HPA_L is zero if BC normal is orthogonal to gravity
        // find the upper and lower point of the domain.
        // 
        // On Dirichlet side walls you still have to add HPA
        const Level* level = patch->getLevel();
        GridP grid = level->getGrid();
        BBox b;
        grid->getSpatialRange(b);
        Vector gridMin = b.min().asVector();
        Vector dx_L0 = grid->getLevel(0)->dCell();
 
        // Pressure reference point is assumed to be 
        //at the cell center of cell 0,0,0 
        Vector press_ref_pt = gridMin + 1.5*dx_L0;

        int p_dir = patch->faceAxes(face)[0];     // normal  face direction
        
        // Only apply this correction in case of Neumann or Dirichlet BC
        bool Neumann_BC = (bc_kind=="Neumann" || bc_kind=="zeroNeumann");
        if ( topLevelTimestep > 0 ){
          if ((gravity[p_dir] != 0 && Neumann_BC) || 
              (gravity.length() != 0 && bc_kind =="Dirichlet")){  
            
            double oneZero = 1;  
            if (bc_kind=="Dirichlet") {
              oneZero = 0.0;
            }    

            IntVector oneCell = patch->faceDirection(face);

            vector<IntVector>::const_iterator iter;
            for (iter=bound.begin();iter != bound.end(); iter++) {
              IntVector R = *iter;
              IntVector L = *iter - oneCell;
              Point here_L = level->getCellPosition(L);
              Point here_R = level->getCellPosition(R);
              Vector dist_L = (here_L.asVector() - press_ref_pt);
              Vector dist_R = (here_R.asVector() - press_ref_pt);
              double rho_L = rho_micro[surroundingMatl_indx][L];
              double rho_R = rho_micro[surroundingMatl_indx][R];
              // Need the dot product to get the sideWall dirichlet BC's right 
              double correction_L = Dot(gravity,dist_L) * rho_L;
              double correction_R = Dot(gravity,dist_R) * rho_R;

              press_CC[R] += correction_R - oneZero * correction_L;
            }
            IveSetBC = true;    
          } // Dirichlet || Neumann
        } // // not initialization step 

        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1)
	        << endl;
        }
      }  // if bcKind != notSet
    }  // child loop
  }  // faces loop
}
/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care any CCvariable<double>, except Pressure
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& var_CC,
           const string& desc,
           const CCVariable<double>& gamma,
           const CCVariable<double>& cv,
           const Patch* patch,
           SimulationStateP& sharedState, 
           const int mat_id,
           DataWarehouse*,
           customBC_var_basket* custom_BC_basket)    // NG hack
{
  BC_doing << "setBC (double) "<< desc << " mat_id = " << mat_id << endl;
  Vector cell_dx = patch->dCell();
  int topLevelTimestep = sharedState->getCurrentTopLevelTimeStep();

  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase and when
  //   lv->setLodiBcs = false
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;

    bool is_tempBC_lodi=  patch->haveBC(face,mat_id,"LODI","Temperature");  
    bool is_rhoBC_lodi =  patch->haveBC(face,mat_id,"LODI","Density");
    
    Lodi_vars* lv = custom_BC_basket->lv;
    if( desc == "Temperature"  && is_tempBC_lodi 
        && topLevelTimestep >0 && custom_BC_basket->setLodiBcs ){
      FaceTemp_LODI(patch, face, var_CC, lv, cell_dx, sharedState);
    }   
    if (desc == "Density"  && is_rhoBC_lodi 
        && topLevelTimestep >0 && custom_BC_basket->setLodiBcs){
      FaceDensity_LODI(patch, face, var_CC, lv, cell_dx);
    }
  }
  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  // Iterate over the faces encompassing the domain
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
          
    bool IveSetBC = false;

    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      bool foundIterator = 
        getIteratorBCValueBCKind<double>( patch, face, child, desc, mat_id,
					       bc_value, bound,bc_kind); 
      
      if (foundIterator && bc_kind != "LODI") {
        //__________________________________
        // LOGIC
        // Any CC Variable
        if (desc == "set_if_sym_BC" && bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }
        if ( bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }

        //__________________________________
        // Apply the boundary condition
        IveSetBC =  setNeumanDirichletBC<double>
	  (patch, face, var_CC,bound, bc_kind, bc_value, cell_dx,mat_id,child);
         
        //__________________________________
        //  hardwiring for NGC nozzle simulation   
        if ( (desc == "Temperature" || desc == "Density") && 
              bc_kind == "Custom" && custom_BC_basket->setLodiBcs) {
          setNGC_Nozzle_BC<CCVariable<double>,double>
                (patch, face, var_CC, desc,"CC", bound, 
                 bc_kind,mat_id, child, sharedState, 
                 custom_BC_basket->ng);
        }
        
        if ( desc == "Temperature" &&custom_BC_basket->setMicroSlipBcs) {
          set_MicroSlipTemperature_BC(patch,face,var_CC,
                              desc, bound, bc_kind, bc_value,
                              custom_BC_basket->sv);
        }

        if ( desc == "Temperature" && custom_BC_basket->set_MMS_BCs) {
          set_MMS_Temperature_BC(patch, face, var_CC, 
                              desc, bound, bc_kind, 
                              custom_BC_basket->mms_var_basket,
                              custom_BC_basket->mms_v);
        }
        //__________________________________
        // Temperature and Gravity and ICE Matls
        // -Ignore this during intialization phase,
        //  since we backout the temperature field
        Vector gravity = sharedState->getGravity();                             
        Material *matl = sharedState->getMaterial(mat_id);
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
        int P_dir =  patch->faceAxes(face)[0];  // principal direction
        
        if (gravity[P_dir] != 0 && desc == "Temperature" && ice_matl 
             && topLevelTimestep >0) {
          ice_matl->getEOS()->
              hydrostaticTempAdjustment(face, patch, bound, gravity,
                                        gamma, cv, cell_dx, var_CC);
        }
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1)
	        << endl;
        }
      }  // if bc_kind != notSet  
    }  // child loop
  }  // faces loop
}

/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care vector boundary condition
 ---------------------------------------------------------------------  */
void setBC(CCVariable<Vector>& var_CC,
           const string& desc,
           const Patch* patch,
           SimulationStateP& sharedState, 
           const int mat_id,
           DataWarehouse* ,
           customBC_var_basket* custom_BC_basket)
{
  BC_doing <<"setBC (Vector_CC) "<< desc <<" mat_id = " <<mat_id<< endl;
  Vector cell_dx = patch->dCell();
  //__________________________________
  //  -Set the LODI BC's first, then let the other BC's wipe out what
  //   was set in the corners and edges. 
  //  -Ignore lodi bcs during intialization phase and when
  //   lv->setLodiBcs = false
  //__________________________________
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType>::const_iterator iter;
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    bool is_velBC_lodi   =  patch->haveBC(face,mat_id,"LODI","Velocity");
    int topLevelTimestep = sharedState->getCurrentTopLevelTimeStep();
    
    Lodi_vars* lv = custom_BC_basket->lv;
    
    if( desc == "Velocity"      && is_velBC_lodi 
        && topLevelTimestep > 0 && custom_BC_basket->setLodiBcs) {
      FaceVel_LODI( patch, face, var_CC, lv, cell_dx, sharedState);
    }
  }
  //__________________________________
  //  N O N  -  L O D I
  //__________________________________
  // Iterate over the faces encompassing the domain
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    bool IveSetBC = false;
    

    IntVector oneCell = patch->faceDirection(face);
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      Vector bc_value = Vector(-9,-9,-9);
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      
      bool foundIterator = 
          getIteratorBCValueBCKind<Vector>(patch, face, child, desc, mat_id,
				                bc_value, bound,bc_kind);
     
      if (foundIterator && bc_kind != "LODI") {
 
        IveSetBC = setNeumanDirichletBC<Vector>(patch, face, var_CC,bound, 
						bc_kind, bc_value, cell_dx,
						mat_id,child);
        //__________________________________
        //  Custom Boundary Conditions
        if ( custom_BC_basket->setLodiBcs) {
          setNGCVelocity_BC(patch,face,var_CC,desc,
                            bound, bc_kind,  mat_id, child, sharedState,
                            custom_BC_basket->ng);
        }
        
        if ( custom_BC_basket->setMicroSlipBcs) {
          set_MicroSlipVelocity_BC(patch,face,var_CC,desc,
                            bound, bc_kind, bc_value,
                            custom_BC_basket->sv);
        }
        
        if ( custom_BC_basket->set_MMS_BCs) {
          set_MMS_Velocity_BC(patch, face, var_CC, desc,
                            bound, bc_kind, sharedState,
                            custom_BC_basket->mms_var_basket,
                            custom_BC_basket->mms_v);
        }
         
        //__________________________________
        //  Tangent components Neumann = 0
        //  Normal components = -variable[Interior]
        //  It's negInterior since it's on the opposite side of the
        //  plane of symetry  
        if ( bc_kind == "symmetric" &&
            (desc == "Velocity" || desc == "set_if_sym_BC" ) ) {
          int P_dir = patch->faceAxes(face)[0];  // principal direction
          IntVector sign = IntVector(1,1,1);
          sign[P_dir] = -1;
          vector<IntVector>::const_iterator iter;

          for (iter=bound.begin(); iter != bound.end(); iter++) {
            IntVector adjCell = *iter - oneCell;
            var_CC[*iter] = sign.asVector() * var_CC[adjCell];
          }
          IveSetBC = true;
          bc_value = Vector(0,0,0); // so the debugging output is accurate
        }
        //__________________________________
        //  debugging
        if( BC_dbg.active() ) {
          BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = " <<*bound.begin()<<" "<< *(bound.end()-1)
	        << endl;
        }
      }  // if (bcKind != "notSet") 
    }  // child loop
  }  // faces loop
}

/* --------------------------------------------------------------------- 
 Function~  is_BC_specified--
 Purpose~   examines the each face in the boundary condition section
            of the input file and tests to make sure that each (variable)
            has a boundary conditions specified.
 ---------------------------------------------------------------------  */
void is_BC_specified(const ProblemSpecP& prob_spec, string variable)
{
  // search the BoundaryConditions problem spec
  // determine if variable bcs have been specified
  
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
 
  // loop over all faces
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
   
    bool bc_specified = false;
    map<string,string> face;
    face_ps->getAttributes(face);
    
    // loop over all BCTypes  
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      if (bc_type["label"] == variable || bc_type["label"] == "Symmetric") {
         bc_specified = true;
      }
    }
    //  I haven't found them so get mad.
    if (!bc_specified){
      ostringstream warn;
      warn <<"\n__________________________________\n"  
           << "ERROR: The boundary condition for ( " << variable
           << " ) was not specified on face " << face["side"] << endl;
      throw ProblemSetupException(warn.str());
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  BC_bulletproofing--
 Purpose~   if BCType id = "all" and it's a single material problem 
            throw an exception.  
 ---------------------------------------------------------------------  */
void BC_bulletproofing(const ProblemSpecP& prob_spec,SimulationStateP& sharedState )
{
  
  ProblemSpecP grid_ps= prob_spec->findBlock("Grid");
  ProblemSpecP bc_ps  = grid_ps->findBlock("BoundaryConditions");
  int numAllMatls = sharedState->getNumMatls();
  
  // loop over all faces
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");face_ps != 0; 
                    face_ps=face_ps->findNextBlock("Face")) {
   
    bool getMad = false;
    map<string,string> face;
    face_ps->getAttributes(face);
    
    // loop over all BCTypes  
    for(ProblemSpecP bc_iter = face_ps->findBlock("BCType"); bc_iter != 0;
                     bc_iter = bc_iter->findNextBlock("BCType")){
      map<string,string> bc_type;
      bc_iter->getAttributes(bc_type);
      
      if (bc_type["id"] == "all" && numAllMatls == 0) {
        ostringstream warn;
        warn <<"\n__________________________________\n"   
             << "ERROR: This is a single material problem and you've specified 'BCType id = all' \n"
             << "The boundary condition machinery essentially treats 'all' and '0' as two separate materials, \n"
             << "setting the boundary conditions twice on each face.  Set BCType id = '0' \n" 
             << " Face:  " << face["side"] << " BCType " << bc_type["label"]<< endl;
        throw ProblemSetupException(warn.str());
      }
    }
  }
}
//______________________________________________________________________
//______________________________________________________________________
//      S T U B   F U N C T I O N S

void setBC(CCVariable<double>& var,     
          const std::string& type,     // so gcc compiles
          const Patch* patch,  
          SimulationStateP& sharedState,
          const int mat_id,
          DataWarehouse* new_dw)
{
  customBC_var_basket* basket  = scinew customBC_var_basket();
  constCCVariable<double> placeHolder;
  
  basket->setLodiBcs      = false;
  basket->setNGBcs        = false;
  basket->setMicroSlipBcs = false;
  basket->set_MMS_BCs     = false;
  
  setBC(var, type, placeHolder, placeHolder, patch, sharedState, 
        mat_id, new_dw,basket);
  
  delete basket;
} 
//__________________________________  
void setBC(CCVariable<double>& press_CC,          
         StaticArray<CCVariable<double> >& rho_micro,
         StaticArray<constCCVariable<double> >& sp_vol,
         const int surroundingMatl_indx,
         const std::string& whichVar, 
         const std::string& kind, 
         const Patch* p, 
         SimulationStateP& sharedState,
         const int mat_id, 
         DataWarehouse* new_dw) {
         
  customBC_var_basket* basket  = scinew customBC_var_basket();
  basket->setLodiBcs      = false;
  basket->setNGBcs        = false;
  basket->setMicroSlipBcs = false;
  basket->set_MMS_BCs     = false;
  
  setBC(press_CC, rho_micro, sp_vol, surroundingMatl_indx,
        whichVar, kind, p, sharedState, mat_id, new_dw, basket); 

  delete basket;         
}   
//__________________________________       
void setBC(CCVariable<Vector>& variable,
          const std::string& type,
          const Patch* p,
          SimulationStateP& sharedState,
          const int mat_id,
          DataWarehouse* new_dw)
{ 
  customBC_var_basket* basket  = scinew customBC_var_basket();
  basket->setLodiBcs      = false;
  basket->setNGBcs        = false;
  basket->setMicroSlipBcs = false;
  basket->set_MMS_BCs     = false;
   
  setBC( variable, type, p, sharedState, mat_id, new_dw,basket);
  
  delete basket; 
}


}  // using namespace Uintah
