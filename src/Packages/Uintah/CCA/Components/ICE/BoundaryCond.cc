#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/BoundCond.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/fillFace.h>
#include <typeinfo>

/*`==========TESTING==========*/
#define JET_BC 0
/*==========TESTING==========`*/


using namespace Uintah;
namespace Uintah {

/*`==========TESTING==========*/
/* ---------------------------------------------------------------------
    Add a source at the boundaries
    Currently hard coded to a jet
 ---------------------------------------------------------------------*/    
bool insideOfObject(const int i, 
                    const int k, 
                    const Patch* patch)
{
  Vector dx = patch->dCell();
  //__________________________________
  //  Hard coded for a jet 
  Vector origin(0.0, 0.0, 0.0);         // origin of jet
  double radius = 0.05;                  // radius of jet
  double x = (double) (i) * dx.x() + dx.x()/2.0;
  double z = (double) (k) * dx.z() + dx.z()/2.0;

  double delX = origin.x() - x;
  double delZ = origin.z() - z;
  double h    = sqrt(delX * delX + delZ * delZ);
  
  if (h < radius) {               // if inside the jet then
    return true;
  }
  return false;
}

/* ---------------------------------------------------------------------
    Add a source at the boundaries
    Currently hard coded to a jet
   ---------------------------------------------------------------------*/
  template <class V, class T> void 
      AddSourceBC(V& var, const Patch* patch, 
		    Patch::FaceType face,                                
		    const T& value,                                      
		    IntVector offset = IntVector(0,0,0))                    
{ 
  //__________________________________
  //  hard coded to only apply on yminus
  Patch::FaceType faceToApplyBC = Patch::yminus;
  
  IntVector low,hi;
  low = var.getLowIndex() + offset;
  hi = var.getHighIndex() - offset;
  
  //__________________________________
  // 
  int oneZero = 0;
  if (typeid(V) == typeid(SFCXVariable<double>) ||
      typeid(V) == typeid(SFCYVariable<double>) || 
      typeid(V) == typeid(SFCZVariable<double>)){
    oneZero = 1;
  }

  if(face == faceToApplyBC) {
    switch (face) {
    case Patch::xplus:
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
         if (insideOfObject(j, k,patch)) {            
	    var[IntVector(hi.x()-1,j,k)] = value;
         }
        }
      }
      break;
    case Patch::xminus:
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
          if (insideOfObject(j, k,patch)) {
	     var[IntVector(low.x() + oneZero,j,k)] = value;
          }
        }
      }
      break;
    case Patch::yplus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {
          if (insideOfObject(i, k,patch)) {
	     var[IntVector(i,hi.y()-1,k)] = value;
          }
        }
      }
      break;
    case Patch::yminus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {
          if (insideOfObject(i, k,patch)) {
	     var[IntVector(i,low.y() + oneZero,k)] = value;
//            cout << " I'm applying BC at "<< IntVector(i,low.y() + oneZero,k) << " " << value << endl;
          }
        }
      }
      break;
    case Patch::zplus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {
          if (insideOfObject(i, j,patch)) {
	     var[IntVector(i,j,hi.z()-1)] = value;
          }
        }
      }
      break;
    case Patch::zminus:
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {
          if (insideOfObject(i, j,patch)) {
	     var[IntVector(i,j,low.z() + oneZero)] = value;
          }
        }
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break;
    }
  }
} 
/*==========TESTING==========`*/

//______________________________________________________________________
// Update pressure boundary conditions due to hydrostatic pressure

void setHydrostaticPressureBC(CCVariable<double>& press,Patch::FaceType face, 
			      Vector& gravity,
			      const CCVariable<double>& rho,
			      const Vector& dx, IntVector offset )
{ 
  IntVector low,hi;
  low = press.getLowIndex() + offset;
  hi = press.getHighIndex() - offset;
  
  // cout<< "CCVARIABLE LO" << low <<endl;
  // cout<< "CCVARIABLE HI" << hi <<endl;
  
  switch (face) {
  case Patch::xplus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	press[IntVector(hi.x()-1,j,k)] = 
	  press[IntVector(hi.x()-2,j,k)] + 
	  gravity.x() * rho[IntVector(hi.x()-2,j,k)] * dx.x();
      }
    }
    break;
  case Patch::xminus:
    for (int j = low.y(); j<hi.y(); j++) {
      for (int k = low.z(); k<hi.z(); k++) {
	press[IntVector(low.x(),j,k)] = 
	  press[IntVector(low.x()+1,j,k)] - 
	  gravity.x() * rho[IntVector(low.x()+1,j,k)] * dx.x();;
      }
    }
    break;
  case Patch::yplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	press[IntVector(i,hi.y()-1,k)] = 
	  press[IntVector(i,hi.y()-2,k)] + 
	  gravity.y() * rho[IntVector(i,hi.y()-2,k)] * dx.y();
      }
    }
    break;
  case Patch::yminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int k = low.z(); k<hi.z(); k++) {
	press[IntVector(i,low.y(),k)] = 
	  press[IntVector(i,low.y()+1,k)] - 
	  gravity.y() * rho[IntVector(i,low.y()+1,k)] * dx.y();
      }
    }
    break;
  case Patch::zplus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	press[IntVector(i,j,hi.z()-1)] = 
	  press[IntVector(i,j,hi.z()-2)] +
	  gravity.z() * rho[IntVector(i,j,hi.z()-2)] * dx.z();
      }
    }
    break;
  case Patch::zminus:
    for (int i = low.x(); i<hi.x(); i++) {
      for (int j = low.y(); j<hi.y(); j++) {
	press[IntVector(i,j,low.z())] =
	  press[IntVector(i,j,low.z()+1)] -  
	  gravity.z() * rho[IntVector(i,j,low.z()+1)] * dx.z();
      }
    }
    break;
  case Patch::numFaces:
    break;
  case Patch::invalidFace:
    break;
  }
}

 
/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care Pressure_CC
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& press_CC,
              const CCVariable<double>& rho_micro,
              const string& which_Var,
              const string& kind, 
              const Patch* patch,
              SimulationStateP& sharedState, 
              const int mat_id,
              DataWarehouse* new_dw)
{
  
  Vector dx = patch->dCell();
  Vector gravity = sharedState->getGravity();
  IntVector offset(0,0,0);
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *bcs, *sym_bcs;
    const BoundCond<double> *new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<double> *>(bcs);
    } else
      continue;
 
    if (sym_bcs != 0) { 
      fillFaceFlux(press_CC,face,0.0,dx, 1.0, offset);
    }

    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet") 
       fillFace(press_CC,face,new_bcs->getValue());

      if (new_bcs->getKind() == "Neumann") 
        fillFaceFlux(press_CC,face,new_bcs->getValue(),dx, 1.0, offset);
       
      //__________________________________
      //  When gravity is on 
      if ( fabs(gravity.x()) > 0.0  || 
           fabs(gravity.y()) > 0.0  || 
           fabs(gravity.z()) > 0.0) {
        CCVariable<double> rho_micro_tmp;
        new_dw->allocateTemporary(rho_micro_tmp,  patch);
        if (which_Var == "sp_vol") {
          for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
            IntVector c = *iter;
            rho_micro_tmp[c] = 1.0/rho_micro[c];
          }
        }
        if (which_Var == "rho_micro") {
          for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
            IntVector c = *iter;
            rho_micro_tmp[c] = rho_micro[c];
          }
        }
       setHydrostaticPressureBC(press_CC,face, gravity, rho_micro_tmp, dx, offset);
      }
    }
  }
}

/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care of Density_CC and Temperature_CC
 or any CC Variable with zeroNeumann
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& variable, const string& kind, 
              const Patch* patch, 
              SimulationStateP& sharedState,
              const int mat_id)
{
  Vector dx = patch->dCell();
  Vector grav = sharedState->getGravity();
  IntVector offset(0,0,0);
  bool onEdgeOfDomain = false; 
  
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){

  const BoundCondBase *bcs, *sym_bcs;
  const BoundCond<double> *new_bcs; 
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id, kind,      face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<double> *>(bcs);
      onEdgeOfDomain = true;
    } else {
      onEdgeOfDomain = false;
      continue;
    }
    
    //__________________________________
    // any CC Variable with zeroNeumann
    if (onEdgeOfDomain && kind == "zeroNeumann") {
      fillFaceFlux(variable,face,0.0,dx, 1.0, offset);
    }
    //__________________________________
    // symmetric BC
    if (sym_bcs != 0) { 
      if (kind == "Density" || kind == "Temperature" || kind == "set_if_sym_BC") {
        fillFaceFlux(variable,face,0.0,dx, 1.0, offset);
      }
    }   
    
    if (new_bcs != 0) {
      //__________________________________
      //  Density_CC
      if (kind == "Density") {
        if (new_bcs->getKind() == "Dirichlet") { 
         fillFace(variable,face,new_bcs->getValue(), offset);
        }

        if (new_bcs->getKind() == "Neumann") {
         fillFaceFlux(variable,face,new_bcs->getValue(),dx, 1.0, offset);
        }
/*`==========TESTING==========*/
#if JET_BC
//        cout << " I'm in density "<< face << endl;
        double hardCodedDensity = 1.1792946927* (300.0/1000.0);
//        double hardCodedDensity = 1.1792946927;
        AddSourceBC<CCVariable<double>,double >(variable, patch, face,
                              hardCodedDensity, offset);  
 #endif 
/*==========TESTING==========`*/
      }
 
      //__________________________________
      // Temperature_CC
      if (kind == "Temperature" ){ 
        if (new_bcs->getKind() == "Dirichlet") { 
           fillFace(variable,face,new_bcs->getValue(), offset);
        }
           
         // Neumann && gravity                 
        if (new_bcs->getKind() == "Neumann" ) {
          fillFaceFlux(variable,face,new_bcs->getValue(),dx,1.0, offset);
            
          if(fabs(grav.x()) >0.0 ||fabs(grav.y()) >0.0 ||fabs(grav.z()) >0.0) {
            Material *matl = sharedState->getMaterial(mat_id);
            ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
            if(ice_matl) {
              double cv     = ice_matl->getSpecificHeat();
              double gamma  = ice_matl->getGamma();
                    
              ice_matl->getEOS()->hydrostaticTempAdjustment(face, 
                                  patch,  grav, gamma,
                                  cv,     dx,   variable);
            }  // if(ice_matl) 
          }  // if(gravity)
        }  // if(Neumann)
/*`==========TESTING==========*/
#if JET_BC
//        cout << " I'm in Temperature "<< face << endl;
        double hardCodedTemperature = 1000;
        AddSourceBC<CCVariable<double>, double >(variable, patch, face,
                               hardCodedTemperature, offset);  
 #endif 
/*==========TESTING==========`*/ 
      }  //  if(Temperature)
    }  // if(new_bc)
  }  // Patch loop
}


/* --------------------------------------------------------------------- 
 Function~  setBC--        
 Purpose~   Takes care of Velocity_CC Boundary conditions
 Notes:      CheckValveBC removes any inflow from outside
             the domain.
 ---------------------------------------------------------------------  */
void setBC(CCVariable<Vector>& variable, const string& kind, 
              const Patch* patch, const int mat_id) 
{
  IntVector  low, hi;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    const BoundCondBase *bcs,*sym_bcs;
    const BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);
    } else
      continue;
    //__________________________________
    //  Tangent components Neumann = 0
    //  Normal components = negInterior
    //  It's negInterior since it's on the opposite side of the
    //  plane of symetry
    if (sym_bcs != 0 && (kind == "Velocity" || kind =="set_if_sym_BC") ) {
      fillFaceFlux(variable,face,Vector(0.,0.,0.),dx, 1.0, offset);
      fillFaceNormal(variable,patch,face,offset);
    }
    
    if (new_bcs != 0 && kind == "Neumann" ) {
      fillFaceFlux(variable,face,Vector(0.,0.,0.),dx, 1.0, offset);
    }
      
    if (new_bcs != 0 && kind == "Velocity") {
      if (new_bcs->getKind() == "Dirichlet"){ 
       fillFace(variable,face,new_bcs->getValue(), offset);
      }

      if (new_bcs->getKind() == "Neumann") {
        fillFaceFlux(variable,face,new_bcs->getValue(),dx, 1.0, offset);
      }      
      if (new_bcs->getKind() == "NegInterior") {
         fillFaceFlux(variable,face,Vector(0.0,0.0,0.0),dx, -1.0, offset);
      }
      if (new_bcs->getKind() == "Neumann_CkValve") {
        fillFaceFlux(variable,face,new_bcs->getValue(),dx, 1.0, offset);
        checkValveBC( variable, patch, face); 
      }
      
/*`==========TESTING==========*/
#if JET_BC
//        cout << " I'm in VelocityBC "<< face <<endl;
        Vector hardCodedVelocity(0,100.0,0);
        AddSourceBC<CCVariable<Vector>,Vector >(variable, patch, face, 
                                            hardCodedVelocity, offset);  
 #endif 
/*==========TESTING==========`*/ 
    }  // end velocity loop
  }  // end face loop
}

/*---------------------------------------------------------------------
 Function~  Neuman_SFC--
 Purpose~   Set neumann BC conditions for SFC(*)Variable<double>
 ---------------------------------------------------------------------  */  
 template<class T> void Neuman_SFC(T& var_FC,
                                   const Patch* patch, 
                                   Patch::FaceType face, 
				   const double value, 
                                   const Vector& dx,
                                   IntVector offset)
{ 
  //__________________________________
  // Add 1 to low index when no neighbor patches are present
  IntVector low,hi;  
  int numGC = 0;
  low = patch->getCellLowIndex();
  int XYZ_var;
  if (typeid(T) == typeid(SFCXVariable<double>))
    XYZ_var = 0;
  if (typeid(T) == typeid(SFCYVariable<double>))
    XYZ_var = 1;
  if (typeid(T) == typeid(SFCZVariable<double>))
    XYZ_var = 2;

  if (XYZ_var == 0) {         // SFCX_var
    low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:1,
                   patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
                   patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
  }
  if (XYZ_var == 1) {         // SFCY_var
    low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
                   patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:1,
                   patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:0);
  }
  if (XYZ_var == 2) {         // SFCZ_var
    low+=IntVector(patch->getBCType(Patch::xminus)==Patch::Neighbor?numGC:0,
                   patch->getBCType(Patch::yminus)==Patch::Neighbor?numGC:0,
                   patch->getBCType(Patch::zminus)==Patch::Neighbor?numGC:1);
  }
  low-= offset;
  hi  = patch->getCellHighIndex();
  hi +=IntVector(patch->getBCType(Patch::xplus) ==Patch::Neighbor?numGC:0,
                 patch->getBCType(Patch::yplus) ==Patch::Neighbor?numGC:0,
                 patch->getBCType(Patch::zplus) ==Patch::Neighbor?numGC:0);
  hi += offset;
  // cout<< "Neuman_SFC: BoundaryCond.cc"<<endl;
  // cout<< "low: "<<low<<endl;
  // cout<< "hi:  "<<hi <<endl;
  // cout<< "face:"<<face<< " XYZ_var: "<<XYZ_var<<endl;     
  //__________________________________
  // Only modify the velocities that are tangential to a face.
  // The normal component is computed by ICE
  switch (face) {
  case Patch::xplus:
    if (XYZ_var != 0 ) {
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
          var_FC[IntVector(hi.x()-1,j,k)] = 
            var_FC[IntVector(hi.x()-2,j,k)] + value * dx.x();
        }
      }
    }
   break;
  case Patch::xminus:
    if (XYZ_var != 0 ) {
      for (int j = low.y(); j<hi.y(); j++) {
        for (int k = low.z(); k<hi.z(); k++) {
          var_FC[IntVector(low.x(),j,k)] = 
            var_FC[IntVector(low.x()+1,j,k)] - value * dx.x();
        }
      }
    }
   break;
  case Patch::yplus:
    if (XYZ_var != 1 ) {
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {
          var_FC[IntVector(i,hi.y()-1,k)] = 
            var_FC[IntVector(i,hi.y()-2,k)] + value * dx.y();
        }
      }
    }
   break;
  case Patch::yminus:
    if (XYZ_var != 1 ) {
      for (int i = low.x(); i<hi.x(); i++) {
        for (int k = low.z(); k<hi.z(); k++) {
          var_FC[IntVector(i,low.y(),k)] = 
            var_FC[IntVector(i,low.y()+1,k)] - value * dx.y();
        }
      }
    }
   break;
  case Patch::zplus:
    if (XYZ_var != 2 ) {
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {
          var_FC[IntVector(i,j,hi.z()-1)] = 
            var_FC[IntVector(i,j,hi.z()-2)] + value * dx.z();
        }
      }
    }
   break;
  case Patch::zminus:
    if (XYZ_var != 2 ) {
      for (int i = low.x(); i<hi.x(); i++) {
        for (int j = low.y(); j<hi.y(); j++) {
          var_FC[IntVector(i,j,low.z())] = 
            var_FC[IntVector(i,j,low.z()+1)] -  value * dx.z();
        }
      }
    }
   break;
  case Patch::numFaces:
   break;
  case Patch::invalidFace:
   break;
  }
}

/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes care of vel_FC.x()
 Note:      Neumann BC values are not set on xminus or xplus, 
            hey are computed in AddExchangeContributionToFCVel.
 ---------------------------------------------------------------------  */
void setBC(SFCXVariable<double>& variable, const  string& kind, 
              const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    const BoundCondBase *bcs, *sym_bcs;
    const BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);
    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      // Set the tangential components
      Neuman_SFC<SFCXVariable<double> >(variable,patch,face, 0.0,dx,offset);
          
      // Set normal component = 0
      if( face == Patch::xplus || face == Patch::xminus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars"); 
                                                  !iter.done(); iter++) { 
         IntVector c = *iter;
          variable[c] = 0.0;  
        }
      }
    }
    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      string kind = new_bcs->getKind();
      if (kind == "Dirichlet" && comp == "x") {
        fillFace<SFCXVariable<double>,double>(variable,patch, face,
					      new_bcs->getValue().x(),
					      offset);
      }
      if (kind == "Neumann" && comp == "x") {
        Vector dx = patch->dCell();
        Neuman_SFC<SFCXVariable<double> >(variable, patch, face,
                                          new_bcs->getValue().x(), dx, offset);
      }
    }
  }
}
/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes care of vel_FC.y()
 Note:      Neumann BC values are not set on yminus or yplus, 
            hey are computed in AddExchangeContributionToFCVel.
 ---------------------------------------------------------------------  */
void setBC(SFCYVariable<double>& variable, const  string& kind, 
              const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    const BoundCondBase *bcs, *sym_bcs;
    const BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);
    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      // Set the tangential components
      Neuman_SFC<SFCYVariable<double> >(variable,patch,face,0.0,dx, offset); 
      
      // set normal compoent = 0
      if( face == Patch::yminus || face == Patch::yplus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars");
                                                    !iter.done(); iter++) { 
          IntVector c = *iter;
          variable[c] = 0.0;  
        }
      }
    }
    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      string kind = new_bcs->getKind();
      if (kind == "Dirichlet" && comp == "y") {
        fillFace<SFCYVariable<double>, double >(variable,patch, face,
			   new_bcs->getValue().y(),
			   offset);
      }

      if (kind == "Neumann" && comp == "y") {
        Vector dx = patch->dCell();    
        Neuman_SFC<SFCYVariable<double> >(variable, patch, face,
					  new_bcs->getValue().y(), dx, offset);
      }
    }
/*`==========TESTING==========*/
#if JET_BC
//        cout << " I'm in SFCYVariable "<< endl;
        double hardCodedVelocity = 100.0;
        AddSourceBC<SFCYVariable<double>,double >(variable, patch, face, 
                                            hardCodedVelocity, offset);  
 #endif 
/*==========TESTING==========`*/ 

  }
}
/* --------------------------------------------------------------------- 
 Function~  setBC--      
 Purpose~   Takes care of vel_FC.z()
 Note:      Neumann BC values are not set on zminus or zplus, 
            hey are computed in AddExchangeContributionToFCVel.
 ---------------------------------------------------------------------  */
void setBC(SFCZVariable<double>& variable, const  string& kind, 
              const string& comp, const Patch* patch, const int mat_id) 
{
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    const BoundCondBase *bcs, *sym_bcs;
    const BoundCond<Vector>* new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);
    } else
      continue;

    IntVector offset(0,0,0);

    //__________________________________
    //  Symmetry boundary conditions
    //  -set Neumann = 0 on all walls
    if (sym_bcs != 0) {
      Vector dx = patch->dCell();
      // Set the tangential components
      Neuman_SFC<SFCZVariable<double> >(variable,patch,face,0.0,dx,offset); 
      
      // set normal component = 0
      if( face == Patch::zminus || face == Patch::zplus ) {
        for(CellIterator iter = patch->getFaceCellIterator(face,"FC_vars"); 
                                                      !iter.done(); iter++) { 
         IntVector c = *iter;
          variable[c] = 0.0;  
        }
      }
    }
    //__________________________________
    // Neumann or Dirichlet
    if (new_bcs != 0) {
      string kind = new_bcs->getKind();
      if (kind == "Dirichlet" && comp == "z") {
        fillFace<SFCZVariable<double>, double >(variable,patch, face,
			   new_bcs->getValue().z(),
			   offset);
      }

      if (kind == "Neumann" && comp == "z") {
        Vector dx = patch->dCell();
        Neuman_SFC<SFCZVariable<double> >(variable, patch, face,
                                          new_bcs->getValue().z(), dx, offset);
      }
    }
  }
}
/* --------------------------------------------------------------------- 
 Function~  checkValveBC--      
 Purpose~   Velocity/momentum can only go out of the domain,  If setBC(Neumann)
            calculated an inflow condition this routine sets the velocity 
            to 0.0;  Note call this function after setBC(Neumann)
 ---------------------------------------------------------------------  */
void checkValveBC( CCVariable<Vector>& var, 
                   const Patch* patch,
                   Patch::FaceType face)        
{ 
  switch (face) {
  case Patch::xplus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].x( std::max( var[c].x(), 0.0) );
    }
    break;
  case Patch::xminus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].x(std::min( var[c].x(), 0.0) );
    }
    break;
  case Patch::yplus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].y(std::max( var[c].y(), 0.0) );
    }
    break;
  case Patch::yminus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].y(std::min( var[c].y(), 0.0) );
    }
    break;
  case Patch::zplus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].z(std::max( var[c].z(), 0.0) );
    }
    break;
  case Patch::zminus:
    for(CellIterator iter = patch->getFaceCellIterator(face); 
                                             !iter.done(); iter++) { 
      IntVector c = *iter;
      var[c].z(std::min( var[c].z(), 0.0) );
    }
    break;
  case Patch::numFaces:
    break;
  case Patch::invalidFace:
    break; 
  }
}
/* --------------------------------------------------------------------- 
 Function~  ImplicitMatrixBC--      
 Purpose~   Along each face of the domain set the stencil weight in
           that face = 0
 Naming convention
      +x -x +y -y +z -z
       e, w, n, s, t, b 
 ---------------------------------------------------------------------  */
void ImplicitMatrixBC( CCVariable<Stencil7>& A, 
                   const Patch* patch)        
{ 
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){
    switch (face) {
    case Patch::xplus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter - IntVector(1,0,0));
        A[c].e = 0.0;
      }
      break;
    case Patch::xminus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter + IntVector(1,0,0));
        A[c].w = 0.0;
      }
      break;
    case Patch::yplus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter - IntVector(0,1,0));
        A[c].n = 0.0;
      }
      break;
    case Patch::yminus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter + IntVector(0,1,0)); 
        A[c].s = 0.0;
      }
      break;
    case Patch::zplus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter - IntVector(0,0,1));
        A[c].t = 0.0;
      }
      break;
    case Patch::zminus:
      for(CellIterator iter = patch->getFaceCellIterator(face); 
                                               !iter.done(); iter++) { 
        IntVector c(*iter + IntVector(0,0,1));
        A[c].b = 0.0;
      }
      break;
    case Patch::numFaces:
      break;
    case Patch::invalidFace:
      break; 
    }
  }
}

}  // using namespace Uintah
