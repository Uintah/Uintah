#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/LODIBaseFuncs.h>
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
#include <Core/Util/DebugStream.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Math/MiscMath.h>

 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);

using namespace Uintah;
namespace Uintah {

/*`==========TESTING==========*/
/* ---------------------------------------------------------------------
    Add a source at the boundaries
    Currently hard coded for a jet on the x- face
 ---------------------------------------------------------------------*/    
bool insideOfObject(const int j, 
                    const int k, 
                    const Patch* patch)
{
  Vector dx = patch->dCell();
  //__________________________________
  //  Hard coded for a jet 
  Vector origin(0.0, 0.0, 0.0);         // origin of jet
  double radius = 0.5;                  // radius of jet
  double y = (double) (j) * dx.y() + dx.y()/2.0;
  double z = (double) (k) * dx.z() + dx.z()/2.0;

  double delY = origin.y() - y;
  double delZ = origin.z() - z;
  double h    = sqrt(delY * delY + delZ * delZ);
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
  //  hard coded to only apply on xminus
  Patch::FaceType faceToApplyBC = Patch::xminus;
  if( face == faceToApplyBC) {
   cout << "    AddSourceBC "<< face << " " << value <<endl;
  }
  
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
 //           cout << " I'm applying BC at "<< IntVector(low.x() + oneZero,j,k) << " " << value << endl;
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
  BC_doing << "setHydrostaticPressureBC"<< endl;
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
//______________________________________________________________________
//______________________________________________________________________

#ifdef ORG_BCS
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
  BC_doing << "ORG setBC (Pressure) "<< kind << endl;
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
  BC_doing << "ORG setBC (Temp, Density) "<< kind << endl;
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
#ifdef JET_BC
        double hardCodedDensity = 13.937282;
 //     double hardCodedDensity = 6.271777;
 //     double hardCodedDensity = 1.1792946927* (300.0/1000.0);
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
#ifdef JET_BC
        BC_dbg << "ORG AddSourceBC Temperature "<< face << endl;
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
 ---------------------------------------------------------------------  */
void setBC(CCVariable<Vector>& variable, const string& kind, 
              const Patch* patch, const int mat_id) 
{
  BC_doing << "ORG setBC (Velocity) "<< kind <<endl;
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
      
/*`==========TESTING==========*/
#ifdef JET_BC
        BC_dbg << "ORG AddSourceBC VelocityBC "<< face <<endl;
     //   Vector hardCodedVelocity(10,0,0);
        Vector hardCodedVelocity(209,0,0);
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
  BC_doing << "ORG setBC (SFCXVariable) "<< kind <<endl;
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
/*`==========TESTING==========*/
#ifdef JET_BC
      // double hardCodedVelocity = 10.0;
        double hardCodedVelocity = 209.0;
        AddSourceBC<SFCXVariable<double>,double >(variable, patch, face, 
                                            hardCodedVelocity, offset);  
 #endif 
/*==========TESTING==========`*/ 
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
  BC_doing << "ORG setBC (SFCYVariable) "<< kind <<endl;
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
  BC_doing << "ORG setBC (SFCZVariable) "<< kind <<endl;
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
#endif
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
      
    // only apply the BC when on the edge of the computational domain.
    // skip the faces between neighboring patches  
    bool onEdgeOfDomain = false;  
    if (patch->getBCType(face) == Patch::None) {
      onEdgeOfDomain = true;
    }
    
    if (onEdgeOfDomain){
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
    }  // face on edge of domain?
  }  // face loop
}



/* --------------------------------------------------------------------- 
 Function~  are_We_Using_LODI_BC--                      L   O   D   I
 Purpose~   returns if we are using LODI BC on any face, 
 ---------------------------------------------------------------------  */
bool are_We_Using_LODI_BC(const Patch* patch,
                          vector<bool>& is_LODI_face,
                          const int mat_id)
{ 
  BC_doing << "are_We_Using_LODI_BC on patch"<<patch->getID()<< endl;
  
  bool usingLODI = false;
  
  vector<string> kind(3);
  kind[0] = "Density";
  kind[1] = "Temperature";
  kind[2] = "Pressure";
      
  is_LODI_face.reserve(6);
  //__________________________________
  // Iterate over the faces encompassing the domain
  // not the faces between neighboring patches.
  vector<Patch::FaceType>::const_iterator iter;
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;  
    
    is_LODI_face[face] = false;
    //__________________________________
    // check if temperature, pressure or density
    //  is using LODI
    for(int i = 0; i < 3; i++ ) {
      const BoundCondBase *bcs;
      const BoundCond<double>* new_bcs; 
      bcs     = patch->getBCValues(mat_id,kind[i],face);
      new_bcs = dynamic_cast<const BoundCond<double> *>(bcs);
      if( new_bcs !=0 && new_bcs->getKind() == "LODI" ) {
        usingLODI = true;
        is_LODI_face[face] = true;
      }
      delete new_bcs;
    }
    //__________________________________
    //  check if velocity is using LODI
    const BoundCondBase *bcs;
    const BoundCond<Vector>* new_bcs;

    bcs     = patch->getBCValues(mat_id,"Velocity",face);      
    new_bcs = dynamic_cast<const BoundCond<Vector> *>(bcs);

    if( new_bcs != 0 && new_bcs->getKind() == "LODI" ) {
      usingLODI = true;
      is_LODI_face[face] = true;
    }
    BC_dbg  <<" using LODI on face "<<  is_LODI_face[face]<<endl;
    delete new_bcs;
  }
  return usingLODI;
}



//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//                  J O H N ' S   B C

#ifdef JOHNS_BC

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
  BC_doing << "Johns setBC (press_CC) "<< kind <<" " << which_Var
           << " mat_id = " << mat_id << endl;
  Vector cell_dx = patch->dCell();


  // Iterate over the faces encompassing the domain
  // only set BC on Boundariesfaces
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    IntVector dir= patch->faceAxes(face);
    double dx = cell_dx[dir[0]];
    bool IveSetBC = false;
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      double bc_value = -9;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      getIteratorBCValueBCKind<double>( patch, face, child, kind, mat_id,
					bc_value, bound,bc_kind); 
      if(bc_kind != "NotSet" ) {
        // define what a symmetric  pressure BC means
        if( bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }

        int p_dir = patch->faceAxes(face)[0];     // principal  face direction
        Vector gravity = sharedState->getGravity(); 
        //__________________________________
        // Apply the boundary condition
        if (gravity[p_dir] == 0) { 
          IveSetBC = setNeumanDirichletBC<double>(patch, face, press_CC,bound, 
						  bc_kind, bc_value, cell_dx);
        }
        //__________________________________
        // With Gravity
        // change gravity sign according to the face direction
        if (gravity[p_dir] != 0) {  

          Vector faceDir = patch->faceDirection(face).asVector();
          double grav = gravity[p_dir] * (double)faceDir[p_dir]; 
          IntVector oneCell = patch->faceDirection(face);
          vector<IntVector>::const_iterator iter;

          int plusMinusOne = 1;
          if(which_Var == "sp_vol") {
            plusMinusOne = -1;
          }
          for (iter=bound.begin();iter != bound.end(); iter++) { 
            IntVector adjCell = *iter - oneCell;
            press_CC[*iter] = press_CC[adjCell] 
                            + grav * dx * pow(rho_micro[adjCell],plusMinusOne);
          }
          IveSetBC = true;
        }  // with gravity
        //__________________________________
        //  debugging
        BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1)
	        << endl;
      }  // if bcKind != notSet
    }  // if face == none
  }  // faces loop
}
/* --------------------------------------------------------------------- 
 Function~  setBC--
 Purpose~   Takes care any CC variable
 ---------------------------------------------------------------------  */
void setBC(CCVariable<double>& var_CC,
           const string& desc, 
           const Patch* patch,
           SimulationStateP& sharedState, 
           const int mat_id)
{
  BC_doing << "Johns setBC (double) "<< desc << " mat_id = " << mat_id << endl;
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
      double bc_value = -9;
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      getIteratorBCValueBCKind<double>( patch, face, child, desc, mat_id,
					bc_value, bound,bc_kind); 
      if (bc_kind != "NotSet" ) {
        //__________________________________
        // LOGIC
        // Any CC Variable
        if (desc == "set_if_sym_BC" && bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }
        if (desc == "zeroNeumann" || bc_kind == "symmetric"){
          bc_kind = "zeroNeumann";
        }

        // mass Fraction/scalar have a zeroNeumann default BC
        bool defaultZeroNeumann = false;
        string::size_type pos1 = desc.find ("massFraction");
        string::size_type pos2 = desc.find ("scalar");
        string::size_type pos3 = desc.find ("mixtureFraction");
        string::size_type found = std::string::npos;
        if ( pos1 != found || pos2 !=  found || pos3 != found){
          defaultZeroNeumann = true;
        }
        if (defaultZeroNeumann || bc_kind == "NotSet") {
          bc_kind == "zeroNeumann";
        }      

        //__________________________________
        // Apply the boundary condition
        IveSetBC =  setNeumanDirichletBC<double>
                       (patch, face, var_CC,bound, bc_kind, bc_value, cell_dx);

        //__________________________________
        // Temperature and Gravity and ICE Matls
        // --change gravity sign according to the face direction
        int p_dir = patch->faceAxes(face)[0];     // principal  face direction
        Vector gravity = sharedState->getGravity();                             
        Material *matl = sharedState->getMaterial(mat_id);
        ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

        if (gravity[p_dir] != 0 && desc == "Temperature" && ice_matl) {  

          Vector faceDir = patch->faceDirection(face).asVector();
          double grav  = gravity[p_dir] * (double)faceDir[p_dir]; 
          double cv    = ice_matl->getSpecificHeat();
          double gamma = ice_matl->getGamma();
          double dx    = cell_dx[p_dir];

          vector<IntVector>::const_iterator iter;  
          for (iter=bound.begin(); iter != bound.end(); iter++) { 
            var_CC[*iter] += grav * dx/((gamma - 1.)*cv);
          }
        }
        //__________________________________
        //  debugging
        BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = "<< *bound.begin()<< " "<< *(bound.end()-1)
	        << endl;
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
           const int mat_id)
{
  BC_doing <<"Johns setBC (Vector_CC) "<< desc <<" mat_id = " <<mat_id<< endl;
  Vector cell_dx = patch->dCell();
  
  //__________________________________
  // Iterate over the faces encompassing the domain
  // not the faces between neighboring patches.
  vector<Patch::FaceType>::const_iterator iter;
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    IntVector oneCell = patch->faceDirection(face);
    bool IveSetBC = false;
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id);

    for (int child = 0;  child < numChildren; child++) {
      Vector bc_value = Vector(-9,-9,-9);
      string bc_kind = "NotSet";
      vector<IntVector> bound;
      getIteratorBCValueBCKind<Vector>(patch, face, child, desc, mat_id,
				       bc_value, bound,bc_kind);
     
      if (bc_kind != "NotSet" ) {
        //__________________________________
        // Apply the boundary condition
        IveSetBC =  setNeumanDirichletBC<Vector>(patch, face, var_CC,bound, 
						 bc_kind, bc_value, cell_dx);

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
        BC_dbg <<"Face: "<< face <<" I've set BC " << IveSetBC
               <<"\t child " << child  <<" NumChildren "<<numChildren 
               <<"\t BC kind "<< bc_kind <<" \tBC value "<< bc_value
               <<"\t bound limits = " <<*bound.begin()<<" "<< *(bound.end()-1)
	        << endl;
      }  // if (bcKind != "notSet"    
    }  // child loop
  }  // faces loop
}
#endif // end of #if John'sBC
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//______________________________________________________________________
//
#ifdef LODI_BCS


/* --------------------------------------------------------------------- 
 Function~  setBCDensityLODI--                      L   O   D   I
 Purpose~   Takes care of Symmetry BC, Dirichelet BC, Characteristic BC
            for Density
 ---------------------------------------------------------------------  */
void setBCDensityLODI(CCVariable<double>& rho_CC,
                StaticArray<CCVariable<Vector> >& di,
                const CCVariable<Vector>& nu,
                constCCVariable<double>& rho_tmp,
                const CCVariable<double>& p,
                constCCVariable<Vector>& vel,
                const double delT,
                const Patch* patch,
                const int mat_id)
{ 
  BC_doing << "LODI setBC (Density) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);  

  //__________________________________
  //  Set the LODI BC's first and then let
  //  the other faces and BC's wipe out what
  //  LODI set in the corners and edges
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){ 

    const BoundCondBase *rho_bcs;
    const BoundCond<double> *rho_new_bcs;
    std::string rho_kind   = "Density";

    if(patch->getBCType(face) == Patch::None) {
      rho_bcs = patch->getBCValues(mat_id, rho_kind,face);
      rho_new_bcs   = dynamic_cast<const BoundCond<double> *>(rho_bcs);
    } else {
      continue;
    }
 
    if(rho_new_bcs != 0 && rho_new_bcs->getKind() == "LODI"){ 
       FaceDensityLODI(patch, face, rho_CC, di, nu, rho_tmp, vel, delT, dx);
    }
  }  
  
  
  //__________________________________
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){ //loop over faces for a given patch: 2

    const BoundCondBase *rho_bcs, *sym_bcs;
    const BoundCond<double> *rho_new_bcs;

    std::string rho_kind   = "Density";

    if(patch->getBCType(face) == Patch::None) {
      rho_bcs = patch->getBCValues(mat_id, rho_kind,face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      rho_new_bcs   = dynamic_cast<const BoundCond<double> *>(rho_bcs);
    } else {
      continue;
    }

    if(sym_bcs != 0) {
      fillFaceFlux(rho_CC, face, 0.0, dx, 1.0, offset);
    }

    if(rho_new_bcs != 0 && rho_new_bcs->getKind() == "Dirichlet") {
      fillFace(rho_CC, face, rho_new_bcs->getValue(), offset);
    }

    if (rho_new_bcs != 0 && rho_new_bcs->getKind() == "Neumann") {
      fillFaceFlux(rho_CC, face, 
                 rho_new_bcs->getValue(), dx, 1.0, offset);
    }    
/*`==========TESTING==========*/
#ifdef JET_BC
          double hardCodedDensity = 13.937282;
    //      double hardCodedDensity = 6.271777;
   //   double hardCodedDensity = 1.1792946927* (300.0/1000.0);
    AddSourceBC<CCVariable<double>,double >(rho_CC, patch, face,
                           hardCodedDensity, offset);  
#endif 
/*==========TESTING==========`*/ 
  } //faces
}


/* --------------------------------------------------------------------- 
 Function~  setBCVelLODI--                    L   O   D   I
 Purpose~   Takes care of Symmetry BC, Dirichelet BC, Characteristic BC
            for momentum equations
 ---------------------------------------------------------------------  */
void setBCVelLODI(CCVariable<Vector>& vel_CC,
            StaticArray<CCVariable<Vector> >& di,
            const CCVariable<Vector>& nu,
            constCCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            constCCVariable<Vector>& vel,
            const double delT,
            const Patch* patch,
            const int mat_id)
{ 
  BC_doing << "LODI setBC (Vel) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);
  
  //__________________________________
  //  Set the LODI BC's first and then let
  //  the other faces and BC's wipe out what
  //  LODI set in the corners and edges
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){ 

    const BoundCondBase *vel_bcs;
    const BoundCond<Vector> *vel_new_bcs;

    std::string kind   = "Velocity";

    if(patch->getBCType(face) == Patch::None) {
      vel_bcs   = patch->getBCValues(mat_id, kind, face);
      vel_new_bcs = dynamic_cast<const BoundCond<Vector> *>(vel_bcs);
    } else {
      continue;
    } 
    if (vel_new_bcs != 0 && kind == "Velocity" && vel_new_bcs->getKind() == "LODI") {
      
      FaceVelLODI( patch, face, vel_CC, di, nu,
                  rho_tmp, p, vel, delT, dx);
    }
  } 
  
  //__________________________________
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){ //loop over faces for a given patch: 2

    const BoundCondBase *vel_bcs, *sym_bcs;
    const BoundCond<Vector> *vel_new_bcs;

    std::string kind   = "Velocity";

    if(patch->getBCType(face) == Patch::None) {
      vel_bcs   = patch->getBCValues(mat_id, kind, face);
      sym_bcs = patch->getBCValues(mat_id, "Symmetric", face);
      vel_new_bcs   = dynamic_cast<const BoundCond<Vector> *>(vel_bcs);
    } else {
      continue;
    }

    if (sym_bcs != 0 && (kind == "Velocity" || kind =="set_if_sym_BC") ) {
      fillFaceFlux(vel_CC,face,Vector(0.,0.,0.),dx, 1.0, offset);
      fillFaceNormal(vel_CC,patch,face,offset);
    }
    
    if (vel_new_bcs != 0 && kind == "Neumann" ) {
      fillFaceFlux(vel_CC, face, Vector(0.,0.,0.), dx, 1.0, offset);
    }
      
    if (vel_new_bcs != 0 && kind == "Velocity") {
      if (vel_new_bcs->getKind() == "Dirichlet"){ 
       fillFace(vel_CC, face, vel_new_bcs->getValue(), offset);
      }

      if (vel_new_bcs->getKind() == "Neumann") {
        fillFaceFlux(vel_CC, face, vel_new_bcs->getValue(), dx, 1.0, offset);
      }
/*`==========TESTING==========*/
#ifdef JET_BC
       // Vector hardCodedVelocity(10.0,0.0,0);
        Vector hardCodedVelocity(209.0,0.0,0);
        AddSourceBC<CCVariable<Vector>,Vector >(vel_CC, patch, face, 
                                            hardCodedVelocity, offset);  
 #endif 
/*==========TESTING==========`*/
      }  //end velocity
    } // loop over faces
  } 
/* --------------------------------------------------------------------- 
 Function~  setBCTempLODI--                   L   O   D   I
 Purpose~   Takes care of Symmetry BC, Dirichelet BC, Characteristic BC 
            for temperature
 ---------------------------------------------------------------------  */
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
              const int mat_id )
{ 
  BC_doing << "LODI setBC (Temp) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);

  //__________________________________
  //  Set the LODI BC's first and then let
  //  the other faces and BC's wipe out what
  //  LODI set in the corners and edges
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){

    const BoundCondBase *temp_bcs;
    const BoundCond<double> *temp_new_bcs;

    std::string temp_kind  = "Temperature";
    if(patch->getBCType(face) == Patch::None) {
      temp_bcs      = patch->getBCValues(mat_id, temp_kind,  face);
      temp_new_bcs  = dynamic_cast<const BoundCond<double> *>(temp_bcs);
    } else {
      continue;
    }

    if (temp_new_bcs != 0 && temp_new_bcs->getKind() == "LODI") {
       FaceTempLODI(patch, face, temp_CC, di,
                    e, rho_CC,nu,
                    rho_tmp, p, vel, 
                    delT, cv, gamma, dx);
    }
  } 
  
  //__________________________________
  for(Patch::FaceType face = Patch::startFace;
      face <= Patch::endFace; face=Patch::nextFace(face)){ //loop over faces for a given patch: 2

    const BoundCondBase *temp_bcs, *sym_bcs;
    const BoundCond<double> *temp_new_bcs;

    std::string temp_kind  = "Temperature";
    if(patch->getBCType(face) == Patch::None) {
      temp_bcs  = patch->getBCValues(mat_id, temp_kind,  face);
      sym_bcs = patch->getBCValues(mat_id,"Symmetric",face);
      temp_new_bcs  = dynamic_cast<const BoundCond<double> *>(temp_bcs);
    } else {
      continue;
    }

    if(sym_bcs != 0) {
      fillFaceFlux(temp_CC , face, 0.0, dx, 1.0, offset);
    }

    if(temp_new_bcs != 0 && temp_new_bcs->getKind() == "Dirichlet") {
      fillFace(temp_CC, face, temp_new_bcs->getValue(), offset);
    }

    if (temp_new_bcs != 0 && temp_new_bcs->getKind() == "Neumann") {
      fillFaceFlux(temp_CC, face, 
                 temp_new_bcs->getValue(), dx, 1.0, offset);
    }
/*`==========TESTING==========*/
  #ifdef JET_BC
        double hardCodedTemperature = 1000;
        AddSourceBC<CCVariable<double>, double >(temp_CC, patch, face,
                               hardCodedTemperature, offset); 
 #endif  
/*==========TESTING==========`*/ 
    } 
  } 


/*__________________________________________________________________
 Function~ computeDi--              L   O   D   I
 Purpose~  compute Di's at the boundary cells using upwind first-order 
           differenceing scheme
____________________________________________________________________*/
void computeDi(StaticArray<CCVariable<Vector> >& d,
               const vector<bool>& d_is_LODI_face,
               constCCVariable<double>& rho,              
               const CCVariable<double>& press,                   
               constCCVariable<Vector>& vel,                  
               constCCVariable<double>& speedSound_,                    
               const Patch* patch,                            
               const int mat_id)                              
{
  BC_doing << "LODI computeLODIFirstOrder "<< endl;
  Vector dx = patch->dCell();

  vector<IntVector> R_Offset(6);
  R_Offset[Patch::xminus] = IntVector(1,0,0);  // right cell offset
  R_Offset[Patch::xplus]  = IntVector(0,0,0);
  R_Offset[Patch::yminus] = IntVector(0,1,0);
  R_Offset[Patch::yplus]  = IntVector(0,0,0);
  R_Offset[Patch::zminus] = IntVector(0,0,1);
  R_Offset[Patch::zplus]  = IntVector(0,0,0);

  vector<IntVector> L_Offset(6);
  L_Offset[Patch::xminus] = IntVector(0, 0, 0);   // left cell offset
  L_Offset[Patch::xplus]  = IntVector(-1,0, 0);
  L_Offset[Patch::yminus] = IntVector(0, 0, 0);
  L_Offset[Patch::yplus]  = IntVector(0,-1, 0);
  L_Offset[Patch::zminus] = IntVector(0, 0, 0);
  L_Offset[Patch::zplus]  = IntVector(0, 0, -1);

  // Iterate over the faces encompassing the domain
  // only set DI on Boundariesfaces that are LODI
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
 
    if (d_is_LODI_face[face] ) {
      BC_dbg << " computing DI on face " << face 
             << " patch " << patch->getID()<<endl;
      //_____________________________________
      //Compute Di at
      IntVector axes = patch->faceAxes(face);
      int dir0 = axes[0]; // find the principal dir and other 2 directions
      int dir1 = axes[1]; 
      int dir2 = axes[2];    

      double delta = dx[dir0];

      IntVector normal = patch->faceDirection(face);
      double norm = (double)normal[dir0];

      for(CellIterator iter=patch->getFaceCellIterator(face, "plusEdgeCells"); 
          !iter.done();iter++) {
        IntVector c = *iter;
        IntVector r = c + R_Offset[face];
        IntVector l = c + L_Offset[face];

        double speedSound = speedSound_[c];
        double speedSoundsqr = speedSound * speedSound;
        double vel_bndry = vel[c][dir0];

        double drho_dx = (rho[r] - rho[l])/delta;
        double dp_dx   = (press[r] - press[l])/delta;
        Vector dVel_dx = (vel[r] - vel[l])/(delta);

        //Due to numerical noice , we filter them out by hard coding
        //    
        if(fabs(drho_dx) < 1.0e-10) drho_dx = 0.0;
        if(fabs(dp_dx) < 1.0e-10)   dp_dx = 0.0;
        if(fabs(dVel_dx[dir0]) < 1.0e-10) dVel_dx[dir0]=0.0;
        if(fabs(dVel_dx[dir1]) < 1.0e-10) dVel_dx[dir1]=0.0;
        if(fabs(dVel_dx[dir2]) < 1.0e-10) dVel_dx[dir2]=0.0;

        //__________________________________
        // L1 Wave Amplitude
        int L1_sign;
        double L1 = 0;
        L1_sign = Sign(norm * (vel_bndry - speedSound));
        if(L1_sign > 0) {       // outgoing waves
          L1 = (vel_bndry - speedSound) 
             * (dp_dx - rho[c] * speedSound * dVel_dx[dir0]);
        } 
        //__________________________________
        // L2, 3, 4 Wave Amplitude
        int L234_sign;
        double L2=0, L3=0, L4=0;
        L234_sign = Sign(norm * vel_bndry);
        if(L234_sign > 0) {     // outgoing waves
          L2 = vel_bndry * (speedSoundsqr * drho_dx - dp_dx);
          L3 = vel_bndry * dVel_dx[dir1];
          L4 = vel_bndry * dVel_dx[dir2];
        } 
        //__________________________________
        // L5 Wave Amplitude
        int L5_sign;
        double L5=0;
        L5_sign =  Sign(norm * (vel_bndry + speedSound));
        if(L5_sign > 0) {      // outgoing wave
          L5 = (vel_bndry + speedSound) 
             * (dp_dx + rho[c] * speedSound * dVel_dx[dir0]);
        } 
        //__________________________________
        // Compute d1-5
        d[1][c][dir0] = (L2 + 0.5 * (L1 + L5))/(speedSoundsqr);
        d[2][c][dir0] = 0.5 * (L5 + L1);
        d[3][c][dir0] = 0.5 * (L5 - L1)/(rho[c] * speedSound);
        d[4][c][dir0] = L3;
        d[5][c][dir0] = L4;
      }
    } // if(onEdgeOfDomain) 
  } //end of for loop over faces
}//end of function

/*__________________________________________________________________
 Function~ computeNu--                    L   O   D   I
 Purpose~  compute dissipation coefficients 
__________________________________________________________________*/ 
void computeNu(CCVariable<Vector>& nu,
               const vector<bool>& is_LODI_face,
               const CCVariable<double>& p, 
               const Patch* patch)
{
  BC_doing << "LODI computeNu "<< endl;
  double d_SMALL_NUM = 1.0e-100;
    
  // Iterate over the faces encompassing the domain
  // only set DI on Boundariesfaces that are LODI
  vector<Patch::FaceType>::const_iterator iter;
  
  for (iter  = patch->getBoundaryFaces()->begin(); 
       iter != patch->getBoundaryFaces()->end(); ++iter){
    Patch::FaceType face = *iter;
    
    if (is_LODI_face[face] ) {
      BC_dbg << " computing Nu on face " << face 
             << " patch " << patch->getID()<<endl;   
              
      vector<int> otherDir(2);
      IntVector axes = patch->faceAxes(face);
      int P_dir   = axes[0]; // principal direction
      otherDir[0] = axes[1]; // other vector directions
      otherDir[1] = axes[2];  

      for(CellIterator iter=patch->getFaceCellIterator(face, "minusEdgeCells"); 
                                                          !iter.done();iter++) {
        IntVector c = *iter;

        for ( int i = 0; i < 2 ; i++ ) {  // set both orthogonal components
          int dir = otherDir[i];
          IntVector r = c;
          IntVector l = c;
          r[dir] += 1;  // tweak the r and l cell indices
          l[dir] -= 1; 
                        // 2nd order cell centered difference
          nu[c][dir] = fabs(p[r] - 2.0 * p[c] + p[l])/
                        (fabs(p[r] - p[c]) + fabs(p[c] - p[l])  + d_SMALL_NUM);
        }
      }
      //__________________________________
      //    E D G E S
      // use cell centered and one sided differencing
      // only hit outside faces, not faces between 2 patches
      vector<Patch::FaceType>::const_iterator iter;
      
      for (iter = patch->getBoundaryFaces()->begin(); 
           iter != patch->getBoundaryFaces()->end(); ++iter){
      
        Patch::FaceType face0 = *iter;
        //__________________________________
        //  Find the Vector components Edir1 and Edir2
        //  for this particular edge
        IntVector faceDir = patch->faceDirection(face0);
        IntVector axes = patch->faceAxes(face0);
        int Edir1 = axes[0];
        int Edir2 = otherDirection(P_dir, Edir1);

        CellIterator iterLimits =  
                      patch->getEdgeCellIterator(face,face0,"minusCornerCells");

        for(CellIterator iter = iterLimits;!iter.done();iter++){ 

          IntVector c = *iter;
          IntVector r  = c;
          IntVector rr = c;
          r[Edir1]  -= faceDir[Edir1];      // tweak the r and l cell indices
          rr[Edir1] -= 2 * faceDir[Edir1];  // One sided differencing
          nu[c][Edir1] = fabs(p[c] - 2.0 * p[r] + p[rr])/
                        (fabs(p[c] - p[r]) + fabs(p[r] - p[rr])  + d_SMALL_NUM);

          IntVector r2 = c;
          IntVector l2 = c;
          r2[Edir2] += 1;  // tweak the r and l cell indices
          l2[Edir2] -= 1;  // cell centered differencing
          nu[c][Edir2] = fabs(p[r2] - 2.0 * p[c] + p[l2])/
                        (fabs(p[r2] - p[c]) + fabs(p[c] - p[l2])  + d_SMALL_NUM);
        }
      }
      //________________________________________________________
      // C O R N E R S   
  /*`==========TESTING==========*/
  // Need a clever way to figure out the r and rr indicies
  //  for the two different directions
  #if 0 
      vector<IntVector> crn(4);
      computeCornerCellIndices(patch, face, crn);

      for( int corner = 0; corner < 4; corner ++ ) {
        IntVector c = crn[corner];



        IntVector r  = c;
        IntVector rr = c;
        for ( dir.begin();
          r[Edir2]  -= 1;  // tweak the r and l cell indices
          rr[Edir2] -= 2;  // One sided differencing

          IntVector adj = c - offset;
          nu[c][Edir1] = fabs(p[c] - 2.0 * p[r] + p[rr])/
                        (fabs(p[c] - p[r]) + fabs(p[r] - p[rr])  + d_SMALL_NUM);
      } 
  #endif     
  /*==========TESTING==========`*/
    }  // on the right face with LODI BCs
  }
}
/* --------------------------------------------------------------------- 
 Function~  setBCPress_LODI--                   L   O   D   I
 Purpose~   Takes care Pressure_CC
 ---------------------------------------------------------------------  */
void  setBCPress_LODI(CCVariable<double>& press_CC,
                      StaticArray<CCVariable<double> >& var_CC,
                      StaticArray<constCCVariable<double> >& Temp_CC,
                      StaticArray<CCVariable<double> >& f_theta,
                      const string& which_Var,
                      const string& kind, 
                      const Patch* patch,
                      SimulationStateP& sharedState, 
                      const int mat_id,
                      DataWarehouse* new_dw)
{
  BC_doing << "LODI setBC(Press) "<< endl;
  Vector dx = patch->dCell();
  Vector gravity = sharedState->getGravity();
  IntVector offset(0,0,0);
  
  int numALLMatls = sharedState->getNumMatls();  
  StaticArray<CCVariable<double> > rho_micro(numALLMatls);
  //__________________________________
  // compute rho_micro from var_CC
  for (int m = 0; m < numALLMatls; m++) {
    new_dw->allocateTemporary(rho_micro[m],  patch);
    if (which_Var == "sp_vol") {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        rho_micro[m][c] = 1.0/var_CC[m][c];
      }
    }
    if (which_Var == "rho_micro") {
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++) {
        IntVector c = *iter;
        rho_micro[m][c] = var_CC[m][c];
      }
    }
  }  // matls

  if( which_Var !="rho_micro" && which_Var !="sp_vol" ){
    throw InternalError("setBCPress_LODI: Invalid option for which_var");
  }

  //__________________________________
  //  Set the LODI BC's first and then let
  //  the other faces and BC's wipe out what
  //  LODI set in the corners and edges
  for(Patch::FaceType face = Patch::startFace;
                      face <= Patch::endFace; face=Patch::nextFace(face)){
    const BoundCondBase *bcs;
    const BoundCond<double> *new_bcs;
    if (patch->getBCType(face) == Patch::None) {
      bcs     = patch->getBCValues(mat_id,kind,face);
      new_bcs = dynamic_cast<const BoundCond<double> *>(bcs);
    } else {
      continue;
    }

    if ( new_bcs != 0 && new_bcs->getKind() == "LODI"){
      fillFacePress_LODI(patch, press_CC, rho_micro, Temp_CC, f_theta, numALLMatls,
                         sharedState, face);
    }
  } 
  
  //__________________________________
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
      fillFaceFlux(press_CC,face,0.0,dx,1.0,offset);
    }

    if (new_bcs != 0) {
      if (new_bcs->getKind() == "Dirichlet") {
       fillFace(press_CC,face,new_bcs->getValue());
      }

      if (new_bcs->getKind() == "Neumann") { 
        fillFaceFlux(press_CC,face,new_bcs->getValue(),dx, 1.0, offset);
      }
               
      //__________________________________
      //  When gravity is on 
      if ( fabs(gravity.x()) > 0.0  || 
           fabs(gravity.y()) > 0.0  || 
           fabs(gravity.z()) > 0.0) {
       int SURROUND_MATL = 0;        // Mat index of surrounding matl.
       setHydrostaticPressureBC(press_CC,face, gravity, 
                                rho_micro[SURROUND_MATL], dx, offset);
      }
    }
  }
}
#endif  // LODI_BC
}  // using namespace Uintah
