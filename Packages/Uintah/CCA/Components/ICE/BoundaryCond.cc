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

 // setenv SCI_DEBUG "ICE_BC_DBG:+,ICE_BC_DOING:+"
static DebugStream BC_dbg(  "ICE_BC_DBG", false);
static DebugStream BC_doing("ICE_BC_DOING", false);
using namespace Uintah;
namespace Uintah {



//template<class T>void doNothing(T& Temp_CC)
//{
//}

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
        double hardCodedDensity = 1.1792946927* (300.0/1000.0);
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
        Vector hardCodedVelocity(10,0,0);
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
        double hardCodedVelocity = 10.0;
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








//______________________________________________________________________
//                  J O H N ' S   B C

#ifdef JOHNS_BC
// Takes care of Pressure_CC
void setBC(CCVariable<double>& press_CC,
              const CCVariable<double>& rho_micro,
              const string& which_Var,
              const string& kind, 
              const Patch* patch,
              SimulationStateP& sharedState, 
              const int mat_id,
              DataWarehouse* new_dw)
{
  BC_doing << "Johns setBC (press_CC) "<< kind <<" " << which_Var<<endl;
  Vector dx = patch->dCell();
  for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
       face=Patch::nextFace(face)) {
    double spacing;
    double gravity;
    determineSpacingAndGravity(face,dx,sharedState,spacing,gravity);

    if (patch->getBCType(face) == Patch::None) {
      // find the correct BC
      // check its type (symmetric, neumann, dirichlet)
      // do the fill face
      
      // fill face
      // as iterate over the values, need to check what the proper value
      // should be,  need a inside function for getValue(), since
      // the BC value could be different -- union, difference, etc.

      // For a given Intvector boundary, find the appropriate bc, determine if
      // it is symmetric, neumann, dirichlet, and its value,
      int numChildren = patch->getBCDataArray(face)->getNumberChildren();
      for (int child = 0;  child < numChildren; child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
                                                   kind,bound,inter,
                                                   sfx,sfy,sfz,
                                                   child);
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);

       const BoundCond<double> *new_bcs = 
         dynamic_cast<const BoundCond<double> *>(bc);       
       double bc_value=0;
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
#if 0
         cout << "BC kind = " << bc_kind << endl;
         cout << "BC value = " << bc_value << endl;
#endif
       }
       
       // Apply the boundary conditions
       vector<IntVector>::const_iterator boundary,interior;
         
       if (bc_kind == "Dirichlet")
         for (boundary=bound.begin(); boundary != bound.end(); 
              boundary++) 
           press_CC[*boundary] = bc_value;
       
       if (bc_kind == "Neumann") {
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end();   boundary++,interior++) {
           press_CC[*boundary] = press_CC[*interior] - bc_value*spacing;
         }
       }
         
       
       if (sym_bc != 0)
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end(); boundary++,interior++) 
           press_CC[*boundary] = press_CC[*interior];
       
       // If we have gravity, need to apply hydrostatic condition
       if ((sharedState->getGravity()).length() > 0.) {
         if (which_Var == "sp_vol") {
           for (boundary=bound.begin(),interior=inter.begin();
               boundary != bound.end(); boundary++,interior++) 
             press_CC[*boundary] = press_CC[*interior] + 
              gravity*spacing/rho_micro[*interior];
         }
         if (which_Var == "rho_micro") {
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             press_CC[*boundary] = press_CC[*interior] +
              gravity*spacing*rho_micro[*interior];
         }
       }
      }
    }
  }
}
//______________________________________________________________________
//
void determineSpacingAndGravity(Patch::FaceType face, 
                            Vector& dx,SimulationStateP& sharedState,
                            double& spacing, double& gravity)
{
  if (face == Patch::xminus || face == Patch::xplus) spacing = dx.x();
  if (face == Patch::yminus || face == Patch::yplus) spacing = dx.y();
  if (face == Patch::zminus || face == Patch::zplus) spacing = dx.z();
  
  if (face == Patch::xminus) gravity = -(sharedState->getGravity().x());
  if (face == Patch::xplus) gravity = (sharedState->getGravity().x());
  if (face == Patch::yminus) gravity = -(sharedState->getGravity().y());
  if (face == Patch::yplus) gravity = (sharedState->getGravity().y());
  if (face == Patch::zminus) gravity = -(sharedState->getGravity().z());
  if (face == Patch::zplus) gravity = (sharedState->getGravity().z());
}


//______________________________________________________________________
// Take care of the Density Temperature and MassFraction
void setBC(CCVariable<double>& variable, const string& kind, 
              const Patch* patch,  SimulationStateP& sharedState,
              const int mat_id)
{
  BC_doing << "Johns setBC (Temp, Density) "<< kind <<endl;
  Vector dx = patch->dCell();
  for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
       face=Patch::nextFace(face)) {
    double spacing;
    double gravity;
    determineSpacingAndGravity(face,dx,sharedState,spacing,gravity);

    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren();
      for (int child = 0; child < numChildren; child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
                                                   kind, bound,inter,
                                                   sfx,sfy,sfz,
                                                   child);
      
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);
       const BoundCond<double> *new_bcs = 
         dynamic_cast<const BoundCond<double> *>(bc);       
       
       double bc_value=0;
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
       }
       
       // Apply the "zeroNeumann"
       vector<IntVector>::const_iterator boundary,interior;
       if (kind == "zeroNeumann")
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end(); boundary++,interior++) 
           variable[*boundary] = variable[*interior];
       
       if (sym_bc != 0) {
         if (kind == "Density" || kind == "Temperature" 
             || kind == "set_if_sym_BC")
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior];
       }
   
       if (kind == "Density") {
         if (bc_kind == "Dirichlet")
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = bc_value;
         if (bc_kind == "Neumann")
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] =  variable[*interior] - bc_value*spacing;
       }
       

       //__________________________________
       //  mass Fraction BCs
       bool massFractionBC = false;   // check for massFraction BC
       string::size_type pos = kind.find ("massFraction");
       if ( pos != std::string::npos ){
         massFractionBC = true;
       }

       if (massFractionBC) {
        cout << kind << endl;
         if (bc_kind == "Dirichlet"){
            for (boundary=bound.begin(); boundary != bound.end(); boundary++){ 
              variable[*boundary] = bc_value;
            }
          }else if(bc_kind == "Neumann"){
            for (boundary=bound.begin(),interior=inter.begin(); 
                boundary != bound.end(); boundary++,interior++){
              variable[*boundary] =  variable[*interior] - bc_value*spacing;
            }
          }else{   // if it hasn't been specified then assume it's zeroNeumann
            for (boundary=bound.begin(),interior=inter.begin(); 
                boundary != bound.end(); boundary++,interior++){
              variable[*boundary] =  variable[*interior];
            }
          }
        }
        //__________________________________
        //  Temperature
        if (kind == "Temperature") {
         if (bc_kind == "Dirichlet")
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = bc_value;
         if (bc_kind == "Neumann") {
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior] - bc_value*spacing;
           // We have gravity
           if ((sharedState->getGravity()).length() > 0.) {
             // Do the hydrostatic temperature adjustment
             Material *matl = sharedState->getMaterial(mat_id);
             ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
             if (ice_matl) {
              double cv = ice_matl->getSpecificHeat();
              double gamma = ice_matl->getGamma();
              for (boundary=bound.begin(); boundary != bound.end(); 
                   boundary++) 
                variable[*boundary] += gravity*spacing/((gamma-1.)*cv);
             }
           }
         }
       }
      }
    } 
    
  }
}


/* --------------------------------------------------------------------- 
 Function~  setBC--        
 Purpose~   Takes care of Velocity_CC Boundary conditions
 ---------------------------------------------------------------------  */
void setBC(CCVariable<Vector>& variable, const string& kind, 
              const Patch* patch, const int mat_id) 
{
  BC_doing << "Johns setBC (Vector) "<< kind << endl;
  Vector dx = patch->dCell();
  for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
       face=Patch::nextFace(face)) {
    
    Vector sign;
    double spacing;
    if (face == Patch::xminus || face == Patch::xplus) {
      sign=Vector(-1.,1.,1.);
      spacing = dx.x();
    }
    if (face == Patch::yminus || face == Patch::yplus) {
      sign=Vector(1.,-1.,1.);
      spacing = dx.y();
    }
    if (face == Patch::zminus || face == Patch::zplus) {
      sign=Vector(1.,1.,-1.);
      spacing = dx.z();
    }

    if (patch->getBCType(face) == Patch::None) {
      for (int child = 0; 
          child < patch->getBCDataArray(face)->getNumberChildren(); child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,
                                                   kind,bound,inter,
                                                   sfx,sfy,sfz,
                                                   child);
      
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);
       const BoundCond<Vector> *new_bcs = 
         dynamic_cast<const BoundCond<Vector> *>(bc);       
       
       Vector bc_value(0.,0.,0.);
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
#if 0
         cout << "BC kind = " << bc_kind << endl;
         cout << "BC value = " << bc_value << endl;
#endif
       }
       vector<IntVector>::const_iterator boundary,interior;
       if (sym_bc != 0 && (kind == "Velocity" || kind == "set_if_sym_BC")) {
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end(); boundary++,interior++) {
           variable[*boundary] = variable[*interior];  
           variable[*boundary] = sign*variable[*interior];
         }
       }
       
       if (bc_kind == "Neumann") 
         for (boundary=bound.begin(),interior=inter.begin(); 
              boundary != bound.end(); boundary++,interior++) 
           variable[*boundary] = variable[*interior];
       
       if (kind == "Velocity") {
         if (bc_kind == "Dirichlet")
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = bc_value;
         
         if (bc_kind == "Neumann")
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior] - bc_value*spacing;
       }
      }
    }
  }
}
//______________________________________________________________________
//
void determineSpacingAndSign(Patch::FaceType face, Vector& dx,double& spacing,
                          double& sign)
{
  if (face == Patch::xminus || face == Patch::xplus) {
      spacing = dx.x();
  }
  if (face == Patch::yminus || face == Patch::yplus) {
    spacing = dx.y();
  }
  if (face == Patch::zminus || face == Patch::zplus) {
    spacing = dx.z();
  }
  
  if (face == Patch::xminus || face == Patch::yminus || 
      face == Patch::zminus) sign = -1.;
  
  if (face == Patch::xplus || face == Patch::yplus || 
      face == Patch::zplus) sign = 1.;
  
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
  BC_doing << "Johns setBC (SFCXVariable) "<< kind <<endl;
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    double spacing,sign;
    determineSpacingAndSign(face,dx,spacing,sign);

    if (patch->getBCType(face) == Patch::None) {
      for (int child = 0; 
          child < patch->getBCDataArray(face)->getNumberChildren(); child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,kind,
                                                   bound,inter,
                                                   sfx,sfy,sfz,child);
       
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);
       
       const BoundCond<Vector>* new_bcs  = 
         dynamic_cast<const BoundCond<Vector> *>(bc);
    
       Vector bc_value(0.,0.,0.);
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
       }
       vector<IntVector>::const_iterator boundary,interior,sfcx;
       //__________________________________
       //  Symmetry boundary conditions
       //  -set Neumann = 0 on all walls
       if (sym_bc != 0) {
         // Since this is for the SFCXVariable, only set things on the 
         // y and z faces  -- set the tangential components
         if (face == Patch::yminus || face == Patch::yplus || 
             face == Patch::zminus || face == Patch::zplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior];
         
         // Set normal component = 0
         if(face == Patch::xplus || face == Patch::xminus ) 
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = 0.;
        }
         
       //__________________________________
       // Neumann or Dirichlet
       if (bc_kind == "Dirichlet") {
         // Use the interior index for the xminus face and 
         // no neighboring patches
         // Use the boundary index for the xplus face
         for (sfcx=sfx.begin(); sfcx != sfx.end(); sfcx++)
           variable[*sfcx] = bc_value.x();
       }
       if (bc_kind == "Neumann") {
         if (face == Patch::yminus || face == Patch::yplus || 
             face == Patch::zminus || face == Patch::zplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior] + 
              bc_value.x()*spacing*sign;
       }
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
  BC_doing << "Johns setBC (SFCYVariable) "<< kind <<endl;
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    double spacing,sign;
    determineSpacingAndSign(face,dx,spacing,sign);

    if (patch->getBCType(face) == Patch::None) {
      for (int child = 0; 
          child < patch->getBCDataArray(face)->getNumberChildren(); child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,kind,
                                                   bound,inter,
                                                   sfx,sfy,sfz,child);
       
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);
       
       const BoundCond<Vector>* new_bcs  = 
         dynamic_cast<const BoundCond<Vector> *>(bc);
    
       Vector bc_value(0.,0.,0.);
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
       }
       vector<IntVector>::const_iterator boundary,interior,sfcy;
       //__________________________________
       //  Symmetry boundary conditions
       //  -set Neumann = 0 on all walls
       if (sym_bc != 0) {
         // Since this is for the SFCYVariable, only set things on the 
         // x and z faces  -- set the tangential components
         if (face == Patch::xminus || face == Patch::xplus || 
             face == Patch::zminus || face == Patch::zplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior];
         
         // Set normal component = 0
         if( face == Patch::yplus || face == Patch::yminus ) 
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = 0.;
        }
         
       //__________________________________
       // Neumann or Dirichlet
       if (bc_kind == "Dirichlet") {
         for (sfcy=sfy.begin();sfcy != sfy.end(); sfcy++) {
           variable[*sfcy] = bc_value.y();
         }
       }
       
       if (bc_kind == "Neumann") {
         if (face == Patch::xminus || face == Patch::xplus || 
             face == Patch::zminus || face == Patch::zplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior] + 
              bc_value.y()*spacing*sign;
       }
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
  BC_doing << "Johns setBC (SFCZVariable) "<< kind <<endl;
  Vector dx = patch->dCell();
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
      face=Patch::nextFace(face)){
    double spacing,sign;
    determineSpacingAndSign(face,dx,spacing,sign);
    int numChildren = patch->getBCDataArray(face)->getNumberChildren();
    if (patch->getBCType(face) == Patch::None) {
      for (int child = 0;  child < numChildren; child++) {
       vector<IntVector> bound,inter,sfx,sfy,sfz;
       const BoundCondBase* bc = patch->getArrayBCValues(face,mat_id,kind,
                                                   bound,inter,
                                                   sfx,sfy,sfz,child);
       
       const BoundCondBase* sym_bc = patch->getArrayBCValues(face,mat_id,
                                                       "Symmetric",
                                                       bound,inter,
                                                       sfx,sfy,sfz,
                                                       child);
       
       const BoundCond<Vector>* new_bcs  = 
         dynamic_cast<const BoundCond<Vector> *>(bc);
    
       Vector bc_value(0.,0.,0.);
       string bc_kind="";
       if (new_bcs != 0) {
         bc_value = new_bcs->getValue();
         bc_kind = new_bcs->getKind();
       }
       vector<IntVector>::const_iterator boundary,interior,sfcz;
       //__________________________________
       //  Symmetry boundary conditions
       //  -set Neumann = 0 on all walls
       if (sym_bc != 0) {
         // Since this is for the SFCZVariable, only set things on the 
         // x and y faces  -- set the tangential components
         if (face == Patch::xminus || face == Patch::xplus || 
             face == Patch::yminus || face == Patch::yplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior];
         
         // Set normal component = 0
         if( face == Patch::zplus || face == Patch::zminus ) 
           for (boundary=bound.begin(); boundary != bound.end(); boundary++) 
             variable[*boundary] = 0.;
        }
         
       //__________________________________
       // Neumann or Dirichlet
       if (bc_kind == "Dirichlet") 
         for (sfcz=sfz.begin();sfcz != sfz.end(); sfcz++)
           variable[*sfcz] = bc_value.z();
       
       if (bc_kind == "Neumann") {
         if (face == Patch::xminus || face == Patch::xplus || 
             face == Patch::yminus || face == Patch::yplus)
           for (boundary=bound.begin(),interior=inter.begin(); 
               boundary != bound.end(); boundary++,interior++) 
             variable[*boundary] = variable[*interior] + 
              bc_value.z()*spacing*sign;
       }
      }
    }
  }
}
#endif



//______________________________________________________________________

///______________________________________________________________________
//
#ifdef LODI_BCS
/* --------------------------------------------------------------------- 
 Function~  setBCDensityLODI--                      L   O   D   I
 Purpose~   Takes care of Symmetry BC, Dirichelet BC, Characteristic BC
            for Density
 ---------------------------------------------------------------------  */
void setBCDensityLODI(CCVariable<double>& rho_CC,
                const CCVariable<double>& d1_x,                       
                const CCVariable<double>& d1_y,                       
                const CCVariable<double>& d1_z,                       
                const CCVariable<double>& nux,
                const CCVariable<double>& nuy,
                const CCVariable<double>& nuz,
                CCVariable<double>& rho_tmp,
                CCVariable<double>& p,
                CCVariable<Vector>& vel,
                constCCVariable<double>& c,
                const double delT,
                const double gamma,
                const double R_gas,
                const Patch* patch, 
                const int mat_id)
{ 
  BC_doing << "LODI setBC (Density) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);  
  
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
 
    if(rho_new_bcs != 0 && rho_new_bcs->getKind() == "LODI"){ 
     fillFaceDensityLODI(rho_CC, d1_x, d1_y, d1_z,
                       nux, nuy, nuz, rho_tmp, p, vel, c, 
                       face, delT, gamma, R_gas, 
                       mat_id, dx);
    }  
      
/*`==========TESTING==========*/
#ifdef JET_BC
    double hardCodedDensity = 1.1792946927* (300.0/1000.0);
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
            const CCVariable<double>& d1_x, 
            const CCVariable<double>& d3_x, 
            const CCVariable<double>& d4_x, 
            const CCVariable<double>& d1_y,  
            const CCVariable<double>& d3_y, 
            const CCVariable<double>& d4_y, 
            const CCVariable<double>& d1_z,  
            const CCVariable<double>& d3_z, 
            const CCVariable<double>& d4_z, 
            const CCVariable<double>& nux,
            const CCVariable<double>& nuy,
            const CCVariable<double>& nuz,
            CCVariable<double>& rho_tmp,
            const CCVariable<double>& p,
            CCVariable<Vector>& vel,
            constCCVariable<double>& c,
            const double delT,
            const double gamma,
            const double R_gas,
            const Patch* patch, 
            const int mat_id)
{ 
  BC_doing << "LODI setBC (Vel) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);

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
      
      if(vel_new_bcs->getKind() == "LODI") {
       fillFaceVelLODI(vel_CC, d1_x, d3_x, d4_x, 
                               d1_y, d3_y, d4_y,
                               d1_z, d3_z, d4_z, 
                     nux, nuy, nuz, rho_tmp, p, vel, c, 
                     face, delT, gamma, R_gas, 
                     mat_id, dx);
      }
/*`==========TESTING==========*/
#ifdef JET_BC
        Vector hardCodedVelocity(10.0,0.0,0);
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
              const CCVariable<double>& d1_x, 
              const CCVariable<double>& d2_x, 
              const CCVariable<double>& d3_x, 
              const CCVariable<double>& d4_x, 
              const CCVariable<double>& d5_x,
              const CCVariable<double>& d1_y, 
              const CCVariable<double>& d2_y, 
              const CCVariable<double>& d3_y, 
              const CCVariable<double>& d4_y, 
              const CCVariable<double>& d5_y,
              const CCVariable<double>& d1_z, 
              const CCVariable<double>& d2_z, 
              const CCVariable<double>& d3_z, 
              const CCVariable<double>& d4_z, 
              const CCVariable<double>& d5_z,
              const CCVariable<double>& e,
              const CCVariable<double>& rho_CC,
              const CCVariable<double>& nux,
              const CCVariable<double>& nuy,
              const CCVariable<double>& nuz,
              CCVariable<double>& rho_tmp,
              CCVariable<double>& p,
              CCVariable<Vector>& vel,
              constCCVariable<double>& c,
              const double delT,
              const double cv,
              const double gamma,
              const Patch* patch, 
              const int mat_id)
{ 
  BC_doing << "LODI setBC (Temp) "<< endl;
  Vector dx = patch->dCell();
  IntVector offset(0,0,0);

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

    if (temp_new_bcs != 0 && temp_new_bcs->getKind() == "LODI") {
       fillFaceTempLODI(temp_CC, d1_x, d2_x, d3_x, d4_x, d5_x, 
                                 d1_y, d2_y, d3_y, d4_y, d5_y,
                                 d1_z, d2_z, d3_z, d4_z, d5_z, 
                        e, rho_CC, nux, nuy, nuz, rho_tmp, 
                        p, vel, c, face, delT, cv, gamma,
                        mat_id, dx);
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
/*________________________________________________________________
 Function~ computeDiFirstOrder--                L   O   D   I
 Purpose~  Compute amplitudes of characteristic waves using First-order 
           upwind difference and the di's which are necessary to calculate 
           convection terms at boundary cells in the dircetion 
           across the boundary.  
__________________________________________________________________*/
void computeDiFirstOrder(const double& faceNormal, double& d1, double& d2,
                        double& d3, double& d4, double& d5, const double& rho1,
                         const double& rho2, const double& p1, const double& p2, 
                         const double& c, const Vector& vel1, const Vector& vel2, 
                         const double& vel_cross_bound, const double& dx) 
{   
//  BC_doing << "LODI ComputeDIFirstOrder "<< endl;    
  double d_SMALL_NUM = 1.0e-100;
  //________________________________________________________
  double drho_dx,dp_dx,du_dx,dv_dx,dw_dx,L1,L2,L3,L4,L5;
  drho_dx = (rho1 - rho2)/dx;
  dp_dx   = (  p1 - p2  )/dx;
  du_dx   = (vel1.x() - vel2.x())/dx;
  dv_dx   = (vel1.y() - vel2.y())/dx;
  dw_dx   = (vel1.z() - vel2.z())/dx;

  //__________________________________
  // L1 Wave Amplitude
  double L1_sign;
  L1_sign = faceNormal * (vel_cross_bound - c)/
                    (fabs(vel_cross_bound - c) + d_SMALL_NUM);
    if(L1_sign > 0) {      // outgoing waves
      L1 = (vel_cross_bound - c) * 
           (dp_dx - rho1 * c * du_dx);
    } else {               // incomming waves
      L1 = 0.0;
    }
  //__________________________________
  // L2, 3, 4 Wave Amplitude
  double L234_sign;
  L234_sign = faceNormal * vel_cross_bound
          / (fabs(vel_cross_bound) + d_SMALL_NUM);
    if(L234_sign > 0) {     // outgoing waves
      L2 = vel_cross_bound * (c * c * drho_dx - dp_dx);
      L3 = vel_cross_bound * dv_dx;
      L4 = vel_cross_bound * dw_dx;
     } else {               // incomming waves
      L2 = 0.0;
      L3 = 0.0;
      L4 = 0.0;
    }

  //__________________________________
  // L5 Wave Amplitude
  double L5_sign;
  L5_sign = faceNormal * (vel_cross_bound + c)/
                    (fabs(vel_cross_bound + c) + d_SMALL_NUM);
    if(L5_sign > 0) {      // outgoing wave
      L5 = (vel_cross_bound + c) * 
           (dp_dx + rho1 * c * du_dx);
    } else {               // incomming waves
      L5 = 0.0;
    }

  //__________________________________
  // Compute d1-5
  d1 = (L2 + 0.5 * (L1 + L5))/c/c;
  d2 = 0.5 * (L5 + L1);
  d3 = 0.5 * (L5 - L1)/rho1/c;
  d4 = L3;
  d5 = L4;
}

/*________________________________________________________________
 Function~ computeDiSecondOrder--           L   O   D   I
 Purpose~  Compute amplitudes of characteristic waves using second-order 
           upwind difference and the di's which are necessary to calculate 
           convection terms at boundary cells in the dircetion 
           across the boundary.  
__________________________________________________________________*/

void computeDiSecondOrder(const double& faceNormal, double& d1, double& d2, double& d3, double& d4, 
                          double& d5, const double& rho1, const double& rho2, const double& rho3, 
                         const double& p1, const double& p2, const double& p3, const double& c, 
                          const Vector& vel1, const Vector& vel2, const Vector& vel3, 
                          const double& vel_cross_bound, const double& dx) 

{       
  double d_SMALL_NUM = 1.0e-100;
  //________________________________________________________
  double drho_dx,dp_dx,du_dx,dv_dx,dw_dx,L1,L2,L3,L4,L5;
  drho_dx = faceNormal * (3.0 * rho1 - 4.0 * rho2 + rho3)/dx;
  dp_dx   = faceNormal * (3.0 *   p1 - 4.0 *   p2 + p3)/dx;
  du_dx   = faceNormal * (3.0 * vel1.x() - 4.0 * vel2.x() + vel3.x())/dx;
  dv_dx   = faceNormal * (3.0 * vel1.y() - 4.0 * vel2.y() + vel3.y())/dx;
  dw_dx   = faceNormal * (3.0 * vel1.z() - 4.0 * vel2.z() + vel3.z())/dx;

  //__________________________________
  // L1 Wave Amplitude
  double L1_sign;
  L1_sign = faceNormal * (vel_cross_bound - c)/
                    (fabs(vel_cross_bound - c) + d_SMALL_NUM);
    if(L1_sign > 0) {      // outgoing waves
      L1 = (vel_cross_bound - c) * 
           (dp_dx - rho1 * c * du_dx);
    } else {               // incomming waves
      L1 = 0.0;
    }
  //__________________________________
  // L2, 3, 4 Wave Amplitude
  double L234_sign;
  L234_sign = faceNormal * vel_cross_bound
           / (fabs(vel_cross_bound) + d_SMALL_NUM);
    if(L234_sign > 0) {     // outgoing waves
      L2 = vel_cross_bound * (c * c * drho_dx - dp_dx);
      L3 = vel_cross_bound * dv_dx;
      L4 = vel_cross_bound * dw_dx;
     } else {               // incomming waves
      L2 = 0.0;
      L3 = 0.0;
      L4 = 0.0;
    }

  //__________________________________
  // L5 Wave Amplitude
  double L5_sign;
  L5_sign = faceNormal * (vel_cross_bound + c)/
                    (fabs(vel_cross_bound + c) + d_SMALL_NUM);
    if(L5_sign > 0) {      // outgoing wave
      L5 = (vel_cross_bound + c) * 
           (dp_dx + rho1 * c * du_dx);
    } else {               // incomming waves
      L5 = 0.0;
    }

  //__________________________________
  // Compute d1-5
  d1 = (L2 + 0.5 * (L1 + L5))/c/c;
  d2 = 0.5 * (L5 + L1);
  d3 = 0.5 * (L5 - L1)/rho1/c;
  d4 = L3;
  d5 = L4;
}


/*___________________________________________
 Function~ computeEnergy--              L   O   D   I
 Purpose~  compute the total energy at the boundary face
____________________________________________________________________*/
void computeEnergy(CCVariable<double>& e,
                   CCVariable<Vector>& vel,
                   CCVariable<double>& rho,
                   const Patch* patch)
{
  BC_doing << "LODI computeEnergy "<< endl;
    IntVector low = e.getLowIndex();
    IntVector hi  = e.getHighIndex();
    int hi_x = hi.x() - 1;
    int hi_y = hi.y() - 1;
    int hi_z = hi.z() - 1;
    
    for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
    switch (face) { // switch:1
      case Patch::xplus:
        { //case: 2
          //_____________________________________
          //Compute total energy on xplus plane
           //cout << "energy 1111" << endl;
          for(int j = low.y(); j <= hi_y; j++) {
            for (int k = low.z(); k <= hi_z; k++) {
              IntVector r  =  IntVector(hi_x,j,k);
              double vel_sqr = (vel[r].x() * vel[r].x()
                             +  vel[r].y() * vel[r].y()
                             +  vel[r].z() * vel[r].z());
              double   KE = 0.5 * rho[r] * vel_sqr;
                     e[r] = e[r] + KE;
          
          }//end of k loop
         }//end of j loop
        }//end of case Patch::xplus:
      break;
      //Insert other faces here
      default:
      break;
    }//end of switch
  } //end of for loop over faces
}//end of function


/*__________________________________________________________________
 Function~ computeLODIFirstOrder--              L   O   D   I
 Purpose~  compute Di's at the boundary cells using upwind first-order 
           differenceing scheme
____________________________________________________________________*/

void computeLODIFirstOrder(CCVariable<double>& d1_x, 
                           CCVariable<double>& d2_x, 
                           CCVariable<double>& d3_x, 
                           CCVariable<double>& d4_x, 
                           CCVariable<double>& d5_x,
                           CCVariable<double>& d1_y, 
                           CCVariable<double>& d2_y, 
                           CCVariable<double>& d3_y, 
                           CCVariable<double>& d4_y, 
                           CCVariable<double>& d5_y,
                           CCVariable<double>& d1_z, 
                           CCVariable<double>& d2_z, 
                           CCVariable<double>& d3_z, 
                           CCVariable<double>& d4_z, 
                           CCVariable<double>& d5_z,
                      CCVariable<double>& rho_tmp,  
                      CCVariable<double>& p, 
                      CCVariable<Vector>& vel, 
                      constCCVariable<double>& c, 
                      const Patch* patch,
                      const int mat_id)

{
    BC_doing << "LODI computeLODIFirstOrder "<< endl;
    IntVector low = p.getLowIndex();
    IntVector hi  = p.getHighIndex();
    Vector dx = patch->dCell();
    int hi_x = hi.x() - 1;
    int hi_y = hi.y() - 1;
    int hi_z = hi.z() - 1;
    double d1, d2, d3, d4, d5, vel_cross_bound, delta;
    double faceNormal, rhoR, rhoL, pR, pL, cSpeed;
    Vector velR, velL;
    for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
    switch (face) { // switch:1
      case Patch::xplus:
        {
          //_____________________________________
          //Compute Di at xplus plane
          for(int j = low.y(); j <= hi_y; j++) {
            for (int k = low.z(); k <= hi_z; k++) {
              IntVector r  =  IntVector(hi_x,  j, k);
              IntVector l  =  IntVector(hi_x-1,j, k);
              faceNormal = 1.0;
              rhoR = rho_tmp[r];
              rhoL = rho_tmp[l];
              pR   = p[r];
              pL   = p[l];
              cSpeed = c[r];
              velR = vel[r];
              velL = vel[l];
              vel_cross_bound = velR.x();
              delta = dx.x();
              computeDiFirstOrder(faceNormal, d1, d2, d3, d4, d5, rhoR, rhoL, 
                                  pR, pL, cSpeed, velR, velL, vel_cross_bound, delta);
               d1_x[r] = d1;
              d2_x[r] = d2;
              d3_x[r] = d3;
              d4_x[r] = d4;
              d5_x[r] = d5;
          }//end of k loop
         }//end of j loop
       }//end of case Patch::xplus:
      break;
      case Patch::xminus: 
          //do nothing
      break;
      case Patch::yplus:
           //do nothing
      break;
      case Patch::yminus:
           //do nothing
        break;
      case Patch::zplus:
          //do nothing
      break;
      case Patch::zminus:
          //do nothing
      break;
 
      default:
      break;
    }
  } //end of for loop over faces
}//end of function


/*__________________________________________________________________
 Function~ computeLODISecondOrder--                 L   O   D   I
 Purpose~  compute Di's at the boundary cells using upwind second-order 
           differenceing scheme
____________________________________________________________________*/
void computeLODISecondOrder(CCVariable<double>& d1_x, 
                            CCVariable<double>& d2_x, 
                            CCVariable<double>& d3_x, 
                            CCVariable<double>& d4_x, 
                            CCVariable<double>& d5_x,
                            CCVariable<double>& d1_y, 
                            CCVariable<double>& d2_y, 
                            CCVariable<double>& d3_y, 
                            CCVariable<double>& d4_y, 
                            CCVariable<double>& d5_y,
                            CCVariable<double>& d1_z, 
                            CCVariable<double>& d2_z, 
                            CCVariable<double>& d3_z, 
                            CCVariable<double>& d4_z, 
                            CCVariable<double>& d5_z,
                       constCCVariable<double>& rho_tmp,  
                       const CCVariable<double>& p, 
                       constCCVariable<Vector>& vel, 
                       constCCVariable<double>& c, 
                       const Patch* patch,
                       const int mat_id)

{
    BC_doing << "LODI computeLODISecondOrder "<< endl;
    IntVector low = p.getLowIndex();
    IntVector hi  = p.getHighIndex();
    Vector dx = patch->dCell();
    int hi_x = hi.x() - 1;
    int hi_y = hi.y() - 1;
    int hi_z = hi.z() - 1;
    double d1, d2, d3, d4, d5, vel_cross_bound, delta;
    double faceNormal, rhoR, rhoM, rhoL, pR, pM, pL, cSpeed;
    Vector velR, velM, velL;
    for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace;
        face=Patch::nextFace(face)){
    switch (face) { // switch:1
      case Patch::xplus:
        { //case: 2
          //_____________________________________
          //Compute Di at xplus plane
          for(int j = low.y(); j <= hi_y; j++) {
            for (int k = low.z(); k <= hi_z; k++) {
              IntVector r  =  IntVector(hi_x,  j, k);
              IntVector m  =  IntVector(hi_x-1,j, k);
              IntVector l  =  IntVector(hi_x-2,j, k);
              faceNormal = 1.0;
              rhoR = rho_tmp[r];
              rhoM = rho_tmp[m];
              rhoL = rho_tmp[l];
              pR   = p[r];
              pM   = p[m];
              pL   = p[l];
              cSpeed = c[r];
              velR = vel[r];
              velM = vel[m];
              velL = vel[l];
              vel_cross_bound = velR.x();
              delta = dx.x();
              computeDiSecondOrder(faceNormal, d1, d2, d3, d4, d5, rhoR, rhoM, rhoL, 
                                   pR, pM, pL, cSpeed, velR, velM, velL, vel_cross_bound, delta);
              d1_x[r] = d1;
              d2_x[r] = d2;
              d3_x[r] = d3;
              d4_x[r] = d4;
              d5_x[r] = d5;
          }//end of k loop
         }//end of j loop
        }//end of case Patch::xplus:
      break;
      /*
      case Patch::xminus:
        //do nothing
      break;
      case Patch::yplus:
        // do nothing
      break;
      case Patch::yminus:
        // donothing
      break;
      case Patch::zplus:
        //do nothing
      break;
      case Patch::zminus:
        //do nothing
      break;
      */
      default:
      break;
    }
  } 
}

/*___________________________________________
 Function~ computeNu--                    L   O   D   I
 Purpose~  compute dissipation coefficients 
 __________________________________________*/ 

void computeNu(CCVariable<double>& nux, CCVariable<double>& nuy, CCVariable<double>& nuz,
               CCVariable<double>& p, const Patch* patch)
{
  BC_doing << "LODI computeNu "<< endl;
  IntVector low = patch->getLowIndex();
  IntVector hi  = patch->getHighIndex();
  double d_SMALL_NUM = 1.0e-100;
  //cout << "beging the computeNU" << endl;
  for(Patch::FaceType face = Patch::startFace; face <= Patch::endFace; 
                   face = Patch::nextFace(face)){
     switch (face) {
       case Patch::xplus:
         {//case: 3
          //---------------------------------
          //Coefficients for artifical dissipation term
          //cout << "I am in the switch, case" << endl;
         
        int hi_x = hi.x() - 1;
        int hi_y = hi.y() - 1;
        int hi_z = hi.z() - 1;
        
          for(int j = low.y()+1; j < hi_y; j++) {
           for (int k = low.z()+1; k < hi_z; k++) {
             IntVector r  =  IntVector(hi_x,  j,   k);
             IntVector l  =  IntVector(hi_x-1,j,   k);
             IntVector t  =  IntVector(hi_x,  j+1, k);
             IntVector b  =  IntVector(hi_x,  j-1, k);
             IntVector f  =  IntVector(hi_x,  j,   k+1);
             IntVector bk =  IntVector(hi_x,  j,   k-1);
             nuy[r] = fabs(p[t] - 2.0 * p[r] + p[b])/
                     (fabs(p[t] - p[r]) + fabs(p[r] - p[b])  + d_SMALL_NUM);
             nuz[r] = fabs(p[f] - 2.0 * p[r] + p[bk])/
                     (fabs(p[f] - p[r]) + fabs(p[r] - p[bk]) + d_SMALL_NUM);
           }
         }
         
        //_____________________________________________________________________
        //Compute the dissipation coefficients on the edge and corner of xplus
        for (int k = low.z()+1; k < hi_z; k++) {
          nuy[IntVector(hi_x,hi_y,k)] = nuy[IntVector(hi_x,hi_y-1,k)];
        }
        for (int k = low.z()+1; k < hi_z; k++) {
          nuy[IntVector(hi_x,low.y(),k)] = nuy[IntVector(hi_x,low.y()+1,k)];
        }
        
        for(int j = low.y(); j <= hi_y; j++) {
          nuz[IntVector(hi_x,j,low.z())] = nuz[IntVector(hi_x,j,low.z()+1)];
        }
        for(int j = low.y(); j <= hi_y; j++) {
          nuz[IntVector(hi_x,j,hi_z)] = nuz[IntVector(hi_x,j,hi_z-1)];
        }
      }// end of case Patch::xplus: 3
      break;
      //Insert other faces here
      default:
      break;
    }//end of switch: 2
  }//end of loop over faces
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

      if (new_bcs->getKind() == "LODI"){
        fillFacePress_LODI(press_CC, rho_micro, Temp_CC, f_theta, numALLMatls,
                           sharedState, face);
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
#endif  // LODI_BCS

}  // using namespace Uintah
