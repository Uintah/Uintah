#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_MMS_BC_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_MMS_BC_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct mms_variable_basket{
    double A;
    double viscosity;
    double gamma;
    double cv;
    string whichMMS;
    double delT;
  };    
  //____________________________________________________________
  // This struct contains all of the additional variables needed by setBC.
  struct mms_vars{
    constCCVariable<double> press_CC;
    constCCVariable<double> rho_CC;
    string where;
    double delT;
  };
  //____________________________________________________________
  
  
  bool read_MMS_BC_inputs(const ProblemSpecP&,
                          mms_variable_basket* mms_vb);
                  
  void addRequires_MMS(Task* t, 
                       const string& where,
                       ICELabel* lb,
                       const MaterialSubset* ice_matls);
                       
  void  preprocess_MMS_BCs(DataWarehouse* new_dw,
                           DataWarehouse* old_dw,
                           ICELabel* lb,
                           const int indx,
                           const Patch* patch,
                           const string& where,
                           bool& setMMS_BCs,
                           mms_vars* mss_v);
                           
  void set_MMS_Velocity_BC(const Patch* patch,
                           const Patch::FaceType face,
                           CCVariable<Vector>& vel_CC,
                           const string& var_desc,
                           const vector<IntVector> bound,
                           const string& bc_kind,
                           SimulationStateP& sharedState,
                           mms_variable_basket* mms_var_basket,
                           mms_vars* mms_v);
                           
  void set_MMS_Temperature_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<double>& temp_CC,
                              const string& var_desc,
                              const vector<IntVector> bound,
                              const string& bc_kind,
                              mms_variable_basket* mms_var_basket,
                              mms_vars* mms_v);
                              
  void set_MMS_press_BC(const Patch* patch,
                        const Patch::FaceType face,
                        CCVariable<double>& press_CC,
                        const vector<IntVector> bound,
                        const string& bc_kind,
                        SimulationStateP& sharedState,
                        mms_variable_basket* mms_var_basket,
                        mms_vars* mms_v);  
                        
                        
/*______________________________________________________________________ 
 Function~  set_MMS_BCs_FC--
 Purpose~   Sets the face center velocity boundary conditions
 ______________________________________________________________________*/
 template<class T>
 bool set_MMS_BCs_FC( const Patch* patch,
                      const Patch::FaceType face,
                      T& vel_FC,
                      const vector<IntVector> bound,
                      string& bc_kind,
                      const Vector& dx,
                      const IntVector& P_dir,
                      const string& whichVel,
                      SimulationStateP& sharedState,
                      mms_variable_basket* mms_var_basket,
                      mms_vars* mms_v)
{
  cout_doing<< "Doing set_MMS_BCs_FC: \t\t" << whichVel
            << " face " << face << endl;
  
  bool IveSetBC = false;
 
  vector<IntVector>::const_iterator iter;
  
  //__________________________________
  // on (x,y,z)minus faces move in one cell
  IntVector one_or_zero(0,0,0);
  if ( (whichVel == "X_vel_FC" && face == Patch::xminus) || 
       (whichVel == "Y_vel_FC" && face == Patch::yminus) || 
       (whichVel == "Z_vel_FC" && face == Patch::zminus)){
    one_or_zero = patch->faceDirection(face);
  } 
  //__________________________________
  //  set one or zero flags
  double x_one_zero = 0.0;
  if (whichVel =="X_vel_FC") 
    x_one_zero = 1.0;
  
  double y_one_zero = 0.0;
  if (whichVel =="Y_vel_FC") 
    y_one_zero = 1.0;
    
  double z_one_zero = 0.0;
  if (whichVel =="Z_vel_FC") 
    z_one_zero = 1.0;
  
  //__________________________________
  // 
  if (bc_kind == "MMS_1") {
    double nu = mms_var_basket->viscosity;
    double A =  mms_var_basket->A;
    double t  = sharedState->getElapsedTime();
//    t += mms_v->delT;
    
    for (iter=bound.begin(); iter != bound.end(); iter++) {
      IntVector c = *iter - one_or_zero;
      Point pt = patch->cellPosition(c);
      double x_CC = pt.x(); 
      double y_CC = pt.y();
      
      double x_FC = x_CC - (dx.x()/2) * x_one_zero;
      double y_FC = y_CC - (dx.y()/2) * y_one_zero;
      
/*`==========TESTING==========*/
#if 0
      cout.setf(ios::scientific,ios::floatfield);
      cout.precision(6);
      if (c.y() ==25 && c.z() == 0 && (face == 0 || face == 1)){
        cout << "face " << face << " " << c 
              <<  " x_CC " << x_CC << " x_FC " << x_FC
              <<  " y_CC " << y_CC << " y_FC " << y_FC 
              << " t " << t << " nu " << nu << " A " << A <<endl;
      }
      
      if (c.x() ==25 && c.z() == 0 && (face == 2 || face == 3)){
        cout << "face " << face << " " << c 
              <<  " x_CC " << x_CC << " x_FC " << x_FC
              <<  " y_CC " << y_CC << " y_FC " << y_FC
              << " t " << t << " nu " << nu << " A " << A <<endl;
      }
#endif
/*===========TESTING==========`*/
      Vector vel(0.0,0.0,0.0);
      vel.x( 1.0 - A * cos(x_FC -t) * sin(y_FC -t) * exp(-2.0*nu*t));
      vel.y( 1.0 + A * sin(x_FC -t) * cos(y_FC -t) * exp(-2.0*nu*t));
      
      vel_FC[c] = x_one_zero * vel.x() 
                + y_one_zero * vel.y()
                + z_one_zero * vel.z();
    }
  }
  IveSetBC = true; 
  return IveSetBC; 
}                        
                                                
} // End namespace Uintah
#endif
