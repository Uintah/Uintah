#ifndef Packages_Uintah_CCA_Components_Ice_CustomBCs_Sine_h
#define Packages_Uintah_CCA_Components_Ice_CustomBCs_Sine_h

#include <CCA/Ports/DataWarehouse.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <typeinfo>

using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. global variables that are needed
  // by most setBC routines.
  struct sine_variable_basket{
    double omega;
    double A;
    double gamma;
    double cv;
    double delT;
  };    
  //____________________________________________________________
  // This struct contains all of the additional variables needed by setBC.
  struct sine_vars{
    constCCVariable<double> press_CC;
    constCCVariable<double> rho_CC;
    string where;
    double delT;
  };
  //____________________________________________________________
  bool read_Sine_BC_inputs(const ProblemSpecP&,
                          sine_variable_basket* sine_vb);
                  
  void addRequires_Sine(Task* t, 
                        const string& where,
                        ICELabel* lb,
                        const MaterialSubset* ice_matls);
                       
  void  preprocess_Sine_BCs(DataWarehouse* new_dw,
                            DataWarehouse* old_dw,
                            ICELabel* lb,
                            const int indx,
                            const Patch* patch,
                            const string& where,
                            bool& setSine_BCs,
                            sine_vars* mss_v);
                           
  void set_Sine_Velocity_BC(const Patch* patch,
                            const Patch::FaceType face,
                            CCVariable<Vector>& vel_CC,
                            const string& var_desc,
                            const vector<IntVector>* bound_ptr,
                            const string& bc_kind,
                            SimulationStateP& sharedState,
                            sine_variable_basket* sine_var_basket,
                            sine_vars* sine_v);
                           
  void set_Sine_Temperature_BC(const Patch* patch,
                               const Patch::FaceType face,
                               CCVariable<double>& temp_CC,
                               const string& var_desc,
                               const vector<IntVector>* bound_ptr,
                               const string& bc_kind,
                               sine_variable_basket* sine_var_basket,
                               sine_vars* sine_v);
                              
  void set_Sine_press_BC(const Patch* patch,
                         const Patch::FaceType face,
                         CCVariable<double>& press_CC,
                         const vector<IntVector>* bound_ptr,
                         const string& bc_kind,
                         SimulationStateP& sharedState,
                         sine_variable_basket* sine_var_basket,
                         sine_vars* sine_v);  
                        
                        
/*______________________________________________________________________ 
 Function~  set_Sine_BCs_FC--
 Purpose~   Sets the face center velocity boundary conditions
            THIS IS JUST A PLACEHOLDER FOR NOW
 ______________________________________________________________________*/
 template<class T>
 bool set_Sine_BCs_FC( const Patch* patch,
                       const Patch::FaceType face,
                       T& vel_FC,
                       const vector<IntVector>* bound_ptr,
                       string& bc_kind,
                       const Vector& dx,
                       const IntVector& /*P_dir*/,
                       const string& whichVel,
                       SimulationStateP& sharedState,
                       sine_variable_basket* sine_var_basket,
                       sine_vars* sine_v)
{
  cout<< "Doing set_sine_BCs_FC: \t\t" << whichVel   << " face " << face << endl;
  
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
//  double A     = sine_var_basket->A;
//  double omega = sine_var_basket->omega;                                
//  double t     = sharedState->getElapsedTime();                         
//  t += sine_v->delT;                                                 
                                                                     
  for (iter=bound_ptr->begin(); iter != bound_ptr->end(); iter++) {  
    IntVector c = *iter - one_or_zero;                  
      
    Vector vel(0.0,0.0,0.0);                                         
    vel.x( 0.0 );  
    vel.y( 0.0 );
    vel.z( 0.0 );  
                                                                     
    vel_FC[c] = x_one_zero * vel.x()                                 
              + y_one_zero * vel.y()                                 
              + z_one_zero * vel.z();                                
  }                                                                  
  IveSetBC = true; 
  return IveSetBC; 
}                        
                                                
} // End namespace Uintah
#endif
