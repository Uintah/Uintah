#ifndef Packages_Uintah_CCA_Components_Ice_Slip_h
#define Packages_Uintah_CCA_Components_Ice_Slip_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Core/Containers/StaticArray.h>
#include <typeinfo>
using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  struct Slip_variable_basket{
    double alpha_momentum;        // momentum accommidation coeff
    double alpha_temperature;     // temperature accomidation coeff.
  };    
  //____________________________________________________________
  // This struct contains the additional variables required to compute
  // mean free path and the gradients
  struct Slip_vars{
    constCCVariable<double> gamma;   
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> press_CC;        
    constCCVariable<double> Temp_CC;
    constCCVariable<double> viscosity;
    CCVariable<double> lamda;    // mean free path
    double alpha_momentum;
    double alpha_temperature; 
  };
  
  bool read_MicroSlip_BC_inputs(const ProblemSpecP&,
                                Slip_variable_basket* svb);
                                 
  void addRequires_MicroSlip(Task* t, 
                             const string& where,
                             ICELabel* lb,
                             const MaterialSubset* ice_matls,
                             Slip_variable_basket* sv);
                      
  void preprocess_MicroSlip_BCs(DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                ICELabel* lb,
                                const Patch* patch,
                                const string& where,
                                const int indx,
                                SimulationStateP& sharedState,
                                bool& setSlipBcs,
                                Slip_vars* sv,
                                Slip_variable_basket* svb);
                                  
  bool is_MicroSlip_face(const Patch* patch,
                         Patch::FaceType face,
                         SimulationStateP& sharedState);
                  
  void set_MicroSlipVelocity_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<Vector>& vel_CC,
                              const string& var_desc,
                              const vector<IntVector> bound,
                              const string& bc_kind,
                              const Vector wall_velocity,
                              Slip_vars* sv);

  void set_MicroSlipTemperature_BC(const Patch* patch,
                              const Patch::FaceType face,
                              CCVariable<double>& temp_CC,
                              const string& var_desc,
                              const vector<IntVector> bound,
                              const string& bc_kind,
                              const double wall_temperature,
                              Slip_vars* sv);                          
} // End namespace Uintah
#endif
