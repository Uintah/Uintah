#ifndef Packages_Uintah_CCA_Components_Ice_LODI_h
#define Packages_Uintah_CCA_Components_Ice_LODI_h

#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Core/Containers/StaticArray.h>
#include <typeinfo>
using namespace Uintah;
namespace Uintah {

  //_____________________________________________________________
  // This struct contains misc. variables that are carried around
  // press_infinity:  user input
  // sigma:           user input constant
  struct Lodi_variable_basket{
    double press_infinity;  
    double sigma;
    vector<Patch::FaceType> LodiFaces;
  };
  //_____________________________________________________________
  // This struct contains the additional variables required to 
  // apply the Lodi pressure bcs.
  struct Lodi_vars_pressBC{
    Lodi_vars_pressBC(int numMatls):Li(6){};
    CCVariable<double> speedSound;
    CCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    StaticArray<CCVariable<Vector> > Li;
    bool usingLODI;
    bool setLodiBcs;                 
  };
      
  //____________________________________________________________
  // This struct contains the additional variables required to 
  // apply the Lodi Temperature, density and velocity BC.
  struct Lodi_vars{                
    Lodi_vars() : Li(6) {}  

    constCCVariable<double> speedSound;
    constCCVariable<double> gamma;   

    CCVariable<double> rho_CC;
    CCVariable<Vector> vel_CC;
    CCVariable<double> press_tmp;        
    CCVariable<double> temp_CC;            
    StaticArray<CCVariable<Vector> > Li; 
    bool setLodiBcs; 
    Lodi_variable_basket* var_basket;           
  };
  


  void lodi_bc_preprocess( const Patch* patch,
                            Lodi_vars* lv,
                            ICELabel* lb,            
                            const int indx,
                            CCVariable<double>& rho_CC,
                            CCVariable<Vector>& vel_CC,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            SimulationStateP& sharedState);
                            
  void lodi_getVars_pressBC( const Patch* patch,
                             Lodi_vars_pressBC* lodi_vars,
                             ICELabel* lb,
                             SimulationStateP sharedState,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw);

  bool read_LODI_BC_inputs(const ProblemSpecP&,
                           Lodi_variable_basket*);
                                               
  VarLabel* getMaxMach_face_VarLabel( Patch::FaceType face);                                           
                                                             
  void Lodi_maxMach_patchSubset(const LevelP& level,
                                 SimulationStateP& sharedState,
                                 vector<PatchSubset*> &);
                                  
  bool is_LODI_face(const Patch* patch,
                  Patch::FaceType face,
                  SimulationStateP& sharedState);                            
                            

  void computeLi(StaticArray<CCVariable<Vector> >& L,
                 const CCVariable<double>& rho,              
                 const CCVariable<double>& press,                   
                 const CCVariable<Vector>& vel,                  
                 const CCVariable<double>& speedSound,              
                 const Patch* patch,
                 DataWarehouse* new_dw,
                 SimulationStateP& sharedState,
                 const Lodi_variable_basket* user_inputs, 
                 const bool recursiveTasks);

  void getBoundaryEdges(const Patch* patch,
                        const Patch::FaceType face,
                        vector<Patch::FaceType>& face0);
                                 
  int remainingVectorComponent(int dir1, int dir2);
  
  void FaceDensity_LODI(const Patch* patch,
                       const Patch::FaceType face,
                       CCVariable<double>& rho_CC,
                       Lodi_vars* lv,
                       const Vector& dx);
                  
  void FaceVel_LODI(const Patch* patch,
                   Patch::FaceType face,                 
                   CCVariable<Vector>& vel_CC,           
                   Lodi_vars* lv,
                   const Vector& dx,
                   SimulationStateP& sharedState);
                    
  void FaceTemp_LODI(const Patch* patch,
                    const Patch::FaceType face,
                    CCVariable<double>& temp_CC,
                    Lodi_vars* lv, 
                    const Vector& dx,
                    SimulationStateP& sharedState);
               
  void FacePress_LODI(const Patch* patch,
                      CCVariable<double>& press_CC,
                      StaticArray<CCVariable<double> >& rho_micro,
                      SimulationStateP& sharedState, 
                      Patch::FaceType face,
                      Lodi_vars_pressBC* lv);

                          
} // End namespace Uintah
#endif
