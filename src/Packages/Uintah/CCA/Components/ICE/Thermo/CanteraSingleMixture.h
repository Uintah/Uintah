#ifndef Uintah_CanteraSingleMixture_h
#define Uintah_CanteraSingleMixture_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

namespace Cantera {
  class IdealGasMix;
};

namespace Uintah {

/**************************************

CLASS
   CanteraSingleMixture
   
   Short description...

GENERAL INFORMATION

   CanteraSingleMixture.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS


DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class CanteraSingleMixture : public ThermoInterface {
  public:
    CanteraSingleMixture(ProblemSpecP& ps);
    virtual ~CanteraSingleMixture();

    virtual void scheduleInitializeThermo(SchedulerP& sched,
                                          const PatchSet* patches,
                                          ICEMaterial* ice_matl);
    virtual void scheduleReactions(SchedulerP& sched,
                                   const PatchSet* patches,
                                   ICEMaterial* ice_matl);

    virtual void addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                        int numGhostCells);
    virtual void addTaskDependencies_thermalConductivity(Task* t, Task::WhichDW dw,
                                                         int numGhostCells);
    virtual void addTaskDependencies_cp(Task* t, Task::WhichDW dw,
                                        int numGhostCells);
    virtual void addTaskDependencies_cv(Task* t, Task::WhichDW dw,
                                        int numGhostCells);
    virtual void addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                           int numGhostCells);
    virtual void addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                       int numGhostCells);
    virtual void addTaskDependencies_Temp(Task* t, Task::WhichDW dw,
                                          int numGhostCells);
    virtual void addTaskDependencies_int_eng(Task* t, Task::WhichDW dw,
                                             int numGhostCells);

    virtual void compute_thermalDiffusivity(CellIterator iter,
                                            CCVariable<double>& thermalDiffusivity,
                                            DataWarehouse* dw,
                                            constCCVariable<double>& int_eng,
                                            constCCVariable<double>& sp_vol);
    virtual void compute_thermalConductivity(CellIterator iter,
                                             CCVariable<double>& thermalDiffusivity,
                                             DataWarehouse* dw);
    virtual void compute_cp(CellIterator iter, CCVariable<double>& cp,
                            DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_cv(CellIterator iter, CCVariable<double>& cv,
                            DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                            DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_R(CellIterator iter, CCVariable<double>& R,
                            DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_Temp(CellIterator iter, CCVariable<double>& temp,
                              DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                 DataWarehouse* dw, constCCVariable<double>& temp);
  private:
    std::string d_speciesMix;
    double d_thermalConductivity;
    Cantera::IdealGasMix* d_gas;
  };
} // End namespace Uintah
      
#endif  // Uintah_CanteraSingleMixture_h
