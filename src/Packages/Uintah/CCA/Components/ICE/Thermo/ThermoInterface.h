#ifndef Uintah_ThermoInterface_h
#define Uintah_ThermoInterface_h

#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/ICE/PropertyBase.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

namespace Uintah {

/**************************************

CLASS
   ThermoInterface
   
   Short description...

GENERAL INFORMATION

   ThermoInterface.h

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

  class ThermoInterface : public PropertyBase {
  public:
    ThermoInterface();
    virtual ~ThermoInterface();

    virtual void scheduleInitializeThermo(SchedulerP& sched,
                                          const PatchSet* patches,
                                          ICEMaterial* ice_matl) = 0;
    virtual void addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_cp(Task* t, Task::WhichDW dw,
                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_cv(Task* t, Task::WhichDW dw,
                                        int numGhostCells) = 0;
    virtual void addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                           int numGhostCells) = 0;
    virtual void addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                       int numGhostCells) = 0;
    virtual void addTaskDependencies_Temp(Task* t, Task::WhichDW dw,
                                          int numGhostCells) = 0;
    virtual void addTaskDependencies_int_eng(Task* t, Task::WhichDW dw,
                                             int numGhostCells) = 0;

    virtual void compute_thermalDiffusivity(CellIterator iter,
                                            CCVariable<double>& thermalDiffusivity,
                                            DataWarehouse* dw,
                                            constCCVariable<double>& sp_vol) = 0;
    virtual void compute_cp(CellIterator iter, CCVariable<double>& cp,
                            DataWarehouse* dw) = 0;
    virtual void compute_cv(CellIterator iter, CCVariable<double>& cv,
                            DataWarehouse* dw) = 0;
    virtual void compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                               DataWarehouse* dw) = 0;
    virtual void compute_R(CellIterator iter, CCVariable<double>& R,
                           DataWarehouse* dw) = 0;
    virtual void compute_Temp(CellIterator iter, CCVariable<double>& temp,
                              DataWarehouse* dw, constCCVariable<double>& int_eng) = 0;
    virtual void compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                 DataWarehouse* dw, constCCVariable<double>& temp) = 0;
  };
} // End namespace Uintah
      
#endif  // Uintah_ThermoInterface_h


