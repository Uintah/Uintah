#ifndef Uintah_ConstantThermo_h
#define Uintah_ConstantThermo_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

namespace Uintah {

/**************************************

CLASS
   ConstantThermo
   
   Short description...

GENERAL INFORMATION

   ConstantThermo.h

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

  class ConstantThermo : public ThermoInterface {
  public:
    ConstantThermo(ProblemSpecP& ps);
    virtual ~ConstantThermo();

    virtual void scheduleInitializeThermo(SchedulerP& sched,
                                          const PatchSet* patches,
                                          ICEMaterial* ice_matl);

    virtual void addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
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
                                            constCCVariable<double>& sp_vol);
    virtual void compute_cp(CellIterator iter, CCVariable<double>& cp,
                            DataWarehouse* dw);
    virtual void compute_cv(CellIterator iter, CCVariable<double>& cv,
                            DataWarehouse* dw);
    virtual void compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                               DataWarehouse* dw);
    virtual void compute_R(CellIterator iter, CCVariable<double>& R,
                           DataWarehouse* dw);
    virtual void compute_Temp(CellIterator iter, CCVariable<double>& temp,
                              DataWarehouse* dw, constCCVariable<double>& int_eng);
    virtual void compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                 DataWarehouse* dw, constCCVariable<double>& temp);
  private:
    double d_thermalConductivity;
    double d_specificHeat;
    double d_gamma;
  };
} // End namespace Uintah
      
#endif  // Uintah_ConstantThermo_h
