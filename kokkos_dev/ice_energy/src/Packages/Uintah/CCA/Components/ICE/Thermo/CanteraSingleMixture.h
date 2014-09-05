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

  class ModelSetup;
  class CanteraSingleMixture : public ThermoInterface {
  public:
    CanteraSingleMixture(ProblemSpecP& ps, ModelSetup* setup, ICEMaterial* ice_matl);
    virtual ~CanteraSingleMixture();

    virtual bool doThermalConduction();

    virtual void scheduleInitializeThermo(SchedulerP& sched,
                                          const PatchSet* patches);
    virtual void scheduleReactions(SchedulerP& sched,
                                   const PatchSet* patches);

    virtual void addTaskDependencies_thermalDiffusivity(Task* t, State state,
                                                        int numGhostCells);
    virtual void addTaskDependencies_thermalConductivity(Task* t, State state,
                                                         int numGhostCells);
    virtual void addTaskDependencies_cp(Task* t, State state,
                                        int numGhostCells);
    virtual void addTaskDependencies_cv(Task* t, State state,
                                        int numGhostCells);
    virtual void addTaskDependencies_gamma(Task* t, State state,
                                           int numGhostCells);
    virtual void addTaskDependencies_R(Task* t, State state,
                                       int numGhostCells);
    virtual void addTaskDependencies_Temp(Task* t, State state,
                                          int numGhostCells);
    virtual void addTaskDependencies_int_eng(Task* t, State state,
                                             int numGhostCells);

    virtual void compute_thermalDiffusivity(CellIterator iter,
                                            CCVariable<double>& thermalDiffusivity,
                                            DataWarehouse* old_dw, DataWarehouse* new_dw,
                                            State state, const Patch* patch, int matl,
                                            int numGhostCells,
                                            const constCCVariable<double>& int_eng,
                                            const constCCVariable<double>& sp_vol);
    virtual void compute_thermalConductivity(CellIterator iter,
                                             CCVariable<double>& thermalDiffusivity,
                                             DataWarehouse* old_dw, DataWarehouse* new_dw,
                                             State state, const Patch* patch, int matl,
                                             int numGhostCells,
                                             const constCCVariable<double>& int_eng,
                                             const constCCVariable<double>& sp_vol);
    virtual void compute_cp(CellIterator iter, CCVariable<double>& cp,
                            DataWarehouse* old_dw, DataWarehouse* new_dw,
                            State state, const Patch* patch, int matl,
                            int numGhostCells,
                            const constCCVariable<double>& int_eng,
                            const constCCVariable<double>& sp_vol);
    virtual void compute_cv(CellIterator iter, CCVariable<double>& cv,
                            DataWarehouse* old_dw, DataWarehouse* new_dw,
                            State state, const Patch* patch, int matl,
                            int numGhostCells,
                            const constCCVariable<double>& int_eng,
                            const constCCVariable<double>& sp_vol);
    virtual void compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                               DataWarehouse* old_dw, DataWarehouse* new_dw,
                               State state, const Patch* patch, int matl,
                               int numGhostCells,
                               const constCCVariable<double>& int_eng,
                               const constCCVariable<double>& sp_vol);
    virtual void compute_R(CellIterator iter, CCVariable<double>& R,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           State state, const Patch* patch, int matl,
                           int numGhostCells,
                           const constCCVariable<double>& int_eng,
                           const constCCVariable<double>& sp_vol);
    virtual void compute_Temp(CellIterator iter, CCVariable<double>& temp,
                              DataWarehouse* old_dw, DataWarehouse* new_dw,
                              State state, const Patch* patch, int matl,
                              int numGhostCells,
                              const constCCVariable<double>& int_eng,
                              const constCCVariable<double>& sp_vol);
    virtual void compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch, int matl,
                                 int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol);

    virtual void compute_cp(cellList::iterator begin, cellList::iterator end,
                            CCVariable<double>& cp,
                            DataWarehouse* old_dw, DataWarehouse* new_dw,
                            State state, const Patch* patch, int matl,
                            int numGhostCells,
                            const constCCVariable<double>& int_eng,
                            const constCCVariable<double>& sp_vol);
    virtual void compute_cv(cellList::iterator begin, cellList::iterator end,
                            CCVariable<double>& cv,
                            DataWarehouse* old_dw, DataWarehouse* new_dw,
                            State state, const Patch* patch, int matl,
                            int numGhostCells,
                            const constCCVariable<double>& int_eng,
                            const constCCVariable<double>& sp_vol);
    virtual void compute_gamma(cellList::iterator begin, cellList::iterator end,
                               CCVariable<double>& gamma,
                               DataWarehouse* old_dw, DataWarehouse* new_dw,
                               State state, const Patch* patch, int matl,
                               int numGhostCells,
                               const constCCVariable<double>& int_eng,
                               const constCCVariable<double>& sp_vol);
    virtual void compute_R(cellList::iterator begin, cellList::iterator end,
                           CCVariable<double>& R,
                           DataWarehouse* old_dw, DataWarehouse* new_dw,
                           State state, const Patch* patch, int matl,
                           int numGhostCells,
                           const constCCVariable<double>& int_eng,
                           const constCCVariable<double>& sp_vol);
    virtual void compute_Temp(cellList::iterator begin, cellList::iterator end,
                              CCVariable<double>& temp,
                              DataWarehouse* old_dw, DataWarehouse* new_dw,
                              State state, const Patch* patch, int matl,
                              int numGhostCells,
                              const constCCVariable<double>& int_eng,
                              const constCCVariable<double>& sp_vol);
    virtual void compute_int_eng(cellList::iterator begin, cellList::iterator end,
                                 CCVariable<double>& int_eng,
                                 DataWarehouse* old_dw, DataWarehouse* new_dw,
                                 State state, const Patch* patch, int matl,
                                 int numGhostCells,
                                 const constCCVariable<double>& int_eng,
                                 const constCCVariable<double>& sp_vol);
  private:
    std::string d_speciesMix;
    double d_thermalConductivity;
    Cantera::IdealGasMix* d_gas;
  };
} // End namespace Uintah
      
#endif  // Uintah_CanteraSingleMixture_h
