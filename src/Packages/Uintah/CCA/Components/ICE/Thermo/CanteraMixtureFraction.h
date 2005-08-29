#ifndef Uintah_CanteraMixtureFraction_h
#define Uintah_CanteraMixtureFraction_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>

namespace Cantera {
  class IdealGasMix;
}

namespace Uintah {

/**************************************

CLASS
   CanteraMixtureFraction
   
   Short description...

GENERAL INFORMATION

   CanteraMixtureFraction.h

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
  class ICELabel;
  class ModelSetup;
  class GeometryPiece;
  class CanteraMixtureFraction : public ThermoInterface {
  public:
    CanteraMixtureFraction(ProblemSpecP& ps, ModelSetup* setup, ICEMaterial* ice_matl);
    virtual ~CanteraMixtureFraction();

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
                                            DataWarehouse* dw, const Patch* patch,
                                            int matl, int numGhostCells,
                                            constCCVariable<double>& int_eng,
                                            constCCVariable<double>& sp_vol);
    virtual void compute_thermalConductivity(CellIterator iter,
                                             CCVariable<double>& thermalDiffusivity,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_cp(CellIterator iter, CCVariable<double>& cp,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_cv(CellIterator iter, CCVariable<double>& cv,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_R(CellIterator iter, CCVariable<double>& R,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_Temp(CellIterator iter, CCVariable<double>& temp,
                            DataWarehouse* dw, const Patch* patch,
                            int matl, int numGhostCells,
                            constCCVariable<double>& int_eng);
    virtual void compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                 DataWarehouse* dw, const Patch* patch,
                                 int matl, int numGhostCells,
                                 constCCVariable<double>& int_eng,
                                 constCCVariable<double>& sp_vol);
  private:
    void initialize(const ProcessorGroup*, 
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse*,
                    DataWarehouse* new_dw);
    double d_thermalConductivity;
    Cantera::IdealGasMix* d_gas;
    std::string species0;
    std::string species1;
    vector<double> mix0;
    vector<double> mix1;

    class Region {
    public:
      Region(GeometryPiece* piece, ProblemSpecP&);

      GeometryPiece* piece;
      double initialMixtureFraction;
    };

    string name;
    VarLabel* mixtureFraction_CCLabel;
    VarLabel* mixtureFraction_reacted_CCLabel;
    vector<Region*> regions;

    MaterialSet* mymatls;
    ICELabel* lb;

    void addTaskDependencies_general(Task* t, State state, int numGhostCells);
  };
} // End namespace Uintah
      
#endif  // Uintah_CanteraMixtureFraction_h
