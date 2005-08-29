
#include <Packages/Uintah/CCA/Components/ICE/Thermo/CanteraSingleMixture.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>

using namespace Cantera;
using namespace Uintah;

CanteraSingleMixture::CanteraSingleMixture(ProblemSpecP& ps, ModelSetup*, ICEMaterial* ice_matl)
  : ThermoInterface(ice_matl)
{
  ps->require("thermal_conductivity", d_thermalConductivity);
  ps->require("speciesMix",d_speciesMix);
  // Parse the Cantera XML file
  string fname;
  ps->require("file", fname);
  string id;
  ps->require("id", id);
  try {
    d_gas = new IdealGasMix(fname, id);
    int nsp = d_gas->nSpecies();
    int nel = d_gas->nElements();
    cerr.precision(17);
    cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
    d_gas->setState_TPY(300., 101325., "CH4:0.1, O2:0.2, N2:0.7");
    cerr << *d_gas;
  } catch (CanteraError) {
    showErrors(cerr);
    throw InternalError("Cantera initialization failed", __FILE__, __LINE__);
  }
}

CanteraSingleMixture::~CanteraSingleMixture()
{
  delete d_gas;
}

void CanteraSingleMixture::scheduleInitializeThermo(SchedulerP& sched,
                                                    const PatchSet* patches)
{
  // No initialization
}

void CanteraSingleMixture::scheduleReactions(SchedulerP& sched,
                                             const PatchSet* patches)
{
  // No reactions
}

void CanteraSingleMixture::addTaskDependencies_thermalDiffusivity(Task* t, Task::WhichDW dw,
                                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_thermalConductivity(Task* t, Task::WhichDW dw,
                                                             int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_cp(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_cv(Task* t, Task::WhichDW dw,
                                            int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_gamma(Task* t, Task::WhichDW dw,
                                               int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_R(Task* t, Task::WhichDW dw,
                                           int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_Temp(Task* t, Task::WhichDW dw,
                                              int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::addTaskDependencies_int_eng(Task* t, Task::WhichDW dw,
                                                 int numGhostCells)
{
  // No additional requirements
}

void CanteraSingleMixture::compute_thermalDiffusivity(CellIterator iter,
                                                      CCVariable<double>& thermalDiffusivity,
                                                      DataWarehouse*, const Patch* patch,
                                                      int matl, int numGhostCells,
                                                      constCCVariable<double>& int_eng,
                                                      constCCVariable<double>& sp_vol)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    thermalDiffusivity[*iter] = d_thermalConductivity/d_gas->cp_mass() * sp_vol[*iter];
  }
}

void CanteraSingleMixture::compute_thermalConductivity(CellIterator iter,
                                                       CCVariable<double>& thermalConductivity,
                                                       DataWarehouse*, const Patch* patch,
                                                       int matl, int numGhostCells,
                                                       constCCVariable<double>& int_eng)
{
  for(;!iter.done();iter++)
    thermalConductivity[*iter] = d_thermalConductivity;
}

void CanteraSingleMixture::compute_cp(CellIterator iter, CCVariable<double>& cp,
                                      DataWarehouse*, const Patch* patch,
                                      int matl, int numGhostCells,
                                      constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cp[*iter] = d_gas->cp_mass();
  }
}

void CanteraSingleMixture::compute_cv(CellIterator iter, CCVariable<double>& cv,
                                      DataWarehouse*, const Patch* patch,
                                      int matl, int numGhostCells,
                                      constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    cv[*iter] = d_gas->cv_mass();
  }
}

void CanteraSingleMixture::compute_gamma(CellIterator iter, CCVariable<double>& gamma,
                                         DataWarehouse*, const Patch* patch,
                                         int matl, int numGhostCells,
                                         constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    gamma[*iter] = d_gas->cp_mass()/d_gas->cv_mass();
  }
}

void CanteraSingleMixture::compute_R(CellIterator iter, CCVariable<double>& R,
                                     DataWarehouse*, const Patch* patch,
                                     int matl, int numGhostCells,
                                     constCCVariable<double>& int_eng)
{
  cerr << "csm not done: " << __LINE__ << '\n';
#if 0
  double tmp = (d_gamma-1) * d_specificHeat;
  for(;!iter.done();iter++)
    R[*iter] = tmp;
#endif
}

void CanteraSingleMixture::compute_Temp(CellIterator iter, CCVariable<double>& temp,
                                        DataWarehouse*, const Patch* patch,
                                        int matl, int numGhostCells,
                                        constCCVariable<double>& int_eng)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_UV(int_eng[*iter], 1.0);
    temp[*iter] = d_gas->temperature();
  }
}

void CanteraSingleMixture::compute_int_eng(CellIterator iter, CCVariable<double>& int_eng,
                                           DataWarehouse*, const Patch* patch,
                                           int matl, int numGhostCells,
                                           constCCVariable<double>& temp,
                                           constCCVariable<double>& sp_vol)
{
  d_gas->setMassFractionsByName(d_speciesMix);
  for(;!iter.done();iter++){
    d_gas->setState_TR(temp[*iter], 1.0);
    int_eng[*iter] = d_gas->intEnergy_mass();
  }
}
