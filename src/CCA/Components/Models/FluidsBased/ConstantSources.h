#ifndef Uintah_Component_Models_FluidsBased_ConstantSources_h
#define Uintah_Component_Models_FluidsBased_ConstantSources_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>
#include <vector>

namespace Uintah {

class ICELabel;

class ConstantSources : public FluidsBasedModel {
public:
  ConstantSources(const ProcessorGroup* myworld,
                  const MaterialManagerP& materialManager,
                  const ProblemSpecP& params);

  virtual ~ConstantSources();

  // Required model interface
  virtual void problemSetup(GridP& grid, const bool isRestart);

  virtual void scheduleInitialize(SchedulerP& sched,
                                  const LevelP& level);

  virtual void initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  virtual void scheduleComputeModelSources(SchedulerP& sched,
                                           const LevelP& level);

  virtual void computeModelSources(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);

  virtual void scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                       const LevelP& level,
                                                       const MaterialSet* matls);

  virtual void modifyThermoTransportProperties(const ProcessorGroup*,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matls,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw);

  // Required pure virtuals from FluidsBasedModel
  virtual void outputProblemSpec(ProblemSpecP& ps);
  virtual void scheduleRestartInitialize(SchedulerP&, const LevelP&);
  virtual void scheduleTestConservation(SchedulerP&, const PatchSet*);

  // Unused hooks
  virtual void scheduleComputeStableTimeStep(SchedulerP&, const LevelP&) {}
  virtual void computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int) {}
  virtual void scheduleErrorEstimate(const LevelP&, SchedulerP&) {}

private:
  ConstantSources(const ConstantSources&) = delete;
  ConstantSources& operator=(const ConstantSources&) = delete;

  //------------------------------------------------------------------
  // 1D profile loaded from a .dat file at problem setup.
  // UPS block: <qdotProfile>
  // File columns (comment lines start with '#'):
  //   x[m]  qdot[W/m3]                          (2-column format)
  //   x[m]  qdot[W/m3]  cv[J/kg-K]  gamma[-]   (4-column format)
  // Values are linearly interpolated onto cell centres; cells outside
  // the profile range are clamped to the nearest endpoint.
  // When hasCvGamma is true, cv and gamma are fixed to profile values
  // every timestep via modifyThermoTransportProperties.
  //------------------------------------------------------------------
  struct ProfileInit {
    bool        isActive  {false};
    bool        hasCvGamma{false};
    bool        hasMuK    {false};
    int         axis      {0};
    std::string filename;
    std::vector<double> x;
    std::vector<double> qdot;
    std::vector<double> cv;
    std::vector<double> gamma;
    std::vector<double> mu;
    std::vector<double> k;
  };

  //------------------------------------------------------------------
  // 1D flow-variable initialization from a .dat file (one-time, at t=0).
  // UPS block: <initProfile>
  // File columns (comment lines start with '#'):
  //   x[m]  T[K]  u[m/s]  rho[kg/m3]  press[Pa]
  // Values are linearly interpolated onto cell centres; cells outside
  // the profile range are clamped to the nearest endpoint.
  //------------------------------------------------------------------
  struct FlowProfileInit {
    bool        isActive{false};
    int         axis    {0};
    std::string filename;
    std::vector<double> x;
    std::vector<double> T;
    std::vector<double> u;
    std::vector<double> rho;
    std::vector<double> press;
  };

  ICELabel*    Ilb{nullptr};
  Material*    d_matl{nullptr};
  MaterialSet* d_matl_set{nullptr};
  ProblemSpecP d_params;

  VarLabel* d_phi_label{nullptr};
  VarLabel* d_phi_src_label{nullptr};
  VarLabel* d_HRR_label{nullptr};

  // USER PARAMETERS
  double d_qdot_Wm3{0.0};
  double d_scalar_remove_per_s{0.0};
  double d_phi_init{0.0};

  GeometryPieceP   d_sourceRegion{nullptr};  // null = apply everywhere
  ProfileInit      d_profileInit;
  FlowProfileInit  d_flowProfileInit;
};

} // namespace Uintah

#endif
