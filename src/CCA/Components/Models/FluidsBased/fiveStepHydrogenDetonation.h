#ifndef Uintah_Component_Models_FluidsBased_fiveStepHydrogenDetonation_h
#define Uintah_Component_Models_FluidsBased_fiveStepHydrogenDetonation_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>

#include <vector>
#include <string>

#include <cmath>

namespace Uintah {


class ICELabel;

namespace SpeciesIndexHydrogen {
  constexpr int H2   = 0;
  constexpr int O2   = 1;
  constexpr int N2   = 2;
  constexpr int H2O  = 3;
  constexpr int H    = 4;
  constexpr int O    = 5;
  constexpr int OH   = 6;
  constexpr int HO2  = 7;
  constexpr int H2O2 = 8;
}

/**
 * 5- step Hydrogen Detonation
 *
 * Single-material ICE combustion model with multiple passive scalars
 * representing pseudospecies mass fractions.
 */
class fiveStepHydrogenDetonation : public FluidsBasedModel {

public:
  fiveStepHydrogenDetonation(const ProcessorGroup* myworld,
                     const MaterialManagerP& materialManager,
                     const ProblemSpecP& params);

  virtual ~fiveStepHydrogenDetonation();

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


  virtual void outputProblemSpec(ProblemSpecP& ps);
  virtual void scheduleRestartInitialize(SchedulerP&, const LevelP&);
  virtual void scheduleTestConservation(SchedulerP&, const PatchSet*);

  virtual void scheduleComputeStableTimeStep(SchedulerP&, const LevelP&) {}
  virtual void scheduleModifyThermoTransportProperties(SchedulerP&, const LevelP&, const MaterialSet*) {}
  virtual void computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int) {}
  virtual void scheduleErrorEstimate(const LevelP&, SchedulerP&) {}

private:
  fiveStepHydrogenDetonation(const fiveStepHydrogenDetonation&) = delete;
  fiveStepHydrogenDetonation& operator=(const fiveStepHydrogenDetonation&) = delete;

  //------------------------------------------------------------------
  // Geometry-based initialization
  //------------------------------------------------------------------
  struct Region {
    GeometryPieceP piece;
    std::vector<double> Yinit; // size = 6 (tracked species)

    Region(GeometryPieceP p, const std::vector<double>& Y)
      : piece(p), Yinit(Y) {}
  };
  //------------------------------------------------------------------
  // Combustion Function declarations
  //------------------------------------------------------------------
  double enthalpy(double T, int idx);
  double gibbs(double T, int idx);
  std::vector<double> globalRates(double T, const std::vector<double>& C);
  double heatRelease(double q5, double T);
  std::vector<double> massSource(const std::vector<double>& q);
  //------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------

  // Molecular Weights                         [H2,    O2,     N2,     H2O,     H,     O,      OH,     HO2] g/mol
  inline static const std::vector<double> Mw = {2.016, 31.998, 28.014, 18.015,  1.008, 15.999, 17.007, 33.006};

  // Chaperon Efficiencies
  inline static const std::vector<double> s3Efficiencies = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> s5Efficiencies = {0.73, 1.0, 0.0, 3.65, 1.0, 1.0, 1.0, 1.0};

  // Universial gas constant
  static constexpr double Ru = 8.314462618; //(J / mol K)

  // Arrhenius Parameters
  inline static const std::vector<double> A = {
        4.48e+13, 2.65e+16, 2.8e+18, 8.4e+13, 2.2e+22
  };

  inline static const std::vector<double> n = {
        0.0, -0.6707, -0.86, 0.0, -2.0
  };

  inline static const std::vector<double> Ea = {
        1.4644e+05, 7.1299544e+04, 0.0, 2.65684e+03, 0.0 // J / mol
  };

  // NASA7 polynomial coefficients for molar enthalpy [H2, O2, N2, H2O, H, O, OH, HO2] 
  // Note tradtionally enthalpy is calculated like a0 + a1T + (a2/2)T^2 + (a3/3)T^3... 
  // division of coefficients (a2/2, a3/3, a4/4) is already computed below
  inline static const std::vector<double> a0 = {
    3.3372792, 3.28253784, 2.92664, 3.03399249, 2.50000001, 2.56942078, 3.09288767, 4.0172109};

  inline static const std::vector<double> a1 = {
    -2.4701236549999999e-05, 7.4154376999999996e-04, 7.4398840000000003e-04, 1.0884590200000001e-03,
    -1.1542148649999999e-11,-4.2987056850000002e-05, 2.7421485799999999e-04, 1.1199100650000000e-03};
  
  inline static const std::vector<double> a2 = {
    1.6648559266666665e-07,-2.5265555633333331e-07,-1.8949200000000001e-07, -5.4690839333333327e-08,
    5.3853982666666673e-15, 1.3982819633333334e-08, 4.2168409333333331e-08, -2.1121938333333332e-07};
  
  inline static const std::vector<double> a3 = {
    -4.4891598500000001e-11, 5.2367638749999998e-11, 2.5242595000000000e-11,-2.4260496750000000e-11,
    -1.1837880875000001e-18,-2.5044449749999998e-12,-2.1986538899999999e-11, 2.8561592500000001e-11};
  
  inline static const std::vector<double> a4 = {
    4.0051075199999998e-15,-4.3343558799999996e-15,-1.3506701999999999e-15, 3.3640198399999998e-15,
    9.9639471400000006e-23, 2.4566738199999997e-16, 2.3482475199999999e-15,-2.1581707000000000e-15};
  
  inline static const std::vector<double> a5 = {
    -950.158922,-1088.45772,-922.7977,-30004.2971, 25473.6599, 29217.5791, 3858.657, 111.856713};
  
  // NASA7 polynomial coefficients for gibbs free energy (dimensionless) [H2, O2, N2, H2O, H, O, OH, HO2]
  inline static const std::vector<double> b0 = {
    3.3372792, 3.28253784, 2.92664, 3.03399249, 2.50000001, 2.56942078, 3.09288767, 4.0172109};

  inline static const std::vector<double> b1 = {
    -2.4701236549999999e-05, 7.4154376999999996e-04, 7.4398840000000003e-04, 1.0884590200000001e-03,
    -1.1542148649999999e-11,-4.2987056850000002e-05, 2.7421485799999999e-04, 1.1199100650000000e-03};

  inline static const std::vector<double> b2 = {
    8.3242796333333327e-08,-1.2632777816666666e-07,-9.4746000000000007e-08,-2.7345419666666664e-08,
    2.6926991333333336e-15, 6.9914098166666672e-09, 2.1084204666666666e-08,-1.0560969166666666e-07};

  inline static const std::vector<double> b3 = {
    -1.4963866166666668e-11, 1.7455879583333332e-11, 8.4141983333333334e-12,-8.0868322499999999e-12,
    -3.9459602916666671e-19,-8.3481499166666657e-13,-7.3288463000000001e-12, 9.5205308333333342e-12};

  inline static const std::vector<double> b4 = {
    1.0012768799999999e-15,-1.0835889699999999e-15,-3.3766754999999997e-16, 8.4100495999999995e-16,
    2.4909867850000001e-23, 6.1416845499999994e-17, 5.8706187999999998e-16,-5.3954267500000000e-16,};

  inline static const std::vector<double> b5 = {
    -950.158922,-1088.45772,-922.7977,-30004.2971, 25473.6599, 29217.5791, 3858.657, 111.856713};

  inline static const std::vector<double> b6 = {
    -3.20502331, 5.45323129, 5.980528, 4.9667701,-0.446682914, 4.78433864, 4.4766961, 3.78510215};

  //------------------------------------------------------------------
  // Data members
  //------------------------------------------------------------------
  ICELabel*    Ilb{nullptr};
  Material*    d_matl{nullptr};
  MaterialSet* d_matl_set{nullptr};
  ProblemSpecP d_params;

  // Species bookkeeping
  static constexpr int N_SPECIES = 6;

  // VarLabels for passive scalars and their sources
  std::vector<VarLabel*> d_Y_labels;      // scalar-YH2, scalar-YO2, ...
  std::vector<VarLabel*> d_Y_src_labels;  // scalar_YH2_src, ...

  // Geometry regions for initialization
  std::vector<Region*> d_regions;

  bool d_debug{false};
};

} // namespace Uintah

#endif
