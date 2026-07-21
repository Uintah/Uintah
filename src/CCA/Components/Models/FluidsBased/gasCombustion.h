/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Uintah_Component_Models_FluidsBased_gasCombustion_h
#define Uintah_Component_Models_FluidsBased_gasCombustion_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <CCA/Components/Models/FluidsBased/gasCombustionMechanism.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>

#include <string>
#include <vector>

//---------------------------------------------------------------
// Model simulates gas phase combustion given a reaction mechanism file.
// Mechanism file requires the species and their thermodynamic properties
// along with reactions.  Reactions can be omitted for state dependent
// modelling of gas mixture transport properties.
//
// All mechanism data and pure thermo/kinetics/transport evaluations
// live in ReactionMech (gasCombustionMechanism.h); this class owns the
// Uintah coupling: task scheduling, DataWarehouse traffic, the
// chemistry ODE sub-integrator, and species diffusion fluxes.
//
// Example ups block:
//   <Model type="gasCombustion">
//     <gasCombustion>
//       <material>           reactant       </material>
//       <mechanismFile>      burke2012.mech </mechanismFile>
//       <closureSpecies>     N2             </closureSpecies>
//       <Y_init>             [0.028, 0.226, 0, 0, 0, 0, 0, 0] </Y_init>
//       ...
//     </gasCombustion>
//   </Model>
// Y_init (and geom_object <Y>) are tracked-species ordered: mechanism
// species order with the closure species removed.
//
// Written by James Karr July 2026
//--------------------------------------------------------------

namespace Uintah {

class ICELabel;

/**
 * Single-material ICE combustion model with multiple passive scalars
 * representing species mass fractions; species set, kinetics, and
 * transport fits come from a mechanism file parsed at problemSetup.
 */
class gasCombustion : public FluidsBasedModel {

public:
  gasCombustion(const ProcessorGroup* myworld,
                const MaterialManagerP& materialManager,
                const ProblemSpecP& params);

  virtual ~gasCombustion();

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
  virtual void scheduleModifyThermoTransportProperties(SchedulerP&, const LevelP&, const MaterialSet*);
  virtual void modifyThermoTransportProperties(const ProcessorGroup*, const PatchSubset*,
                                               const MaterialSubset*, DataWarehouse*, DataWarehouse*);

  virtual void scheduleComputeStableTimeStep(SchedulerP&, const LevelP&) {}
  virtual void computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int) {}
  virtual void scheduleErrorEstimate(const LevelP&, SchedulerP&) {}

  // Caloric EOS ownership: ICE's internal energy carrier for this material
  // is the true sensible energy e_s(T,Y) rather than cv*T.  ICE and the
  // exchange model delegate every T <-> energy conversion to these hooks.
  virtual bool ownsCaloricEOS(const int indx) const
  { return d_matl_set && d_matl_set->getSubset(0)->contains(indx); }

  virtual void computeSensibleEnergy(CCVariable<double>   & es,
                                     const Array3<double> & temp,
                                     CellIterator           iter,
                                     const Patch          * patch,
                                     DataWarehouse        * comp_dw,
                                     const YForm            yform,
                                     const int              indx);

  virtual void computeTempFromSensibleEnergy(CCVariable<double>   & temp,
                                             const Array3<double> & es,
                                             CellIterator           iter,
                                             const Patch          * patch,
                                             DataWarehouse        * comp_dw,
                                             const YForm            yform,
                                             const int              indx);

private:
  gasCombustion(const gasCombustion&) = delete;
  gasCombustion& operator=(const gasCombustion&) = delete;

  //------------------------------------------------------------------
  // Mechanism: species set, thermo, kinetics, transport (parsed once
  // in problemSetup; read-only afterwards)
  //------------------------------------------------------------------
  ReactionMech d_mech;
  std::string  d_mechFile;
  std::string  d_closureName{"N2"};

  int d_nTracked{0};   // = d_mech.nTracked(), cached after parse
  int d_nAll{0};       // = d_mech.nAll()
  int d_closure{-1};   // = d_mech.closureIndex()

  // Per-cell temperature bulletproofing, derived from the mechanism's own
  // declared NASA7 range (see problemSetup): [d_Twarn_lo, d_Twarn_hi] is
  // where every species' polynomial is a fit rather than an extrapolation
  // (warn outside it); [d_Thard_lo, d_Thard_hi] is a wider sanity band
  // that catches solver blow-up (throw outside it).
  double d_Twarn_lo{0.0};
  double d_Twarn_hi{0.0};
  double d_Thard_lo{0.0};
  double d_Thard_hi{0.0};

  // Background (whole-domain) initial mass fractions, tracked order
  std::vector<double> d_Yinit_bg;

  // ODE integrator controls (from ups, not the mechanism file).
  // rtol/atol_Y/atol_T act on the SI integration state (T in Kelvin),
  // independent of the ups <units> selection.
  double d_rtol;
  double d_atol_Y;
  double d_atol_T;
  double d_safety;
  double d_max_shrink;
  double d_max_grow;

  //------------------------------------------------------------------
  // Unit system (ups <units> block).  The mechanism and every internal
  // evaluation are SI; the DataWarehouse carries the user's units.
  // Inputs are multiplied by these factors (user -> SI) at the point of
  // read, outputs divided (SI -> user) at the point of write.
  //------------------------------------------------------------------
  std::string d_lenUnit {"m"};
  std::string d_massUnit{"kg"};
  std::string d_timeUnit{"s"};
  std::string d_tempUnit{"K"};

  // Base factors, user -> SI
  double d_lenConv {1.0};
  double d_massConv{1.0};
  double d_timeConv{1.0};
  double d_tempConv{1.0};

  // Derived factors, user -> SI (computed once in problemSetup)
  double d_rhoConv    {1.0};   // density            [kg/m^3]
  double d_velConv    {1.0};   // velocity           [m/s]
  double d_specEngConv{1.0};   // specific energy    [J/kg]
  double d_engConv    {1.0};   // energy             [J]
  double d_cvConv     {1.0};   // specific heat      [J/kg-K]
  double d_pressConv  {1.0};   // pressure           [Pa]
  double d_viscConv   {1.0};   // dynamic viscosity  [Pa-s]
  double d_condConv   {1.0};   // thermal cond.      [W/m-K]
  double d_diffConv   {1.0};   // diffusion coeff.   [m^2/s]
  double d_hrrConv    {1.0};   // heat release rate  [W/m^3]

  //------------------------------------------------------------------
  // Geometry-based initialization
  //------------------------------------------------------------------
  struct Region {
    GeometryPieceP piece;
    std::vector<double> Yinit;   // size nTracked

    Region(GeometryPieceP p, const std::vector<double>& Y)
      : piece(p), Yinit(Y) {}
  };

  //------------------------------------------------------------------
  // 1D profile initialization from a .dat file (e.g. from Cantera/SD Toolbox).
  // File columns: x  T  u  rho  press  (in the ups <units> system; SI default)
  //               then nTracked mass fraction columns in tracked order
  //               (mechanism species order, closure species omitted).
  // Lines beginning with '#' are ignored.  Values are linearly interpolated
  // onto cell centres; cells outside the profile range are clamped.
  //------------------------------------------------------------------
  struct ProfileInit {
    bool        isActive {false};
    int         axis     {0};
    std::string filename;
    std::vector<double>              x;
    std::vector<double>              T;
    std::vector<double>              u;
    std::vector<double>              rho;
    std::vector<double>              press;
    std::vector<std::vector<double>> Y;   // rows sized nTracked
  };

  //------------------------------------------------------------------
  // Chemistry ODE sub-integration (constant volume).  ChemStepResult
  // doubles as reusable workspace: create two outside the cell loop,
  // pass them to chemStep every call -- no per-cell allocation.
  //------------------------------------------------------------------
  struct ChemStepResult {
    double rhsEnergy{0.0};           // dT/dt [K/s]
    double engSrc{0.0};              // qdot * cellVol [W]
    std::vector<double> rhsMass;     // dY/dt, tracked order [1/s]

    // scratch reused across calls
    std::vector<double> cp, conc, q, S;
    ReactionMech::Workspace w;
  };

  void chemStep(double T, const std::vector<double>& Y,
                double rho_kg, double cellVol, ChemStepResult& res);

  // Fetch the tracked mass fractions from the DW form requested by the
  // caloric EOS hooks (see FluidsBasedModel::YForm)
  void gatherMassFractions(std::vector<constCCVariable<double> >& Y,
                           constCCVariable<double>& massL,
                           const Patch* patch,
                           DataWarehouse* comp_dw,
                           const YForm yform,
                           const int indx) const;

  //------------------------------------------------------------------
  // Data members
  //------------------------------------------------------------------
  Ghost::GhostType d_gn = Ghost::None;

  ICELabel*    Ilb{nullptr};
  Material*    d_matl{nullptr};
  MaterialSet* d_matl_set{nullptr};
  ProblemSpecP d_params;

  // VarLabels for passive scalars and their sources, tracked order
  std::vector<VarLabel*> d_Y_labels;      // scalar-YH2, scalar-YO2, ...
  std::vector<VarLabel*> d_Y_src_labels;  // scalar_YH2_src, ...

  VarLabel* d_dtChem_label{nullptr};        // minimum chemistry substep taken per cell
  VarLabel* d_dtChemLimiter_label{nullptr}; // limiter at that substep: all-species index, nAll = T, -1 = no substepping
  VarLabel* d_HRR_label{nullptr};           // heat release rate [W/m³ in user units]
  std::vector<VarLabel*> d_diffCoef_labels; // D_k [m^2/s in user units], all-species indexed

  // Geometry regions for initialization
  std::vector<Region*> d_regions;

  ProfileInit d_profileInit;

  bool d_doChemistry{true};
  bool d_doDiffusion{true};
  bool d_debug{false};
  IntVector d_debugCell=IntVector(-9);   // cell for debugging output
};

} // namespace Uintah

#endif
