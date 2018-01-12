/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#ifndef Uintah_Component_Arches_Properties_h
#define Uintah_Component_Arches_Properties_h

/***************************************************************************
CLASS
    Properties
       Sets up the Properties ????

GENERAL INFORMATION
    Properties.h - Declaration of Properties class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)

    Creation Date : 05-30-2000

    C-SAFE

KEYWORDS

DESCRIPTION

PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
    None
***************************************************************************/

#include <sci_defs/uintah_defs.h>

#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/Filter.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace Uintah {
class ArchesLabel;
class MPMArchesLabel;
class MixingModel;
class MixingRxnTable;
class TabPropsInterface;
class MixingRxnModel;
class TimeIntegratorLabel;
class PhysicalConstants;
class Properties {

public:

  // GROUP: Constructors:
  ///////////////////////////////////////////////////////////////////////
  // Constructor taking
  //   [in]

  Properties(ArchesLabel* label, \
             const MPMArchesLabel* MAlb,
             PhysicalConstants* phys_const,
             const ProcessorGroup* myworld);

  // GROUP: Destructors :
  ///////////////////////////////////////////////////////////////////////
  // Destructor

  ~Properties();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database

  void problemSetup(const ProblemSpecP& params);

  void sched_averageRKProps(SchedulerP&,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const TimeIntegratorLabel* timelabels );

  void sched_saveTempDensity(SchedulerP&,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const TimeIntegratorLabel* timelabels);

  void sched_computeDrhodt(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const TimeIntegratorLabel* timelabels);


  // GROUP: Get Methods :
  ///////////////////////////////////////////////////////////////////////
  // Get the number of mixing variables

  inline int getNumMixVars() const{
    return d_numMixingVars;
  }

  // GROUP: Set Methods :
  ///////////////////////////////////////////////////////////////////////
  // Set the boundary consition pointer

  inline void setBC(BoundaryCondition* bc) {
    d_bc = bc;
  }

  inline void setFilter(Filter* filter) {
    d_filter = filter;
  }

  inline void set3dPeriodic(bool periodic) {
    d_3d_periodic = periodic;
  }

  inline double getAdiabaticAirEnthalpy() const{
    return d_H_air;
  }

  inline double getCarbonContent(double f) const{
    return d_carbon_fuel*f+d_carbon_air*(1.0-f);
  }

  void addLookupSpecies( );

  void sched_computeProps( const LevelP&,
                           SchedulerP&,
                           const bool initialize,
                           const bool modify_ref_den,
                           const int time_substep );

  void sched_checkTableBCs( const LevelP&,
                            SchedulerP& sched );

  void doTableMatching();

private:


  ///////////////////////////////////////////////////////////////////////
  // Carry out actual computation of density reference array

  void averageRKProps(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels );

  void saveTempDensity(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels);

  void computeDrhodt(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels);

  // GROUP: Constructors Not Instantiated:
  ///////////////////////////////////////////////////////////////////////
  // Copy Constructor (never instantiated)
  //   [in]
  //        const Properties&

  Properties(const Properties&);

  // GROUP: Operators Not Instantiated:
  ///////////////////////////////////////////////////////////////////////
  // Assignment Operator (never instantiated)
  //   [in]
  //        const Properties&

  Properties& operator=(const Properties&);

  // Variable labels used by simulation controller
  ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  const VarLabel* d_mf_label;

  bool d_reactingFlow;
  PhysicalConstants* d_physicalConsts;
  bool d_radiationCalc;
  bool d_DORadiationCalc;

  bool d_co_output;
  bool d_sulfur_chem;
  bool d_soot_precursors;

  bool d_filter_drhodt;
  bool d_first_order_drhodt;
  int d_numMixingVars;
  double d_opl;
  IntVector d_denRef;

  MixingRxnModel* d_mixingRxnTable;

  BoundaryCondition* d_bc;
  bool d_empirical_soot;
  double d_sootFactor;
  bool d_3d_periodic;
  bool d_inverse_density_average;
  double d_H_air;
  bool d_tabulated_soot;
  double d_f_stoich, d_carbon_fuel, d_carbon_air;
  Filter* d_filter;
  const ProcessorGroup* d_myworld;

  // New Table Interface Stuff:

  // for doing adiabatic gas with non-adiabatic particles
  bool d_adiabGas_nonadiabPart;

  std::string mixModel;

}; // end class Properties
} // End namespace Uintah


#endif
