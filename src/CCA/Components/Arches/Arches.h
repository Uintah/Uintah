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

#ifndef Uintah_Component_Arches_Arches_h
#define Uintah_Component_Arches_Arches_h

/**
 * \file
 *
 * \mainpage
 *
 * \author Various Generations of CRSim
 * \date   September 1997 (?)
 *
 * Arches is a variable density, low-mach (pressure projection) CFD formulation, with radiation, chemistry and
 * turbulence subgrid closure.
 *
 */

#include <CCA/Components/Application/ApplicationCommon.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModuleFactory.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Handle.h>

#include <sci_defs/petsc_defs.h>
#include <sci_defs/uintah_defs.h>

#include <string>

// Divergence constraint instead of drhodt in pressure equation
//#define divergenceconstraint

// Exact Initialization for first time step in
// MPMArches problem, to eliminate problem of
// sudden appearance of mass in second step
//
// #define ExactMPMArchesInitialize

namespace Uintah {

  class VarLabel;
  class PhysicalConstants;
  class NonlinearSolver;
  class MPMArchesLabel;
  class ArchesParticlesHelper;

class Arches : public ApplicationCommon {

public:

  enum DIRNAME { NODIR, XDIR, YDIR, ZDIR };
  enum STENCILNAME { AP, AE, AW, AN, AS, AT, AB };
  enum NUMGHOSTS {ZEROGHOSTCELLS , ONEGHOSTCELL, TWOGHOSTCELLS,
                  THREEGHOSTCELLS, FOURGHOSTCELLS, FIVEGHOSTCELLS };
  
  Arches(const ProcessorGroup* myworld,
	 const MaterialManagerP materialManager);

  virtual ~Arches();

  virtual void problemSetup(const ProblemSpecP& params,
                            const ProblemSpecP& materials_ps,
                            GridP& grid);

  virtual void scheduleInitialize(const LevelP& level,
                                  SchedulerP&);

  virtual void scheduleRestartInitialize(const LevelP& level,
                                         SchedulerP& sched);

  virtual void restartInitialize();

  virtual void scheduleComputeStableTimeStep(const LevelP& level,
                                             SchedulerP&);

  virtual void scheduleTimeAdvance( const LevelP& level,
                                    SchedulerP&);

  virtual void scheduleAnalysis( const LevelP& level,
				 SchedulerP&);

  virtual int computeTaskGraphIndex( const int timeStep );

  void setMPMArchesLabel(const MPMArchesLabel* MAlb){
    m_MAlab = MAlb;
  }

  virtual double recomputeDelT(const double delT);

  void setWithMPMARCHES() {
    m_with_mpmarches = true;
  };

  //________________________________________________________________________________________________
  //  Multi-level/AMR
  // stub functions.  Needed for multi-level RMCRT and
  // if you want to change the coarse level patch configuration
  virtual void scheduleCoarsen(const LevelP& ,
                                SchedulerP& ){ /* do Nothing */};

  virtual void scheduleRefineInterface(const LevelP&,
                                       SchedulerP&,
                                       bool, bool){ /* do Nothing */};

  virtual void scheduleInitialErrorEstimate( const LevelP& ,
                                             SchedulerP&  ){/* do Nothing */};
  virtual void scheduleErrorEstimate( const LevelP& ,
                                      SchedulerP&  ){/* do Nothing */};
  //________ end stub functions ____________________________________________________________________

private:

  Arches();

  Arches(const Arches&);

  Arches& operator=(const Arches&);

  /** @brief Assign a unique name to each BC where name is not specified in the UPS.
             Note that this functionality was taken from Wasatch. (credit: Tony Saad) **/
  void assign_unique_boundary_names( Uintah::ProblemSpecP bcProbSpec );

  /** @brief convert a number to a string **/
  template <typename T>
  std::string number_to_string ( T n )
  {
    std::stringstream ss;
    ss << n;
    return ss.str();
  }

  PhysicalConstants* m_physicalConsts;
  NonlinearSolver* m_nlSolver;
  const MPMArchesLabel* m_MAlab;
  Uintah::ProblemSpecP m_arches_spec;
  ArchesParticlesHelper* m_particlesHelper;

  std::vector<AnalysisModule*> m_analysis_modules;

  bool m_with_mpmarches;

  bool m_do_lagrangian_particles;

  int m_arches_level_index;

}; // end class Arches

} // End namespace Uintah

#endif
