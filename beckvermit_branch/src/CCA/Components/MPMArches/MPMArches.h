/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef Uintah_Component_MPMArches_h
#define Uintah_Component_MPMArches_h

/**************************************

CLASS
   MPMArches
   
   Short description...

GENERAL INFORMATION

   MPMArches.h

   Author: Rajesh Rawat (rawat@crsim.utah.edu)
   Department of Chemical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   MPMArches

DESCRIPTION
   Long description...
  
WARNING

****************************************/

#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SimulationInterface.h>

namespace Uintah {
 struct cutcell { double d_cutcell[13]; }; //centroids/surface normals/areafractions
} 

namespace SCIRun {

  void swapbytes( Uintah::cutcell& );

}

#include <CCA/Components/MPM/Contact/Contact.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/OnTheFlyAnalysis/AnalysisModule.h>
#include <Core/Geometry/Vector.h>

#undef RIGID_MPM
// #define RIGID_MPM

namespace Uintah {

using namespace SCIRun;


 const TypeDescription* fun_getTypeDescription(cutcell*);

class MPMArches : public UintahParallelComponent, public SimulationInterface {
public:
  MPMArches(const ProcessorGroup* myworld, const bool doAMR);
  virtual ~MPMArches();

  // Read inputs from ups file for MPMArches case
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& materials_ps,
                            GridP& grid, SimulationStateP&);

  virtual void restartInitialize();

  virtual void outputProblemSpec(ProblemSpecP& ps);

  // Set up initial conditions for MPMArches problem	 
  virtual void scheduleInitialize(const LevelP& level,
				  SchedulerP&);

  virtual void scheduleInitializeKStability(SchedulerP& sched,
					    const PatchSet* patches,
					    const MaterialSet* arches_matls);

  virtual void scheduleInitializeCutCells(SchedulerP& sched,
					  const PatchSet* patches,
					  const MaterialSet* arches_matls);

  // Interpolate particle information from particles to grid
  // for the initial condition
  virtual void scheduleInterpolateParticlesToGrid(SchedulerP& sched,
						  const PatchSet* patches,
						  const MaterialSet* matls);

  //////////
  virtual void scheduleComputeStableTimestep(const LevelP& level,
					     SchedulerP&);
	 
  //////////
  virtual void scheduleTimeAdvance( const LevelP& level, 
				    SchedulerP&);

  // Copy cut cell information from time step to next time step
  void scheduleCopyCutCells(SchedulerP& sched,
			    const PatchSet* patches,
			    const MaterialSet* arches_matls);

  // Interpolate Particle variables from Nodes to cell centers
  void scheduleInterpolateNCToCC(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls);

  // Interpolate relevant particle variables from cell center to
  // appropriate face centers
  void scheduleInterpolateCCToFC(SchedulerP&,
				 const PatchSet* patches,
				 const MaterialSet* matls);

  // Calculate gas void fraction by subtraction of solid volume
  // fraction from one
  void scheduleComputeVoidFracMPM(SchedulerP& sched,
			       const PatchSet* patches,
			       const MaterialSet* arches_matls,
			       const MaterialSet* mpm_matls,
			       const MaterialSet* all_matls);

  void scheduleComputeVoidFrac(SchedulerP& sched,
			       const PatchSet* patches,
			       const MaterialSet* arches_matls,
			       const MaterialSet* mpm_matls,
			       const MaterialSet* all_matls);

  // Calculate integrated solid temperature from individual 
  // materials in a given cell
  void scheduleComputeIntegratedSolidProps(SchedulerP& sched,
					   const PatchSet* patches,
					   const MaterialSet* arches_matls,
					   const MaterialSet* mpm_matls,
					   const MaterialSet* all_matls);

  // Calculate Total Heat Flux to Solid (combination of x-, y-,
  // z- and cc-fluxes) for any wall cell

  void scheduleComputeTotalHT(SchedulerP& sched,
			      const PatchSet* patches,
			      const MaterialSet* arches_matls);

  // Calculate momentum exchange terms for gas-solid interface
  void scheduleMomExchange(SchedulerP& sched,
			   const PatchSet* patches,
			   const MaterialSet* arches_matls,
			   const MaterialSet* mpm_matls,
			   const MaterialSet* all_matls);

  // Calculate heat exchange terms for gas-solid interface
  void scheduleEnergyExchange(SchedulerP& sched,
			      const PatchSet* patches,
			      const MaterialSet* arches_matls,
			      const MaterialSet* mpm_matls,
			      const MaterialSet* all_matls);

  // Interpolate all momentum and energy exchange sources from
  // face centers and accumulate with cell-center sources
  void schedulePutAllForcesOnCC(SchedulerP& sched,
			        const PatchSet* patches,
			        const MaterialSet* mpm_matls);

  // Interpolate all momentum and energy sources from cell-centers
  // to node centers for MPM use
  void schedulePutAllForcesOnNC(SchedulerP& sched,
			        const PatchSet* patches,
			        const MaterialSet* mpm_matls);

  void scheduleComputeAndIntegrateAcceleration(SchedulerP&, const PatchSet*,
				               const MaterialSet*);

  void computeAndIntegrateAcceleration(const ProcessorGroup*,
			               const PatchSubset* patches,
			               const MaterialSubset* matls,
			               DataWarehouse* old_dw,
			               DataWarehouse* new_dw);


  void scheduleSolveHeatEquations(SchedulerP&, const PatchSet*,
				  const MaterialSet*);

  void solveHeatEquations(const ProcessorGroup*,
			  const PatchSubset* patches,
			  const MaterialSubset* matls,
			  DataWarehouse* old_dw,
			  DataWarehouse* new_dw);

  

  ///////////////////////////////////////////////////////////////////////
    // Function to return boolean for recompiling taskgraph

    virtual bool needRecompile(double time, double dt,
			       const GridP& grid);
      virtual double recomputeTimestep(double current_dt);
      
      virtual bool restartableTimesteps();

 protected:

    void initializeKStability(const ProcessorGroup*,
			      const PatchSubset* patches,
			      const MaterialSubset* arches_matls,
			      DataWarehouse* old_dw,
			      DataWarehouse* new_dw);

    void initializeCutCells(const ProcessorGroup*,
			    const PatchSubset* patches,
			    const MaterialSubset* arches_matls,
			    DataWarehouse* old_dw,
			    DataWarehouse* new_dw);

    void interpolateParticlesToGrid(const ProcessorGroup*,
				  const PatchSubset* patches,
				  const MaterialSubset* ,
				  DataWarehouse* old_dw,
				  DataWarehouse* new_dw);

  void copyCutCells(const ProcessorGroup*,
		    const PatchSubset* patches,
		    const MaterialSubset*,
		    DataWarehouse* old_dw,
		    DataWarehouse* new_dw);

  void interpolateNCToCC(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw,
			 DataWarehouse* new_dw);

  void interpolateCCToFC(const ProcessorGroup*,
			 const PatchSubset* patches,
			 const MaterialSubset* matls,
			 DataWarehouse* old_dw,
			 DataWarehouse* new_dw);

  void computeVoidFracMPM(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset*,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);


  void computeVoidFrac(const ProcessorGroup*,
		       const PatchSubset* patches,
		       const MaterialSubset*,
		       DataWarehouse* old_dw,
		       DataWarehouse* new_dw);

  void computeIntegratedSolidProps(const ProcessorGroup*,
				   const PatchSubset* patches,
				   const MaterialSubset*,
				   DataWarehouse* /*old_dw*/,
				   DataWarehouse* new_dw);

  void computeTotalHT(const ProcessorGroup*,
		      const PatchSubset* patches,
		      const MaterialSubset*,
		      DataWarehouse* /*old_dw*/,
		      DataWarehouse* new_dw);

  void doMomExchange(const ProcessorGroup*,
		     const PatchSubset* patches,
		     const MaterialSubset*,
		     DataWarehouse* old_dw,
		     DataWarehouse* new_dw);

  void collectToCCGasMomExchSrcs(const ProcessorGroup*,
				 const PatchSubset* patches,
				 const MaterialSubset*,
				 DataWarehouse* old_dw,
				 DataWarehouse* new_dw);

  void interpolateCCToFCGasMomExchSrcs(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* ,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

#if 0
  void redistributeDragForceFromCCtoFC(const ProcessorGroup*,
				       const PatchSubset* patches,
				       const MaterialSubset* ,
				       DataWarehouse* old_dw,
				       DataWarehouse* new_dw);

#endif

  void doEnergyExchange(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset*,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw);

  void collectToCCGasEnergyExchSrcs(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset*,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw);

  void putAllForcesOnCC(const ProcessorGroup*,
		        const PatchSubset* patches,
		        const MaterialSubset*,
		        DataWarehouse* old_dw,
		        DataWarehouse* new_dw);

  void putAllForcesOnNC(const ProcessorGroup*,
		        const PatchSubset* patches,
		        const MaterialSubset*,
		        DataWarehouse* old_dw,
		        DataWarehouse* new_dw);

  double d_SMALL_NUM;
  // GROUP: Constructors (Private):
  ////////////////////////////////////////////////////////////////////////
  // Default MPMArches constructor
  MPMArches();
  ////////////////////////////////////////////////////////////////////////
  // MPMArches copy constructor

  MPMArches(const MPMArches&);
  MPMArches& operator=(const MPMArches&);
	 
  SimulationStateP d_sharedState;
  MPMLabel* Mlb;
  const ArchesLabel* d_Alab;
  const MPMArchesLabel* d_MAlb;
  const VarLabel* d_enthalpy_label; 

  std::string d_enthalpy_name; 

#ifdef RIGID_MPM
  RigidMPM*        d_mpm;
#else
  SerialMPM*       d_mpm;
#endif

  Arches*          d_arches;
  
  std::vector<AnalysisModule*> d_analysisModules;

  double d_tcond;
  bool d_calcEnergyExchange;
  bool d_DORad;
  bool d_radiation;
  int nofTimeSteps;
  bool d_recompile;
  double prturb;
  double cpfluid;
  bool d_useCutCell;
  bool d_stationarySolid;
  bool d_inviscid;
  bool d_restart;
  bool fixCellType;
  bool d_fixTemp;
  bool d_ifTestingCutCells;
  bool calcVolFracMPM;
  bool calcVel;
  bool d_stairstep;
  bool d_doingRestart; 

  enum CENTROID {CENX=1, CENY, CENZ};    
  enum SURFNORM {NORMX=4, NORMY, NORMZ};
  enum AREAFRN {AREAE=7, AREAW, AREAN, AREAS, AREAT, AREAB, TOTAREA};

};
      
} // End namespace Uintah
      

#endif
