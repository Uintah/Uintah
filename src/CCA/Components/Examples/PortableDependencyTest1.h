/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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


#ifndef Packages_Uintah_CCA_Components_Examples_PortableDependencyTest1_h
#define Packages_Uintah_CCA_Components_Examples_PortableDependencyTest1_h

#include <CCA/Components/Application/ApplicationCommon.h>

#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>

#include <CCA/Components/Examples/ExamplesLabel.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Core/Parallel/Portability.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/KokkosViews.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace std;
using namespace Uintah;

namespace Uintah {
class SimpleMaterial;


/**************************************

CLASS
   PortableDependencyTest1

   PortableDependencyTest1 simulation

GENERAL INFORMATION

   PortableDependencyTest1.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   PortableDependencyTest1

DESCRIPTION
   Long description...

WARNING

 ****************************************/
#define task_parameters \
		const PatchSubset* patches,	\
		const MaterialSubset* matls,	\
		OnDemandDataWarehouse* old_dw,	\
		OnDemandDataWarehouse* new_dw,	\
		UintahParams& uintahParams,		\
		ExecutionObject<ExecSpace, MemSpace>& execObj

//#define CUDA_CALL( call )               \
//{                                       \
//	if ( cudaSuccess != call ){         \
//		printf("CUDA Error at %s %d: %s %s\n", __FILE__, __LINE__,cudaGetErrorName( call ),  cudaGetErrorString( call ) );  \
//		exit(1);						\
//	}									\
//}

class PortableDependencyTest1 : public ApplicationCommon {
public:
	PortableDependencyTest1( const ProcessorGroup   * myworld, const MaterialManagerP   materialManager)
: ApplicationCommon( myworld, materialManager )	  {
		phi_label = VarLabel::create("phi", NCVariable<double>::getTypeDescription());
	}

	virtual ~PortableDependencyTest1(){
		VarLabel::destroy(phi_label);
	}

	virtual void problemSetup(const ProblemSpecP& params, const ProblemSpecP& restart_prob_spec, GridP& grid);

	virtual void scheduleInitialize(const LevelP& level, SchedulerP& sched){
		Task* task = scinew Task("PortableDependencyTest1::initialize", this, &PortableDependencyTest1::initialize);
		task->computes(phi_label);
		sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
	}

	virtual void scheduleRestartInitialize(const LevelP& level, SchedulerP& sched){}

	virtual void scheduleComputeStableTimeStep(const LevelP& level, SchedulerP& sched){
		Task* task = scinew Task("PortableDependencyTest1::computeStableTimeStep", this, &PortableDependencyTest1::computeStableTimeStep);
		task->computes(getDelTLabel(), level.get_rep());
		sched->addTask(task, level->eachPatch(), m_materialManager->allMaterials());
	}

	//main tests start here. scheduleTimeAdvance will schedule tests as per environment variables set.

	virtual void scheduleTimeAdvance( const LevelP& level, SchedulerP&);
	template <typename ExecSpace, typename MemSpace>
	void scheduleComputeTask( const LevelP& level, SchedulerP& sched);
	template <typename ExecSpace, typename MemSpace>
	void scheduleModifyTask( const LevelP& level, SchedulerP& sched);
	template <typename ExecSpace, typename MemSpace>
	void scheduleRequireTask( const LevelP& level, SchedulerP& sched);
	//the first compute task - creates phi in the new dw.
	template <typename ExecSpace, typename MemSpace>
	void computeTask(task_parameters);

	template <typename ExecSpace, typename MemSpace>	//modifies phi after computeTask, if environment flag is set
	void modifyTask( task_parameters );

	template <typename ExecSpace, typename MemSpace>	//requires phi - verify values either after compute or after modify.
	void requireTask ( task_parameters );

private:
	void initialize(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw){
		for (int p = 0; p < patches->size(); p++) {
			const Patch* patch = patches->get(p);

			NCVariable<double> phi;
			new_dw->allocateAndPut(phi, phi_label, 0, patch);
			phi.initialize(0.);
		}
	}


	void computeStableTimeStep(const ProcessorGroup*,
			const PatchSubset* patches,
			const MaterialSubset* matls,
			DataWarehouse* old_dw,
			DataWarehouse* new_dw){
		new_dw->put(delt_vartype(delt_), getDelTLabel(), getLevel(patches));
	}

	double delt_;
	SimpleMaterial* mymat_;
	const VarLabel* phi_label;
	std::string tasks, exespaces;

	PortableDependencyTest1(const PortableDependencyTest1&);
	PortableDependencyTest1& operator=(const PortableDependencyTest1&);

};
}

#endif
