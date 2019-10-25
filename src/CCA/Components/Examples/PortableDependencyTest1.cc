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

#include <CCA/Components/Examples/PortableDependencyTest1.h>

double g_expected = 0.0;


void PortableDependencyTest1::problemSetup( const ProblemSpecP & params
		, const ProblemSpecP & restart_prob_spec
		,       GridP        & /*grid*/
)
{
	ProblemSpecP portabledependencytest1 = params->findBlock("portabledependencytest1");
	portabledependencytest1->require("delt", delt_);
	portabledependencytest1->require("task", tasks);
	portabledependencytest1->require("exespace", exespaces);
	mymat_ = scinew SimpleMaterial();
	m_materialManager->registerSimpleMaterial(mymat_);
}

int cid=0, mid=0, rid=0;

template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::scheduleComputeTask(const LevelP& level, SchedulerP& sched){
	std::string name = "PortableDependencyTest1::computeTask" + to_string(cid++);
	auto TaskDependencies = [&](Task* task) {
		task->computes(phi_label, nullptr, Uintah::Task::NormalDomain);
	};
	create_portable_tasks(TaskDependencies, this, name.data(),
			&PortableDependencyTest1::computeTask<ExecSpace, MemSpace>,
			sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::scheduleModifyTask( const LevelP& level, SchedulerP& sched){
	std::string name = "PortableDependencyTest1::modifyTask" + to_string(mid++);
	auto TaskDependencies = [&](Task* task) {
		task->modifies(phi_label);
	};
	create_portable_tasks(TaskDependencies, this, name.data(),
			&PortableDependencyTest1::modifyTask<ExecSpace, MemSpace>,
			sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::scheduleRequireTask( const LevelP& level, SchedulerP& sched){
	std::string name = "PortableDependencyTest1::requireTask" + to_string(rid++);
	auto TaskDependencies = [&](Task* task) {
		task->requires(Task::NewDW, phi_label, Ghost::None, 0);
	};
	create_portable_tasks(TaskDependencies, this, name.data(),
			&PortableDependencyTest1::requireTask<ExecSpace, MemSpace>,
			sched, level->eachPatch(), m_materialManager->allMaterials(), TASKGRAPH::DEFAULT);
}

void PortableDependencyTest1::scheduleTimeAdvance( const LevelP  & level, SchedulerP & sched){
	for(size_t i=0; i<tasks.length(); i++){
		char task = tasks[i];
		char exespace = exespaces[i];

			 if(task == 'c' && exespace=='c') scheduleComputeTask<UINTAH_CPU_TAG>(level, sched);
		else if(task == 'c' && exespace=='g') scheduleComputeTask<KOKKOS_CUDA_TAG>(level, sched);
		else if(task == 'm' && exespace=='c') scheduleModifyTask<UINTAH_CPU_TAG>(level, sched);
		else if(task == 'm' && exespace=='g') scheduleModifyTask<KOKKOS_CUDA_TAG>(level, sched);
		else if(task == 'r' && exespace=='c') scheduleRequireTask<UINTAH_CPU_TAG>(level, sched);
		else if(task == 'r' && exespace=='g') scheduleRequireTask<KOKKOS_CUDA_TAG>(level, sched);
		else {
			printf("wrong combination of tasks and exe spaces\n");
			exit(1);
		}
	}
}


template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::computeTask( task_parameters )
{
	for (int p = 0; p < patches->size(); p++) {
		const Patch* patch = patches->get(p);
		auto newphi = new_dw->getGridVariable<NCVariable<double>, double, MemSpace> (phi_label, 0, patch);	//getGridVariable should call allocateAndPut

		IntVector l = patch->getNodeLowIndex(), h = patch->getNodeHighIndex();
		BlockRange range(l, h);
		Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA(int i, int j, int k){
			newphi(i, j, k) = 1.0;
		});
	}
	//CUDA_CALL(cudaDeviceSynchronize());
	g_expected = 1.0;
	std::cout << "computeTask: " << typeid(ExecSpace).name() << ", expected value for the NEXT task: " << g_expected << "\n";
}

template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::modifyTask( task_parameters )
{
	//CUDA_CALL(cudaDeviceSynchronize());
	int wrong=0;
	double expected = g_expected;
	for (int p = 0; p < patches->size(); p++) {
		const Patch* patch = patches->get(p);
		auto newphi = new_dw->getGridVariable<NCVariable<double>, double, MemSpace> (phi_label, 0, patch, Ghost::None, 0, true);	//getGridVariable should call getModifiable

		IntVector l = patch->getNodeLowIndex(), h = patch->getNodeHighIndex();
		BlockRange range(l, h);
		Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA(int i, int j, int k, int &wrong){
			if(newphi(i, j, k) != expected){	//verify expected value before updating
				wrong++;
				printf("modifies: expected mismatch: %d %d %d %f %f\n", i, j, k, newphi(i,j,k), expected);
			}
			newphi(i, j, k) = newphi(i, j, k) + 0.5;
		}, wrong);
	}
	//CUDA_CALL(cudaDeviceSynchronize());
	printf("modifyTask. wrong values: %d\n", wrong);

	g_expected+=0.5;
	std::cout << "modifyTask: " << typeid(ExecSpace).name() << ", expected value for the NEXT task: " << g_expected << "\n";


}

template <typename ExecSpace, typename MemSpace>
void PortableDependencyTest1::requireTask( task_parameters )
{
	//CUDA_CALL(cudaDeviceSynchronize());
	int wrong=0;
	double expected = g_expected;
	for (int p = 0; p < patches->size(); p++) {
		const Patch* patch = patches->get(p);
		auto newphi = new_dw->getConstGridVariable<constNCVariable<double>, double, MemSpace> (phi_label, 0, patch, Ghost::None, 0);

		IntVector l = patch->getNodeLowIndex(), h = patch->getNodeHighIndex();
		BlockRange range(l, h);
		Uintah::parallel_reduce_sum(execObj, range, KOKKOS_LAMBDA(int i, int j, int k, int &wrong){
			if(newphi(i, j, k) != expected){
				wrong++;
				printf("requires: expected mismatch: %d %d %d %f %f\n", i, j, k, newphi(i,j,k), expected);
			}
		}, wrong);
	}
	//CUDA_CALL(cudaDeviceSynchronize());
	printf("requireTask. wrong values: %d\n", wrong);

	std::cout << "requireTask: " << typeid(ExecSpace).name() << ", expected value for the NEXT task: " << g_expected << "\n";

}
