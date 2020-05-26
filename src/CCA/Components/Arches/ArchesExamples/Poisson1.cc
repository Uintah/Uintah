#include <CCA/Components/Arches/ArchesExamples/Poisson1.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace Uintah;
using namespace Uintah::ArchesExamples;

//---------------------------------------------- load function pointers ----------------------------------------------------
TaskAssignedExecutionSpace Poisson1::loadTaskComputeBCsFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::BC>( this
									 , &Poisson1::compute_bcs<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
									 , &Poisson1::compute_bcs<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
									 , &Poisson1::compute_bcs<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
									 );
}

TaskAssignedExecutionSpace Poisson1::loadTaskInitializeFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &Poisson1::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Poisson1::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Poisson1::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Poisson1::loadTaskEvalFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &Poisson1::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     , &Poisson1::eval<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     , &Poisson1::eval<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Poisson1::loadTaskTimestepInitFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_INITIALIZE>( this
									 , &Poisson1::timestep_init<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
									 , &Poisson1::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
									 , &Poisson1::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
									 );
}

TaskAssignedExecutionSpace Poisson1::loadTaskRestartInitFunctionPointers(){
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//----------------------------------------------- problem setup ---------------------------------------------------
void Poisson1::problemSetup( ProblemSpecP& db ){
//updating delT (as done in stand alone example) is not needed here. Arches will take care of delT
}

void
Poisson1::create_local_labels(){
  register_new_variable<CCVariable<double> >( "phi" );
}


//--------------------------------------------- init -----------------------------------------------------
void
Poisson1::register_initialize( ArchesVIVector& variable_registry , const bool packed_tasks){
  register_variable( "phi", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
}


template <typename ExecSpace, typename MemSpace>
void Poisson1::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  //copied from Poisson1::initialize
  typedef CCVariable<double> T;
  auto phi = tsk_info->get_field<T, double, MemSpace>("phi");
  parallel_initialize(execObj,0.0, phi);

  //arches bc
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){
	double val = i_bc->second.find("Phi")->value;
    const bool on_this_patch = i_bc->second.has_patch(patch->getID());
    if ( on_this_patch ){
      //Handle cell type first
      Uintah::ListOfCellsIterator& cell_iter_ct  = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
      parallel_for_unstructured(execObj,cell_iter_ct.get_ref_to_iterator(execObj),cell_iter_ct.size(), KOKKOS_LAMBDA (int i,int j,int k) {
        phi(i,j,k) = val;
      });
    }
  }
}


//--------------------------------------------- timestep_init -----------------------------------------------------
void
Poisson1::register_timestep_init( ArchesVIVector& variable_registry , const bool packed_tasks){
  register_variable( "phi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, m_task_name );
  register_variable( "phi", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
}

template <typename ExecSpace, typename MemSpace> void
Poisson1::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  typedef CCVariable<double> T;
  typedef constCCVariable<double> CT;
  auto phi    = tsk_info->get_field<CT, const double, MemSpace>("phi");
  auto newphi = tsk_info->get_field< T, double, MemSpace>("phi");
  // Prepare the ranges for both boundary conditions and main loop
  IntVector l = patch->getExtraCellLowIndex();
  IntVector h = patch->getExtraCellHighIndex();

  Uintah::BlockRange rangeBoundary( l, h);

  // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
  //Uintah::parallel_for<ExecSpace, LaunchBounds< 640,1 > >( execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
  Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
    newphi(i, j, k) = 0.0;
  });
}


//--------------------------------------------- eval -----------------------------------------------------
void
Poisson1::register_timestep_eval( ArchesVIVector& variable_registry , const int time_substep, const bool packed_tasks){
  register_variable( "phi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep, m_task_name );
  register_variable( "phi", ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
}


template <typename ExecSpace, typename MemSpace> void
Poisson1::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  //copied from Poisson1::timeAdvance
  typedef CCVariable<double> T;
  typedef constCCVariable<double> CT;
  auto phi    = tsk_info->get_field<CT, const double, MemSpace>("phi");
  auto newphi = tsk_info->get_field< T, double, MemSpace>("phi");

  // Prepare the ranges for both boundary conditions and main loop
  IntVector l = patch->getExtraCellLowIndex();
  IntVector h = patch->getExtraCellHighIndex();

  l += IntVector( patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1
                , patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1
                , patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1
                );

  h -= IntVector( patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1
                , patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1
                , patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1
                );

  Uintah::BlockRange range( l, h );

  // Perform the main loop
  Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
    newphi(i, j, k) = ( 1. / 6 ) *
                      ( phi(i + 1, j, k) + phi(i - 1, j, k) + phi(i, j + 1, k) +
                        phi(i, j - 1, k) + phi(i, j, k + 1) + phi(i, j, k - 1) );
  });
}


//--------------------------------------------- compute_bcs -----------------------------------------------------
void
Poisson1::register_compute_bcs( ArchesVIVector& variable_registry, const int time_substep, const bool packed_tasks ){
  register_variable( "phi", ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep, m_task_name );
  register_variable( "phi", ArchesFieldContainer::MODIFIES, variable_registry, m_task_name );
}

template <typename ExecSpace, typename MemSpace> void
Poisson1::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  typedef CCVariable<double> T;
  typedef constCCVariable<double> CT;
  auto phi    = tsk_info->get_field<CT, const double, MemSpace>("phi");
  auto newphi = tsk_info->get_field< T, double, MemSpace>("phi");

  //arches bc
  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){
	const bool on_this_patch = i_bc->second.has_patch(patch->getID());
	if ( on_this_patch ){
	  Uintah::ListOfCellsIterator& cell_iter_ct  = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());
	  parallel_for_unstructured(execObj,cell_iter_ct.get_ref_to_iterator(execObj),cell_iter_ct.size(), KOKKOS_LAMBDA (int i,int j,int k) {
		newphi(i,j,k) = phi(i,j,k);
	  });
	}
  }
}
