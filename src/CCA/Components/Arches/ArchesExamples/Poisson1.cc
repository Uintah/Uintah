#include <CCA/Components/Arches/ArchesExamples/Poisson1.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>

using namespace Uintah;
using namespace Uintah::ArchesExamples;

//---------------------------------------------- load function pointers ----------------------------------------------------
TaskAssignedExecutionSpace Poisson1::loadTaskComputeBCsFunctionPointers(){
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

TaskAssignedExecutionSpace Poisson1::loadTaskInitializeFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                     , &Poisson1::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Poisson1::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Poisson1::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Poisson1::loadTaskEvalFunctionPointers(){
  return create_portable_arches_tasks<TaskInterface::TIMESTEP_EVAL>( this
                                     , &Poisson1::eval<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                     //, &Poisson1::timestep_init<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                     //, &Poisson1::timestep_init<KOKKOS_CUDA_TAG>  // Task supports Kokkos::Cuda builds
                                     );
}

TaskAssignedExecutionSpace Poisson1::loadTaskTimestepInitFunctionPointers(){
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
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
  //TODO: port this code
  int matl = 0;
  typedef CCVariable<double> T;
  auto phi = tsk_info->get_field<T, double, MemSpace>("phi");
  parallel_initialize(execObj,0.0, phi);

  for (Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face = Patch::nextFace(face)) {

    if (patch->getBCType(face) == Patch::None) {
      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matl);
      for (int child = 0; child < numChildren; child++) {
        Iterator nbound_ptr, nu;

        const BoundCondBase* bcb = patch->getArrayBCValues(face, matl, "Phi", nu, nbound_ptr, child);

        const BoundCond<double>* bc = dynamic_cast<const BoundCond<double>*>(bcb);
        double value = bc->getValue();
        for (nbound_ptr.reset(); !nbound_ptr.done(); nbound_ptr++) {
          phi( (*nbound_ptr)[0], (*nbound_ptr)[1], (*nbound_ptr)[2]) = value;
        }
        delete bcb;
      }
    }
  }

}

//--------------------------------------------- eval -----------------------------------------------------
void
Poisson1::register_timestep_eval( ArchesVIVector& variable_registry , const int time_substep, const bool packed_tasks){
  register_variable( "phi", ArchesFieldContainer::REQUIRES, 1, ArchesFieldContainer::OLDDW, variable_registry, time_substep, m_task_name );
  register_variable( "phi", ArchesFieldContainer::COMPUTES, variable_registry, m_task_name );
}


template <typename ExecSpace, typename MemSpace> void
Poisson1::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){
  //copied from Poisson1::timeAdvance
  typedef CCVariable<double> T;
  typedef constCCVariable<double> CT;
  auto phi    = tsk_info->get_field<CT, const double, MemSpace>("phi");
  auto newphi = tsk_info->get_field< T, double, MemSpace>("phi");

    // Prepare the ranges for both boundary conditions and main loop
    IntVector l = patch->getNodeLowIndex();
    IntVector h = patch->getNodeHighIndex();

    Uintah::BlockRange rangeBoundary( l, h);

    l += IntVector( patch->getBCType(Patch::xminus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::yminus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::zminus) == Patch::Neighbor ? 0 : 1
                  );

    h -= IntVector( patch->getBCType(Patch::xplus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::yplus) == Patch::Neighbor ? 0 : 1
                  , patch->getBCType(Patch::zplus) == Patch::Neighbor ? 0 : 1
                  );

    Uintah::BlockRange range( l, h );

    // Perform the boundary condition of copying over prior initialized values.  (TODO:  Replace with boundary condition)
    //Uintah::parallel_for<ExecSpace, LaunchBounds< 640,1 > >( execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
    Uintah::parallel_for(execObj, rangeBoundary, KOKKOS_LAMBDA(int i, int j, int k){
      newphi(i, j, k) = phi(i,j,k);
    });

    // Perform the main loop
    Uintah::parallel_for(execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
      newphi(i, j, k) = ( 1. / 6 ) *
                        ( phi(i + 1, j, k) + phi(i - 1, j, k) + phi(i, j + 1, k) +
                          phi(i, j - 1, k) + phi(i, j, k + 1) + phi(i, j, k - 1) );

//      printf("In lambda CUDA at %d,%d,%d), m_phi is at %p %p %g from %g, %g, %g, %g, %g, %g and m_newphi is %g\n", i, j, k,
//             phi.m_view.data(), &(phi(i,j,k)),
//             phi(i,j,k),
//             phi(i + 1, j, k), phi(i - 1, j, k), phi(i, j + 1, k),
//             phi(i, j - 1, k), phi(i, j, k + 1), phi(i, j, k - 1),
//             newphi(i,j,k));


    });

}
