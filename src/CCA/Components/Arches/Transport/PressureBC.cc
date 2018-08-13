#include <CCA/Components/Arches/Transport/PressureBC.h>

using namespace Uintah::ArchesCore;
using namespace Uintah;

typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
PressureBC::PressureBC( std::string task_name, int matl_index ) :
AtomicTaskInterface( task_name, matl_index )
{}

//--------------------------------------------------------------------------------------------------
PressureBC::~PressureBC()
{}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace PressureBC::loadTaskEvalFunctionPointers(){

  return create_portable_arches_tasks( this,
                                       &PressureBC::eval<UINTAH_CPU_TAG>,
                                       &PressureBC::eval<KOKKOS_OPENMP_TAG> );

}

//--------------------------------------------------------------------------------------------------
void PressureBC::problemSetup( ProblemSpecP& db ){
  m_press = "pressure";
}

//--------------------------------------------------------------------------------------------------
void PressureBC::create_local_labels()
{}

//--------------------------------------------------------------------------------------------------
void PressureBC::register_timestep_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                 const int time_substep, const bool pack_tasks ){

  register_variable( m_press, AFC::MODIFIES, variable_registry, _task_name );

}

//--------------------------------------------------------------------------------------------------
template<typename ExecutionSpace, typename MemorySpace>
void PressureBC::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecutionSpace, MemorySpace>& executionObject ){

  CCVariable<double>& p = tsk_info->get_uintah_field_add<CCVariable<double> >( m_press );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    Uintah::ListOfCellsIterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    BndTypeEnum my_type = i_bc->second.type;

    if ( my_type == WALL || my_type == INLET ){

      parallel_for_unstructured(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
        // enforce dp/dn = 0
        p(i,j,k) = p(i-iDir[0],j-iDir[1],k-iDir[2]);

      });

    } else if ( my_type == OUTLET ||my_type == PRESSURE ) {

      //enforce p = 0

      const double sign = -(iDir[0]+iDir[1]+iDir[2]);

      parallel_for_unstructured(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
        p(i,j,k) = sign*p(i-iDir[0],j-iDir[1],k-iDir[2]);
      });
    }
  }
}
