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
void PressureBC::problemSetup( ProblemSpecP& db ){
  m_press = "pressure";
}

//--------------------------------------------------------------------------------------------------
void PressureBC::create_local_labels()
{}

//--------------------------------------------------------------------------------------------------
void PressureBC::register_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                 const int time_substep ){

  register_variable( m_press, AFC::MODIFIES, variable_registry, m_task_name );

}

//--------------------------------------------------------------------------------------------------
void PressureBC::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  CCVariable<double>& p = tsk_info->get_uintah_field_add<CCVariable<double> >( m_press );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    Uintah::Iterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    BndTypeEnum my_type = i_bc->second.type;

    if ( my_type == WALL || my_type == INLET ){

      for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){

        // enforce dp/dn = 0

        IntVector c = *cell_iter;
        IntVector c_interior = c - iDir;

        p[c] = p[c_interior];

      }

    } else if ( my_type == OUTLET ||my_type == PRESSURE ) {

      //enforce p = 0

      const double sign = -(iDir[0]+iDir[1]+iDir[2]);

      for ( cell_iter.reset(); !cell_iter.done(); cell_iter++ ){

        IntVector c = *cell_iter;
        IntVector c_interior = c - iDir;

        p[c] = sign*p[c_interior];

      }
    }
  }
}
