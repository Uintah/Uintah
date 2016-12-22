#include <CCA/Components/Arches/Transport/VelRhoHatBC.h>

using namespace Uintah;
typedef ArchesFieldContainer AFC;

//--------------------------------------------------------------------------------------------------
VelRhoHatBC::VelRhoHatBC( std::string task_name, int matl_index ) :
AtomicTaskInterface( task_name, matl_index )
{
}

//--------------------------------------------------------------------------------------------------
VelRhoHatBC::~VelRhoHatBC()
{
}

//--------------------------------------------------------------------------------------------------
void VelRhoHatBC::problemSetup( ProblemSpecP& db ){
  m_xmom = "x-mom";
  m_ymom = "y-mom";
  m_zmom = "z-mom";
}

//--------------------------------------------------------------------------------------------------
void VelRhoHatBC::create_local_labels(){
}

//--------------------------------------------------------------------------------------------------
void VelRhoHatBC::register_eval( std::vector<AFC::VariableInformation>& variable_registry,
                                 const int time_substep ){
  register_variable( m_xmom, AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( m_ymom, AFC::MODIFIES, variable_registry, m_task_name );
  register_variable( m_zmom, AFC::MODIFIES, variable_registry, m_task_name );
}

void VelRhoHatBC::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

  SFCXVariable<double>* xmom = tsk_info->get_uintah_field<SFCXVariable<double> >( m_xmom );
  SFCYVariable<double>* ymom = tsk_info->get_uintah_field<SFCYVariable<double> >( m_ymom );
  SFCZVariable<double>* zmom = tsk_info->get_uintah_field<SFCZVariable<double> >( m_zmom );

  const BndMapT& bc_info = m_bcHelper->get_boundary_information();
  const double possmall = 1e-16;

  for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

    Uintah::Iterator& cell_iter = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID() );
    IntVector iDir = patch->faceDirection( i_bc->second.face );
    Patch::FaceType face = i_bc->second.face;
    BndTypeEnum my_type = i_bc->second.type;
    int sign = iDir[0] + iDir[1] + iDir[2];
    int bc_sign = 0;
    int move_to_face_value = ( sign < 1 ) ? 1 : 0;

    IntVector move_to_face(std::abs(iDir[0])*move_to_face_value,
                           std::abs(iDir[1])*move_to_face_value,
                           std::abs(iDir[2])*move_to_face_value);

    Array3<double>* var = NULL;
    if ( face == Patch::xminus || face == Patch::xplus ){
      var = xmom;
    }  else if ( face == Patch::yminus || face == Patch::yplus ){
      var = ymom;
    } else {
      var = zmom;
    }

    if ( my_type == OUTLET ){
      bc_sign = 1.;
    } else if ( my_type == PRESSURE){
      bc_sign = -1.;
    }

    sign = bc_sign * sign;

    if ( my_type == OUTLET || my_type == PRESSURE ){
      // This applies the mostly in (pressure)/mostly out (outlet) boundary condition
      for (cell_iter.reset(); !cell_iter.done(); cell_iter++ ){

        IntVector cface = *cell_iter + move_to_face; // cell on the face
        IntVector cint = *cell_iter - iDir; // first interior cell
        IntVector cef = cface + iDir; // extra cell face in the last index (most outwardly position)

        if ( sign * (*var)[cface] > possmall ){
          // du/dx = 0
          (*var)[cface] = (*var)[cint];
        } else {
          // shut off the hatted value to encourage the mostly-* condition
          (*var)[cface] = 0.0;
        }

        (*var)[cef] = (*var)[cface];

      }
    }
  }
}
