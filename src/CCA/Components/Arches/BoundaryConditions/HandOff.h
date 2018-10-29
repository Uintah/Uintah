#ifndef Uintah_Component_Arches_HandOff_h
#define Uintah_Component_Arches_HandOff_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/IO/UintahZlibUtil.h>

namespace Uintah{

  template <typename T>
  class HandOff : public TaskInterface {

public:

    HandOff<T>( std::string task_name, int matl_index ) : TaskInterface(task_name, matl_index ){}
    ~HandOff<T>(){}

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void create_local_labels();


    //Build instructions for this (HandOff) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      HandOff* build()
      { return scinew HandOff<T>( m_task_name, m_matl_index ); }

      private:

      std::string m_task_name;
      int m_matl_index;

    };

private:

    typedef ArchesFieldContainer AFC;

    typedef std::map<IntVector, double> CellToValue;
    struct FFInfo {
      CellToValue values;
      Vector relative_xyz;
      double dx;
      double dy;
      IntVector relative_ijk;
      std::string default_type;
      std::string name;
      double default_value;
    };

    FFInfo m_boundary_info;
    std::string m_default_label;

    void readInputFile( std::string file_name, HandOff::FFInfo& struct_result, const int index );

    std::string m_filename;

    int m_index;

  };

  //Function definitions ---------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::problemSetup( ProblemSpecP& db ){

    db->require( "filename", m_filename );
    db->require( "relative_xyz", m_boundary_info.relative_xyz );

    ProblemSpecP db_default = db->findBlock("default");

    if ( db_default != nullptr ){
      db_default->getAttribute("value", m_boundary_info.default_value );
      db_default->getAttribute("type", m_boundary_info.default_type );
      db_default->getAttribute("label", m_default_label );
    }
    else {
      throw ProblemSetupException("Error: no default bc found for: "+m_task_name, __FILE__, __LINE__);
    }

    ArchesCore::VariableHelper<T> helper;

    m_index = helper.dir;

    readInputFile( m_filename, m_boundary_info, m_index );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::create_local_labels(){

    register_new_variable<T>(m_task_name);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::register_initialize(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    T& var = *(tsk_info->get_uintah_field<T>( m_task_name ));
    var.initialize(0.0);

    ArchesCore::VariableHelper<T> helper;

    Point xyz(m_boundary_info.relative_xyz[0], m_boundary_info.relative_xyz[1],
              m_boundary_info.relative_xyz[2]);
    m_boundary_info.relative_ijk = patch->getLevel()->getCellIndex(xyz);

    bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus  = patch->getBCType(Patch::xplus)  != Patch::Neighbor;
    bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus  = patch->getBCType(Patch::yplus)  != Patch::Neighbor;
    bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus  = patch->getBCType(Patch::zplus)  != Patch::Neighbor;

    IntVector low_index = patch->getCellLowIndex();
    IntVector high_index = patch->getCellHighIndex();

    for ( auto i_info = m_boundary_info.values.begin(); i_info != m_boundary_info.values.end();
          i_info++ ){

      IntVector ijk(0,0,0);
      int face = i_info->first[0];
      IntVector shift(0,0,0);

      // First column is the face (-1,1,-2,2,-3,3) corresponding to the face
      // (-x,x,-y,y,-z,z) and the other two are the remaining indices for
      // the other two coordinate directions listed in a right-handed coordinate system
      if ( face == -1 && xminus ){
        ijk[0] = low_index[0];
        ijk[1] = i_info->first[1];
        ijk[2] = i_info->first[2];
        shift = IntVector(-1,0,0);
      } else if ( face == 1 && xplus ){
        ijk[0] = high_index[0];
        ijk[1] = i_info->first[1];
        ijk[2] = i_info->first[2];
        shift = IntVector(-1,0,0);
      } else if ( face == -2 && yminus ){
        ijk[0] = i_info->first[2];
        ijk[1] = low_index[1];
        ijk[2] = i_info->first[1];
        shift = IntVector(0,-1,0);
      } else if ( face == 2 && yplus ){
        ijk[0] = i_info->first[2];
        ijk[1] = high_index[1];
        ijk[2] = i_info->first[1];
        shift = IntVector(0,-1,0);
      } else if ( face == -3 && zminus ){
        ijk[0] = i_info->first[1];
        ijk[1] = i_info->first[2];
        ijk[2] = low_index[2];
        shift = IntVector(0,0,-1);
      } else if ( face == 3 && zplus ){
        ijk[0] = i_info->first[1];
        ijk[1] = i_info->first[2];
        ijk[2] = high_index[2];
        shift = IntVector(0,0,-1);
      }

      // Translate the ijk to remove the relative position:
      ijk += m_boundary_info.relative_ijk;

      if ( patch->containsCell(ijk) ){

        var[ijk] = i_info->second;
        var[ijk + shift] = i_info->second;

      }

    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::register_timestep_init(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const bool packed_tasks ){

    register_variable( m_task_name, ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_task_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW,
                       variable_registry );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    T& var = *(tsk_info->get_uintah_field<T>(m_task_name));
    CT& old_var = *(tsk_info->get_const_uintah_field<CT>(m_task_name));

    var.copyData(old_var);

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  HandOff<T>::register_compute_bcs(
    std::vector<ArchesFieldContainer::VariableInformation>& variable_registry,
    const int time_substep,
    const bool packed_tasks ){

    register_variable( m_task_name, ArchesFieldContainer::MODIFIES, variable_registry );
    register_variable( m_default_label, ArchesFieldContainer::REQUIRES, 0,
                       ArchesFieldContainer::NEWDW, variable_registry );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T> void
  HandOff<T>::compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    // NOTE: If the BC is any kind of custom bc AND isn't covered in the
    //       handoff list then we will just apply the default BC even though
    //       we might not be using this information in any way on non-handoff
    //       boundaries.

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    CT& default_var = *(tsk_info->get_const_uintah_field<CT>(m_default_label));
    T& var = *(tsk_info->get_uintah_field<T>(m_task_name));

    const BndMapT& bc_info = m_bcHelper->get_boundary_information();
    Vector DX = patch->dCell();

    for ( auto i_bc = bc_info.begin(); i_bc != bc_info.end(); i_bc++ ){

      const BndCondSpec* spec = i_bc->second.find(m_default_label);
      Uintah::Patch::FaceType face = i_bc->second.face;

      IntVector shift(0,0,0);
      double delta=0.;
      int I = 0;

      if ( face == Patch::xminus ){
        shift = IntVector(1,0,0);
        delta = DX.x();
        I = -1;
      } else if ( face == Patch::xplus ){
        shift = IntVector(-1,0,0);
        delta = DX.x();
        I = 1;
      } else if ( face == Patch::yminus ){
        shift = IntVector(0,1,0);
        delta = DX.y();
        I = -2;
      } else if ( face == Patch::yplus ){
        shift = IntVector(0,-1,0);
        delta = DX.y();
        I = 2;
      } else if ( face == Patch::zminus ){
        shift = IntVector(0,0,1);
        delta = DX.z();
        I = -3;
      } else if ( face == Patch::zplus ){
        shift = IntVector(0,0,-1);
        delta = DX.z();
        I = 3;
      }

      if ( spec->bcType == CUSTOM ){
        Uintah::ListOfCellsIterator& cell_iter
          = m_bcHelper->get_uintah_extra_bnd_mask( i_bc->second, patch->getID());

        parallel_for(cell_iter.get_ref_to_iterator(),cell_iter.size(), [&] (const int i,const int j,const int k) {
           
          IntVector ijk(i,j,k);
          IntVector orig_ijk(i,j,k);

          ijk -= m_boundary_info.relative_ijk;
          ijk[0] = I;
          auto i_check = m_boundary_info.values.find(ijk);

          if ( i_check == m_boundary_info.values.end() ){
            if ( m_boundary_info.default_type == "dirichlet" ){
              var[orig_ijk] = m_boundary_info.default_value;
              var[orig_ijk+shift] = m_boundary_info.default_value;
            } else if ( m_boundary_info.default_type == "nuemann" ){
              var[orig_ijk+shift] = m_boundary_info.default_value * delta + default_var[orig_ijk];
            }
          }
        });
      }
    }
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void HandOff<T>::readInputFile( std::string file_name, HandOff::FFInfo& struct_result,
                                  const int index )
  {

    gzFile file = gzopen( file_name.c_str(), "r" );
    if ( file == nullptr ) {
      proc0cout << "Error opening file: " << file_name <<
        " for boundary conditions. Errno: " << errno << std::endl;
      throw ProblemSetupException("Unable to open the given input file: " + file_name,
        __FILE__, __LINE__);
    }

    struct_result.name = getString( file );

    struct_result.dx = getDouble( file );
    struct_result.dy = getDouble( file );

    int num_points = getInt( file );

    std::map<IntVector, double> values;

    //Shift indices to set the extra cell value
    int is=0; int js=0; int ks = 0;
    if ( index == 0 ){
        is = 1;
    } else if ( index == 1 ){
        js = 1;
    } else if ( index == 2 ){
        ks = 1;
    }

    for ( int i = 0; i < num_points; i++ ) {

      int I = getInt( file );
      int J = getInt( file );
      int K = getInt( file );

      Vector v;
      v[0] = getDouble( file );

      if ( index > -1 && index < 3 ){

        v[1] = getDouble( file );
        v[2] = getDouble( file );

        IntVector C(I,J,K);

        values.insert( std::make_pair( C, v[index] ));

        IntVector C2(I-is, J-js, K-ks);

        values.insert( std::make_pair( C2, v[index] ));

      } else {

        IntVector C(I,J,K);

        values.insert( std::make_pair( C, v[0]));

      }
    }

    struct_result.values = values;

    gzclose( file );

  }

}
#endif
