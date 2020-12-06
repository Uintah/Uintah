#ifndef Uintah_Component_Arches_FileInit_h
#define Uintah_Component_Arches_FileInit_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <Core/IO/UintahZlibUtil.h>

namespace Uintah{

  template <typename T>
  class FileInit : public TaskInterface {

public:

    FileInit<T>
      ( std::string task_name, int matl_index, const std::string var_name );
    ~FileInit<T>();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool pack_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    template <typename ExecSpace, typename MemSpace>
    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj );

    template <typename ExecSpace, typename MemSpace>
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    template <typename ExecSpace, typename MemSpace>
    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){}

    void create_local_labels(){};

    //Build instructions for this (FileInit) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index, std::string var_name ) :
        m_task_name(task_name), m_matl_index(matl_index), m_var_name(var_name){}
      ~Builder(){}

      FileInit* build()
      { return scinew FileInit( m_task_name, m_matl_index, m_var_name ); }

      private:

      std::string m_task_name;
      int m_matl_index;
      std::string m_var_name;

    };

//private:

    int m_var_idx;
    const std::string m_var_name;
    std::string m_var2_name;
    std::string m_var3_name;

    std::string m_filename;
    typedef std::vector<double> CellToValues;
    CellToValues m_data;
    int nx,ny,nz;

    void readInputFile( std::string, const bool read_vector );

  };

  //
  // Class member implementations
  //

  //------------------------------------------------------------------------------------------------
  template <typename T>
  FileInit<T>::FileInit( std::string task_name, int matl_index, const std::string var_name ) :
  TaskInterface( task_name, matl_index ), m_var_name(var_name)
  {}

  //------------------------------------------------------------------------------------------------
  template <typename T>
  FileInit<T>::~FileInit()
  {}

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FileInit<T>::loadTaskComputeBCsFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FileInit<T>::loadTaskInitializeFunctionPointers()
  {
    return create_portable_arches_tasks<TaskInterface::INITIALIZE>( this
                                       , &FileInit<T>::initialize<UINTAH_CPU_TAG>     // Task supports non-Kokkos builds
                                       , &FileInit<T>::initialize<KOKKOS_OPENMP_TAG>  // Task supports Kokkos::OpenMP builds
                                       //, &FileInit<T>::initialize<KOKKOS_CUDA_TAG>    // Task supports Kokkos::Cuda builds
                                       );
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FileInit<T>::loadTaskEvalFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FileInit<T>::loadTaskTimestepInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //--------------------------------------------------------------------------------------------------
  template <typename T>
  TaskAssignedExecutionSpace FileInit<T>::loadTaskRestartInitFunctionPointers()
  {
    return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void FileInit<T>::problemSetup( ProblemSpecP& db ){

    bool is_vector = false;

    db->require("filename", m_filename );
    if ( db->findBlock("vector") ){
      db->findBlock("vector")->getAttribute("index", m_var_idx );
      is_vector = true;
    }

    readInputFile( m_filename, is_vector );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void FileInit<T>::register_initialize(std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

      register_variable( m_var_name, ArchesFieldContainer::MODIFIES, variable_registry );

  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  template <typename ExecSpace, typename MemSpace>
  void FileInit<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, ExecutionObject<ExecSpace, MemSpace>& execObj ){

    auto phi = tsk_info->get_field<T, double, MemSpace>(m_var_name);

    Uintah::parallel_initialize( execObj, 0.0, phi );

    int x=nx,y=ny,z=nz;
    double * data = m_data.data();

    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( execObj, range, KOKKOS_LAMBDA (int i, int j, int k){
      phi(i,j,k) = data[i*y*z+j*z+k];

    });
  }

  //------------------------------------------------------------------------------------------------
  template <typename T>
  void FileInit<T>::readInputFile( std::string filename, const bool read_vector )
  {

    gzFile file = gzopen( filename.c_str(), "r" );
    if ( file == nullptr ) {
      proc0cout << "Error opening file: " << filename << " for initialization: " << errno << std::endl;
      throw ProblemSetupException("Unable to open the given input file: " + filename, __FILE__, __LINE__);
    }

    int N;
    nx = getInt(file);
    ny = getInt(file);
    nz = getInt(file);
    N = nx*ny*nz;
    m_data=std::vector<double> (N);

    for ( int II = 0; II < N; II++ ){

      int i = getInt(file);
      int j = getInt(file);
      int k = getInt(file);

      double value = 0;

      if ( read_vector ){

        std::vector<double> phi(3);

        phi[0] = getDouble(file);
        phi[1] = getDouble(file);
        phi[2] = getDouble(file);

        value = phi[m_var_idx];

      } else {

        double phi;

        phi = getDouble(file);

        value = phi;

      }

      m_data[i*ny*nz+j*nz+k]=value;

    }

    gzclose( file );

  }
}

#endif
