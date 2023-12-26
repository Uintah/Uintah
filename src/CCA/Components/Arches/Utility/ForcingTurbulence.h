#ifndef Uintah_Component_Arches_ForcingTurbulence_h
#define Uintah_Component_Arches_ForcingTurbulence_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
// #include <CCA/Components/Arches/ArchesVariables.h>

#include <complex>

/**
 *  \file     ForcingTurbulence.h
 *  \class    ForcingTurbulence
 *  \author   Jebin Elias
 *  \date     May 16, 2020          v1
 *
 *  \brief    Integrate TKE of each spherical shell of spectral space at every
 *            timestep. Scale each shell (or any number of shells of choice)
 *            to its previous energy state.
 *            - Currently, operatable only on singe patch
 */

//------------------------------------------------------------------------------

namespace Uintah{

  class ForcingTurbulence : public TaskInterface {

  public:

    ForcingTurbulence( std::string task_name, int matl_index );
    ~ForcingTurbulence();

    TaskAssignedExecutionSpace loadTaskComputeBCsFunctionPointers();

    TaskAssignedExecutionSpace loadTaskInitializeFunctionPointers();

    TaskAssignedExecutionSpace loadTaskEvalFunctionPointers();

    TaskAssignedExecutionSpace loadTaskTimestepInitFunctionPointers();

    TaskAssignedExecutionSpace loadTaskRestartInitFunctionPointers();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks);

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){};

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    // Build instructions
    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index ) : m_task_name(task_name), m_matl_index(matl_index){}
      ~Builder(){}

      ForcingTurbulence* build()
      { return scinew ForcingTurbulence( m_task_name, m_matl_index ); }

    private:

      std::string m_task_name;
      int m_matl_index;

    };

  private:

    int compute_iShell( int i, int j, int k, int Nx, int Ny, int Nz );

    void compute_TKE( const Patch* patch, ArchesTaskInfoManager* tsk_info, std::map<int, double> &TKE_spectrum );

    void eval_scale_TKE( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void stockham( std::complex<double> x[], std::complex<double> y[], int n, int n2, int flag );

    void cooley_tukey( std::complex<double> x[], int n, int n2, int flag );

    // Compute 3D fft
    void fft3D( const Patch* patch, int n1, int n2, int n3, int flag,
                SFCXVariable<double>& Uvel_Real, std::complex<double> * Uvel_Spectral,
                SFCYVariable<double>& Vvel_Real, std::complex<double> * Vvel_Spectral,
                SFCZVariable<double>& Wvel_Real, std::complex<double> * Wvel_Spectral );

    int Nx = 0, Ny = 0, Nz = 0, Nt = 0;
    Vector m_gridRes{ 0, 0, 0 };
    int m_Nbins = 0;

    std::string m_uVel_name, m_vVel_name, m_wVel_name, m_density_name;

    std::map<int, double> TKE_spectrum_nm1, TKE_spectrum_n;

  }; // class ForcingTurbulence
} // namespace Uintah
#endif
