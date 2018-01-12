#ifndef Uintah_Component_Arches_RateDeposition_h
#define Uintah_Component_Arches_RateDeposition_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

namespace Uintah{

  class Discretization_new;
  class RateDeposition : public TaskInterface {

  public:

    RateDeposition( std::string task_name, int matl_index, const int N );
    ~RateDeposition();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){};

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    const std::string get_env_name( const int i, const std::string base_name ){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }


    //Build instructions for this (RateDeposition) class.
    class Builder : public TaskInterface::TaskBuilder {
    public:

      Builder( std::string task_name, int matl_index, const int N ) :
              _task_name(task_name), _matl_index(matl_index), _Nenv(N){}
      ~Builder(){}

      RateDeposition* build()
      { return scinew RateDeposition( _task_name, _matl_index , _Nenv ); }
    private:
      std::string _task_name;
      int _matl_index;
      const int _Nenv;
    };

  private:

    typedef ArchesFieldContainer AFC;
    typedef ArchesFieldContainer::VariableInformation AFC_VI;

    int _Nenv;
    double _Tmelt;
    double _MgO;    double _AlO;double _CaO; double _SiO;

    std::string _ParticleTemperature_base_name;
    std::string _MaxParticleTemperature_base_name;
    std::string _ProbParticleX_base_name;
    std::string _ProbParticleY_base_name;
    std::string _ProbParticleZ_base_name;

    std::string _ProbDepositionX_base_name;
    std::string _ProbDepositionY_base_name;
    std::string _ProbDepositionZ_base_name;

    std::string _RateDepositionX_base_name;
    std::string _RateDepositionY_base_name;
    std::string _RateDepositionZ_base_name;
    std::string _diameter_base_name;

    std::string _ProbSurfaceX_name;
    std::string _ProbSurfaceY_name;
    std::string _ProbSurfaceZ_name;
    std::string _xvel_base_name;
    std::string _yvel_base_name;
    std::string _zvel_base_name;

    std::string  _weight_base_name;
    std::string  _rho_base_name;

    std::string _FluxPx_base_name;
    std::string _FluxPy_base_name;
    std::string _FluxPz_base_name;

    std::string  _WallTemperature_name;
    double _pi_div_six;

    //--------------------compute the deposition probability ---------------------------------------
    inline double compute_prob_stick( const double A, const double B,
                                      const double Tvol, const double MaxTvol)
    {
      // Urbain model 1981
      const double ReferVisc=10000.0;

      //-----------------------Actual work here----------------------
      // Compute the melting probability
      double ProbMelt=0;
      double ProbVisc=0;
      double ProbStick=0;
      double Visc=0;
      ProbMelt=( MaxTvol >= _Tmelt ? 1:0);

      //compute the viscosity probability
      Visc = 0.1 * A * std::max(Tvol,273.0) * exp(1000.0*B /(std::max(Tvol,273.0)) );
      ProbVisc = ( ReferVisc/(Visc) > 1 ? 1 :  ReferVisc/(Visc) );
      ProbStick= ( ProbVisc * ProbMelt );
      return ProbStick;
    };

  };

}
#endif
