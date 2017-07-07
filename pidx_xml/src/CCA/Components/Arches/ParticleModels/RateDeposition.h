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

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep );

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info );

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

      Builder( std::string task_name, int matl_index, const int N ) : _task_name(task_name), _matl_index(matl_index), _Nenv(N){}
      ~Builder(){}

      RateDeposition* build()
      { return scinew RateDeposition( _task_name, _matl_index , _Nenv ); }
      private:
      std::string _task_name;
      int _matl_index;
      const int _Nenv;
    };

private:
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

    //--------------------compute the deposition probability --------------------------------------------------------------
    inline double compute_prob_stick(double areaFraction,double Tvol,double MaxTvol)
    {  // Urbain model 1981
      //double CaO=26.49/100;const double MgO=4.47/100; double AlO=14.99/100;const double SiO=38.9/100; //const double alpha=0;
     double CaO=_CaO;double MgO=_MgO; double AlO=_AlO;double SiO=_SiO; //const double alpha=0;
     // const double B0=0; const doulbe B1=0; const double B3=0;
     double CaOmolar=0.0;               double AlOmolar=0.0;
     CaOmolar=CaO/(CaO+MgO+AlO+SiO);            AlOmolar=AlO/(CaO+MgO+AlO+SiO);
     const double alpha=CaOmolar/(AlOmolar+CaOmolar);
     const double B0=13.8+39.9355*alpha-44.049*alpha*alpha;
     const double B1=30.481-117.1505*alpha+129.9978*alpha*alpha;
     const double B2=-40.9429+234.0486*alpha-300.04*alpha*alpha;
     const double B3= 60.7619-153.9276*alpha+211.1616*alpha*alpha;
     const double Bactivational=B0+B1*SiO+B2*SiO*SiO+B3*SiO*SiO*SiO;
     const double Aprepontional=exp(-(0.2693*Bactivational+11.6725));  //const double Bactivational= 47800;
     const double ReferVisc=10000.0;

  //-----------------------Actual work here----------------------
   // Compute the melting probability
    double ProbMelt=0;
    double ProbVisc=0;
    double ProbStick=0;
    double Visc=0;
    ProbMelt=( MaxTvol >= _Tmelt ? 1:0);

   //compute the viscosity probability
     Visc = 0.1*Aprepontional* std::max(Tvol,273.0) * exp(1000.0*Bactivational /(std::max(Tvol,273.0)) );
     ProbVisc = (ReferVisc/(Visc) > 1 ? 1 :  ReferVisc/(Visc) );
     ProbStick= ((1-areaFraction) * ProbVisc * ProbMelt);
     return ProbStick;
     };

   };

}
#endif
