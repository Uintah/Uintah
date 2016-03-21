#ifndef Uintah_Component_Arches_CoalTemperature_h
#define Uintah_Component_Arches_CoalTemperature_h

#include <CCA/Components/Arches/Task/TaskInterface.h>

#include <CCA/Components/Arches/FunctorSwitch.h>
 
namespace Uintah{ 

  class Operators; 
  class CoalTemperature : public TaskInterface { 
    struct computeCoalTemperature{
      computeCoalTemperature(double _dt,
                             int    _ix,
                             constCCVariable<double>& _gas_temperature,
                             constCCVariable<double>& _vol_frac,
                             constCCVariable<double>& _rcmass, 
                             constCCVariable<double>& _charmass,
                             constCCVariable<double>& _enthalpy,
                             constCCVariable<double>& _temperatureold,
                             constCCVariable<double>& _diameter,
                             CCVariable<double>& _temperature, 
                             CCVariable<double>& _dTdt,
                             CoalTemperature* theClassAbove ) :
                             dt(_dt),
                             ix(_ix),
#ifdef UINTAH_ENABLE_KOKKOS
                             gas_temperature(_gas_temperature.getKokkosView()),
                             vol_frac(_vol_frac.getKokkosView()),
                             rcmass(_rcmass.getKokkosView()), 
                             charmass(_charmass.getKokkosView()),
                             enthalpy(_enthalpy.getKokkosView()),
                             temperatureold(_temperatureold.getKokkosView()), 
                             diameter(_diameter.getKokkosView()),
                             temperature(_temperature.getKokkosView()),
                             dTdt(_dTdt.getKokkosView()),
#else
                             gas_temperature(_gas_temperature),
                             vol_frac(_vol_frac),
                             rcmass(_rcmass), 
                             charmass(_charmass),
                             enthalpy(_enthalpy),
                             temperatureold(_temperatureold), 
                             diameter(_diameter),
                             temperature(_temperature),
                             dTdt(_dTdt),
#endif
                             TCA(theClassAbove){ }

      void operator()(int i , int j, int k ) const {

        int icount = 0;
        double delta = 1.0;

        double tol = 1.0;
        double hint = 0.0;
        double Ha = 0.0;
        double Hc = 0.0;
        double H = 0.0;
        double f1 = 0.0;
        double f2 = 0.0;
        double dT = 0.0;

        double pT = temperature(i,j,k);
        double gT = gas_temperature(i,j,k);
        double pT_olddw = temperatureold(i,j,k);
        double oldpT = temperature(i,j,k);
        double RC = rcmass(i,j,k);
        double CH = charmass(i,j,k);
        double pE = enthalpy(i,j,k);
        double vf = vol_frac(i,j,k);

        double massDry=0.0;
        double initAsh=0.0;
        double dp=0.0;

        if (vf < 1.0e-10 ){
          temperature(i,j,k)=gT; // gas temperature
          dTdt(i,j,k)=(pT-pT_olddw)/dt;
        } else {
          int max_iter=15;
          int iter =0;

          if ( !TCA->_const_size ) {
            dp = diameter(i,j,k);
            massDry = TCA->_pi/6.0 * std::pow( dp, 3.0 ) * TCA->_rhop_o;
            initAsh = massDry * TCA->_ash_mf;
          } else {
            initAsh = TCA->_init_ash[ix];
          }

          if ( initAsh > 0.0 ) {
            for ( ; iter < max_iter; iter++) {
              icount++;
              oldpT = pT;
              // compute enthalpy given Tguess
              hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
              Ha = -202849.0 + TCA->_Ha0 + pT * (593. + pT * 0.293);
              Hc = TCA->_Hc0 + hint * TCA->_RdMW;
              H = Hc * (RC + CH) + Ha * initAsh;
              f1 = pE - H;
              // compute enthalpy given Tguess + delta
              pT = pT + delta;
              hint = -156.076 + 380/(-1 + exp(380 / pT)) + 3600/(-1 + exp(1800 / pT));
              Ha = -202849.0 + TCA->_Ha0 + pT * (593. + pT * 0.293);
              Hc = TCA->_Hc0 + hint * TCA->_RdMW;
              H = Hc * (RC + CH) + Ha * initAsh;
              f2 = pE - H;
              // correct temperature
              dT = f1 * delta / (f2-f1) + delta;
              pT = pT - dT;    //to add an coefficient for steadness
              // check to see if tolernace has been met
              tol = std::abs(oldpT - pT);

              if (tol < 0.01 )
                break;
            }
            if (iter ==max_iter-1 || pT <273.0 || pT > 3500.0 ){
              double pT_low=273;
              hint = -156.076 + 380/(-1 + exp(380 / pT_low)) + 3600/(-1 + exp(1800 / pT_low));
              Ha = -202849.0 + TCA->_Ha0 + pT_low * (593. + pT_low * 0.293);
              Hc = TCA->_Hc0 + hint * TCA->_RdMW;
              double H_low = Hc * (RC + CH) + Ha * initAsh;
              double pT_high=3500;
              hint = -156.076 + 380/(-1 + exp(380 / pT_high)) + 3600/(-1 + exp(1800 / pT_high));
              Ha = -202849.0 + TCA->_Ha0 + pT_high * (593. + pT_high * 0.293);
              Hc = TCA->_Hc0 + hint * TCA->_RdMW;
              double H_high = Hc * (RC + CH) + Ha * initAsh;
              if (pE < H_low || pT < 273.0){
                pT = 273.0;
              } else if (pE > H_high || pT > 3500.0) {
                pT = 3500.0;
              }
            }
          } else {
            pT = TCA->_initial_temperature; //prevent nans when dp & ash = 0.0 in cqmom
          }

          temperature(i,j,k)=pT;
          dTdt(i,j,k)=(pT-pT_olddw)/dt;
        }

      }
      private:
      double dt;
      int    ix;

#ifdef UINTAH_ENABLE_KOKKOS
      KokkosView3<const double> gas_temperature;
      KokkosView3<const double> vol_frac;
      KokkosView3<const double> rcmass; 
      KokkosView3<const double> charmass;
      KokkosView3<const double> enthalpy;
      KokkosView3<const double> temperatureold; 
      KokkosView3<const double> diameter;
      KokkosView3<double> temperature; 
      KokkosView3<double> dTdt;
#else
      constCCVariable<double>& gas_temperature;
      constCCVariable<double>& vol_frac;
      constCCVariable<double>& rcmass; 
      constCCVariable<double>& charmass;
      constCCVariable<double>& enthalpy;
      constCCVariable<double>& temperatureold; 
      constCCVariable<double>& diameter;
      CCVariable<double>& temperature; 
      CCVariable<double>& dTdt;
#endif
      CoalTemperature* TCA;     
    };

public: 

    CoalTemperature( std::string task_name, int matl_index, const int N );
    ~CoalTemperature(); 

    void problemSetup( ProblemSpecP& db ); 

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ); 

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ); 

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){}; 

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                      SpatialOps::OperatorDatabase& opr ){}; 

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                     SpatialOps::OperatorDatabase& opr );
    
    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
                        SpatialOps::OperatorDatabase& opr );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info, 
               SpatialOps::OperatorDatabase& opr );

    void create_local_labels(); 

    const std::string get_env_name( const int i, const std::string base_name ){ 
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
               
    const std::string get_qn_env_name( const int i, const std::string base_name ){ 
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_qn" + env;
    }

    //Build instructions for this (CoalTemperature) class. 
    class Builder : public TaskInterface::TaskBuilder { 

      public: 

      Builder( std::string task_name, int matl_index, const int N ) : _task_name(task_name), _matl_index(matl_index), _Nenv(N){}
      ~Builder(){}

      CoalTemperature* build()
      { return new CoalTemperature( _task_name, _matl_index, _Nenv ); }

      private: 

      std::string _task_name; 
      int _matl_index;
      int _Nenv;

    };

private: 

    bool _const_size;
    int _Nenv;
    double _rhop_o;
    double _pi; 
    double _initial_temperature; 
    double _Ha0; 
    double _Hc0; 
    double _Hh0; 
    double _Rgas; 
    double _RdC; 
    double _RdMW; 
    double _MW_avg;
    double _ash_mf;

    std::vector<double> _init_ash;
    std::vector<double> _init_rawcoal;
    std::vector<double> _init_char;
    std::vector<double> _sizes;
    std::vector<double> _denom; 

    std::string _diameter_base_name;
    std::string _rawcoal_base_name;
    std::string _char_base_name; 
    std::string _enthalpy_base_name; 
    std::string _dTdt_base_name; 
    std::string _gas_temperature_name;
    std::string _vol_fraction_name;

    struct CoalAnalysis{
      double C;
      double H; 
      double O; 
      double N; 
      double S; 
      double CHAR; 
      double ASH; 
      double H2O; 
    };
  
  };
}
#endif 
