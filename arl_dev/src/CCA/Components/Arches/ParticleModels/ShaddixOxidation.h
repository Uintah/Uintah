#ifndef Uintah_Component_Arches_ShaddixOxidation_h
#define Uintah_Component_Arches_ShaddixOxidation_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

//-------------------------------------------------------

/**
 * @class    ShaddixOxidation
 * @author   Alex Abboud
 * @date     September 2015
 *
 * @brief    This class calculates the Shaddix Char oxidation rate for coal particles
 *
 * @details  This class calculates the Shaddix Char oxidation rate for coal, the method
 *           is adapted from the previous implementation in Arches/CoalModels/ to utilize
 *           the nebo formulation of the code here
 */

//-------------------------------------------------------

namespace Uintah{
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class ShaddixOxidation : public TaskInterface {
    
  public:
    
    ShaddixOxidation<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~ShaddixOxidation<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}
      
      ShaddixOxidation* build()
      { return scinew ShaddixOxidation<IT, DT>( _task_name, _matl_index, _base_var_name, _Nenv ); }
      
    private:
      
      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _Nenv;
      
    };
    
  protected:
    
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
    
  private:
    //resulting model names
    const std::string _base_var_name;
    std::string _base_gas_var_name;
    std::string  _gas_var_name; // save integrated value
    std::string _base_temp_rate_name;
    std::string _base_surface_rate_name;
    std::string _base_pO2_surface_name;
    //required model names
    std::string _base_raw_coal_name;
    std::string _base_char_mass_name;
    std::string _base_particle_temp_name;
    std::string _base_particle_size_name;
    std::string _base_particle_char_prod_name;
    std::string _base_birth_name;
    
    //gas properties
    std::string _gas_o2_name;
    std::string _gas_co2_name;
    std::string _gas_h20_name;
    std::string _gas_n2_name;
    std::string _gas_mw_mix_name;
    std::string _gas_temp_name;
    std::string _gas_density_name;
    
    const int _Nenv;                 // The number of environments
    
    //diffsuion rate parameters
    double _D1;
    double _D2;
    double _D3;
    double _T0;
    //reaction rate parameters
    double _As;
    double _Es;
    double _n;
    //constants
    double _HFCO2;
    double _HFCO;
    double _R;
    double _MWC;
    double _MWO2;
    double _MWCO2;
    double _MWH2O;
    double _MWN2;
    double _small;
    double _pi;
    
    double _weightClip;
 
    const std::string get_name(const int i, const std::string base_name){
      std::stringstream out;
      std::string env;
      out << i;
      env = out.str();
      return base_name + "_" + env;
    }
    
  };
  
  //Function definitions:
  
  template <typename IT, typename DT>
  ShaddixOxidation<IT, DT>::ShaddixOxidation( std::string task_name, int matl_index,
                                              const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _Nenv(N){}
  
  template <typename IT, typename DT>
  ShaddixOxidation<IT, DT>::~ShaddixOxidation()
  {}
  
  template <typename IT, typename DT>
  void ShaddixOxidation<IT, DT>::problemSetup( ProblemSpecP& db ){
    //required particle properties
    _base_raw_coal_name = ParticleTools::parse_for_role_to_label(db, "raw_coal");
    _base_char_mass_name = ParticleTools::parse_for_role_to_label(db, "char");
    _base_particle_size_name = ParticleTools::parse_for_role_to_label(db, "size");
    _base_particle_temp_name = ParticleTools::parse_for_role_to_label(db, "temperature");
    
    db->require("char_production_label",_base_particle_char_prod_name);
    db->getWithDefault("birth_label",_base_birth_name,"none");
    if ( db->findBlock("gas_source_name") ) {
      db->get("gas_source_name",_gas_var_name);
    } else {
      _gas_var_name = "gas_" + _base_var_name + "tot";
    }
    _base_gas_var_name = "gas_" + _base_var_name;
    
    //constants
    _HFCO2 = -393509.0; // J/mol
    _HFCO = -110525.0;
    _R = 8.314; // J/K/mol
    _MWC = 12.0e-3; // kg/mol
    _MWO2 = 32.0; // g/mol
    _MWCO2 = 44.0;
    _MWH2O = 18.0;
    _MWN2 = 28.0;
    _small = 1.0e-30;
    _pi = acos(-1.0);
    
    //binary diffsuion at 293 K
    _D1 = 0.153e-4; // O2-CO2 m^2/s
    _D2 = 0.240e-4; // O2-H2O
    _D3 = 0.219e-4; // O2-N2
    _T0 = 293;

    //required gas properties
    _gas_o2_name = "O2";
    _gas_co2_name = "CO2";
    _gas_h20_name = "H2O";
    _gas_n2_name = "N2";
    _gas_mw_mix_name = "mixture_molecular_weight";
    _gas_temp_name = "temperature";
    _gas_density_name = "densityCP";
    
    db->getWithDefault("weight_clip",_weightClip,1.0e-10);
    
    _base_temp_rate_name = "temp_rate_" + _base_var_name;
    _base_surface_rate_name = "surf_rate_" + _base_var_name;
    _base_pO2_surface_name = "PO2_surf_" + _base_var_name;
    
    const ProblemSpecP db_root = db->getRootNode();
    if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){
      ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
      
      //get rate params from coal
      if (db_coal_props->findBlock("ShaddixChar")) {
        ProblemSpecP db_Shad = db_coal_props->findBlock("ShaddixChar");
        //get reaction rate params
        db_Shad->require("As",_As);
        db_Shad->require("Es",_Es);
        db_Shad->require("n",_n);
      } else {
        throw ProblemSetupException("Error: ShaddixChar Oxidation coefficients missing in <ParticleProperties>.", __FILE__, __LINE__);
      }
    }
  }
  
  template <typename IT, typename DT>
  void ShaddixOxidation<IT, DT>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);
      
      register_new_variable<DT>( name );
      register_new_variable<DT>( gas_name );
      register_new_variable<DT>( temp_rate_name );
      register_new_variable<DT>( surf_rate_name );
      register_new_variable<DT>( pO2_surf_name );
    }
    register_new_variable<DT>( _gas_var_name );
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void ShaddixOxidation<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);
      
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( temp_rate_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( surf_rate_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( pO2_surf_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }
  
  template <typename IT, typename DT>
  void ShaddixOxidation<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                            SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);
      
      DTptr charOxRate = tsk_info->get_so_field<DT>(name);
      DTptr gasCharOxRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr partTRate = tsk_info->get_so_field<DT>(temp_rate_name);
      DTptr surfRate = tsk_info->get_so_field<DT>(surf_rate_name);
      DTptr pO2Surf = tsk_info->get_so_field<DT>(pO2_surf_name);
      
      *charOxRate <<= 0.0;
      *gasCharOxRate <<= 0.0;
      *partTRate <<= 0.0;
      *surfRate <<= 0.0;
      *pO2Surf <<= 0.0;
    }
    DTptr gasTotalRate = tsk_info->get_so_field<DT>( _gas_var_name );
    *gasTotalRate <<= 0;
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void ShaddixOxidation<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void ShaddixOxidation<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                               SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void ShaddixOxidation<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _Nenv; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);
      
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( temp_rate_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( surf_rate_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( pO2_surf_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      
      //independent variables
      const std::string weight_name = get_name( i, "w" );
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name );
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string particle_char_prod_name = get_name( i, _base_particle_char_prod_name );
      
      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_char_prod_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      
      if (_base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        register_variable( birth_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    
    //required gas indep vars
    register_variable( _gas_co2_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_h20_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_o2_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_n2_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_mw_mix_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  }
  
  template <typename IT, typename DT>
  void ShaddixOxidation<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                      SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    
    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();
    
    //gas values
    ITptr CO2 = tsk_info->get_const_so_field<IT>(_gas_co2_name);
    ITptr H2O = tsk_info->get_const_so_field<IT>(_gas_h20_name);
    ITptr O2 = tsk_info->get_const_so_field<IT>(_gas_o2_name);
    ITptr N2 = tsk_info->get_const_so_field<IT>(_gas_n2_name);
    ITptr gasT = tsk_info->get_const_so_field<IT>(_gas_temp_name);
    ITptr gasMW = tsk_info->get_const_so_field<IT>(_gas_mw_mix_name);
    ITptr rhoG = tsk_info->get_const_so_field<IT>(_gas_density_name);
    
    DTptr gasTotalRate = tsk_info->get_so_field<DT>(_gas_var_name);
    *gasTotalRate <<= 0.0;
    //temporary variables used for intermediate calculations
    SpatialOps::SpatFldPtr<DT> pO2Inf = SpatialFieldStore::get<DT>( *gasTotalRate );    //O2 at gas BL
    SpatialOps::SpatFldPtr<DT> areaSum = SpatialFieldStore::get<DT>( *gasTotalRate );    //sum of area of all particles
    SpatialOps::SpatFldPtr<DT> delta = SpatialFieldStore::get<DT>( *gasTotalRate );  //iterative solve param
    SpatialOps::SpatFldPtr<DT> DO2 = SpatialFieldStore::get<DT>( *gasTotalRate );   //diffusino of O2
    SpatialOps::SpatFldPtr<DT> conc = SpatialFieldStore::get<DT>( *gasTotalRate );       //gas concentration
    
    // find gas PO2 area of particles and diffusion, which is the same across all quadrature nodes
    *pO2Inf <<= *O2/( _MWO2/ *gasMW );
    *areaSum <<= 0.0;
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string w_name = get_name( i, "w" );
      ITptr partSize = tsk_info->get_const_so_field<IT>(particle_size_name);
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      
      *areaSum <<= *areaSum + *weight * *partSize * *partSize;
    }
    *delta <<= *pO2Inf / 4.0;
    *DO2 <<= ( *CO2/_MWCO2 + *H2O/_MWH2O + *N2/_MWN2 ) / ( *CO2/(_MWCO2*_D1) + *H2O/(_MWH2O*_D2) + *N2/(_MWN2*_D3) ) * pow(*gasT/_T0,1.5);
    *conc <<= *gasMW * *rhoG * 1000.0;
    
    double nIter = 15; //max number iterations in solver
    double tol = 1.0e-15; //tolerance for iterative solve
    
    for ( int i = 0; i < _Nenv; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);
      
      DTptr charOxRate = tsk_info->get_so_field<DT>(name);
      DTptr gasCharOxRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr partTRate = tsk_info->get_so_field<DT>(temp_rate_name);
      DTptr surfRate = tsk_info->get_so_field<DT>(surf_rate_name);
      DTptr pO2Surf = tsk_info->get_so_field<DT>(pO2_surf_name);
      
      //temporary variables used for intermediate calculations
      SpatialOps::SpatFldPtr<DT> surfAreaFrac = SpatialFieldStore::get<DT>( *charOxRate );    //area fraction for this quad node
      SpatialOps::SpatFldPtr<DT> kS = SpatialFieldStore::get<DT>( *charOxRate ); //rate coefficient
      SpatialOps::SpatFldPtr<DT> pO2SurfGuess = SpatialFieldStore::get<DT>( *charOxRate ); //1st iterative guess
      SpatialOps::SpatFldPtr<DT> pO2SurfOld = SpatialFieldStore::get<DT>( *charOxRate ); //old iterative guess
      SpatialOps::SpatFldPtr<DT> pO2SurfNew = SpatialFieldStore::get<DT>( *charOxRate ); //new iterative guess
      SpatialOps::SpatFldPtr<DT> pO2SurfTmp = SpatialFieldStore::get<DT>( *charOxRate ); //temp val for iterative solve
      SpatialOps::SpatFldPtr<DT> CO2CO = SpatialFieldStore::get<DT>( *charOxRate ); //co2co
      SpatialOps::SpatFldPtr<DT> OF = SpatialFieldStore::get<DT>( *charOxRate ); //OF
      SpatialOps::SpatFldPtr<DT> gamma = SpatialFieldStore::get<DT>( *charOxRate ); //gamma
      SpatialOps::SpatFldPtr<DT> q = SpatialFieldStore::get<DT>( *charOxRate ); //rate of surface oxidation
      SpatialOps::SpatFldPtr<DT> f0 = SpatialFieldStore::get<DT>( *charOxRate ); //slope for iterative solve
      SpatialOps::SpatFldPtr<DT> f1 = SpatialFieldStore::get<DT>( *charOxRate ); //slope for iterative solve
      SpatialOps::SpatFldPtr<DT> gamma1 = SpatialFieldStore::get<DT>( *charOxRate ); //gamma 1
      SpatialOps::SpatFldPtr<DT> maxRateO2Limit = SpatialFieldStore::get<DT>( *charOxRate ); //oxygen limited reaction rate
      SpatialOps::SpatFldPtr<DT> maxRateCharLimit = SpatialFieldStore::get<DT>( *charOxRate ); //char limited reaction rate
      SpatialOps::SpatFldPtr<DT> maxRate = SpatialFieldStore::get<DT>( *charOxRate ); //minimum of limited reaction rates
      SpatialOps::SpatFldPtr<DT> charRate = SpatialFieldStore::get<DT>( *charOxRate ); //actual reaction rate
      
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name);
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string particle_char_prod_name = get_name( i, _base_particle_char_prod_name );
      const std::string w_name = get_name( i, "w" );
      
      ITptr rawCoal = tsk_info->get_const_so_field<IT>(raw_coal_name);
      ITptr charMass = tsk_info->get_const_so_field<IT>(char_mass_name);
      ITptr partTemp = tsk_info->get_const_so_field<IT>(particle_temp_name);
      ITptr partSize = tsk_info->get_const_so_field<IT>(particle_size_name);
      ITptr charProdRate = tsk_info->get_const_so_field<IT>(particle_char_prod_name);
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      
      ITptr birth;
      if ( _base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        birth = tsk_info->get_const_so_field<IT>(birth_name);
      }

      *surfAreaFrac <<= cond( *areaSum < 0.0, *weight * *partSize * *partSize / *areaSum )
                            (0.0);
      *kS <<= _As * exp( -_Es/( _R * *partTemp));
      
      //set up first iterative step to solve PO2Surf & q
      *pO2SurfGuess <<= *pO2Inf/2.0;
      *pO2SurfOld <<= *pO2SurfGuess - *delta;
      *CO2CO <<= 0.02 * pow( *pO2SurfOld, 0.21 ) * exp ( 3070.0/ *partTemp );
      *OF <<= 0.5 * ( 1.0 + *CO2CO * ( 1.0 + *CO2CO ) );
      *gamma <<= -(1.0 - *OF );
      *q <<= *kS * pow( *pO2SurfOld, _n);
      *f0 <<= *pO2SurfOld - *gamma - ( *pO2Inf - *gamma ) * exp( -( *q * *partSize)/(2.0 * *conc * *DO2) );
      
      *pO2SurfNew <<= *pO2SurfGuess + *delta;
      *CO2CO <<= 0.02 * pow( *pO2SurfNew, 0.21 ) * exp ( 3070.0/ *partTemp );
      *OF <<= 0.5 * ( 1.0 + *CO2CO * ( 1.0 + *CO2CO ) );
      *gamma <<= -(1.0 - *OF );
      *q <<= *kS * pow( *pO2SurfNew, _n);
      *f0 <<= *pO2SurfNew - *gamma - ( *pO2Inf - *gamma ) * exp( -( *q * *partSize)/(2.0 * *conc * *DO2) );
      
      for ( int iter = 0; iter < nIter; iter++ ) {
        *pO2SurfTmp <<= *pO2SurfOld;
        *pO2SurfOld <<= *pO2SurfNew;
        *pO2SurfNew <<= *pO2SurfTmp - ( *pO2SurfNew - *pO2SurfTmp )/ ( *f1 - *f0 ) * *f0;
        *pO2SurfNew <<= max( 0.0, min( *pO2Inf, *pO2SurfNew ) );
        
        double tolMax = field_max_interior( *pO2SurfNew - *pO2SurfOld );
        if ( abs(tolMax) < tol ) { //converged solution exit iterations
          *pO2Surf <<= *pO2SurfNew;
          *CO2CO <<= 0.02 * pow( *pO2Surf, 0.21 ) * exp( 3070.0/ *partTemp );
          *OF <<= 0.5 * ( 1.0 + *CO2CO * ( 1.0 + *CO2CO ) );
          *gamma <<= -( 1.0 - *OF );
          *q <<= *kS * pow( *pO2Surf, _n );
          break;
        }
        //redo rate calcualtion for next iteration
        *f0 <<= *f1;
        *CO2CO <<= 0.02 * pow( *pO2SurfNew, 0.21 ) * exp ( 3070.0/ *partTemp );
        *OF <<= 0.5 * ( 1.0 + *CO2CO * ( 1.0 + *CO2CO ) );
        *gamma <<= -( 1.0 - *OF );
        *q <<= *kS * pow( *pO2SurfNew, _n );
        *f1 <<= *pO2SurfNew - *gamma - ( *pO2Inf - *gamma ) * exp( -( *q * *partSize)/(2.0 * *conc * *DO2) );
        *pO2Surf <<= *pO2SurfNew;
      }
      
      //clip values if O2 conc too small or RC+char too small
      *pO2Surf <<= cond( *pO2Inf < 1.0e-12 || (*rawCoal + *charMass) < _small , 0.0 )
                       ( *pO2Surf );
      *CO2CO <<= cond( *pO2Inf < 1.0e-12 || (*rawCoal + *charMass) < _small , 0.0 )
                     ( *CO2CO );
      *q <<= cond( *pO2Inf < 1.0e-12 || (*rawCoal + *charMass) < _small , 0.0 )
                 ( *q );
      
      *gamma1 <<= (_MWC/_MWO2)* ( *CO2CO + 1.0 )/( *CO2CO + 0.5 );
      *maxRateO2Limit <<= max( *O2 * *rhoG * *gamma1 * *surfAreaFrac/ ( dt * *weight ), 0.0 );

      //cliping block to add later
      //if (doDQMOM) { //placeholder for addign clipping
      //  if ( _base_birth_name == "none" ) { //vol = cellVol
      //    *maxRateCharLimit <<= (*rhsSource + *charRHSSource)/(vol * *weight )
      //  } else {
      //    *maxRateCharLimit <<= (*rhsSource + *charRHSSource)/((vol + *birth) * *weight )
      //  }
      //}
      //if (doCQMOM) {
      // *maxRateCharLimit <<=
      //}
      //*maxRate <<= min( *maxRateCharLimit, *maxRateO2Limit );
      
      *maxRate <<= *maxRateO2Limit;
      *charRate <<= min( _pi * *partSize * *partSize * _MWC * *q, *maxRate ); // kg/s
      
      //if small weight or model_val > 0.0, then set all to 0
      *charOxRate <<= cond( *weight < _weightClip || *charRate < 0.0, 0.0)
                          ( - *charRate + *charProdRate );
      *gasCharOxRate <<= cond( *weight < _weightClip || *charRate < 0.0, 0.0 )
                             ( *charRate * *weight );
      *gasTotalRate <<= *gasTotalRate + *gasCharOxRate;
      
      *partTRate <<= cond( *weight < _weightClip || *charRate < 0.0, 0.0 )
                         ( - *charRate/_MWC/( 1.0 + *CO2CO) * ( *CO2CO * _HFCO2 + _HFCO ) );
      *surfRate <<= cond( *weight < _weightClip || *charRate < 0.0, 0.0 )
                        ( - _MWC * *q);
      *pO2Surf <<= cond( *weight < _weightClip, 0.0 )
                       ( *pO2Surf );
//      if ( i == 0 ) {
//        std::cout << "Oxidation Vars---------------------" << std::endl;
//        typename DT::iterator it = charOxRate->interior_begin();
//        std::cout << "charox " << *it << std::endl;
//        it = surfRate->interior_begin();
//        std::cout << "surfrate " << *it << std::endl;
//        it = maxRateO2Limit->interior_begin();
//        std::cout << "O2 limit " << *it << std::endl;
//        it = gamma1->interior_begin();
//        std::cout << "gamma " << *it << std::endl;
//        it = q->interior_begin();
//        std::cout << "q " << *it << std::endl;
//        it = CO2CO->interior_begin();
//        std::cout << "CO2CO " << *it << std::endl;
//        it = pO2Surf->interior_begin();
//        std::cout << "pO2surf " << *it << std::endl;
//        it = f1->interior_begin();
//        std::cout << "f1 " << *it << std::endl;
//        it = f0->interior_begin();
//        std::cout << "f0 " << *it << std::endl;
//        it = q->interior_begin();
//        std::cout << "q " << *it << std::endl;
//        it = DO2->interior_begin();
//        std::cout << "D_O2 " << *it << std::endl;
//        it = conc->interior_begin();
//        std::cout << "conc " << *it << std::endl;
//      }
    }
  }
}
#endif
