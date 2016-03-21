#ifndef Uintah_Component_Arches_ShaddixEnthalpy_h
#define Uintah_Component_Arches_ShaddixEnthalpy_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

//-------------------------------------------------------

/**
 * @class    ShaddixEnthalpy
 * @author   Alex Abboud
 * @date     September 2015
 *
 * @brief    This class calculates the Shaddix Enthalpy rate for coal particles
 *
 * @details  This class calculates the Shaddix Enthalpy rate for coal, the method
 *           is adapted from the previous implementation in Arches/CoalModels/ to utilize
 *           the nebo formulation of the code here
 */

//-------------------------------------------------------

namespace Uintah{
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class ShaddixEnthalpy : public TaskInterface {
    
  public:
    
    ShaddixEnthalpy<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~ShaddixEnthalpy<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}
      
      ShaddixEnthalpy* build()
      { return new ShaddixEnthalpy<IT, DT>( _task_name, _matl_index, _base_var_name, _Nenv ); }
      
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
    std::string _gas_var_name;
    std::string _base_qconv_name;
    std::string _base_qrad_name;
    //required model names
    std::string _base_raw_coal_name;
    std::string _base_char_mass_name;
    std::string _base_particle_temp_name;
    std::string _base_particle_size_name;

    std::string _base_char_oxi_temp_name;
    std::string _base_abskp_name;
    std::string _base_surf_rate_name;
    std::string _base_char_gas_name;
    std::string _base_devol_gas_name;
    std::string _base_u_velocity_name;
    std::string _base_v_velocity_name;
    std::string _base_w_velocity_name;
    
    //gas properties
    std::string _gas_temp_name;
    std::string _gas_cp_name;
    std::string _volq_name;
    std::string _gas_u_velocity_name;
    std::string _gas_v_velocity_name;
    std::string _gas_w_velocity_name;
    std::string _gas_density_name;
    
    const int _Nenv;                 // The number of environments

    bool d_radiateAtGasTemp;
    bool d_radiation;
    //constants

    double _Pr;
    double _sigma;
    double _Rgas;
    double _RdC;
    double _RdMW;
    double _visc;
    double _MWAvg;
    double _ksi;
    double _Hc0;
    double _Hh0;
    double _initRhoP;
    double _weightClip;
    double _initAshMassFrac;
    double _pi;
    
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
  ShaddixEnthalpy<IT, DT>::ShaddixEnthalpy( std::string task_name, int matl_index,
                                             const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _Nenv(N){}
  
  template <typename IT, typename DT>
  ShaddixEnthalpy<IT, DT>::~ShaddixEnthalpy()
  {}
  
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT, DT>::problemSetup( ProblemSpecP& db ){
    //required particle properties
    _base_raw_coal_name = ParticleTools::parse_for_role_to_label(db, "raw_coal");
    _base_char_mass_name = ParticleTools::parse_for_role_to_label(db, "char");
    _base_particle_size_name = ParticleTools::parse_for_role_to_label(db, "size");
    _base_particle_temp_name = ParticleTools::parse_for_role_to_label(db, "temperature");
    _base_u_velocity_name = ParticleTools::parse_for_role_to_label(db, "uvel");
    _base_v_velocity_name = ParticleTools::parse_for_role_to_label(db, "vvel");
    _base_w_velocity_name = ParticleTools::parse_for_role_to_label(db, "wvel");

    //required rates
    db->require("char_temprate_label",_base_char_oxi_temp_name);
    db->require("surf_rate_label",_base_surf_rate_name);
    db->require("char_gas_label",_base_char_gas_name);
    db->require("devol_gas_label",_base_devol_gas_name);
    
    if ( db->findBlock("gas_source_name") ) {
      db->get("gas_source_name",_gas_var_name);
    } else {
      _gas_var_name = "gas_" + _base_var_name + "tot";
    }
    _base_gas_var_name = "gas_" + _base_var_name;
    
    //constants
    _sigma = 5.67e-8; // J/s/m^2/K^4 Stefan Boltzmann constant
    _Rgas = 8314.3; // J/k/kmol
    _Pr = 0.7; // Prandlt number
    db->getWithDefault("weight_clip",_weightClip,1.0e-10);
    
    //required gas properties
    _gas_temp_name = "temperature";
    _gas_cp_name = "specificheat";
    _volq_name = "radiationVolq";
    _gas_u_velocity_name = "CCUVelocity";
    _gas_v_velocity_name = "CCVVelocity";
    _gas_w_velocity_name = "CCWVelocity";
    _gas_density_name = "densityCP";
    
    //resulting other variables
    _base_qconv_name = "qconv_" + _base_var_name;
    _base_qrad_name = "qrad_" + _base_var_name;
    
    // get coal properties
    CoalHelper& coal_helper = CoalHelper::self();
    CoalHelper::CoalDBInfo& coal_db = coal_helper.get_coal_db();
    _initRhoP = coal_db.rhop_o;
    _pi = coal_db.pi;
    _initAshMassFrac = coal_db.ash_mf;
    _MWAvg = coal_db.mw_avg;
    _Hc0 = coal_db.h_c0;
    _Hh0 = coal_db.h_ch0;
    _ksi = coal_db.ksi;
    _RdC = _Rgas/12.0;
    _RdMW = _Rgas/_MWAvg;
    
    // check for a radiation model:
    d_radiation = false;
    SourceTermFactory& srcs = SourceTermFactory::self();
    if ( srcs.source_type_exists("do_radiation") ){
      d_radiation = true;
    }
    if ( srcs.source_type_exists( "rmcrt_radiation") ){
      d_radiation = true;
    }
    
    //user can specifically turn off radiative heat transfer
    if (db->findBlock("noRadiation"))
      d_radiation = false;
    
    if (d_radiation ) {
      ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels");
      for ( ProblemSpecP db_model = db_prop->findBlock("model"); db_model != 0;
           db_model = db_model->findNextBlock("model")){
        std::string modelName;
        db_model->getAttribute("type", modelName);
        if (modelName=="radiation_properties"){
          if  (db_model->findBlock("calculator") == 0){
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
            d_radiation = false;
            break;
          }else if(db_model->findBlock("calculator")->findBlock("particles") == 0){
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
            d_radiation = false;
            break;
          }
          db_model->findBlock("calculator")->findBlock("particles")->findBlock("abskp")->getAttribute("label",_base_abskp_name);
          db_model->findBlock("calculator")->findBlock("particles")->getWithDefault( "radiateAtGasTemp", d_radiateAtGasTemp, true );
          break;
        }
        if  (db_model== 0){
          proc0cout <<"\n///-------------------------------------------///\n";
          proc0cout <<"WARNING: No radiation particle properties computed!\n";
          proc0cout <<"Particles will not interact with radiation!\n";
          proc0cout <<"///-------------------------------------------///\n";
          d_radiation = false;
          break;
        }
      }
    }
    
    // check for viscosity
    if (db->getRootNode()->findBlock("PhysicalConstants")) {
      ProblemSpecP db_phys = db->getRootNode()->findBlock("PhysicalConstants");
      db_phys->require("viscosity", _visc);
      if( _visc == 0 ) {
        throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
      }
    } else {
      throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
    }

  }
  
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT, DT>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);
      
      register_new_variable<DT>( name );
      register_new_variable<DT>( gas_name );
      register_new_variable<DT>( qconv_name );
      register_new_variable<DT>( qrad_name );
    }
    register_new_variable<DT>( _gas_var_name );
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);
      
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( qconv_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( qrad_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }
  
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                           SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);
      
      DTptr heatRate = tsk_info->get_so_field<DT>(name);
      DTptr gasHeatRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr qConv = tsk_info->get_so_field<DT>(qconv_name);
      DTptr qRad = tsk_info->get_so_field<DT>(qrad_name);
      
      *heatRate <<= 0.0;
      *gasHeatRate <<= 0.0;
      *qConv <<= 0.0;
      *qRad <<= 0.0;
    }
    DTptr gasTotalRate = tsk_info->get_so_field<DT>(_gas_var_name);
    *gasTotalRate <<= 0.0;
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                              SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
    
    for ( int i = 0; i < _Nenv; i++ ){
      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);
      
      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( qconv_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( qrad_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      
      //independent variables
      const std::string weight_name = get_name( i, "w" );
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name );
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string char_oxi_temp_name = get_name( i, _base_char_oxi_temp_name );
      const std::string surf_rate_name = get_name( i, _base_surf_rate_name );
      const std::string char_gas_name = get_name( i, _base_char_gas_name );
      const std::string devol_gas_name = get_name( i, _base_devol_gas_name );
      const std::string u_velocity_name = get_name( i, _base_u_velocity_name );
      const std::string v_velocity_name = get_name( i, _base_v_velocity_name );
      const std::string w_velocity_name = get_name( i, _base_w_velocity_name );

      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( char_oxi_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( surf_rate_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( char_gas_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( devol_gas_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( u_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( v_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( w_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      
      if ( d_radiation ) {
        const std::string abskp_name = get_name( i, _base_abskp_name );
        register_variable( abskp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

    //required gas indep vars
    register_variable( _gas_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_cp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_u_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_v_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_w_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( _gas_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    if ( d_radiation ) {
      register_variable( _volq_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }
  }
  
  template <typename IT, typename DT>
  void ShaddixEnthalpy<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                     SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    
    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();
    
    //gas values
    ITptr gasT = tsk_info->get_const_so_field<IT>(_gas_temp_name);
    ITptr gasCP = tsk_info->get_const_so_field<IT>(_gas_cp_name);
    ITptr rhoG = tsk_info->get_const_so_field<IT>(_gas_density_name);
    ITptr velU = tsk_info->get_const_so_field<IT>(_gas_u_velocity_name);
    ITptr velV = tsk_info->get_const_so_field<IT>(_gas_v_velocity_name);
    ITptr velW = tsk_info->get_const_so_field<IT>(_gas_w_velocity_name);
    ITptr volQ;
    if ( d_radiation )
      volQ = tsk_info->get_const_so_field<IT>(_volq_name);
    
    DTptr gasTotalRate = tsk_info->get_so_field<DT>( _gas_var_name );
    *gasTotalRate <<= 0.0;
    
    for ( int i = 0; i < _Nenv; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);
      
      DTptr heatRate = tsk_info->get_so_field<DT>(name);
      DTptr gasHeatRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr qConv = tsk_info->get_so_field<DT>(qconv_name);
      DTptr qRad = tsk_info->get_so_field<DT>(qrad_name);
      
      //temporary variables used for intermediate calculations
      SpatialOps::SpatFldPtr<DT> deltaV = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> Re = SpatialFieldStore::get<DT>( *heatRate );    //Reynolds number
      SpatialOps::SpatFldPtr<DT> Nu = SpatialFieldStore::get<DT>( *heatRate );    //Prandtl number
      SpatialOps::SpatFldPtr<DT> filmT = SpatialFieldStore::get<DT>( *heatRate );   //ave temperature of gas/particle
      SpatialOps::SpatFldPtr<DT> rkg = SpatialFieldStore::get<DT>( *heatRate );   //gas termal conductivity
      SpatialOps::SpatFldPtr<DT> kappa = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> blow = SpatialFieldStore::get<DT>( *heatRate );  //blowign factor
      SpatialOps::SpatFldPtr<DT> deltaT = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> alphaRC = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> alphaCP = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> cpAsh = SpatialFieldStore::get<DT>( *heatRate ); //heat capacity ash
      SpatialOps::SpatFldPtr<DT> cpCoal = SpatialFieldStore::get<DT>( *heatRate ); //heat capacity coal
      SpatialOps::SpatFldPtr<DT> initAsh = SpatialFieldStore::get<DT>( *heatRate ); //initial ash mass of this quad node
      SpatialOps::SpatFldPtr<DT> maxQConv = SpatialFieldStore::get<DT>( *heatRate ); //max possible heat flux from convection
      SpatialOps::SpatFldPtr<DT> Eb = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> FSum = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> maxQRad = SpatialFieldStore::get<DT>( *heatRate );  //max possible heat flux from radiation
      SpatialOps::SpatFldPtr<DT> hInt = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> hC = SpatialFieldStore::get<DT>( *heatRate );
      SpatialOps::SpatFldPtr<DT> qRxn = SpatialFieldStore::get<DT>( *heatRate ); //heat flux from char oxidation

      //paritcle variables rqd
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name);
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string char_oxi_temp_name = get_name( i, _base_char_oxi_temp_name );
      const std::string surf_rate_name = get_name( i, _base_surf_rate_name );
      const std::string char_gas_name = get_name( i, _base_char_gas_name );
      const std::string devol_gas_name = get_name( i, _base_devol_gas_name );
      const std::string w_name = get_name( i, "w" );
      const std::string u_velocity_name = get_name( i, _base_u_velocity_name );
      const std::string v_velocity_name = get_name( i, _base_v_velocity_name );
      const std::string w_velocity_name = get_name( i, _base_w_velocity_name );

      ITptr rawCoal = tsk_info->get_const_so_field<IT>(raw_coal_name);
      ITptr charMass = tsk_info->get_const_so_field<IT>(char_mass_name);
      ITptr partTemp = tsk_info->get_const_so_field<IT>(particle_temp_name);
      ITptr partSize = tsk_info->get_const_so_field<IT>(particle_size_name);
      ITptr charOxiTemp = tsk_info->get_const_so_field<IT>(char_oxi_temp_name);
      ITptr surfRate = tsk_info->get_const_so_field<IT>(surf_rate_name);
      ITptr charGas = tsk_info->get_const_so_field<IT>(char_gas_name);
      ITptr devolGas = tsk_info->get_const_so_field<IT>(devol_gas_name);
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      ITptr partVelU = tsk_info->get_const_so_field<IT>(u_velocity_name);
      ITptr partVelV = tsk_info->get_const_so_field<IT>(v_velocity_name);
      ITptr partVelW = tsk_info->get_const_so_field<IT>(w_velocity_name);
      
      //solve convection flux term
      *deltaV <<= sqrt( ( *velU - *partVelU ) * ( *velU - *partVelU ) + ( *velV - *partVelV ) * ( *velV - *partVelV ) + ( *velW - *partVelW ) * ( *velW - *partVelW ) );
      *Re <<= *deltaV * *partSize * *rhoG / _visc;
      *Nu <<= 2.0 + 0.65 * sqrt( *Re ) * pow( _Pr, 1.0/3.0 );
      *filmT <<= ( *partTemp + *gasT )/ 2.0;
      *rkg <<= cond( *filmT < 300.0, 0.0262 )
                   ( *filmT > 1200.0, 0.07184 * pow( *filmT/1200.0, 0.58 ) )
                   ( -2.32575758e-8 * *filmT * *filmT + 8.52627273e-5 * *filmT + 3.88709091e-3 );
      // the old rkg code uses an interpolation in this range with data points
      // double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
      // double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
      // this quadratic correlation is a python polyfit on the data with R^2 = 0.99994
      
      *kappa <<= - *surfRate * *partSize * *gasCP / ( 2.0 * *rkg );
      *blow <<= cond( abs( exp( *kappa ) - 1.0 ) < 1.0e-16, 1.0)
                    ( *kappa/( exp( *kappa ) - 1.0 ) );
      *deltaT <<= ( *gasT - *partTemp );
      *qConv <<= *Nu * _pi * *blow * *rkg * *partSize * *deltaT; // J/s
      //clip convective term if too large
      *alphaRC <<= *rawCoal + *charMass;
      *initAsh <<= _initAshMassFrac * _pi/6.0 * *partSize * *partSize * *partSize * _initRhoP;
      *cpAsh <<= 754.0 + 0.586 * *partTemp;
      *cpCoal <<= _RdMW * ( (144400.0 * exp( 380.0/ *partTemp)/( *partTemp * *partTemp * (exp(380.0/ *partTemp) - 1.0) * (exp(380.0/ *partTemp) - 1.0) ) ) +
                            (6480000.0 * exp( 1800.0/ *partTemp)/( *partTemp * *partTemp * (exp(1800.0/ *partTemp) - 1.0) * (exp(1800.0/ *partTemp) - 1.0) )) );
      *alphaCP <<= *cpCoal * *alphaRC + *cpAsh * *initAsh;
      *maxQConv <<= *alphaCP * ( *deltaT/ dt );
      *qConv <<= cond( abs(*qConv) > abs(*maxQConv), *maxQConv )
                     ( *qConv );
      
      //solve radition flux term
      *qRad <<= 0.0;
      if ( d_radiation ) {
        const std::string abskp_name = get_name( i, _base_abskp_name );
        ITptr absKp = tsk_info->get_const_so_field<IT>(abskp_name);
        if ( d_radiateAtGasTemp ) {
          *Eb <<= 4.0 * _sigma * pow( *gasT, 4.0);
          *maxQRad <<= (pow( *volQ / ( 4.0 * _sigma), 0.25 ) - *gasT)/ dt * *alphaCP;
        } else {
          *Eb <<= 4.0 * _sigma * pow( *partTemp, 4.0);
          *maxQRad <<= (pow( *volQ / ( 4.0 * _sigma), 0.25 ) - *partTemp)/ dt * *alphaCP;
        }
        *FSum <<= *volQ;
        *qRad <<= *absKp * ( *FSum - *Eb );
        //check for maximum radiation value
        *qRad <<= cond( abs(*qRad) > abs(*maxQRad), *maxQRad )
                      ( *qRad );
      }
      //integrated value
      *hInt <<= -156.076 + 380.0/( exp(380.0/ *partTemp) - 1.0) + 3600.0/( exp(1800.0/ *partTemp) - 1.0 );
      *hC <<= _Hc0 + *hInt * _RdMW;
      *qRxn <<= *charOxiTemp;
      
      //get combined heat flux
      *heatRate <<= cond( *weight < _weightClip, 0.0)
                        (*qConv + *qRad + _ksi * *qRxn - ( *devolGas + *charGas ) * *hC);
      *gasHeatRate <<= cond( *weight < _weightClip, 0.0)
                           ( - *weight * *heatRate);
      *gasTotalRate <<= cond( *weight < _weightClip, 0.0)
                            (*gasTotalRate + *gasHeatRate );

//      if ( i == 0) {
//        std::cout << "Enathalpy Vars---------------------" << std::endl;
//        typename DT::iterator it = heatRate->interior_begin();
//        std::cout << "total " << *it << std::endl;
//        it = qConv->interior_begin();
//        std::cout << "Conv " << *it << std::endl;
//        it = qRad->interior_begin();
//        std::cout << "Rad " << *it << std::endl;
//        it = qRxn->interior_begin();
//        std::cout << "Rxn " << *it << std::endl;
//        it = hC->interior_begin();
//        std::cout << "hC " << *it << std::endl;
//        it = devolGas->interior_begin();
//        std::cout << "devol " << *it << std::endl;
//        it = charGas->interior_begin();
//        std::cout << "char " << *it << std::endl;
//        it = Re->interior_begin();
//        std::cout << "Re " << *it << std::endl;
//        it = rkg->interior_begin();
//        std::cout << "rkg " << *it << std::endl;
//        it = kappa->interior_begin();
//        std::cout << "kappa " << *it << std::endl;
//        it = blow->interior_begin();
//        std::cout << "blow " << *it << std::endl;
//        it = surfRate->interior_begin();
//        std::cout << "surfRate " << *it << std::endl;
//      }
      
    }
  }
}
#endif


