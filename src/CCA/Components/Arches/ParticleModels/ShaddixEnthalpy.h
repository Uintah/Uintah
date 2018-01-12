#ifndef Uintah_Component_Arches_ShaddixEnthalpy_h
#define Uintah_Component_Arches_ShaddixEnthalpy_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <CCA/Components/Arches/GridTools.h>

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
  template <typename T>
  class ShaddixEnthalpy : public TaskInterface {

  public:

    ShaddixEnthalpy<T>( std::string task_name, int matl_index,
                        const std::string var_name, const int N ) :
                        TaskInterface(task_name, matl_index),
                        _base_var_name(var_name), _Nenv(N){}
    ~ShaddixEnthalpy<T>(){}

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}

      ShaddixEnthalpy* build()
      { return scinew ShaddixEnthalpy<T>( _task_name, _matl_index, _base_var_name, _Nenv ); }

    private:

      std::string _task_name;
      int _matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _Nenv;

    };

  protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

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

  }; //class ShaddixEnthalpy

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::problemSetup( ProblemSpecP& db ){
    proc0cout << "WARNING: ParticleModels ShaddixEnthalpy needs to be made consistent with DQMOM models and use correct DW, use model at your own risk."
      << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n"<< std::endl;
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
      for ( ProblemSpecP db_model = db_prop->findBlock("model"); db_model != nullptr; db_model = db_model->findNextBlock("model")){
        std::string modelName;
        db_model->getAttribute("type", modelName);
        if (modelName=="radiation_properties"){
          if( db_model->findBlock("calculator") == nullptr ) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
            d_radiation = false;
            break;
          }
          else if( db_model->findBlock("calculator")->findBlock("particles") == nullptr ){
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
        if ( db_model == nullptr ){
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
    }
    else {
      throw InvalidValue("ERROR: EnthalpyShaddix: problemSetup(): Missing <PhysicalConstants> section in input file!",__FILE__,__LINE__);
    }

  }

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string qconv_name = get_name(i, _base_qconv_name);
      const std::string qrad_name = get_name(i, _base_qrad_name);

      register_new_variable<T>( name );
      register_new_variable<T>( gas_name );
      register_new_variable<T>( qconv_name );
      register_new_variable<T>( qrad_name );
    }
    register_new_variable<T>( _gas_var_name );
  }

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

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

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string qconv_name = get_name(ienv, _base_qconv_name);
      const std::string qrad_name = get_name(ienv, _base_qrad_name);

      T& heatRate    = *(tsk_info->get_uintah_field<T>(name));
      T& gasHeatRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& qConv       = *(tsk_info->get_uintah_field<T>(qconv_name));
      T& qRad        = *(tsk_info->get_uintah_field<T>(qrad_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        heatRate(i,j,k) = 0.0;
        gasHeatRate(i,j,k) = 0.0;
        qConv(i,j,k) = 0.0;
        qRad(i,j,k) = 0.0;

      });
    }

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>(_gas_var_name));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0.0;
    });

  }

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

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

      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( char_oxi_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( surf_rate_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( char_gas_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( devol_gas_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( u_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( v_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( w_velocity_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      if ( d_radiation ) {
        const std::string abskp_name = get_name( i, _base_abskp_name );
        register_variable( abskp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, variable_registry, time_substep );

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

//--------------------------------------------------------------------------------------------------
  template <typename T>
  void ShaddixEnthalpy<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();

    //gas values
    CT& gasT  = *(tsk_info->get_const_uintah_field<CT>(_gas_temp_name));
    CT& gasCP = *(tsk_info->get_const_uintah_field<CT>(_gas_cp_name));
    CT& rhoG  = *(tsk_info->get_const_uintah_field<CT>(_gas_density_name));
    CT& velU  = *(tsk_info->get_const_uintah_field<CT>(_gas_u_velocity_name));
    CT& velV  = *(tsk_info->get_const_uintah_field<CT>(_gas_v_velocity_name));
    CT& velW  = *(tsk_info->get_const_uintah_field<CT>(_gas_w_velocity_name));
    CT* volQPtr = NULL;
    if ( d_radiation ){
      volQPtr = tsk_info->get_const_uintah_field<CT>(_volq_name);
    }
    CT& volQ = *volQPtr;

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>( _gas_var_name ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0.0;
    });

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string qconv_name = get_name(ienv, _base_qconv_name);
      const std::string qrad_name = get_name(ienv, _base_qrad_name);

      T& heatRate    = *(tsk_info->get_uintah_field<T>(name));
      T& gasHeatRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& qConv       = *(tsk_info->get_uintah_field<T>(qconv_name));
      T& qRad        = *(tsk_info->get_uintah_field<T>(qrad_name));

      //paritcle variables rqd
      const std::string raw_coal_name = get_name( ienv, _base_raw_coal_name );
      const std::string char_mass_name = get_name( ienv, _base_char_mass_name);
      const std::string particle_temp_name = get_name( ienv, _base_particle_temp_name );
      const std::string particle_size_name = get_name( ienv, _base_particle_size_name );
      const std::string char_oxi_temp_name = get_name( ienv, _base_char_oxi_temp_name );
      const std::string surf_rate_name = get_name( ienv, _base_surf_rate_name );
      const std::string char_gas_name = get_name( ienv, _base_char_gas_name );
      const std::string devol_gas_name = get_name( ienv, _base_devol_gas_name );
      const std::string w_name = get_name( ienv, "w" );
      const std::string u_velocity_name = get_name( ienv, _base_u_velocity_name );
      const std::string v_velocity_name = get_name( ienv, _base_v_velocity_name );
      const std::string w_velocity_name = get_name( ienv, _base_w_velocity_name );
      const std::string abskp_name = get_name( ienv, _base_abskp_name );

      CT* absKpPtr = NULL;
      if ( d_radiation ){
        absKpPtr = tsk_info->get_const_uintah_field<CT>(abskp_name);
      }
      CT& absKp = *absKpPtr;
      CT& rawCoal     = *(tsk_info->get_const_uintah_field<CT>(raw_coal_name));
      CT& charMass    = *(tsk_info->get_const_uintah_field<CT>(char_mass_name));
      CT& partTemp    = *(tsk_info->get_const_uintah_field<CT>(particle_temp_name));
      CT& partSize    = *(tsk_info->get_const_uintah_field<CT>(particle_size_name));
      CT& charOxiTemp = *(tsk_info->get_const_uintah_field<CT>(char_oxi_temp_name));
      CT& surfRate    = *(tsk_info->get_const_uintah_field<CT>(surf_rate_name));
      CT& charGas     = *(tsk_info->get_const_uintah_field<CT>(char_gas_name));
      CT& devolGas    = *(tsk_info->get_const_uintah_field<CT>(devol_gas_name));
      CT& weight      = *(tsk_info->get_const_uintah_field<CT>(w_name));
      CT& partVelU    = *(tsk_info->get_const_uintah_field<CT>(u_velocity_name));
      CT& partVelV    = *(tsk_info->get_const_uintah_field<CT>(v_velocity_name));
      CT& partVelW    = *(tsk_info->get_const_uintah_field<CT>(w_velocity_name));


      Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        //solve convection flux term
        const double deltaV = std::sqrt( ( velU(i,j,k) - partVelU(i,j,k) )
                                       * ( velU(i,j,k) - partVelU(i,j,k) )
                                       + ( velV(i,j,k) - partVelV(i,j,k) )
                                       * ( velV(i,j,k) - partVelV(i,j,k) )
                                       + ( velW(i,j,k) - partVelW(i,j,k) )
                                       * ( velW(i,j,k) - partVelW(i,j,k) ) );
        const double Re = deltaV * partSize(i,j,k) * rhoG(i,j,k) / _visc;
        const double Nu = 2.0 + 0.65 * std::sqrt( Re ) * std::pow( _Pr, 1.0/3.0 );
        const double filmT = ( partTemp(i,j,k) + gasT(i,j,k) )/ 2.0;
        const double dfault_rkg = -2.32575758e-8 * filmT * filmT + 8.52627273e-5
                                * filmT + 3.88709091e-3;
        double rkg = (filmT < 300.0) ? 0.0262 : dfault_rkg;
        rkg = (filmT > 1200.0 ) ? 0.07184 * std::pow( filmT/1200.0, 0.58 ) : dfault_rkg;

        // the old rkg code uses an interpolation in this range with data points
        // double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
        // double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
        // this quadratic correlation is a python polyfit on the data with R^2 = 0.99994

        const double kappa = -surfRate(i,j,k) * partSize(i,j,k) * gasCP(i,j,k) / ( 2.0 * rkg );
        const double blow  = ( std::abs( std::exp( kappa ) - 1.0 ) < 1.0e-16 ) ?  1.0 :
                             ( kappa/( std::exp( kappa ) - 1.0 ) );

        const double deltaT = ( gasT(i,j,k) - partTemp(i,j,k) );
        qConv(i,j,k) = Nu * _pi * blow * rkg * partSize(i,j,k) * deltaT; // J/s

        //clip convective term if too large
        const double alphaRC = rawCoal(i,j,k) + charMass(i,j,k);
        const double initAsh = _initAshMassFrac * _pi/6.0
                              * partSize(i,j,k) * partSize(i,j,k) * partSize(i,j,k) * _initRhoP;
        const double cpAsh = 754.0 + 0.586 * partTemp(i,j,k);
        const double cpCoal = _RdMW * ( (144400.0 * std::exp( 380.0/ partTemp(i,j,k)) /
                              ( partTemp(i,j,k) * partTemp(i,j,k)
                              * (std::exp(380.0/ partTemp(i,j,k)) - 1.0)
                              * (std::exp(380.0/ partTemp(i,j,k)) - 1.0) ) ) +
                              (6480000.0 * std::exp( 1800.0/ partTemp(i,j,k)) /
                              ( partTemp(i,j,k) * partTemp(i,j,k)
                              * (std::exp(1800.0/ partTemp(i,j,k)) - 1.0)
                              * (std::exp(1800.0/ partTemp(i,j,k)) - 1.0) )) );
        const double alphaCP = cpCoal * alphaRC + cpAsh * initAsh;
        const double maxQConv = alphaCP * ( deltaT/ dt );
        qConv(i,j,k) = std::abs(qConv(i,j,k)) > std::abs(maxQConv) ? maxQConv : qConv(i,j,k);

        //solve radition flux term
        qRad(i,j,k) = 0.0;
        if ( d_radiation ) {

          double maxQRad = 0.0;
          double Eb = 0.0;
          if ( d_radiateAtGasTemp ) {

            Eb = 4.0 * _sigma * std::pow( gasT(i,j,k), 4.0);
            maxQRad = (std::pow( volQ(i,j,k) / ( 4.0 * _sigma), 0.25 )
                       - gasT(i,j,k))/ dt * alphaCP;

          } else {

            Eb = 4.0 * _sigma * std::pow( partTemp(i,j,k), 4.0);
            maxQRad = (std::pow( volQ(i,j,k) / ( 4.0 * _sigma), 0.25 ) - partTemp(i,j,k))/ dt
                      * alphaCP;

          }
          const double FSum = volQ(i,j,k);
          qRad(i,j,k) = absKp(i,j,k) * ( FSum - Eb );

          //check for maximum radiation value
          qRad(i,j,k) = std::abs(qRad(i,j,k)) > std::abs(maxQRad) ?  maxQRad : qRad(i,j,k);
        }

        //integrated value
        const double hInt = -156.076 + 380.0 / ( std::exp(380.0/partTemp(i,j,k)) - 1.0)
                            + 3600.0/( std::exp(1800.0/ partTemp(i,j,k)) - 1.0 );
        const double hC = _Hc0 + hInt * _RdMW;
        const double qRxn = charOxiTemp(i,j,k);

        //get combined heat flux
        heatRate(i,j,k) = (weight(i,j,k) < _weightClip ) ?  0.0 :
                          qConv(i,j,k) + qRad(i,j,k) + _ksi * qRxn
                          - ( devolGas(i,j,k) + charGas(i,j,k) ) * hC;

        gasHeatRate(i,j,k) = (weight(i,j,k) < _weightClip) ?  0.0 : -weight(i,j,k) * heatRate(i,j,k);

        gasTotalRate(i,j,k) = (weight(i,j,k) < _weightClip) ? 0.0 : gasTotalRate(i,j,k) + gasHeatRate(i,j,k);

//        if ( i == 0) {
//          std::cout << "Enathalpy Vars---------------------" << std::endl;
//          typename DT::iterator it = heatRate->interior_begin();
//          std::cout << "total " << *it << std::endl;
//          it = qConv->interior_begin();
//          std::cout << "Conv " << *it << std::endl;
//          it = qRad->interior_begin();
//          std::cout << "Rad " << *it << std::endl;
//          it = qRxn->interior_begin();
//          std::cout << "Rxn " << *it << std::endl;
//          it = hC->interior_begin();
//          std::cout << "hC " << *it << std::endl;
//          it = devolGas->interior_begin();
//          std::cout << "devol " << *it << std::endl;
//          it = charGas->interior_begin();
//          std::cout << "char " << *it << std::endl;
//          it = Re->interior_begin();
//          std::cout << "Re " << *it << std::endl;
//          it = rkg->interior_begin();
//          std::cout << "rkg " << *it << std::endl;
//          it = kappa->interior_begin();
//          std::cout << "kappa " << *it << std::endl;
//          it = blow->interior_begin();
//          std::cout << "blow " << *it << std::endl;
//          it = surfRate->interior_begin();
//          std::cout << "surfRate " << *it << std::endl;
//        }
      });
    }
  }
}
#endif
