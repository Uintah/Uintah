#ifndef Uintah_Component_Arches_FOWYDevol_h
#define Uintah_Component_Arches_FOWYDevol_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/Utility/InverseErrorFunction.h>

//-------------------------------------------------------

/**
 * @class    FOWYDevol
 * @author   Alex Abboud
 * @date     September 2015
 *
 * @brief    This class calculates the FOWY devolatilization rate for coal particles
 *
 * @details  This class calculates the FOWY devolatilization rate for coal, the method
 *           is adapted from the previous implementation in Arches/CoalModels/ to utilize
 *           the nebo formulation of the code here
 */

//-------------------------------------------------------

namespace Uintah{

  //CT is the independent variable type
  //T is the dependent variable type
  template <typename T>
  class FOWYDevol : public TaskInterface {

  public:

    FOWYDevol<T>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~FOWYDevol<T>();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}

      FOWYDevol* build()
      { return scinew FOWYDevol<T>( m_task_name, m_matl_index, _base_var_name, _Nenv ); }

    private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _Nenv;

    };

  protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

  private:

    const std::string _base_var_name;
    std::string _base_gas_var_name;
    std::string _gas_var_name;
    std::string _base_vinf_name;
    std::string _base_raw_coal_name;
    std::string _base_char_mass_name;
    std::string _base_particle_temp_name;
    std::string _base_particle_size_name;
    std::string _base_birth_name;

    const int _Nenv;                 // The number of environments

    //various rate parameters
    double _v_hiT;
    double _A;
    double _Ta;
    double _Tbp_graphite;
    double _T_mu;
    double _T_sigma;
    double _T_hardened_bond;
    double _sigma;

    double _initRawCoalMassFrac;
    double _weightClip;
    double _initRhoP;
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

  template <typename T>
  FOWYDevol<T>::FOWYDevol( std::string task_name, int matl_index,
                               const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _Nenv(N){}

  template <typename T>
  FOWYDevol<T>::~FOWYDevol()
  {}

  template <typename T>
  void FOWYDevol<T>::problemSetup( ProblemSpecP& db ){
    proc0cout << "WARNING: ParticleModels FOWYDevol needs to be made consistent with DQMOM models and use correct DW, use model at your own risk."
      << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n"<< std::endl;
    //required particle properties
    _base_raw_coal_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
    _base_char_mass_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
    _base_particle_size_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    _base_particle_temp_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);

    db->getWithDefault("birth_label",_base_birth_name,"none");
    db->getWithDefault("weight_clip",_weightClip,1.0e-10);
    if ( db->findBlock("gas_source_name") ) {
      db->get("gas_source_name",_gas_var_name);
    } else {
      _gas_var_name = "gas_" + _base_var_name + "tot";
    }

    _base_gas_var_name = "gas_" + _base_var_name;
    _base_vinf_name = "vinf_" + _base_var_name;

    // get coal properties
    CoalHelper& coal_helper = CoalHelper::self();
    CoalHelper::CoalDBInfo& coal_db = coal_helper.get_coal_db();
    _initRhoP = coal_db.rhop_o;
    _pi = coal_db.pi;
    _initRawCoalMassFrac = coal_db.raw_coal_mf;

    const ProblemSpecP db_root = db->getRootNode();
    if ( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){
      ProblemSpecP db_coal_props = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");

      //get devol rate params from coal
      if (db_coal_props->findBlock("FOWYDevol")) {
        ProblemSpecP db_FOWY = db_coal_props->findBlock("FOWYDevol");
        db_FOWY->require("Ta", _Ta);
        db_FOWY->require("A", _A);
        db_FOWY->require("v_hiT", _v_hiT);
        db_FOWY->require("Tbp_graphite", _Tbp_graphite); // 
        db_FOWY->require("T_mu", _T_mu); // 
        db_FOWY->require("T_sigma", _T_sigma); // 
        db_FOWY->require("T_hardened_bond", _T_hardened_bond); // 
        db_FOWY->require("sigma", _sigma);

      } else {
        throw ProblemSetupException("Error: FOWY coefficients missing in <CoalProperties>.", __FILE__, __LINE__);
      }
    }
  }

  template <typename T>
  void FOWYDevol<T>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);

      register_new_variable<T>( name );
      register_new_variable<T>( gas_name );
      register_new_variable<T>( vinf_name );
    }
    register_new_variable<T>( _gas_var_name );
  }

  //======INITIALIZATION:
  template <typename T>
  void FOWYDevol<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);

      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
      register_variable( vinf_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry );
  }

  template <typename T>
  void FOWYDevol<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string vinf_name = get_name(ienv, _base_vinf_name);

      T& devolRate = *(tsk_info->get_uintah_field<T>(name));
      T& gasDevolRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& vInf = *(tsk_info->get_uintah_field<T>(vinf_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        devolRate(i,j,k) = 0.0;
        gasDevolRate(i,j,k) = 0.0;
        vInf(i,j,k) = 0.0;
      });
    }

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>(_gas_var_name));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0.0;
    });

  }

  //======TIME STEP INITIALIZATION:
  template <typename T>
  void FOWYDevol<T>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){
  }

  template <typename T>
  void FOWYDevol<T>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

  //======TIME STEP EVALUATION:
  template <typename T>
  void FOWYDevol<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

    for ( int i = 0; i < _Nenv; i++ ){

      //dependent variables(s) or model values
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);

      register_variable( name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( gas_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( vinf_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      //independent variables
      const std::string weight_name = get_name( i, "w" );
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name );
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );

      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      if (_base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        register_variable( birth_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      }
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  }

  template <typename T>
  void FOWYDevol<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    typedef typename ArchesCore::VariableHelper<T>::ConstType CT;

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>(_gas_var_name));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0.0;
    });

    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();
    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string vinf_name = get_name(ienv, _base_vinf_name);

      T& devolRate    = *(tsk_info->get_uintah_field<T>(name));
      T& gasDevolRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& vInf         = *(tsk_info->get_uintah_field<T>(vinf_name));

      const std::string raw_coal_name = get_name( ienv, _base_raw_coal_name );
      const std::string char_mass_name = get_name( ienv, _base_char_mass_name);
      const std::string particle_temp_name = get_name( ienv, _base_particle_temp_name );
      const std::string particle_size_name = get_name( ienv, _base_particle_size_name );
      const std::string w_name = get_name( ienv, "w" );

      CT& rawCoal  = *(tsk_info->get_const_uintah_field<CT>(raw_coal_name));
      CT& charMass = *(tsk_info->get_const_uintah_field<CT>(char_mass_name));
      CT& partTemp = *(tsk_info->get_const_uintah_field<CT>(particle_temp_name));
      CT& partSize = *(tsk_info->get_const_uintah_field<CT>(particle_size_name));
      CT& weight   = *(tsk_info->get_const_uintah_field<CT>(w_name));

      //Alex wasn't using the birth term yet. This needs to be fixed.
      // ITptr birth;
      // if ( _base_birth_name != "none" ) {
      //   const std::string birth_name = get_name( i, _base_birth_name );
      //   birth = tsk_info->get_const_so_field<CT>(birth_name);
      // }
      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){

        const double initRawCoal = _initRawCoalMassFrac * _initRhoP * _pi / 6.0
                                    * partSize(i,j,k) * partSize(i,j,k) * partSize(i,j,k);
        const double mVol = initRawCoal - ( rawCoal(i,j,k) + charMass(i,j,k) );
        vInf(i,j,k) = 0.5*_v_hiT*(1.0 + std::erf( (partTemp(i,j,k) - _T_mu) / (std::sqrt(2.0) * _T_sigma)));
        vInf(i,j,k) += (partTemp(i,j,k) > _T_hardened_bond) ? (partTemp(i,j,k) - _T_hardened_bond)/_Tbp_graphite : 0.0; // linear contribution
        const double fDrive = std::max( initRawCoal * vInf(i,j,k) - mVol, 0.0 );

        const double clipVal = 0.0; //Placeholder block for adding in generic clipping
        //if (doDQMOM) {
        //  if ( _base_birth_name == "none" ) { //vol = cellVol
        //    clip <<= (*rhsSource + *charRHSSource)/(vol * *weight )
        //  } else {
        //    clip <<= (*rhsSource + *charRHSSource)/((vol + *birth) * *weight )
        //  }
        //}
        //if (doCQMOM) { //only check rate*dt is not greater than volatile mass ??
        //}

        const double rateMax = std::max( fDrive / dt + clipVal, 0.0);
        const double zFact = std::min( std::max( fDrive/ initRawCoal/_v_hiT, 2.5e-5), 1.0-2.5e-5 );
        double einput = 1.0 - 2.0 * zFact;
        const double z = std::sqrt(2.0) * erfinv( einput );

        //rate of devolatilization dmVol/dt
        devolRate(i,j,k) = - std::min( _A * fDrive * std::exp(-(_Ta + z * _sigma)/ partTemp(i,j,k) ), rateMax );

        //check for low values of mass or weights and set rate to 0.0 when it occurs
        bool check = ( weight(i,j,k) < _weightClip ) ? true : false;
        check = ( devolRate(i,j,k) > 0.0 ) ? true : false;
        check = ( rawCoal(i,j,k) + charMass(i,j,k) < 1.e-15 ) ? true : false;
        devolRate(i,j,k) = ( check ) ? 0.0 : -std::min( _A * fDrive * std::exp(-(_Ta + z * _sigma)/ partTemp(i,j,k) ), rateMax );
        gasDevolRate(i,j,k) = - devolRate(i,j,k) * weight(i,j,k);
        gasTotalRate(i,j,k) = gasTotalRate(i,j,k) + gasDevolRate(i,j,k);

      });
    }
  }
}
#endif
