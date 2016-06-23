#ifndef Uintah_Component_Arches_FOWYDevol_h
#define Uintah_Component_Arches_FOWYDevol_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/Operators/Operators.h>
#include <spatialops/structured/FVStaggered.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

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
  
  //IT is the independent variable type
  //DT is the dependent variable type
  template <typename IT, typename DT>
  class FOWYDevol : public TaskInterface {
    
  public:
    
    FOWYDevol<IT, DT>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~FOWYDevol<IT, DT>();
    
    void problemSetup( ProblemSpecP& db );
    
    void create_local_labels();
    
    class Builder : public TaskInterface::TaskBuilder {
      
    public:
      
      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      _task_name(task_name), _matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}
      
      FOWYDevol* build()
      { return scinew FOWYDevol<IT, DT>( _task_name, _matl_index, _base_var_name, _Nenv ); }
      
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
    double _Tig;
    double _A;
    double _Ta;
    double _C1;
    double _C2;
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
  
  template <typename IT, typename DT>
  FOWYDevol<IT, DT>::FOWYDevol( std::string task_name, int matl_index,
                               const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _Nenv(N){}
  
  template <typename IT, typename DT>
  FOWYDevol<IT, DT>::~FOWYDevol()
  {}
  
  template <typename IT, typename DT>
  void FOWYDevol<IT, DT>::problemSetup( ProblemSpecP& db ){
    //required particle properties
    _base_raw_coal_name = ParticleTools::parse_for_role_to_label(db, "raw_coal");
    _base_char_mass_name = ParticleTools::parse_for_role_to_label(db, "char");
    _base_particle_size_name = ParticleTools::parse_for_role_to_label(db, "size");
    _base_particle_temp_name = ParticleTools::parse_for_role_to_label(db, "temperature");
    
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
        db_FOWY->require("Tig", _Tig);
        db_FOWY->require("Ta", _Ta);
        db_FOWY->require("A", _A);
        db_FOWY->require("v_hiT", _v_hiT);
        double b, c, d, e;
        db_FOWY->require("b", b);
        db_FOWY->require("c", c);
        db_FOWY->require("d", d);
        db_FOWY->require("e", e);
        _C1 = b + c*_v_hiT;
        _C2 = d + e*_v_hiT;
        db_FOWY->require("sigma", _sigma);
        
      } else {
        throw ProblemSetupException("Error: FOWY coefficients missing in <CoalProperties>.", __FILE__, __LINE__);
      }
    }
  }
  
  template <typename IT, typename DT>
  void FOWYDevol<IT, DT>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);
      
      register_new_variable<DT>( name );
      register_new_variable<DT>( gas_name );
      register_new_variable<DT>( vinf_name );
    }
    register_new_variable<DT>( _gas_var_name );
  }
  
  //======INITIALIZATION:
  template <typename IT, typename DT>
  void FOWYDevol<IT, DT>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
    
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
  
  template <typename IT, typename DT>
  void FOWYDevol<IT,DT>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                    SpatialOps::OperatorDatabase& opr ){
    
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);
      
      DTptr devolRate = tsk_info->get_so_field<DT>(name);
      DTptr gasDevolRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr vInf = tsk_info->get_so_field<DT>(vinf_name);
      
      *devolRate <<= 0.0;
      *gasDevolRate <<= 0.0;
      *vInf <<= 0.0;
    }
    DTptr gasTotalRate = tsk_info->get_so_field<DT>(_gas_var_name);
    *gasTotalRate <<= 0.0;
  }
  
  //======TIME STEP INITIALIZATION:
  template <typename IT, typename DT>
  void FOWYDevol<IT, DT>::register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry ){
  }
  
  template <typename IT, typename DT>
  void FOWYDevol<IT,DT>::timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                                       SpatialOps::OperatorDatabase& opr ){
  }
  
  //======TIME STEP EVALUATION:
  template <typename IT, typename DT>
  void FOWYDevol<IT, DT>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep ){
    
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
      
      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      
      if (_base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        register_variable( birth_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
      }
    }
    register_variable( _gas_var_name, ArchesFieldContainer::COMPUTES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
  }
  
  template <typename IT, typename DT>
  void FOWYDevol<IT,DT>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info,
                              SpatialOps::OperatorDatabase& opr ) {
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatialOps::SpatFldPtr<DT> DTptr;
    typedef SpatialOps::SpatFldPtr<IT> ITptr;
    
    DTptr gasTotalRate = tsk_info->get_so_field<DT>(_gas_var_name);
    *gasTotalRate <<= 0.0;
    
    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();
    for ( int i = 0; i < _Nenv; i++ ){
      
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string vinf_name = get_name(i, _base_vinf_name);
      
      DTptr devolRate = tsk_info->get_so_field<DT>(name);
      DTptr gasDevolRate = tsk_info->get_so_field<DT>(gas_name);
      DTptr vInf = tsk_info->get_so_field<DT>(vinf_name);
      
      //temporary variables used for intermediate calculations
      SpatialOps::SpatFldPtr<DT> mVol = SpatialFieldStore::get<DT>( *devolRate );    //mass volatiles
      SpatialOps::SpatFldPtr<DT> fDrive = SpatialFieldStore::get<DT>( *devolRate );  //driving force
      SpatialOps::SpatFldPtr<DT> zFact = SpatialFieldStore::get<DT>( *devolRate );   //intermediate factor
      SpatialOps::SpatFldPtr<DT> z = SpatialFieldStore::get<DT>( *devolRate );       //intermediate factor
      SpatialOps::SpatFldPtr<DT> rateMax = SpatialFieldStore::get<DT>( *devolRate ); //maximum rate of devolatilization
      SpatialOps::SpatFldPtr<DT> clipVal = SpatialFieldStore::get<DT>( *devolRate ); //additive term to enforce mass conservation
      SpatialOps::SpatFldPtr<DT> initRawCoal = SpatialFieldStore::get<DT>( *devolRate ); //initial raw coal for this particle size
      
      const std::string raw_coal_name = get_name( i, _base_raw_coal_name );
      const std::string char_mass_name = get_name( i, _base_char_mass_name);
      const std::string particle_temp_name = get_name( i, _base_particle_temp_name );
      const std::string particle_size_name = get_name( i, _base_particle_size_name );
      const std::string w_name = get_name( i, "w" );
      
      ITptr rawCoal = tsk_info->get_const_so_field<IT>(raw_coal_name);
      ITptr charMass = tsk_info->get_const_so_field<IT>(char_mass_name);
      ITptr partTemp = tsk_info->get_const_so_field<IT>(particle_temp_name);
      ITptr partSize = tsk_info->get_const_so_field<IT>(particle_size_name);
      ITptr weight = tsk_info->get_const_so_field<IT>(w_name);
      
      ITptr birth;
      if ( _base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        birth = tsk_info->get_const_so_field<IT>(birth_name);
      }
      
      *initRawCoal <<= _initRawCoalMassFrac * _initRhoP * _pi / 6.0 * *partSize * *partSize * *partSize;
      *mVol <<= *initRawCoal - ( *rawCoal + *charMass );
      *vInf <<= 0.5 * _v_hiT * ( 1.0 - tanh( _C1 * ( _Tig - *partTemp)/ *partTemp + _C2));
      *fDrive <<= max( *initRawCoal * *vInf - *mVol, 0.0 );
      
      *clipVal <<= 0.0; //Placeholder block for adding in generic clipping
      //if (doDQMOM) {
      //  if ( _base_birth_name == "none" ) { //vol = cellVol
      //    clip <<= (*rhsSource + *charRHSSource)/(vol * *weight )
      //  } else {
      //    clip <<= (*rhsSource + *charRHSSource)/((vol + *birth) * *weight )
      //  }
      //}
      //if (doCQMOM) { //only check rate*dt is not greater than volatile mass ??
      //}
      
      *rateMax <<= max( *fDrive/dt + *clipVal, 0.0);
      *zFact <<= min( max( *fDrive/ *initRawCoal/_v_hiT, 2.5e-5), 1.0-2.5e-5 );
      *z <<= sqrt(2.0) * inv_erf( 1.0 - 2.0 * *zFact );
      //rate of devolatilization dmVol/dt
      *devolRate <<= - min( _A * *fDrive * exp(-(_Ta + *z * _sigma)/ *partTemp ), *rateMax );
      
      //check for low values of mass or weights and set rate to 0.0 when it occurs
      *devolRate <<= cond( *weight < _weightClip || *devolRate > 0.0 || (*rawCoal + *charMass) < 1.0e-15, 0.0)
                         ( - min( _A * *fDrive * exp(-(_Ta + *z * _sigma)/ *partTemp ), *rateMax ) );
      *gasDevolRate <<= - *devolRate * *weight;
      *gasTotalRate <<= *gasTotalRate + *gasDevolRate;
    }
  }
}
#endif
