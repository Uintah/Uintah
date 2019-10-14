#ifndef Uintah_Component_Arches_ShaddixOxidation_h
#define Uintah_Component_Arches_ShaddixOxidation_h

//#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
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

  template <typename T>
  class ShaddixOxidation : public TaskInterface {

  public:

    ShaddixOxidation<T>( std::string task_name, int matl_index, const std::string var_name, const int N );
    ~ShaddixOxidation<T>();

    void problemSetup( ProblemSpecP& db );

    void create_local_labels();

    class Builder : public TaskInterface::TaskBuilder {

    public:

      Builder( std::string task_name, int matl_index, std::string base_var_name, const int N ) :
      m_task_name(task_name), m_matl_index(matl_index), _base_var_name(base_var_name), _Nenv(N){}
      ~Builder(){}

      ShaddixOxidation* build()
      { return scinew ShaddixOxidation<T>( m_task_name, m_matl_index, _base_var_name, _Nenv ); }

    private:

      std::string m_task_name;
      int m_matl_index;
      std::string _base_var_name;
      std::string _base_gas_var_name;
      const int _Nenv;

    };

  protected:

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks);

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){}

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks);

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){};

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

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

  template <typename T>
  ShaddixOxidation<T>::ShaddixOxidation( std::string task_name, int matl_index,
                                              const std::string base_var_name, const int N ) :
  TaskInterface( task_name, matl_index ), _base_var_name(base_var_name), _Nenv(N){}

  template <typename T>
  ShaddixOxidation<T>::~ShaddixOxidation()
  {}

  template <typename T>
  void ShaddixOxidation<T>::problemSetup( ProblemSpecP& db ){
    proc0cout << "WARNING: ParticleModels ShaddixOxidation needs to be made consistent with DQMOM models and use correct DW, use model at your own risk."
      << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n" << "\n"<< std::endl;
    //required particle properties
    _base_raw_coal_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);
    _base_char_mass_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);
    _base_particle_size_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    _base_particle_temp_name = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);

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

  template <typename T>
  void ShaddixOxidation<T>::create_local_labels(){
    for ( int i = 0; i < _Nenv; i++ ){
      const std::string name = get_name(i, _base_var_name);
      const std::string gas_name = get_name(i, _base_gas_var_name);
      const std::string temp_rate_name = get_name(i, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(i, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(i, _base_pO2_surface_name);

      register_new_variable<T>( name );
      register_new_variable<T>( gas_name );
      register_new_variable<T>( temp_rate_name );
      register_new_variable<T>( surf_rate_name );
      register_new_variable<T>( pO2_surf_name );
    }
    register_new_variable<T>( _gas_var_name );
  }

  //======INITIALIZATION:
  template <typename T>
  void ShaddixOxidation<T>::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry , const bool packed_tasks){

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

  template <typename T>
  void ShaddixOxidation<T>::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string temp_rate_name = get_name(ienv, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(ienv, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(ienv, _base_pO2_surface_name);

      T& charOxRate    = *(tsk_info->get_uintah_field<T>(name));
      T& gasCharOxRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& partTRate     = *(tsk_info->get_uintah_field<T>(temp_rate_name));
      T& surfRate      = *(tsk_info->get_uintah_field<T>(surf_rate_name));
      T& pO2Surf       = *(tsk_info->get_uintah_field<T>(pO2_surf_name));

      Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
      Uintah::parallel_for( range, [&](int i, int j, int k){
        charOxRate(i,j,k) = 0.0;
        gasCharOxRate(i,j,k) = 0.0;
        partTRate(i,j,k) = 0.0;
        surfRate(i,j,k) = 0.0;
        pO2Surf(i,j,k) = 0.0;
      });
    }

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>( _gas_var_name ));
    Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0;
    });
  }

  //======TIME STEP EVALUATION:
  template <typename T>
  void ShaddixOxidation<T>::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks){

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

      register_variable( weight_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( raw_coal_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( char_mass_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_temp_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_size_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
      register_variable( particle_char_prod_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );

      if (_base_birth_name != "none" ) {
        const std::string birth_name = get_name( i, _base_birth_name );
        register_variable( birth_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
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

  template <typename T>
  void ShaddixOxidation<T>::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ){

    //typedef typename ArchesCore::VariableHelper<T>::ConstType CT;
    //**NOTE: This typedef wasn't behaving properly so I have commented it out for now. Some
    //        future person should fix it.
    typedef constCCVariable<double> CT;

    //timestep size need for rate clipping
    const double dt = tsk_info->get_dt();

    //gas values
    CT& CO2   = *(tsk_info->get_const_uintah_field<CT>(_gas_co2_name));
    CT& H2O   = *(tsk_info->get_const_uintah_field<CT>(_gas_h20_name));
    CT& O2    = *(tsk_info->get_const_uintah_field<CT>(_gas_o2_name));
    CT& N2    = *(tsk_info->get_const_uintah_field<CT>(_gas_n2_name));
    CT& gasT  = *(tsk_info->get_const_uintah_field<CT>(_gas_temp_name));
    CT& gasMW = *(tsk_info->get_const_uintah_field<CT>(_gas_mw_mix_name));
    CT& rhoG  = *(tsk_info->get_const_uintah_field<CT>(_gas_density_name));

    T& gasTotalRate = *(tsk_info->get_uintah_field<T>(_gas_var_name));
    Uintah::BlockRange ecrange(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
    Uintah::parallel_for( ecrange, [&](int i, int j, int k){
      gasTotalRate(i,j,k) = 0.0;
    });

    double pO2Inf = 0.;
    double areaSum = 0.;

    Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() );
    Uintah::parallel_for( range, [&](int i, int j, int k){
    // find gas PO2 area of particles and diffusion, which is the same across all quadrature nodes
      pO2Inf = O2(i,j,k)/( _MWO2/ gasMW(i,j,k) );
    });

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){
      const std::string particle_size_name = get_name( ienv, _base_particle_size_name );
      const std::string w_name = get_name( ienv, "w" );
      CT& partSize = *(tsk_info->get_const_uintah_field<CT>(particle_size_name));
      CT& weight   = *(tsk_info->get_const_uintah_field<CT>(w_name));

      Uintah::parallel_for( range, [&](int i, int j, int k){
        areaSum = areaSum + weight(i,j,k) * partSize(i,j,k) * partSize(i,j,k);
      });

    }

    double delta = 0.;
    double DO2 = 0.;
    double conc = 0.;

    Uintah::parallel_for( range, [&](int i, int j, int k){
      delta = pO2Inf / 4.0;
      DO2 = ( CO2(i,j,k)/_MWCO2 + H2O(i,j,k)/_MWH2O + N2(i,j,k)/_MWN2 ) /
            ( CO2(i,j,k)/(_MWCO2 * _D1) + H2O(i,j,k)/(_MWH2O * _D2)
            + N2(i,j,k)/(_MWN2 * _D3) ) * std::pow(gasT(i,j,k)/_T0,1.5);
      conc = gasMW(i,j,k) * rhoG(i,j,k) * 1000.0;
    });

    double nIter = 15; //max number iterations in solver
    double tol = 1.0e-15; //tolerance for iterative solve

    for ( int ienv = 0; ienv < _Nenv; ienv++ ){

      const std::string name = get_name(ienv, _base_var_name);
      const std::string gas_name = get_name(ienv, _base_gas_var_name);
      const std::string temp_rate_name = get_name(ienv, _base_temp_rate_name);
      const std::string surf_rate_name = get_name(ienv, _base_surface_rate_name);
      const std::string pO2_surf_name = get_name(ienv, _base_pO2_surface_name);

      T& charOxRate    = *(tsk_info->get_uintah_field<T>(name));
      T& gasCharOxRate = *(tsk_info->get_uintah_field<T>(gas_name));
      T& partTRate     = *(tsk_info->get_uintah_field<T>(temp_rate_name));
      T& surfRate      = *(tsk_info->get_uintah_field<T>(surf_rate_name));
      T& pO2Surf       = *(tsk_info->get_uintah_field<T>(pO2_surf_name));

      const std::string raw_coal_name = get_name( ienv, _base_raw_coal_name );
      const std::string char_mass_name = get_name( ienv, _base_char_mass_name);
      const std::string particle_temp_name = get_name( ienv, _base_particle_temp_name );
      const std::string particle_size_name = get_name( ienv, _base_particle_size_name );
      const std::string particle_char_prod_name = get_name( ienv, _base_particle_char_prod_name );
      const std::string w_name = get_name( ienv, "w" );

      CT& rawCoal      = *(tsk_info->get_const_uintah_field<CT>(raw_coal_name));
      CT& charMass     = *(tsk_info->get_const_uintah_field<CT>(char_mass_name));
      CT& partTemp     = *(tsk_info->get_const_uintah_field<CT>(particle_temp_name));
      CT& partSize     = *(tsk_info->get_const_uintah_field<CT>(particle_size_name));
      CT& charProdRate = *(tsk_info->get_const_uintah_field<CT>(particle_char_prod_name));
      CT& weight       = *(tsk_info->get_const_uintah_field<CT>(w_name));

      //CT* birthPtr;
      //if ( _base_birth_name != "none" ) {
        //const std::string birth_name = get_name( ienv, _base_birth_name );
        //birthPtr = tsk_info->get_const_uintah_field<CT>(birth_name);
      //}
      //CT& birth = *birthPtr; //not used

      Uintah::parallel_for( range, [&](int i, int j, int k){

        const double pT = partTemp(i,j,k);
        const double pS = partSize(i,j,k);
        const double surfAreaFrac = ( areaSum < 0.0 ) ?  weight(i,j,k) * pS * pS / areaSum : 0.0;
        const double kS = _As * std::exp( -_Es/( _R * pT));


        //set up first iterative step to solve PO2Surf & q
        double pO2SurfGuess = pO2Inf/2.0;
        double pO2SurfOld   = pO2SurfGuess - delta;
        double CO2CO = 0.02 * std::pow( pO2SurfOld, 0.21 ) * std::exp( 3070.0/ pT );
        double OF = 0.5 * ( 1.0 + CO2CO * ( 1.0 + CO2CO ) );
        double gamma = -(1.0 - OF );
        double q = kS * std::pow( pO2SurfOld, _n);
        double f0 = pO2SurfOld - gamma - ( pO2Inf - gamma ) * std::exp( -( q * pS/(2.0 * conc * DO2) ));
        double f1 = 0.;

        double pO2SurfNew = pO2SurfGuess + delta;
        CO2CO = 0.02 * std::pow( pO2SurfNew, 0.21 ) * exp ( 3070.0/ pT );
        OF = 0.5 * ( 1.0 + CO2CO * ( 1.0 + CO2CO ) );
        gamma = OF - 1.0; //-(1.0 - *OF );
        q = kS * std::pow( pO2SurfNew, _n);
        f0 = pO2SurfNew - gamma - ( pO2Inf - gamma ) * std::exp( -( q * pS)/(2.0 * conc * DO2) );

        for ( int iter = 0; iter < nIter; iter++ ) {

          const double pO2SurfTmp = pO2SurfOld;
          pO2SurfOld = pO2SurfNew;
          pO2SurfNew = pO2SurfTmp - ( pO2SurfNew - pO2SurfTmp )/ ( f1 - f0 ) * f0;
          pO2SurfNew = std::max( 0.0, std::min( pO2Inf, pO2SurfNew ) );

          if ( std::abs(pO2SurfNew-pO2SurfOld) < tol ) { //converged solution exit iterations
            pO2Surf(i,j,k) = pO2SurfNew;
            CO2CO = 0.02 * std::pow( pO2SurfNew, 0.21 ) * std::exp( 3070.0/ partTemp(i,j,k) );
            OF = 0.5 * ( 1.0 + CO2CO * ( 1.0 + CO2CO ) );
            gamma = OF - 1.0; //-( 1.0 - OF );
            q = kS * std::pow( pO2SurfNew, _n );
            break;
          }
          //redo rate calcualtion for next iteration
          f0 = f1;
          CO2CO = 0.02 * std::pow( pO2SurfNew, 0.21 ) * std::exp( 3070.0/ pT );
          OF = 0.5 * ( 1.0 + CO2CO * ( 1.0 + CO2CO ) );
          gamma = -( 1.0 - OF );
          q = kS * std::pow( pO2SurfNew, _n );
          f1 = pO2SurfNew - gamma - ( pO2Inf - gamma ) * std::exp( -( q * pS )/(2.0 * conc * DO2) );
          pO2Surf(i,j,k) = pO2SurfNew;
        }

        bool small_pO2inf = pO2Inf < 1.e-12 ? true : false;
        bool small_coal = rawCoal(i,j,k) + charMass(i,j,k) < _small ? true : false;

        pO2Surf(i,j,k) = ( small_pO2inf ) ? 0.0 : pO2Surf(i,j,k);
        pO2Surf(i,j,k) = ( small_coal )   ? 0.0 : pO2Surf(i,j,k);

        CO2CO = ( small_pO2inf ) ? 0.0 : CO2CO;
        CO2CO = ( small_coal )   ? 0.0 : CO2CO;

        q = ( small_pO2inf ) ? 0.0 : q;
        q = ( small_coal )   ? 0.0 : q;

        const double gamma1 = (_MWC/_MWO2) * ( CO2CO + 1.0 ) / ( CO2CO + 0.5 );
        const double maxRateO2Limit = std::max( O2(i,j,k) * rhoG(i,j,k) * gamma1 * surfAreaFrac
                                                / ( dt * weight(i,j,k) ), 0.0 );

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

        const double maxRate = maxRateO2Limit;
        double charRate = std::min( _pi * pS * pS * _MWC * q, maxRate ); // kg/s

        //if small weight or model_val > 0.0, then set all to 0
        bool small_weight = ( weight(i,j,k) < _weightClip ) ? true : false;
        bool small_rate   = ( charRate < 0.0 ) ? true : false;

        charOxRate(i,j,k) = ( small_weight ) ? 0.0 : -charRate + charProdRate(i,j,k);
        charOxRate(i,j,k) = ( small_rate   ) ? 0.0 : -charRate + charProdRate(i,j,k);

        gasCharOxRate(i,j,k) = ( small_weight ) ?  0.0 : charRate * weight(i,j,k);
        gasCharOxRate(i,j,k) = ( small_rate   ) ?  0.0 : charRate * weight(i,j,k);

        gasTotalRate(i,j,k) = gasTotalRate(i,j,k) + gasCharOxRate(i,j,k);


        partTRate(i,j,k) = ( small_weight ) ?  0.0 : - charRate/_MWC/( 1.0 + CO2CO) * ( CO2CO * _HFCO2 + _HFCO );
        partTRate(i,j,k) = ( small_rate   ) ?  0.0 : - charRate/_MWC/( 1.0 + CO2CO) * ( CO2CO * _HFCO2 + _HFCO );


        surfRate(i,j,k) = ( small_weight ) ?  0.0 : -_MWC * q;
        surfRate(i,j,k) = ( small_rate   ) ?  0.0 : -_MWC * q;

        pO2Surf(i,j,k) = ( small_weight ) ?  0.0 : pO2Surf(i,j,k);

      });
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
} //namespace Uintah
#endif
