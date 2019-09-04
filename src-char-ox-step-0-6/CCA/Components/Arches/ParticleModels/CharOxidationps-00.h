#ifndef Uintah_Component_Arches_CharOxidationps_h
#define Uintah_Component_Arches_CharOxidationps_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <CCA/Components/Arches/ParticleModels/CharOxidationpsHelper.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#include <Core/Util/Timers/Timers.hpp>

#define SQUARE(x) x*x
#define CUBE(x)   x*x*x

namespace Uintah {

  template <typename T>
  class CharOxidationps : public TaskInterface {

public:

    CharOxidationps( std::string task_name, int matl_index );
    ~CharOxidationps();

    void problemSetup( ProblemSpecP& db );

    void register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_init( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks );

    void register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep, const bool packed_tasks );

    void register_compute_bcs( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep, const bool packed_tasks ){}

    void compute_bcs( const Patch* patch, ArchesTaskInfoManager* tsk_info ){}

    void initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void timestep_init( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void eval( const Patch* patch, ArchesTaskInfoManager* tsk_info );

    void create_local_labels();

    //Build instructions for this (CharOxidationps) class.
    class Builder : public TaskInterface::TaskBuilder {

      public:

      Builder( std::string task_name, int matl_index ) : _task_name(task_name), _matl_index(matl_index){}
      ~Builder(){}

      CharOxidationps* build()
      { return scinew CharOxidationps<T>( _task_name, _matl_index ); }

      private:

      std::string _task_name;
      int         _matl_index;
    };

private:

    // constants
    double _R;      // [J/ (K mol) ]
    double _R_cal;  // [cal/ (K mol) ]
    double _HF_CO2; // [J/mol]
    double _HF_CO;  // [J/mol]
    double _T0;
    double _tau;    // tortuosity

    int m_nQn_part;

    // model name lists
    std::vector< std::string > m_modelLabel;
    std::vector< std::string > m_gasLabel;
    std::vector< std::string > m_particletemp;
    std::vector< std::string > m_particleSize;
    std::vector< std::string > m_surfacerate;
    //std::vector< std::string > m_PO2surf;
    std::vector< std::string > m_devolRC;
    std::vector< std::string > m_surfAreaF_name;

    //bool m_add_rawcoal_birth;
    //bool m_add_length_birth;
    //bool m_add_char_birth;

    std::vector< std::string > m_char_birth_qn_name;
    std::vector< std::string > m_rawcoal_birth_qn_name;
    std::vector< std::string > m_length_birth_qn_name;

    // other models name lists
    std::vector< std::string > m_particle_temperature;
    std::vector< std::string > m_particle_length;
    //std::vector< std::string > m_particle_length_qn;
    std::vector< std::string > m_particle_density;
    std::vector< std::string > m_rcmass;
    std::vector< std::string > m_char_name;
    std::vector< std::string > m_weight_name;
    std::vector< std::string > m_weightqn_name;
    std::vector< std::string > m_up_name;
    std::vector< std::string > m_vp_name;
    std::vector< std::string > m_wp_name;
    std::string                number_density_name;

    // RHS
    //std::vector< std::string > m_RC_RHS;
    //std::vector< std::string > m_ic_RHS;
    //std::vector< std::string > m_w_RHS;
    //std::vector< std::string > m_length_RHS;

    // scaling constant
    double              _weight_small;
    std::vector<double> m_weight_scaling_constant;
    std::vector<double> m_char_scaling_constant;
    std::vector<double> m_RC_scaling_constant;
    std::vector<double> m_length_scaling_constant;

    //reactions
    std::vector<bool>        _use_co2co_l;
    std::vector<std::string> _oxid_l;
    std::vector<double>      _MW_l;
    std::vector<double>      _a_l;
    std::vector<double>      _e_l;
    std::vector<double>      _phi_l;
    std::vector<double>      _hrxn_l;
    std::vector<std::string> m_reaction_rate_names;

    std::vector<std::vector<double>> _D_mat;
    std::vector<double>              _MW_species;
    std::vector<int>                 _oxidizer_indices;
    double _S;
    int    _NUM_reactions{0}; //

    double _p_void0; //
    double _rho_ash_bulk; //
    double _Sg0; //
    double _Mh; // 12 kg carbon / kmole carbon
    double _init_particle_density; //
    int    _NUM_species; //
    std::vector<std::string> _species_names;

    double _ksi; // [J/mol]
    std::vector< double > m_mass_ash; //
    std::vector< double > m_rho_org_bulk; //
    std::vector< double > m_p_voidmin;

    // parameter from other model
    double _v_hiT;
    double _dynamic_visc; // [kg/(m s)]
    double _gasPressure; // [ J / m^3 ] or [ N /m^2]

    // gas variables names
    std::string m_u_vel_name;
    std::string m_v_vel_name;
    std::string m_w_vel_name;

    std::string m_cc_u_vel_name;
    std::string m_cc_v_vel_name;
    std::string m_cc_w_vel_name;
    std::string m_density_gas_name;
    std::string m_gas_temperature_label;
    std::string m_MW_name;
  };

//--------------------------------------------------------------------------------------------------
template<typename T>
CharOxidationps<T>::CharOxidationps( std::string task_name
                                   , int         matl_index
                                   )
  : TaskInterface( task_name, matl_index )
{
}

//--------------------------------------------------------------------------------------------------
template<typename T>
CharOxidationps<T>::~CharOxidationps()
{
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::problemSetup( ProblemSpecP & db
                                )
{
  // Set constants
  // Enthalpy of formation (J/mol)
  _HF_CO2 = -393509.0;
  _HF_CO  = -110525.0;

  // ideal gas constants
  _R_cal = 1.9872036; // [cal/ (K mol) ]
  _R     = 8.314472;  // [J/ (K mol) ]

  //binary diffsuion at 293 K
  _T0  = 293.0;
  _tau = 1.9598; // tortuosity

  m_nQn_part = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );

  // get Char source term label and devol label from the devolatilization model
  // model  variables
  std::string modelName         = _task_name; //"char_rate";
  std::string delvol_model_name = "devol_rate";
  std::string surfAreaF_root    = "surfaceAreaFraction";

  for ( int l = 0; l < m_nQn_part; l++ ) {

    // Create a label for this model
    m_modelLabel.push_back( ArchesCore::append_env( modelName, l ) );

    // Create the gas phase source term associated with this model
    std::string gasLabel_temp = modelName + "_gasSource";
    m_gasLabel.push_back( ArchesCore::append_env( gasLabel_temp, l ) );

    // Create the particle temperature source term associated with this model
    std::string particletemp_temp = modelName + "_particletempSource";
    m_particletemp.push_back( ArchesCore::append_env( particletemp_temp, l ) );

    // Create the particle size source term associated with this model
    std::string particleSize_temp = modelName + "_particleSizeSource";
    m_particleSize.push_back( ArchesCore::append_env( particleSize_temp, l ) );

    // Create the char oxidation surface rate term associated with this model
    std::string surfacerate_temp = modelName + "_surfacerate";
    m_surfacerate.push_back( ArchesCore::append_env( surfacerate_temp, l ) );

    // Create the char oxidation PO2 surf term associated with this model
    //std::string PO2surf_temp = modelName + "_PO2surf";
    //m_PO2surf.push_back( ArchesCore::append_env( PO2surf_temp, l ) );

    m_devolRC.push_back( ArchesCore::append_env( delvol_model_name, l ) );

    m_surfAreaF_name.push_back( ArchesCore::append_env( surfAreaF_root, l ) );
  }

  _gasPressure = 101325.; // Fix this

  // gas variables
  m_u_vel_name       = ArchesCore::parse_ups_for_role( ArchesCore::UVELOCITY, db, "uVelocitySPBC" );
  m_v_vel_name       = ArchesCore::parse_ups_for_role( ArchesCore::VVELOCITY, db, "vVelocitySPBC" );
  m_w_vel_name       = ArchesCore::parse_ups_for_role( ArchesCore::WVELOCITY, db, "wVelocitySPBC" );
  m_density_gas_name = ArchesCore::parse_ups_for_role( ArchesCore::DENSITY,   db, "density" );

  m_cc_u_vel_name = m_u_vel_name + "_cc";
  m_cc_v_vel_name = m_v_vel_name + "_cc";
  m_cc_w_vel_name = m_w_vel_name + "_cc";

  m_gas_temperature_label = "temperature";
  m_MW_name               = "mixture_molecular_weight";

  // particle variables

  // check for particle temperature
  std::string temperature_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_TEMPERATURE );

  // check for length
  std::string length_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_SIZE );

  // Need a particle density
  std::string density_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_DENSITY );

  // create raw coal mass var label
  std::string rcmass_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_RAWCOAL );

  // check for char mass and get scaling constant
  std::string char_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_CHAR );
  number_density_name   = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_TOTNUM_DENSITY );

  // check for particle velocity
  std::string up_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_XVEL );
  std::string vp_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_YVEL );
  std::string wp_root = ArchesCore::parse_for_particle_role_to_label( db, ArchesCore::P_ZVEL );

  for ( int l = 0; l < m_nQn_part; l++ ) {
    m_particle_temperature.push_back( ArchesCore::append_env( temperature_root, l ) );
    m_particle_length.push_back     ( ArchesCore::append_env( length_root,      l ) );
    //m_particle_length_qn.push_back  ( ArchesCore::append_qn_env( length_root,   l ) );
    m_particle_density.push_back    ( ArchesCore::append_env( density_root,     l ) );
    m_rcmass.push_back              ( ArchesCore::append_env( rcmass_root,      l ) );
    m_char_name.push_back           ( ArchesCore::append_env( char_root,        l ) );
    m_weight_name.push_back         ( ArchesCore::append_env( "w",              l ) );
    m_up_name.push_back             ( ArchesCore::append_env( up_root,          l ) );
    m_vp_name.push_back             ( ArchesCore::append_env( vp_root,          l ) );
    m_wp_name.push_back             ( ArchesCore::append_env( wp_root,          l ) );
  }

  //for ( int l = 0; l < m_nQn_part; l++ ) {

  //  std::string RC_RHS = ArchesCore::append_qn_env( rcmass_root, l ) + "_RHS";
  //  m_RC_RHS.push_back( RC_RHS );

  //  std::string ic_RHS = ArchesCore::append_qn_env( char_root, l ) + "_RHS";
  //  m_ic_RHS.push_back( ic_RHS );

  //  std::string w_RHS = ArchesCore::append_qn_env( "w", l ) + "_RHS";
  //  m_w_RHS.push_back( w_RHS );

  //  std::string length_RHS = ArchesCore::append_qn_env( length_root, l ) + "_RHS";
  //  m_length_RHS.push_back( length_RHS );
  //}

  //const std::string rawcoal_birth_name = rcmass_eqn.get_model_by_type( "BirthDeath" );
  //const std::string char_birth_name    = "char_BirthDeath";
  //const std::string rawcoal_birth_name = "rcmass_BirthDeath";
  //const std::string length_birth_name  = "length_BirthDeath";

  // birth terms

  //m_add_rawcoal_birth = false;
  //m_add_length_birth  = false;
  //m_add_char_birth    = false;

  //if ( char_birth_name != "NULLSTRING" ) {
  //  m_add_char_birth = true;
  //}

  //if ( rawcoal_birth_name != "NULLSTRING" ) {
  //  m_add_rawcoal_birth = true;
  //}

  //if ( length_birth_name != "NULLSTRING" ) {
  //  m_add_length_birth = true;
  //}


  //for ( int l = 0; l < m_nQn_part; l++ ) {

  //  if ( m_add_char_birth ) {
  //    std::string char_birth_qn_name = ArchesCore::append_qn_env( char_birth_name, l );
  //    m_char_birth_qn_name.push_back( char_birth_qn_name );
  //  }

  //  if ( m_add_rawcoal_birth ) {
  //    std::string rawcoal_birth_qn_name = ArchesCore::append_qn_env( rawcoal_birth_name, l );
  //    m_rawcoal_birth_qn_name.push_back( rawcoal_birth_qn_name );
  //  }

  //  if ( m_add_length_birth ) {
  //    std::string length_birth_qn_name = ArchesCore::append_qn_env( length_birth_name, l );
  //    m_length_birth_qn_name.push_back( length_birth_qn_name );
  //  }
  //}

  // scaling constants
  //_weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_small = 1e-15; // fix

  for ( int l = 0; l < m_nQn_part; l++ ) {

    //_weight_scaling_constant = weight_eqn.getScalingConstant( d_quadNode );
    //_char_scaling_constant   = char_eqn.getScalingConstant  ( d_quadNode );

    m_weight_scaling_constant.push_back( 1. );
    m_char_scaling_constant.push_back  ( 1. );
    m_RC_scaling_constant.push_back    ( 1. );
    m_length_scaling_constant.push_back( 1. );
  }

  // model global constants
  // get model coefficients
  std::string oxidizer_name;
  double      oxidizer_MW; //
  double      a; //
  double      e; //
  double      phi; //
  double      hrxn; //
  bool        use_co2co;

  const ProblemSpecP params_root = db->getRootNode();
  CoalHelper& coal_helper = CoalHelper::self();

  if ( params_root->findBlock( "PhysicalConstants" ) ) {
    ProblemSpecP db_phys = params_root->findBlock( "PhysicalConstants" );
    db_phys->require( "viscosity", _dynamic_visc );
  }
  else {
    throw InvalidValue( "Error: Missing <PhysicalConstants> section in input file required for Smith Char Oxidation model.", __FILE__, __LINE__ );
  }

  ProblemSpecP db_coal_props = params_root->findBlock( "CFD" )->findBlock( "ARCHES" )->findBlock( "ParticleProperties" );
  std::string particleType;
  db_coal_props->getAttribute( "type", particleType );

  if ( particleType != "coal" ) {
    throw InvalidValue( "ERROR: CharOxidationSmith2016: Can't use particles of type: " + particleType, __FILE__, __LINE__ );
  }

  if ( db_coal_props->findBlock( "FOWYDevol" ) ) {
    ProblemSpecP db_BT = db_coal_props->findBlock( "FOWYDevol" );
    db_BT->require( "v_hiT", _v_hiT ); //
  }
  else {
    throw ProblemSetupException( "Error: CharOxidationSmith2016 requires FOWY v_hiT.", __FILE__, __LINE__ );
  }

  ProblemSpecP db_part_properties = params_root->findBlock( "CFD" )->findBlock( "ARCHES" )->findBlock( "ParticleProperties" );
  db_part_properties->getWithDefault( "ksi",           _ksi,          1 );      // Fraction of the heat released by char oxidation that goes to the particle
  db_part_properties->getWithDefault( "rho_ash_bulk",  _rho_ash_bulk, 2300.0 );
  db_part_properties->getWithDefault( "void_fraction", _p_void0,      0.3 );

  if ( _p_void0 == 1. ) {
    throw ProblemSetupException( "Error: CharOxidationSmith2016, Given initial conditions for particles p_void0 is 1!! This will give NaN.", __FILE__, __LINE__ );
  }

  if ( _p_void0 <= 0. ) {
    throw ProblemSetupException( "Error: CharOxidationSmith2016, Given initial conditions for particles p_void0 <= 0 !! ", __FILE__, __LINE__ );
  }

  if ( db_coal_props->findBlock( "SmithChar2016" ) ) {

    ProblemSpecP db_Smith = db_coal_props->findBlock( "SmithChar2016" );

    db_Smith->getWithDefault( "Sg0",     _Sg0, 9.35e5 ); // UNCERTAIN initial specific surface area [m^2/kg], range [1e3,1e6]
    db_Smith->getWithDefault( "char_MW", _Mh,  12.0 );   // kg char / kmole char

    _init_particle_density = ArchesCore::get_inlet_particle_density( db );

    double ash_mass_frac = coal_helper.get_coal_db().ash_mf; 

    for ( int l = 0; l < m_nQn_part; l++ ) {

      double initial_diameter = ArchesCore::get_inlet_particle_size( db, l );
      double p_volume         = M_PI / 6. * CUBE( initial_diameter ); // particle volume [m^3]
      double mass_ash         = p_volume * _init_particle_density * ash_mass_frac;
      double initial_rc       = ( M_PI / 6.0 ) * CUBE( initial_diameter ) * _init_particle_density * ( 1. - ash_mass_frac );
      double rho_org_bulk     = initial_rc / ( p_volume * ( 1 - _p_void0 ) - mass_ash / _rho_ash_bulk );                            // bulk density of char [kg/m^3]
      double p_voidmin        = 1. - ( 1 / p_volume ) * ( initial_rc * ( 1. - _v_hiT ) / rho_org_bulk + mass_ash / _rho_ash_bulk ); // bulk density of char [kg/m^3]

      m_mass_ash.push_back    ( mass_ash );
      m_rho_org_bulk.push_back( rho_org_bulk );
      m_p_voidmin.push_back   ( p_voidmin );
    }

    db_Smith->getWithDefault( "surface_area_mult_factor", _S, 1.0 );

    _NUM_species = 0;

    for ( ProblemSpecP db_species = db_Smith->findBlock( "species" ); db_species != nullptr; db_species = db_species->findNextBlock( "species" ) ) {

      std::string new_species = db_species->getNodeValue();
      //helper.add_lookup_species( new_species );
      _species_names.push_back( new_species );

      _NUM_species += 1;
    }

    // reactions

    for ( ProblemSpecP db_reaction = db_Smith->findBlock( "reaction" ); db_reaction != nullptr; db_reaction = db_reaction->findNextBlock( "reaction" ) ) {

      //get reaction rate params
      db_reaction->require       ( "oxidizer_name",             oxidizer_name );
      db_reaction->require       ( "oxidizer_MW",               oxidizer_MW );
      db_reaction->require       ( "pre_exponential_factor",    a );
      db_reaction->require       ( "activation_energy",         e );
      db_reaction->require       ( "stoich_coeff_ratio",        phi );
      db_reaction->require       ( "heat_of_reaction_constant", hrxn );
      db_reaction->getWithDefault( "use_co2co",                 use_co2co, false );

      _use_co2co_l.push_back( use_co2co );
      _MW_l.push_back       ( oxidizer_MW );
      _oxid_l.push_back     ( oxidizer_name );
      _a_l.push_back        ( a );
      _e_l.push_back        ( e );
      _phi_l.push_back      ( phi );
      _hrxn_l.push_back     ( hrxn );

      _NUM_reactions += 1;
    }

    ChemHelper& helper = ChemHelper::self();

    diffusion_terms binary_diff_terms; // this allows access to the binary diff coefficients etc, in the header file.
    int table_size = binary_diff_terms.num_species;

    // find indices specified by user.
    std::vector<int> specified_indices;
    bool check_species;

    for ( int j = 0; j < _NUM_species; j++ ) {

      check_species = true;

      for ( int i = 0; i < table_size; i++ ) {
        if ( _species_names[j] == binary_diff_terms.sp_name[i] ) {
          specified_indices.push_back( i );
          check_species = false;
        }
      }

      if ( check_species ) {
        throw ProblemSetupException( "Error: Species specified in SmithChar2016 oxidation species, not found in SmithChar2016 data-base (please add it).", __FILE__, __LINE__ );
      }
    }

    std::vector<double> temp_v;

    for ( int i = 0; i < _NUM_species; i++ ) {

      temp_v.clear();

      _MW_species.push_back( binary_diff_terms.MW_sp[specified_indices[i]] );
      helper.add_lookup_species( _species_names[i] ); // request all indicated species from table

      for ( int j = 0; j < _NUM_species; j++ ) {
        temp_v.push_back( binary_diff_terms.D_matrix[specified_indices[i]][specified_indices[j]] );
      }

      _D_mat.push_back( temp_v );
    }

    // find index of the oxidizers.
    for ( int reac = 0; reac < _NUM_reactions; reac++ ) {
      for ( int spec = 0; spec < _NUM_species; spec++ ) {
        if ( _oxid_l[reac] == _species_names[spec] ) {
          _oxidizer_indices.push_back( spec );
        }
      }
    }

    for ( int l = 0; l < m_nQn_part; l++ ) {
      for ( int r = 0; r < _NUM_reactions; r++ ) {
        std::string rate_name = "char_gas_reaction" + std::to_string(r) + "_qn" + std::to_string(l);
        m_reaction_rate_names.push_back( rate_name );
      }
    }
  } // end if ( db_coal_props->findBlock( "SmithChar2016" ) )
}


//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::create_local_labels()
{
  for ( int l = 0; l < m_nQn_part; l++ ) {
    register_new_variable< T >( m_modelLabel[l] );
    register_new_variable< T >( m_gasLabel[l] );
    register_new_variable< T >( m_particletemp[l] );
    register_new_variable< T >( m_particleSize[l] );
    register_new_variable< T >( m_surfacerate[l] );
    //register_new_variable< T >( m_PO2surf[l] );
  }

  int nrtq = _NUM_reactions * m_nQn_part; // number of reaction x number of env

  for ( int l = 0; l < nrtq; l++ ) {
    register_new_variable< T >( m_reaction_rate_names[l] );
  }
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::register_initialize(       std::vector<ArchesFieldContainer::VariableInformation> & variable_registry
                                       , const bool                                                     packed_tasks
                                       )
{
  for ( int l = 0; l < m_nQn_part; l++ ) {
    register_variable( m_modelLabel[l],   ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_gasLabel[l],     ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_particletemp[l], ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_particleSize[l], ArchesFieldContainer::COMPUTES, variable_registry );
    register_variable( m_surfacerate[l],  ArchesFieldContainer::COMPUTES, variable_registry );
    //register_variable( m_PO2surf[l],      ArchesFieldContainer::COMPUTES, variable_registry );
  }

  int nrtq = _NUM_reactions * m_nQn_part; // number of reaction x number of env

  for ( int l = 0; l < nrtq; l++ ) {
    register_variable( m_reaction_rate_names[l], ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::initialize( const Patch* patch
                              , ArchesTaskInfoManager* tsk_info
                              )
{
  for ( int l = 0; l < m_nQn_part; l++ ) {

    // model variables
    T& char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel[l] );
    T& gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel[l] );
    T& particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp[l] );
    T& particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize[l] );
    T& surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate[l] );

    char_rate.initialize         ( 0.0 );
    gas_char_rate.initialize     ( 0.0 );
    particle_temp_rate.initialize( 0.0 );
    particle_Size_rate.initialize( 0.0 );
    surface_rate.initialize      ( 0.0 );
  }

  int nrtq = _NUM_reactions * m_nQn_part; // number of reaction x number of env

  for ( int r = 0; r < nrtq; r++ ) {
   T& reaction_rate = tsk_info->get_uintah_field_add< T >( m_reaction_rate_names[r] );
   reaction_rate.initialize( 0.0 );
  }

  //CCVariable<double>& char_rate = tsk_info->get_uintah_field_add<CCVariable<double> >( d_modelLabel );
  //AreaSumF.initialize( 0.0 );
}
//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::register_timestep_init(       std::vector<ArchesFieldContainer::VariableInformation> & variable_registry
                                          , const bool                                                     packed_tasks
                                          )
{
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::timestep_init( const Patch                 * patch
                                 ,       ArchesTaskInfoManager * tsk_info
                                 )
{
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::register_timestep_eval(       std::vector<ArchesFieldContainer::VariableInformation> & variable_registry
                                          , const int                                                      time_substep
                                          , const bool                                                     packed_tasks
                                          )
{
  // model variables
  for ( int l = 0; l < m_nQn_part; l++ ) {
    register_variable( m_modelLabel[l],   ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    register_variable( m_gasLabel[l],     ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    register_variable( m_particletemp[l], ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    register_variable( m_particleSize[l], ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    register_variable( m_surfacerate[l],  ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    //register_variable( m_PO2surf[l],      ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
  }

  //register_variable( "AreaSum", ArchesFieldContainer::COMPUTES, variable_registry, time_substep );

  int nrtq = _NUM_reactions * m_nQn_part; // number of reaction x number of env

  for ( int l = 0; l < nrtq; l++ ) {
    register_variable( m_reaction_rate_names[l], ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
    register_variable( m_reaction_rate_names[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::OLDDW, variable_registry, time_substep );
  }

  // gas variables
  register_variable( m_cc_u_vel_name,         ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_cc_v_vel_name,         ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_cc_w_vel_name,         ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_density_gas_name,      ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_gas_temperature_label, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  register_variable( m_MW_name,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

  // gas species
  for ( int ns = 0; ns < _NUM_species; ns++ ) {
    register_variable( _species_names[ns], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  }

  // other particle variables
  register_variable( number_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

  for ( int l = 0; l < m_nQn_part; l++ ) {

    // from devol model
    register_variable( m_devolRC[l],        ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_surfAreaF_name[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    //birth terms

    //if ( m_add_char_birth ) {
    //  register_variable( m_char_birth_qn_name[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //}

    //if ( m_add_rawcoal_birth ) {
    //  register_variable( m_rawcoal_birth_qn_name[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //}

    //if ( m_add_length_birth ) {
    //  register_variable( m_length_birth_qn_name[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //}

    // other models
    register_variable( m_particle_temperature[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_particle_length[l],      ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //register_variable( m_particle_length_qn[l],   ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_particle_density[l],     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_rcmass[l],               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_char_name[l],            ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_weight_name[l],          ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //register_variable( m_weightqn_name[l],        ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_up_name[l],              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_vp_name[l],              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_wp_name[l],              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    // RHS
    //register_variable( m_RC_RHS[l],     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //register_variable( m_ic_RHS[l],     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //register_variable( m_w_RHS[l],      ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    //register_variable( m_length_RHS[l], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  } // end for ( int l = 0; l < m_nQn_part; l++ )
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::eval( const Patch                 * patch
                        ,       ArchesTaskInfoManager * tsk_info
                        )
{
  Timers::Simple timer;

  // gas variables
  constCCVariable<double>& CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_u_vel_name );
  constCCVariable<double>& CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_v_vel_name );
  constCCVariable<double>& CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_w_vel_name );

  constCCVariable<double>& den         = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_density_gas_name );
  constCCVariable<double>& temperature = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_gas_temperature_label );
  constCCVariable<double>& MWmix       = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_MW_name ); // in kmol/kg_mix

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT; // check comment from other char model

  const double dt = tsk_info->get_dt();

  //Vector Dx = patch->dCell();
  //const double vol = Dx.x()* Dx.y()* Dx.z();

  std::vector< CT* > species;

  for ( int ns = 0; ns < _NUM_species; ns++ ) {
    CT* species_p = tsk_info->get_const_uintah_field< CT >( _species_names[ns] );
    species.push_back( species_p );
  }

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  CT& number_density = tsk_info->get_const_uintah_field_add< CT >( number_density_name ); // total number density

  //T& AreaSumF = tsk_info->get_uintah_field_add< T >( "AreaSum", 0 ); // temporal variable

  //AreaSumF.initialize( 0.0 );
  //for ( int l = 0; l < m_nQn_part; l++ ) {

  //  CT& weight = tsk_info->get_const_uintah_field_add< CT >( m_weight_name[l] );
  //  CT& length = tsk_info->get_const_uintah_field_add< CT >( m_particle_length[l] );

  //  Uintah::parallel_for( range, [&]( int i,  int j, int k ) {
  //    AreaSumF(i,j,k) +=  weight(i,j,k) * length(i,j,k) * length(i,j,k); // [#/m]
  //  }); //end cell loop
  //}

  InversionBase* invf;

  if ( _NUM_reactions == 2 ) {
    invf = scinew invert_2_2;
  }
  else if ( _NUM_reactions == 3 ) {
    invf = scinew invert_3_3;
  }
  else {
    throw InvalidValue( "ERROR: CharOxidationSmith2016: Matrix inversion not implemented for the number of reactions being used.", __FILE__, __LINE__ );
  }

  //root function: find a better way of doing this
  RootFunctionBase* rf;
  rf = scinew root_functionB;

  std::vector< T* >  reaction_rate;
  std::vector< CT* > old_reaction_rate;

  int m  = 0; // to access reaction list: try to see if I can do 2 D
  int m2 = 0; // to access reaction list: try to see if I can do 2 D

  for ( int l = 0; l < m_nQn_part; l++ ) {

    // model variables
    T& char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel[l] );
    T& gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel[l] );
    T& particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp[l] );
    T& particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize[l] );
    T& surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate[l] );

    //T& PO2surf            = tsk_info->get_uintah_field_add< T >(m_PO2surf[l]);

    // reaction rate
    for ( int r = 0; r < _NUM_reactions; r++ ) {

      T*  reaction_rate_p     = tsk_info->get_uintah_field< T >       ( m_reaction_rate_names[m] );
      CT* old_reaction_rate_p = tsk_info->get_const_uintah_field< CT >( m_reaction_rate_names[m] );

      m += 1;

      reaction_rate.push_back    ( reaction_rate_p );
      old_reaction_rate.push_back( old_reaction_rate_p );
    }

    // from devol model
    CT& devolRC = tsk_info->get_const_uintah_field_add< CT >( m_devolRC[l] );

    // particle variables from other models
    CT& particle_temperature = tsk_info->get_const_uintah_field_add< CT >( m_particle_temperature[l] );
    CT& length               = tsk_info->get_const_uintah_field_add< CT >( m_particle_length[l] );
    CT& particle_density     = tsk_info->get_const_uintah_field_add< CT >( m_particle_density[l] );
    CT& rawcoal_mass         = tsk_info->get_const_uintah_field_add< CT >( m_rcmass[l] );
    CT& char_mass            = tsk_info->get_const_uintah_field_add< CT >( m_char_name[l] );
    CT& weight               = tsk_info->get_const_uintah_field_add< CT >( m_weight_name[l] );
    CT& up                   = tsk_info->get_const_uintah_field_add< CT >( m_up_name[l] );
    CT& vp                   = tsk_info->get_const_uintah_field_add< CT >( m_vp_name[l] );
    CT& wp                   = tsk_info->get_const_uintah_field_add< CT >( m_wp_name[l] );

    // birth terms
    //if (m_add_rawcoal_birth) {
    //  CT& rawcoal_birth = tsk_info->get_const_uintah_field_add< CT >(m_rawcoal_birth_qn_name[l]);
    //}

    //if (m_add_char_birth) {
    //  CT& char_birth    = tsk_info->get_const_uintah_field_add< CT >(m_char_birth_qn_name[l]);
    //}

    //if (m_add_length_birth) {
    //  CT& length_birth  = tsk_info->get_const_uintah_field_add< CT >(m_length_birth_qn_name[l]);
    //}

    //CT& weight_p_diam = tsk_info->get_const_uintah_field_add< CT >( m_particle_length_qn[l] ); //check
    //CT& RC_RHS_source = tsk_info->get_const_uintah_field_add< CT >( m_RC_RHS[l] );
    //CT& RHS_source    = tsk_info->get_const_uintah_field_add< CT >( m_ic_RHS[l] );
    //CT& RHS_weight    = tsk_info->get_const_uintah_field_add< CT >( m_w_RHS[l] );
    //CT& RHS_length    = tsk_info->get_const_uintah_field_add< CT >( m_length_RHS[l] );

    CT& surfAreaF = tsk_info->get_const_uintah_field_add< CT >( m_surfAreaF_name[l] );

    if ( l == 1 ) {
      timer.start();
    }

    // 00 - CellIterator
    //Uintah::parallel_for( range, [&]( int i,  int j, int k ) {
    for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ) {
      IntVector c = *iter;

      DenseMatrix* dfdrh = scinew DenseMatrix( _NUM_reactions,_NUM_reactions );

      std::vector<double> D_oxid_mix_l     ( _NUM_reactions );
      std::vector<double> D_kn             ( _NUM_reactions );
      std::vector<double> D_eff            ( _NUM_reactions );
      std::vector<double> phi_l            ( _NUM_reactions );
      std::vector<double> hrxn_l           ( _NUM_reactions );
      std::vector<double> rh_l             ( _NUM_reactions );
      std::vector<double> rh_l_new         ( _NUM_reactions );
      std::vector<double> species_mass_frac( _NUM_species );
      std::vector<double> oxid_mass_frac   ( _NUM_reactions );

      std::vector<double> Sc            ( _NUM_reactions );
      std::vector<double> Sh            ( _NUM_reactions );
      std::vector<double> co_s          ( _NUM_reactions );
      std::vector<double> oxid_mole_frac( _NUM_reactions );
      std::vector<double> co_r          ( _NUM_reactions );
      std::vector<double> k_r           ( _NUM_reactions );
      std::vector<double> M_T           ( _NUM_reactions );
      std::vector<double> effectivenessF( _NUM_reactions );

      std::vector<double> F         ( _NUM_reactions );
      std::vector<double> rh_l_delta( _NUM_reactions );
      std::vector<double> F_delta   ( _NUM_reactions );
      std::vector<double> r_h_ex    ( _NUM_reactions );
      std::vector<double> r_h_in    ( _NUM_reactions );

      if ( weight[c] / m_weight_scaling_constant[l] < _weight_small ) {

        char_rate[c]          = 0.0;
        gas_char_rate[c]      = 0.0;
        particle_temp_rate[c] = 0.0;
        particle_Size_rate[c] = 0.0;
        surface_rate[c]       = 0.0;

        for ( int r = 0; r < _NUM_reactions; r++ ) {
          (*reaction_rate[m2 + r])[c] = 0.0; // check this
        }
      }
      else {

        // populate the temporary variables.
        //Vector gas_vel;                                  // [m/s]
        //Vector part_vel   = partVel(i,j,k);              // [m/s]
        double gas_rho    = den[c];                  // [kg/m^3]
        double gas_T      = temperature[c];          // [K]
        double p_T        = particle_temperature[c]; // [K]
        double p_rho      = particle_density[c];     // [kg/m^3]
        double p_diam     = length[c];               // [m]
        double rc         = rawcoal_mass[c];         // [kg/#]
        double ch         = char_mass[c];            // [kg/#]
        double w          = weight[c];               // [#/m^3]
        double MW         = 1. / MWmix[c];           // [kg mix / kmol mix] (MW in table is 1/MW)
        double r_devol    = devolRC[c] * m_RC_scaling_constant[l] * m_weight_scaling_constant[l]; // [kg/m^3/s]
        double r_devol_ns = -r_devol; // [kg/m^3/s]
        //double RHS_v      = RC_RHS_source(i,j,k) * m_RC_scaling_constant[l] * m_weight_scaling_constant[l]; // [kg/s]
        //double RHS        = RHS_source(i,j,k) * m_char_scaling_constant[l] * m_weight_scaling_constant[l];  // [kg/s]

        // populate temporary variable vectors
        double delta = 1e-6;

        dfdrh->zero(); // [-]

        for ( int r = 0; r < _NUM_reactions; r++ ) {
          rh_l[r]     = (*old_reaction_rate[m2 + r])[c]; // [kg/m^3/s]
          rh_l_new[r] = (*old_reaction_rate[m2 + r])[c]; // [kg/m^3/s]
        }

        for ( int r = 0; r < _NUM_reactions; r++ ) { // check this
          oxid_mass_frac[r] = (*species[_oxidizer_indices[r]])[c]; // [mass fraction]
        }

        for ( int ns = 0; ns < _NUM_species; ns++ ) {
          species_mass_frac[ns] = (*species[ns])[c]; // [mass fraction]
        }

        double CO_CO2_ratio = 200. * exp( -9000. / ( _R_cal * p_T ) ) * 44.0 / 28.0; // [ kg CO / kg CO2] => [kmoles CO / kmoles CO2]
        double CO2onCO      = 1. / CO_CO2_ratio;                                     // [kmoles CO2 / kmoles CO]

        for ( int r = 0; r < _NUM_reactions; r++ ) {
          phi_l[r]  = ( _use_co2co_l[r] ) ? ( CO2onCO + 1 ) / ( CO2onCO + 0.5 )              : _phi_l[r];
          hrxn_l[r] = ( _use_co2co_l[r] ) ? ( CO2onCO * _HF_CO2 + _HF_CO ) / ( 1 + CO2onCO ) : _hrxn_l[r];
        }

        double relative_velocity = std::sqrt( ( CCuVel[c] - up[c] ) * ( CCuVel[c] - up[c] ) +
                                              ( CCvVel[c] - vp[c] ) * ( CCvVel[c] - vp[c] ) +
                                              ( CCwVel[c] - wp[c] ) * ( CCwVel[c] - wp[c] )   ); // [m/s]

        double Re_p     = relative_velocity * p_diam / ( _dynamic_visc / gas_rho ); // Reynolds number [-]
        double x_org    = (rc + ch) / (rc + ch + m_mass_ash[l] );
        double cg       = _gasPressure / (_R * gas_T * 1000.); // [kmoles/m^3] - Gas concentration
        double p_area   = M_PI * SQUARE( p_diam );             // particle surface area [m^2]
        double p_volume = M_PI / 6. * CUBE( p_diam );          // particle volme [m^3]
        double p_void   = std::fmax( 1e-10, 1. - ( 1. / p_volume ) * ( ( rc + ch ) / m_rho_org_bulk[l] + m_mass_ash[l] / _rho_ash_bulk ) ); // current porosity. (-) required due to sign convention of char.

        double psi = 1. / ( _p_void0 * ( 1. - _p_void0 ) );
        double Sj  = _init_particle_density / p_rho * ( ( 1 - p_void ) / ( 1 - _p_void0 ) ) * std::sqrt( 1 - std::fmin( 1.0, psi * log( ( 1 - p_void ) / ( 1 - _p_void0 ) ) ) );
        double rp  = 2 * p_void * (1. - p_void ) / ( p_rho * Sj * _Sg0 ); // average particle radius [m]

        // Calculate oxidizer diffusion coefficient
        // effect diffusion through stagnant gas (see "Multicomponent Mass Transfer", Taylor and Krishna equation 6.1.14)
        for ( int r = 0; r < _NUM_reactions; r++ ) {

          double sum_x_D = 0;
          double sum_x   = 0;

          for ( int ns = 0; ns < _NUM_species; ns++ ) {
            sum_x_D = ( _oxid_l[r] != _species_names[ns] ) ? sum_x_D + species_mass_frac[ns] / ( _MW_species[ns] * _D_mat[_oxidizer_indices[r]][ns] ) : sum_x_D;
            sum_x   = ( _oxid_l[r] != _species_names[ns] ) ? sum_x   + species_mass_frac[ns] / ( _MW_species[ns] )                                    : sum_x;
          }

          D_oxid_mix_l[r] = sum_x / sum_x_D * std::sqrt( CUBE( gas_T / _T0 ) );
          D_kn[r]         = 97. * rp * std::sqrt( p_T / _MW_species[r] );
          D_eff[r]        = p_void / _tau / ( 1. / D_kn[r] + 1. / D_oxid_mix_l[r] );
        }

        for ( int r = 0; r < _NUM_reactions; r++ ) {

          Sc[r]             = _dynamic_visc / ( gas_rho * D_oxid_mix_l[r] );      // Schmidt number [-]
          Sh[r]             = 2.0 + 0.6 * std::sqrt( Re_p ) * std::cbrt( Sc[r] ); // Sherwood number [-]
          oxid_mole_frac[r] = oxid_mass_frac[r] * MW / _MW_l[r];                  // [mole fraction]
          co_r[r]           = cg * oxid_mole_frac[r];                             // oxidizer concentration, [kmoles/m^3]
          k_r[r]            = ( 10.0 * _a_l[r] * std::exp( -_e_l[r] / ( _R_cal * p_T ) ) * _R * p_T * 1000.0 ) / ( _Mh * phi_l[r] * 101325. ); // [m / s]
          M_T[r]            = p_diam / 2. * std::sqrt( k_r[r] * _Sg0 * Sj * p_rho / D_eff[r] );                   // Thiele modulus, Mitchell's formulation
          effectivenessF[r] = ( M_T[r] < 1e-5 ) ? 1.0 : 3. / M_T[r] * ( 1. / std::tanh( M_T[r] ) - 1. / M_T[r] ); // effectiveness factor
        }

        // Newton-Raphson solve for rh_l.
        // rh_(n+1) = rh_(n) - (dF_(n)/drh_(n))^-1 * F_(n)

        int count = 0;

        for ( int it = 0; it < 100; it++ ) {

          count = count + 1;

          for ( int r = 0; r < _NUM_reactions; r++ ) {
            rh_l[r] = rh_l_new[r];
          }

          // get F and Jacobian -> dF/drh
          rf->root_function( F, rh_l, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, D_oxid_mix_l,
                             phi_l, p_void, effectivenessF, Sj, p_rho, x_org, _NUM_reactions, _Sg0, _Mh );
          //root_function( F, rh_l, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, D_oxid_mix_l, phi_l, p_void, effectivenessF, Sj, p_rho, x_org);

          for ( int j = 0; j < _NUM_reactions; j++ ) {

            for ( int k = 0; k < _NUM_reactions; k++ ) {
              rh_l_delta[k] = rh_l[k]; // why ? OD
            }

            rh_l_delta[j] = rh_l[j] + delta;

            rf->root_function( F_delta, rh_l_delta, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, D_oxid_mix_l,
                               phi_l, p_void, effectivenessF, Sj, p_rho, x_org, _NUM_reactions, _Sg0, _Mh );
            //root_function( F_delta, rh_l_delta, co_r, gas_rho, cg, k_r, MW, r_devol_ns, p_diam, Sh, D_oxid_mix_l, phi_l, p_void, effectivenessF, Sj, p_rho, x_org);

            for ( int r = 0; r < _NUM_reactions; r++ ) {
              (*dfdrh)[r][j] = ( F_delta[r] - F[r] ) / delta;
            }
          }

          // invert Jacobian -> (dF_(n)/drh_(n))^-1
          invf->invert_mat( dfdrh ); // simple matrix inversion for a 2x2 matrix.

          // get rh_(n+1)
          double dominantRate = 0.0;
          //double max_F        = 1e-8;

          for ( int r = 0; r < _NUM_reactions; r++ ) {

            for ( int var = 0; var < _NUM_reactions; var++ ) {
              rh_l_new[r] -= (*dfdrh)[r][var] * F[var];
            }

            dominantRate = std::fmax( dominantRate, std::fabs( rh_l_new[r] ) );
            //max_F        = std::fmax( max_F, std::fabs( F[r] ) );
          }

          double residual = 0.0;

          for ( int r = 0; r < _NUM_reactions; r++ ) {
            residual += std::fabs( F[r] ) / dominantRate;
            //residual += std::fabs( F[r] ) / max_F;
          }

          // make sure rh_(n+1) is inbounds
          for ( int r = 0; r < _NUM_reactions; r++ ) {
            rh_l_new[r] = std::fmin( 100000., std::fmax( 0.0, rh_l_new[r] ) ); // max rate adjusted based on pressure (empirical limit)
          }

          //if ( residual < 1e-3 ) {
          if ( residual < 1e-8 ) {
            //std::cout << "residual: " <<" "<< residual << " " << "Number of iterations "<< count << " " <<" env " << l  << std::endl;
            //std::cout << "F[0]: " << F[0] << std::endl;
            //std::cout << "F[1]: " << F[1] << std::endl;
            //std::cout << "F[2]: " << F[2] << std::endl;
            break;
          }
        } // end for ( int it = 0; it < 100; it++ )

        if ( count > 90 ) {
          std::cout << "warning no solution found in char ox: [env " << l << " "  << c.x() << ", " << c.y() << ", " << c.z() << "] " << std::endl;
          std::cout << "F[0]: "              << F[0]              << std::endl;
          std::cout << "F[1]: "              << F[1]              << std::endl;
          std::cout << "F[2]: "              << F[2]              << std::endl;
          std::cout << "p_void: "            << p_void            << std::endl;
          std::cout << "gas_rho: "           << gas_rho           << std::endl;
          std::cout << "gas_T: "             << gas_T             << std::endl;
          std::cout << "p_T: "               << p_T               << std::endl;
          std::cout << "p_diam: "            << p_diam            << std::endl;
          std::cout << "relative_velocity: " << relative_velocity << std::endl;
          std::cout << "w: "                 << w                 << std::endl;
          std::cout << "MW: "                << MW                << std::endl;
          std::cout << "r_devol_ns: "        << r_devol_ns        << std::endl;
          std::cout << "oxid_mass_frac[0]: " << oxid_mass_frac[0] << std::endl;
          std::cout << "oxid_mass_frac[1]: " << oxid_mass_frac[1] << std::endl;
          std::cout << "oxid_mass_frac[2]: " << oxid_mass_frac[2] << std::endl;
          std::cout << "oxid_mole_frac[0]: " << oxid_mole_frac[0] << std::endl;
          std::cout << "oxid_mole_frac[1]: " << oxid_mole_frac[1] << std::endl;
          std::cout << "oxid_mole_frac[2]: " << oxid_mole_frac[2] << std::endl;
          std::cout << "D_oxid_mix_l[0]: "   << D_oxid_mix_l[0]   << std::endl;
          std::cout << "D_oxid_mix_l[1]: "   << D_oxid_mix_l[1]   << std::endl;
          std::cout << "D_oxid_mix_l[2]: "   << D_oxid_mix_l[2]   << std::endl;
          std::cout << "rh_l_new[0]: "       << rh_l_new[0]       << std::endl;
          std::cout << "rh_l_new[1]: "       << rh_l_new[1]       << std::endl;
          std::cout << "rh_l_new[2]: "       << rh_l_new[2]       << std::endl;
          std::cout << "org: "               << rc + ch           << std::endl;
          std::cout << "x_org: "             << x_org             << std::endl;
          std::cout << "p_rho: "             << p_rho             << std::endl;
          std::cout << "p_void0: "           << _p_void0          << std::endl;
          std::cout << "psi: "               << psi               << std::endl;
        }

        double char_mass_rate      = 0.0;
        double d_mass              = 0.0;
        double d_mass2             = 0.0;
        double h_rxn               = 0.0; // this is to compute the reaction rate averaged heat of reaction. It is needed so we don't need to clip any additional rates.
        double h_rxn_factor        = 0.0; // this is to compute a multiplicative factor to correct for fp.
        double surface_rate_factor = 0.0; // this is to compute a multiplicative factor to correct for external vs interal rxn.
        //double oxi_lim             = 0.0; // max rate due to reactions
        //double rh_l_i              = 0.0;

        double surfaceAreaFraction = surfAreaF[c]; //w*p_diam*p_diam/AreaSumF(i,j,k); // [-] this is the weighted area fraction for the current particle size.

        for ( int r = 0; r < _NUM_reactions; r++ ) {

          (*reaction_rate[m2 + r])[c] = rh_l_new[r]; // [kg/m^2/s] this is for the intial guess during the next time-step

          // check to see if reaction rate is oxidizer limited.
          double oxi_lim = ( oxid_mass_frac[r] * gas_rho * surfaceAreaFraction ) / ( dt * w );   // [kg/s/#] // here the surfaceAreaFraction parameter is allowing us to only consume the oxidizer multiplied by the weighted area fraction for the current particle.
          double rh_l_i  = std::fmin( rh_l_new[r] * p_area * x_org * ( 1. - p_void ), oxi_lim ); // [kg/s/#]

          char_mass_rate      += -rh_l_i; // [kg/s/#] // negative sign because we are computing the destruction rate for the particles.
          d_mass              += rh_l_i;
          co_s[r]              = rh_l_i / ( phi_l[r] * _Mh * k_r[r] * ( 1 + effectivenessF[r] * p_diam * p_rho * _Sg0 * Sj / ( 6. * ( 1 - p_void ) ) ) ); // oxidizer concentration at particle surface [kmoles/m^3]
          r_h_ex[r]            = phi_l[r] * _Mh * k_r[l] * co_s[r]; // [kg/m^2/s]
          r_h_in[r]            = r_h_ex[r] * effectivenessF[r] * p_diam * p_rho * _Sg0 * Sj / ( 6. * ( 1 - p_void ) ); // [kg/m^2/s]
          h_rxn_factor        += r_h_ex[r] * _ksi + r_h_in[r];
          h_rxn               += hrxn_l[r] * ( r_h_ex[r] * _ksi + r_h_in[r] );
          d_mass2             += r_h_ex[r] * _ksi + r_h_in[r];
          surface_rate_factor += r_h_ex[r];
        }

        h_rxn_factor        /= ( d_mass  + 1e-50 );
        surface_rate_factor /= ( d_mass  + 1e-50 );
        h_rxn               /= ( d_mass2 + 1e-50 ); // [J/mole]

        // rate clipping for char_mass_rate
        //if ( m_add_rawcoal_birth && m_add_char_birth ) {
        //  char_mass_rate = std::fmax( char_mass_rate, -( ( rc + ch ) / ( dt ) + ( RHS + RHS_v ) / ( vol * w ) + r_devol / w + char_birth(i,j,k) / w + rawcoal_birth(i,j,k) / w ) ); // [kg/s/#]
        //}
        //else {
        //  char_mass_rate = std::fmax( char_mass_rate, - ( ( rc + ch ) / ( dt ) + ( RHS + RHS_v ) / ( vol * w ) + r_devol / w ) ); // [kg/s/#]
        //}

        char_mass_rate = std::fmin( 0.0, char_mass_rate ); // [kg/s/#] make sure we aren't creating char.

        // organic consumption rate
        char_rate[c] = ( char_mass_rate * w ) / ( m_char_scaling_constant[l] * m_weight_scaling_constant[l] ); // [kg/m^3/s - scaled]

        // off-gas production rate
        gas_char_rate[c] = -char_mass_rate * w; // [kg/m^3/s] (negative sign for exchange between solid and gas)

        // heat of reaction source term for enthalpyshaddix
        particle_temp_rate[c] = h_rxn * 1000. / _Mh * h_rxn_factor * char_mass_rate * w / _ksi; // [J/s/m^4] -- the *1000 is need to convert J/mole to J/kmole. char_mass_rate was already multiplied by x_org * (1-p_void).
                                                                                                    // note: this model is designed to work with EnthalpyShaddix. The effect of ksi has already been added to Qreaction so we divide here.

        // particle shrinkage rate
        //double updated_weight = std::fmax( w / m_weight_scaling_constant[l] + dt / vol * ( RHS_weight(i,j,k) ), 1e-15 );
        //double min_p_diam     = std::pow( m_mass_ash[l] * 6 / _rho_ash_bulk / ( 1. - m_p_voidmin[l] ) / M_PI, 1. / 3. );

        double max_Size_rate = 0.0;

        //if ( m_add_length_birth ) {
        //  max_Size_rate = ( updated_weight * min_p_diam / m_length_scaling_constant[l] - weight_p_diam(i,j,k) ) / dt - ( RHS_length(i,j,k) / vol + length_birth(i,j,k) );
        //}
        //else {
        //  max_Size_rate = ( updated_weight * min_p_diam / m_length_scaling_constant[l] - weight_p_diam(i,j,k) ) / dt - ( RHS_length(i,j,k) / vol);
        //}

        double Size_rate = ( x_org < 1e-8 ) ? 0.0 :
                           w / m_weight_scaling_constant[l] * 2. * x_org * surface_rate_factor * char_mass_rate /
                           m_rho_org_bulk[l] / p_area / x_org / ( 1. - p_void ) / m_length_scaling_constant[l]; // [m/s]

        particle_Size_rate[c] = std::fmax( max_Size_rate, Size_rate ); // [m/s] -- these source terms are negative.
        surface_rate[c]       = char_mass_rate / p_area;               // in [kg/(s # m^2)]
        //PO2surf_(i,j,k)           = 0.0;                                   // multiple oxidizers, so we are leaving this empty.

      } // end if ( weight(i,j,k) / m_weight_scaling_constant[l] < _weight_small ) else

      delete dfdrh;

    //}); // end Uintah::parallel_for
    } // end for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ )

    if ( l == 1 ) {
      timer.stop();
      printf( " loopTime: %g\n", timer().milliseconds() );
    }

    m2 += _NUM_reactions;

  } // end for ( int l = 0; l < m_nQn_part; l++ )

  delete invf;
  delete rf;
  ///
}
//--------------------------------------------------------------------------------------------------

}
#endif
