#ifndef Uintah_Component_Arches_CharOxidationps_h
#define Uintah_Component_Arches_CharOxidationps_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <CCA/Components/Arches/ParticleModels/CharOxidationpsHelper.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#define SQUARE(x) x*x
#define CUBE(x)   x*x*x

#define reactions_count 3
#define species_count   4

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
namespace {

template < typename T, typename CT >
struct solveFunctor {

#ifdef UINTAH_ENABLE_KOKKOS
  KokkosView3<const double> m_weight;
  KokkosView3<double>       m_char_rate;
  KokkosView3<double>       m_gas_char_rate;
  KokkosView3<double>       m_particle_temp_rate;
  KokkosView3<double>       m_particle_Size_rate;
  KokkosView3<double>       m_surface_rate;
  KokkosView3<double>       m_reaction_rate[reactions_count];
  KokkosView3<const double> m_old_reaction_rate[reactions_count];
  KokkosView3<const double> m_den;
  KokkosView3<const double> m_temperature;
  KokkosView3<const double> m_particle_temperature;
  KokkosView3<const double> m_particle_density;
  KokkosView3<const double> m_length;
  KokkosView3<const double> m_rawcoal_mass;
  KokkosView3<const double> m_char_mass;
  KokkosView3<const double> m_MWmix;
  KokkosView3<const double> m_devolRC;
  KokkosView3<const double> m_species[species_count];
  KokkosView3<const double> m_CCuVel;
  KokkosView3<const double> m_CCvVel;
  KokkosView3<const double> m_CCwVel;
  KokkosView3<const double> m_up;
  KokkosView3<const double> m_vp;
  KokkosView3<const double> m_wp;
  KokkosView3<const double> m_surfAreaF;
#else
  CT                      & m_weight;
  T                       & m_char_rate;
  T                       & m_gas_char_rate;
  T                       & m_particle_temp_rate;
  T                       & m_particle_Size_rate;
  T                       & m_surface_rate;
  T                       * m_reaction_rate[reactions_count];
  CT                      * m_old_reaction_rate[reactions_count];
  constCCVariable<double> & m_den;
  constCCVariable<double> & m_temperature;
  CT                      & m_particle_temperature;
  CT                      & m_particle_density;
  CT                      & m_length;
  CT                      & m_rawcoal_mass;
  CT                      & m_char_mass;
  constCCVariable<double> & m_MWmix;
  CT                      & m_devolRC;
  CT                      * m_species[species_count];
  constCCVariable<double> & m_CCuVel;
  constCCVariable<double> & m_CCvVel;
  constCCVariable<double> & m_CCwVel;
  CT                      & m_up;
  CT                      & m_vp;
  CT                      & m_wp;
  CT                      & m_surfAreaF;
#endif
  double m_weight_scaling_constant;
  double m_weight_small;
  double m_RC_scaling_constant;
  double m_R_cal;
  bool   m_use_co2co_l[reactions_count];
  double m_phi_l[reactions_count];
  double m_hrxn_l[reactions_count];
  double m_HF_CO2;
  double m_HF_CO;
  double m_dynamic_visc;
  double m_mass_ash;
  double m_gasPressure;
  double m_R;
  double m_rho_org_bulk;
  double m_rho_ash_bulk;
  double m_p_void0;
  double m_init_particle_density;
  double m_Sg0;
  double m_MW_species[species_count];
  double m_D_mat[reactions_count][species_count];
  int    m_oxidizer_indices[reactions_count];
  double m_T0;
  double m_tau;
  double m_MW_l[reactions_count];
  double m_a_l[reactions_count];
  double m_e_l[reactions_count];
  const double m_dt;
  double m_Mh;
  double m_ksi;
  double m_char_scaling_constant;
  double m_length_scaling_constant;

  solveFunctor(

#ifdef UINTAH_ENABLE_KOKKOS
              KokkosView3<const double> & weight
            , KokkosView3<double>       & char_rate
            , KokkosView3<double>       & gas_char_rate
            , KokkosView3<double>       & particle_temp_rate
            , KokkosView3<double>       & particle_Size_rate
            , KokkosView3<double>       & surface_rate
            , KokkosView3<double>         reaction_rate[reactions_count]
            , KokkosView3<const double>   old_reaction_rate[reactions_count]
            , KokkosView3<const double> & den
            , KokkosView3<const double> & temperature
            , KokkosView3<const double> & particle_temperature
            , KokkosView3<const double> & particle_density
            , KokkosView3<const double> & length
            , KokkosView3<const double> & rawcoal_mass
            , KokkosView3<const double> & char_mass
            , KokkosView3<const double> & MWmix
            , KokkosView3<const double> & devolRC
            , KokkosView3<const double>   species[species_count]
            , KokkosView3<const double> & CCuVel
            , KokkosView3<const double> & CCvVel
            , KokkosView3<const double> & CCwVel
            , KokkosView3<const double> & up
            , KokkosView3<const double> & vp
            , KokkosView3<const double> & wp
            , KokkosView3<const double> & surfAreaF
#else
              CT                      & weight
            , T                       & char_rate
            , T                       & gas_char_rate
            , T                       & particle_temp_rate
            , T                       & particle_Size_rate
            , T                       & surface_rate
            , T                       * reaction_rate[reactions_count]
            , CT                      * old_reaction_rate[reactions_count]
            , constCCVariable<double> & den
            , constCCVariable<double> & temperature
            , CT                      & particle_temperature
            , CT                      & particle_density
            , CT                      & length
            , CT                      & rawcoal_mass
            , CT                      & char_mass
            , constCCVariable<double> & MWmix
            , CT                      & devolRC
            , CT                      * species[species_count]
            , constCCVariable<double> & CCuVel
            , constCCVariable<double> & CCvVel
            , constCCVariable<double> & CCwVel
            , CT                      & up
            , CT                      & vp
            , CT                      & wp
            , CT                      & surfAreaF
#endif
            , double   _weight_scaling_constant
            , double   _weight_small
            , double   _RC_scaling_constant
            , double   _R_cal
            , bool     _use_co2co_l[reactions_count]
            , double   _phi_l[reactions_count]
            , double   _hrxn_l[reactions_count]
            , double   _HF_CO2
            , double   _HF_CO
            , double   _dynamic_visc
            , double   _mass_ash
            , double   _gasPressure
            , double   _R
            , double   _rho_org_bulk
            , double   _rho_ash_bulk
            , double   _p_void0
            , double   _init_particle_density
            , double   _Sg0
            , double   _MW_species[species_count]
            , double   _D_mat[reactions_count][species_count]
            , int      _oxidizer_indices[reactions_count]
            , double   _T0
            , double   _tau
            , double   _MW_l[reactions_count]
            , double   _a_l[reactions_count]
            , double   _e_l[reactions_count]
            , const double dt
            , double   _Mh
            , double   _ksi
            , double   _char_scaling_constant
            , double   _length_scaling_constant
            )
    :
#ifdef UINTAH_ENABLE_KOKKOS
      m_weight                  ( weight )
    , m_char_rate               ( char_rate )
    , m_gas_char_rate           ( gas_char_rate )
    , m_particle_temp_rate      ( particle_temp_rate )
    , m_particle_Size_rate      ( particle_Size_rate )
    , m_surface_rate            ( surface_rate )
    , m_den                     ( den )
    , m_temperature             ( temperature )
    , m_particle_temperature    ( particle_temperature )
    , m_particle_density        ( particle_density )
    , m_length                  ( length )
    , m_rawcoal_mass            ( rawcoal_mass )
    , m_char_mass               ( char_mass )
    , m_MWmix                   ( MWmix )
    , m_devolRC                 ( devolRC )
    , m_CCuVel                  ( CCuVel )
    , m_CCvVel                  ( CCvVel )
    , m_CCwVel                  ( CCwVel )
    , m_up                      ( up )
    , m_vp                      ( vp )
    , m_wp                      ( wp )
    , m_surfAreaF               ( surfAreaF )
#else
      m_weight                  ( weight )
    , m_char_rate               ( char_rate )
    , m_gas_char_rate           ( gas_char_rate )
    , m_particle_temp_rate      ( particle_temp_rate )
    , m_particle_Size_rate      ( particle_Size_rate )
    , m_surface_rate            ( surface_rate )
    , m_den                     ( den )
    , m_temperature             ( temperature )
    , m_particle_temperature    ( particle_temperature )
    , m_particle_density        ( particle_density )
    , m_length                  ( length )
    , m_rawcoal_mass            ( rawcoal_mass )
    , m_char_mass               ( char_mass )
    , m_MWmix                   ( MWmix )
    , m_devolRC                 ( devolRC )
    , m_CCuVel                  ( CCuVel )
    , m_CCvVel                  ( CCvVel )
    , m_CCwVel                  ( CCwVel )
    , m_up                      ( up )
    , m_vp                      ( vp )
    , m_wp                      ( wp )
    , m_surfAreaF               ( surfAreaF )
#endif
    , m_weight_scaling_constant ( _weight_scaling_constant )
    , m_weight_small            ( _weight_small )
    , m_RC_scaling_constant     ( _RC_scaling_constant )
    , m_R_cal                   ( _R_cal )
    , m_HF_CO2                  ( _HF_CO2 )
    , m_HF_CO                   ( _HF_CO )
    , m_dynamic_visc            ( _dynamic_visc )
    , m_mass_ash                ( _mass_ash )
    , m_gasPressure             ( _gasPressure )
    , m_R                       ( _R )
    , m_rho_org_bulk            ( _rho_org_bulk )
    , m_rho_ash_bulk            ( _rho_ash_bulk )
    , m_p_void0                 ( _p_void0 )
    , m_init_particle_density   ( _init_particle_density )
    , m_Sg0                     (_Sg0 )
    , m_T0                      (_T0 )
    , m_tau                     ( _tau )
    , m_dt                      ( dt )
    , m_Mh                      ( _Mh )
    , m_ksi                     ( _ksi )
    , m_char_scaling_constant   ( _char_scaling_constant )
    , m_length_scaling_constant ( _length_scaling_constant )
  {
    for ( int nr = 0; nr < reactions_count; nr++ ) {

      m_reaction_rate[nr]     = reaction_rate[nr];
      m_old_reaction_rate[nr] = old_reaction_rate[nr];
      m_use_co2co_l[nr]       = _use_co2co_l[nr];
      m_phi_l[nr]             = _phi_l[nr];
      m_hrxn_l[nr]            = _hrxn_l[nr];
      m_oxidizer_indices[nr]  = _oxidizer_indices[nr];
      m_MW_l[nr]              = _MW_l[nr];
      m_a_l[nr]               = _a_l[nr];
      m_e_l[nr]               = _e_l[nr];

      for ( int ns = 0; ns < species_count; ns++ ) {
        m_D_mat[nr][ns] = _D_mat[nr][ns];
      }
    }

    for ( int ns = 0; ns < species_count; ns++ ) {
      m_species[ns]    = species[ns];
      m_MW_species[ns] = _MW_species[ns];
    }
  }

    void operator() ( int i, int j, int k ) const {

      //DenseMatrix* dfdrh = scinew DenseMatrix( _NUM_reactions,_NUM_reactions );
      double dfdrh[reactions_count][reactions_count];

      // 02 - Replace per-cell std::vector with plain-old-data arrays
      double D_oxid_mix_l     [ reactions_count ];
      double D_kn             [ reactions_count ];
      double D_eff            [ reactions_count ];
      double phi_l            [ reactions_count ];
      double hrxn_l           [ reactions_count ];
      double rh_l             [ reactions_count ];
      double rh_l_new         [ reactions_count ];
      double species_mass_frac[ species_count ];
      double oxid_mass_frac   [ reactions_count ];

      double Sc            [ reactions_count ];
      double Sh            [ reactions_count ];
      double co_s          [ reactions_count ];
      double oxid_mole_frac[ reactions_count ];
      double co_r          [ reactions_count ];
      double k_r           [ reactions_count ];
      double M_T           [ reactions_count ];
      double effectivenessF[ reactions_count ];

      double F         [ reactions_count ];
      double rh_l_delta[ reactions_count ];
      double F_delta   [ reactions_count ];
      double r_h_ex    [ reactions_count ];
      double r_h_in    [ reactions_count ];

      if ( m_weight(i,j,k) / m_weight_scaling_constant < m_weight_small ) {

        m_char_rate(i,j,k)          = 0.0;
        m_gas_char_rate(i,j,k)      = 0.0;
        m_particle_temp_rate(i,j,k) = 0.0;
        m_particle_Size_rate(i,j,k) = 0.0;
        m_surface_rate(i,j,k)       = 0.0;

        for ( int r = 0; r < reactions_count; r++ ) {

          // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
          m_reaction_rate[r](i,j,k) = 0.0; // check this
#else
//          (*reaction_rate[m2 + r])(i,j,k) = 0.0; // check this
          (*m_reaction_rate[r])(i,j,k) = 0.0; // check this
#endif
        }
      }
      else {

        // populate the temporary variables.
        //Vector gas_vel;                                  // [m/s]
        //Vector part_vel   = partVel(i,j,k);              // [m/s]
        double gas_rho    = m_den(i,j,k);                  // [kg/m^3]
        double gas_T      = m_temperature(i,j,k);          // [K]
        double p_T        = m_particle_temperature(i,j,k); // [K]
        double p_rho      = m_particle_density(i,j,k);     // [kg/m^3]
        double p_diam     = m_length(i,j,k);               // [m]
        double rc         = m_rawcoal_mass(i,j,k);         // [kg/#]
        double ch         = m_char_mass(i,j,k);            // [kg/#]
        double w          = m_weight(i,j,k);               // [#/m^3]
        double MW         = 1. / m_MWmix(i,j,k);           // [kg mix / kmol mix] (MW in table is 1/MW)
        double r_devol    = m_devolRC(i,j,k) * m_RC_scaling_constant * m_weight_scaling_constant; // [kg/m^3/s]
        double r_devol_ns = -r_devol; // [kg/m^3/s]
        //double RHS_v      = RC_RHS_source(i,j,k) * m_RC_scaling_constant[l] * m_weight_scaling_constant[l]; // [kg/s]
        //double RHS        = RHS_source(i,j,k) * m_char_scaling_constant[l] * m_weight_scaling_constant[l];  // [kg/s]

        // populate temporary variable vectors
        double delta = 1e-6;

        for ( int r = 0; r < reactions_count; r++ ) {

          // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
          rh_l_new[r] = m_old_reaction_rate[r](i,j,k); // [kg/m^3/s]
#else
          rh_l_new[r] = (*m_old_reaction_rate[r])(i,j,k); // [kg/m^3/s]
#endif
        }

        for ( int r = 0; r < reactions_count; r++ ) { // check this

          // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
          oxid_mass_frac[r] = m_species[m_oxidizer_indices[r]](i,j,k); // [mass fraction]
#else
          oxid_mass_frac[r] = (*m_species[m_oxidizer_indices[r]])(i,j,k); // [mass fraction]
#endif
        }

        for ( int ns = 0; ns < species_count; ns++ ) {

          // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
          species_mass_frac[ns] = m_species[ns](i,j,k); // [mass fraction]
#else
          species_mass_frac[ns] = (*m_species[ns])(i,j,k); // [mass fraction]
#endif
        }

        double CO_CO2_ratio = 200. * exp( -9000. / ( m_R_cal * p_T ) ) * 44.0 / 28.0; // [ kg CO / kg CO2] => [kmoles CO / kmoles CO2]
        double CO2onCO      = 1. / CO_CO2_ratio;                                     // [kmoles CO2 / kmoles CO]

        for ( int r = 0; r < reactions_count; r++ ) {
          phi_l[r]  = ( m_use_co2co_l[r] ) ? ( CO2onCO + 1 ) / ( CO2onCO + 0.5 )                : m_phi_l[r];
          hrxn_l[r] = ( m_use_co2co_l[r] ) ? ( CO2onCO * m_HF_CO2 + m_HF_CO ) / ( 1 + CO2onCO ) : m_hrxn_l[r];
        }

        double relative_velocity = sqrt( ( m_CCuVel(i,j,k) - m_up(i,j,k) ) * ( m_CCuVel(i,j,k) - m_up(i,j,k) ) +
                                         ( m_CCvVel(i,j,k) - m_vp(i,j,k) ) * ( m_CCvVel(i,j,k) - m_vp(i,j,k) ) +
                                         ( m_CCwVel(i,j,k) - m_wp(i,j,k) ) * ( m_CCwVel(i,j,k) - m_wp(i,j,k) )   ); // [m/s]

        double Re_p     = relative_velocity * p_diam / ( m_dynamic_visc / gas_rho ); // Reynolds number [-]
        double x_org    = (rc + ch) / (rc + ch + m_mass_ash );
        double cg       = m_gasPressure / (m_R * gas_T * 1000.); // [kmoles/m^3] - Gas concentration
        double p_area   = M_PI * SQUARE( p_diam );             // particle surface area [m^2]
        double p_volume = M_PI / 6. * CUBE( p_diam );          // particle volme [m^3]
        double p_void   = fmax( 1e-10, 1. - ( 1. / p_volume ) * ( ( rc + ch ) / m_rho_org_bulk + m_mass_ash / m_rho_ash_bulk ) ); // current porosity. (-) required due to sign convention of char.

        double psi = 1. / ( m_p_void0 * ( 1. - m_p_void0 ) );
        double Sj  = m_init_particle_density / p_rho * ( ( 1 - p_void ) / ( 1 - m_p_void0 ) ) * sqrt( 1 - fmin( 1.0, psi * log( ( 1 - p_void ) / ( 1 - m_p_void0 ) ) ) );
        double rp  = 2 * p_void * (1. - p_void ) / ( p_rho * Sj * m_Sg0 ); // average particle radius [m]

        // Calculate oxidizer diffusion coefficient
        // effect diffusion through stagnant gas (see "Multicomponent Mass Transfer", Taylor and Krishna equation 6.1.14)
        for ( int r = 0; r < reactions_count; r++ ) {

          double sum_x_D = 0;
          double sum_x   = 0;

          for ( int ns = 0; ns < species_count; ns++ ) {

            // 06 - No std::string
            sum_x_D = ( ns != m_oxidizer_indices[r] ) ? sum_x_D + species_mass_frac[ns] / ( m_MW_species[ns] * m_D_mat[m_oxidizer_indices[r]][ns] ) : sum_x_D;
            sum_x   = ( ns != m_oxidizer_indices[r] ) ? sum_x   + species_mass_frac[ns] / ( m_MW_species[ns] )                                      : sum_x;
          }

          D_oxid_mix_l[r] = sum_x / sum_x_D * sqrt( CUBE( gas_T / m_T0 ) );
          D_kn[r]         = 97. * rp * sqrt( p_T / m_MW_species[r] );
          D_eff[r]        = p_void / m_tau / ( 1. / D_kn[r] + 1. / D_oxid_mix_l[r] );
        }

        for ( int r = 0; r < reactions_count; r++ ) {

          Sc[r]             = m_dynamic_visc / ( gas_rho * D_oxid_mix_l[r] );      // Schmidt number [-]
          Sh[r]             = 2.0 + 0.6 * sqrt( Re_p ) * cbrt( Sc[r] ); // Sherwood number [-]
          oxid_mole_frac[r] = oxid_mass_frac[r] * MW / m_MW_l[r];                  // [mole fraction]
          co_r[r]           = cg * oxid_mole_frac[r];                             // oxidizer concentration, [kmoles/m^3]
          k_r[r]            = ( 10.0 * m_a_l[r] * exp( -m_e_l[r] / ( m_R_cal * p_T ) ) * m_R * p_T * 1000.0 ) / ( m_Mh * phi_l[r] * 101325. ); // [m / s]
          M_T[r]            = p_diam / 2. * sqrt( k_r[r] * m_Sg0 * Sj * p_rho / D_eff[r] );                   // Thiele modulus, Mitchell's formulation
          effectivenessF[r] = ( M_T[r] < 1e-5 ) ? 1.0 : 3. / M_T[r] * ( 1. / tanh( M_T[r] ) - 1. / M_T[r] ); // effectiveness factor
        }

        // Newton-Raphson solve for rh_l.
        // rh_(n+1) = rh_(n) - (dF_(n)/drh_(n))^-1 * F_(n)

        int count = 0;
        double rtot    = 0.0;
        double Sfactor = 0.0;
        double Bjm     = 0.0;
        double mtc_r   = 0.0;

        for ( int it = 0; it < 100; it++ ) {

          count = count + 1;

          for ( int r = 0; r < reactions_count; r++ ) {
            rh_l[r] = rh_l_new[r];
          }

          // get F and Jacobian -> dF/drh
          rtot    = ( rh_l[0] + rh_l[1] + rh_l[2] ) * x_org * ( 1. - p_void ) + r_devol_ns;
          Sfactor = 0.0;
          Bjm     = 0.0;
          mtc_r   = 0.0;

          for ( int l = 0; l < reactions_count; l++ ) {

            Bjm     = fmin( 80.0, rtot * p_diam / ( D_oxid_mix_l[l] * gas_rho ) ); // [-] // this is the derived for mass flux  BSL chapter 22
            mtc_r   = ( Sh[l] * D_oxid_mix_l[l] * ( ( Bjm >= 1e-7 ) ?  Bjm / ( exp( Bjm ) - 1. ) : 1.0 ) ) / p_diam; // [m/s]
            Sfactor = 1 + effectivenessF[l] * p_diam * p_rho * m_Sg0 * Sj / ( 6. * ( 1. - p_void ) );
            F[l]    = rh_l[l] - ( m_Mh * MW * phi_l[l] * k_r[l] * mtc_r * Sfactor * co_r[l] * cg ) /
                      ( ( MW * cg * ( k_r[l] * x_org * ( 1. - p_void ) * Sfactor + mtc_r ) ) + rtot ); // [kg-char/m^3/s]
          }

          for ( int j = 0; j < reactions_count; j++ ) {

            for ( int k = 0; k < reactions_count; k++ ) {
              rh_l_delta[k] = rh_l[k]; // why ? OD
            }

            rh_l_delta[j] = rh_l[j] + delta;

            rtot    = ( rh_l_delta[0] + rh_l_delta[1] + rh_l_delta[2] ) * x_org * ( 1. - p_void ) + r_devol_ns;
            Sfactor = 0.0;
            Bjm     = 0.0;
            mtc_r   = 0.0;

            for ( int l = 0; l < reactions_count; l++ ) {

              Bjm        = fmin( 80.0, rtot * p_diam / ( D_oxid_mix_l[l] * gas_rho ) ); // [-] // this is the derived for mass flux  BSL chapter 22
              mtc_r      = ( Sh[l] * D_oxid_mix_l[l] * ( ( Bjm >= 1e-7 ) ?  Bjm / ( exp( Bjm ) - 1. ) : 1.0 ) ) / p_diam; // [m/s]
              Sfactor    = 1 + effectivenessF[l] * p_diam * p_rho * m_Sg0 * Sj / ( 6. * ( 1. - p_void ) );
              F_delta[l] = rh_l_delta[l] - ( m_Mh * MW * phi_l[l] * k_r[l] * mtc_r * Sfactor * co_r[l] * cg ) /
                           ( ( MW * cg * ( k_r[l] * x_org * ( 1. - p_void ) * Sfactor + mtc_r ) ) + rtot ); // [kg-char/m^3/s]
             }

            for ( int r = 0; r < reactions_count; r++ ) {
              dfdrh[r][j] = ( F_delta[r] - F[r] ) / delta;
            }
          }

          // invert Jacobian -> (dF_(n)/drh_(n))^-1
          double a11 = dfdrh[0][0];
          double a12 = dfdrh[0][1];
          double a13 = dfdrh[0][2];
          double a21 = dfdrh[1][0];
          double a22 = dfdrh[1][1];
          double a23 = dfdrh[1][2];
          double a31 = dfdrh[2][0];
          double a32 = dfdrh[2][1];
          double a33 = dfdrh[2][2];

          double det_inv = 1 / ( a11 * a22 * a33 +
                                 a21 * a32 * a13 +
                                 a31 * a12 * a23 -
                                 a11 * a32 * a23 -
                                 a31 * a22 * a13 -
                                 a21 * a12 * a33   );

          dfdrh[0][0] = ( a22 * a33 - a23 * a32 ) * det_inv;
          dfdrh[0][1] = ( a13 * a32 - a12 * a33 ) * det_inv;
          dfdrh[0][2] = ( a12 * a23 - a13 * a22 ) * det_inv;
          dfdrh[1][0] = ( a23 * a31 - a21 * a33 ) * det_inv;
          dfdrh[1][1] = ( a11 * a33 - a13 * a31 ) * det_inv;
          dfdrh[1][2] = ( a13 * a21 - a11 * a23 ) * det_inv;
          dfdrh[2][0] = ( a21 * a32 - a22 * a31 ) * det_inv;
          dfdrh[2][1] = ( a12 * a31 - a11 * a32 ) * det_inv;
          dfdrh[2][2] = ( a11 * a22 - a12 * a21 ) * det_inv;

          // get rh_(n+1)
          double dominantRate = 0.0;
          //double max_F        = 1e-8;

          for ( int r = 0; r < reactions_count; r++ ) {

            for ( int var = 0; var < reactions_count; var++ ) {
              rh_l_new[r] -= dfdrh[r][var] * F[var];
            }

            dominantRate = fmax( dominantRate, fabs( rh_l_new[r] ) );
            //max_F        = fmax( max_F, fabs( F[r] ) );
          }

          double residual = 0.0;

          for ( int r = 0; r < reactions_count; r++ ) {
            residual += fabs( F[r] ) / dominantRate;
            //residual += fabs( F[r] ) / max_F;
          }

          // make sure rh_(n+1) is inbounds
          for ( int r = 0; r < reactions_count; r++ ) {
            rh_l_new[r] = fmin( 100000., fmax( 0.0, rh_l_new[r] ) ); // max rate adjusted based on pressure (empirical limit)
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
          printf( "warning no solution found in char ox: [ %d, %d, %d ]\n", i, j, k );
          printf( "F[0]: %g\n",              F[0] );
          printf( "F[1]: %g\n",              F[1] );
          printf( "F[2]: %g\n",              F[2] );
          printf( "p_void: %g\n",            p_void );
          printf( "gas_rho: %g\n",           gas_rho );
          printf( "gas_T: %g\n",             gas_T );
          printf( "p_T: %g\n",               p_T );
          printf( "p_diam: %g\n",            p_diam );
          printf( "relative_velocity: %g\n", relative_velocity );
          printf( "w: %g\n",                 w );
          printf( "MW: %g\n",                MW );
          printf( "r_devol_ns: %g\n",        r_devol_ns );
          printf( "oxid_mass_frac[0]: %g\n", oxid_mass_frac[0] );
          printf( "oxid_mass_frac[1]: %g\n", oxid_mass_frac[1] );
          printf( "oxid_mass_frac[2]: %g\n", oxid_mass_frac[2] );
          printf( "oxid_mole_frac[0]: %g\n", oxid_mole_frac[0] );
          printf( "oxid_mole_frac[1]: %g\n", oxid_mole_frac[1] );
          printf( "oxid_mole_frac[2]: %g\n", oxid_mole_frac[2] );
          printf( "D_oxid_mix_l[0]: %g\n",   D_oxid_mix_l[0] );
          printf( "D_oxid_mix_l[1]: %g\n",   D_oxid_mix_l[1] );
          printf( "D_oxid_mix_l[2]: %g\n",   D_oxid_mix_l[2] );
          printf( "rh_l_new[0]: %g\n",       rh_l_new[0] );
          printf( "rh_l_new[1]: %g\n",       rh_l_new[1] );
          printf( "rh_l_new[2]: %g\n",       rh_l_new[2] );
          printf( "org: %g\n",               rc + ch );
          printf( "x_org: %g\n",             x_org );
          printf( "p_rho: %g\n",             p_rho );
          printf( "p_void0: %g\n",           m_p_void0 );
          printf( "psi:  %g\n",              psi );
        }

        double char_mass_rate      = 0.0;
        double d_mass              = 0.0;
        double d_mass2             = 0.0;
        double h_rxn               = 0.0; // this is to compute the reaction rate averaged heat of reaction. It is needed so we don't need to clip any additional rates.
        double h_rxn_factor        = 0.0; // this is to compute a multiplicative factor to correct for fp.
        double surface_rate_factor = 0.0; // this is to compute a multiplicative factor to correct for external vs interal rxn.
        //double oxi_lim             = 0.0; // max rate due to reactions
        //double rh_l_i              = 0.0;

        double surfaceAreaFraction = m_surfAreaF(i,j,k); //w*p_diam*p_diam/AreaSumF(i,j,k); // [-] this is the weighted area fraction for the current particle size.

        for ( int r = 0; r < reactions_count; r++ ) {

          // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
          m_reaction_rate[r](i,j,k) = rh_l_new[r]; // [kg/m^2/s] this is for the intial guess during the next time-step
#else
          (*m_reaction_rate[r])(i,j,k) = rh_l_new[r]; // [kg/m^2/s] this is for the intial guess during the next time-step
#endif

          // check to see if reaction rate is oxidizer limited.
          double oxi_lim = ( oxid_mass_frac[r] * gas_rho * surfaceAreaFraction ) / ( m_dt * w );   // [kg/s/#] // here the surfaceAreaFraction parameter is allowing us to only consume the oxidizer multiplied by the weighted area fraction for the current particle.
          double rh_l_i  = fmin( rh_l_new[r] * p_area * x_org * ( 1. - p_void ), oxi_lim ); // [kg/s/#]

          char_mass_rate      += -rh_l_i; // [kg/s/#] // negative sign because we are computing the destruction rate for the particles.
          d_mass              += rh_l_i;
          co_s[r]              = rh_l_i / ( phi_l[r] * m_Mh * k_r[r] * ( 1 + effectivenessF[r] * p_diam * p_rho * m_Sg0 * Sj / ( 6. * ( 1 - p_void ) ) ) ); // oxidizer concentration at particle surface [kmoles/m^3]
          r_h_ex[r]            = phi_l[r] * m_Mh * k_r[r] * co_s[r]; // [kg/m^2/s]
          r_h_in[r]            = r_h_ex[r] * effectivenessF[r] * p_diam * p_rho * m_Sg0 * Sj / ( 6. * ( 1 - p_void ) ); // [kg/m^2/s]
          h_rxn_factor        += r_h_ex[r] * m_ksi + r_h_in[r];
          h_rxn               += hrxn_l[r] * ( r_h_ex[r] * m_ksi + r_h_in[r] );
          d_mass2             += r_h_ex[r] * m_ksi + r_h_in[r];
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

        char_mass_rate = fmin( 0.0, char_mass_rate ); // [kg/s/#] make sure we aren't creating char.

        // organic consumption rate
        m_char_rate(i,j,k) = ( char_mass_rate * w ) / ( m_char_scaling_constant * m_weight_scaling_constant ); // [kg/m^3/s - scaled]

        // off-gas production rate
        m_gas_char_rate(i,j,k) = -char_mass_rate * w; // [kg/m^3/s] (negative sign for exchange between solid and gas)

        // heat of reaction source term for enthalpyshaddix
        m_particle_temp_rate(i,j,k) = h_rxn * 1000. / m_Mh * h_rxn_factor * char_mass_rate * w / m_ksi; // [J/s/m^4] -- the *1000 is need to convert J/mole to J/kmole. char_mass_rate was already multiplied by x_org * (1-p_void).
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
                           w / m_weight_scaling_constant * 2. * x_org * surface_rate_factor * char_mass_rate /
                           m_rho_org_bulk / p_area / x_org / ( 1. - p_void ) / m_length_scaling_constant; // [m/s]

        m_particle_Size_rate(i,j,k) = fmax( max_Size_rate, Size_rate ); // [m/s] -- these source terms are negative.
        m_surface_rate(i,j,k)       = char_mass_rate / p_area;               // in [kg/(s # m^2)]
        //PO2surf_(i,j,k)           = 0.0;                                   // multiple oxidizers, so we are leaving this empty.

      } // end if ( weight(i,j,k) / m_weight_scaling_constant[l] < _weight_small ) else

    }  // end operator()
  };   // end solveFunctor
}      // end namespace

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::eval( const Patch                 * patch
                        ,       ArchesTaskInfoManager * tsk_info
                        )
{
  // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
  // gas variables
  KokkosView3<const double> CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_u_vel_name ).getKokkosView();
  KokkosView3<const double> CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_v_vel_name ).getKokkosView();
  KokkosView3<const double> CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_w_vel_name ).getKokkosView();

  KokkosView3<const double> den         = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_density_gas_name ).getKokkosView();
  KokkosView3<const double> temperature = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_gas_temperature_label ).getKokkosView();
  KokkosView3<const double> MWmix       = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_MW_name ).getKokkosView(); // in kmol/kg_mix
#else
  // gas variables
  constCCVariable<double>& CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_u_vel_name );
  constCCVariable<double>& CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_v_vel_name );
  constCCVariable<double>& CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_w_vel_name );

  constCCVariable<double>& den         = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_density_gas_name );
  constCCVariable<double>& temperature = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_gas_temperature_label );
  constCCVariable<double>& MWmix       = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_MW_name ); // in kmol/kg_mix
#endif

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT; // check comment from other char model

  const double dt = tsk_info->get_dt();

  //Vector Dx = patch->dCell();
  //const double vol = Dx.x()* Dx.y()* Dx.z();

  // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
  KokkosView3<const double> species[species_count];
#else
  CT* species[species_count];
#endif

  for ( int ns = 0; ns < _NUM_species; ns++ ) {

    // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
    species[ns] = tsk_info->get_const_uintah_field_add< CT >( _species_names[ns] ).getKokkosView();
#else
    species[ns] = tsk_info->get_const_uintah_field< CT >( _species_names[ns] );
#endif
  }

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
  KokkosView3<const double> number_density = tsk_info->get_const_uintah_field_add< CT >( number_density_name ).getKokkosView(); // total number density
#else
  CT& number_density = tsk_info->get_const_uintah_field_add< CT >( number_density_name ); // total number density
#endif

  //T& AreaSumF = tsk_info->get_uintah_field_add< T >( "AreaSum", 0 ); // temporal variable

  //AreaSumF.initialize( 0.0 );
  //for ( int l = 0; l < m_nQn_part; l++ ) {

  //  CT& weight = tsk_info->get_const_uintah_field_add< CT >( m_weight_name[l] );
  //  CT& length = tsk_info->get_const_uintah_field_add< CT >( m_particle_length[l] );

  //  Uintah::parallel_for( range, [&]( int i,  int j, int k ) {
  //    AreaSumF(i,j,k) +=  weight(i,j,k) * length(i,j,k) * length(i,j,k); // [#/m]
  //  }); //end cell loop
  //}

  // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
  KokkosView3<double>       reaction_rate[reactions_count];
  KokkosView3<const double> old_reaction_rate[reactions_count];
#else
  T*  reaction_rate[reactions_count];
  CT* old_reaction_rate[reactions_count];
#endif

  int m  = 0; // to access reaction list: try to see if I can do 2 D
  int m2 = 0; // to access reaction list: try to see if I can do 2 D

  bool   _use_co2co_l_pod     [ reactions_count ];
  double _phi_l_pod           [ reactions_count ];
  double _hrxn_l_pod          [ reactions_count ];
  int    _oxidizer_indices_pod[ reactions_count ];
  double _MW_l_pod            [ reactions_count ];
  double _a_l_pod             [ reactions_count ];
  double _e_l_pod             [ reactions_count ];
  double _D_mat_pod           [ reactions_count ][ species_count ];

  for ( int nr = 0; nr < reactions_count; nr++ ) {

    _use_co2co_l_pod[nr]      = _use_co2co_l[nr];
    _phi_l_pod[nr]            = _phi_l[nr];
    _hrxn_l_pod[nr]           = _hrxn_l[nr];
    _oxidizer_indices_pod[nr] = _oxidizer_indices[nr];
    _MW_l_pod[nr]             = _MW_l[nr];
    _a_l_pod[nr]              = _a_l[nr];
    _e_l_pod[nr]              = _e_l[nr];

    for ( int ns = 0; ns < species_count; ns++ ) {
      _D_mat_pod[nr][ns] = _D_mat[nr][ns];
    }
  }

  double _MW_species_pod[ species_count ];

  for ( int ns = 0; ns < species_count; ns++ ) {
    _MW_species_pod[ns] = _MW_species[ns];
  }

  for ( int l = 0; l < m_nQn_part; l++ ) {

    // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
    // model variables
    KokkosView3<double> char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel[l] ).getKokkosView();
    KokkosView3<double> gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel[l] ).getKokkosView();
    KokkosView3<double> particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp[l] ).getKokkosView();
    KokkosView3<double> particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize[l] ).getKokkosView();
    KokkosView3<double> surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate[l] ).getKokkosView();
#else
    // model variables
    T& char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel[l] );
    T& gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel[l] );
    T& particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp[l] );
    T& particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize[l] );
    T& surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate[l] );
#endif

    //T& PO2surf            = tsk_info->get_uintah_field_add< T >(m_PO2surf[l]);

    // reaction rate
    for ( int r = 0; r < _NUM_reactions; r++ ) {

      // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
      reaction_rate[r]     = tsk_info->get_uintah_field_add< T >       ( m_reaction_rate_names[m] ).getKokkosView();
      old_reaction_rate[r] = tsk_info->get_const_uintah_field_add< CT >( m_reaction_rate_names[m] ).getKokkosView();

      m += 1;
#else
      reaction_rate[r]     = tsk_info->get_uintah_field< T >       ( m_reaction_rate_names[m] );
      old_reaction_rate[r] = tsk_info->get_const_uintah_field< CT >( m_reaction_rate_names[m] );

      m += 1;
#endif
    }

    // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
    // from devol model
    KokkosView3<const double> devolRC = tsk_info->get_const_uintah_field_add< CT >( m_devolRC[l] ).getKokkosView();

    // particle variables from other models
    KokkosView3<const double> particle_temperature = tsk_info->get_const_uintah_field_add< CT >( m_particle_temperature[l] ).getKokkosView();
    KokkosView3<const double> length               = tsk_info->get_const_uintah_field_add< CT >( m_particle_length[l] ).getKokkosView();
    KokkosView3<const double> particle_density     = tsk_info->get_const_uintah_field_add< CT >( m_particle_density[l] ).getKokkosView();
    KokkosView3<const double> rawcoal_mass         = tsk_info->get_const_uintah_field_add< CT >( m_rcmass[l] ).getKokkosView();
    KokkosView3<const double> char_mass            = tsk_info->get_const_uintah_field_add< CT >( m_char_name[l] ).getKokkosView();
    KokkosView3<const double> weight               = tsk_info->get_const_uintah_field_add< CT >( m_weight_name[l] ).getKokkosView();
    KokkosView3<const double> up                   = tsk_info->get_const_uintah_field_add< CT >( m_up_name[l] ).getKokkosView();
    KokkosView3<const double> vp                   = tsk_info->get_const_uintah_field_add< CT >( m_vp_name[l] ).getKokkosView();
    KokkosView3<const double> wp                   = tsk_info->get_const_uintah_field_add< CT >( m_wp_name[l] ).getKokkosView();
#else
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
#endif

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

    // 05 - KokkosView3
#ifdef UINTAH_ENABLE_KOKKOS
    KokkosView3<const double> surfAreaF = tsk_info->get_const_uintah_field_add< CT >( m_surfAreaF_name[l] ).getKokkosView();
#else
    CT& surfAreaF = tsk_info->get_const_uintah_field_add< CT >( m_surfAreaF_name[l] );
#endif

solveFunctor< T, CT > func(
#ifdef UINTAH_ENABLE_KOKKOS
                   weight
                 , char_rate
                 , gas_char_rate
                 , particle_temp_rate
                 , particle_Size_rate
                 , surface_rate
                 , reaction_rate
                 , old_reaction_rate
                 , den
                 , temperature
                 , particle_temperature
                 , particle_density
                 , length
                 , rawcoal_mass
                 , char_mass
                 , MWmix
                 , devolRC
                 , species
                 , CCuVel
                 , CCvVel
                 , CCwVel
                 , up
                 , vp
                 , wp
                 , surfAreaF
#else
                   weight
                 , char_rate
                 , gas_char_rate
                 , particle_temp_rate
                 , particle_Size_rate
                 , surface_rate
                 , reaction_rate
                 , old_reaction_rate
                 , den
                 , temperature
                 , particle_temperature
                 , particle_density
                 , length
                 , rawcoal_mass
                 , char_mass
                 , MWmix
                 , devolRC
                 , species
                 , CCuVel
                 , CCvVel
                 , CCwVel
                 , up
                 , vp
                 , wp
                 , surfAreaF
#endif
                 , m_weight_scaling_constant[l]
                 , _weight_small
                 , m_RC_scaling_constant[l]
                 , _R_cal
                 , _use_co2co_l_pod
                 , _phi_l_pod
                 , _hrxn_l_pod
                 , _HF_CO2
                 , _HF_CO
                 , _dynamic_visc
                 , m_mass_ash[l]
                 , _gasPressure
                 , _R
                 , m_rho_org_bulk[l]
                 , _rho_ash_bulk
                 , _p_void0
                 , _init_particle_density
                 , _Sg0
                 , _MW_species_pod
                 , _D_mat_pod
                 , _oxidizer_indices_pod
                 , _T0
                 , _tau
                 , _MW_l_pod
                 , _a_l_pod
                 , _e_l_pod
                 , dt
                 , _Mh
                 , _ksi
                 , m_char_scaling_constant[l]
                 , m_length_scaling_constant[l]
                 );

    Uintah::parallel_for( range, func );

    m2 += _NUM_reactions;

  } // end for ( int l = 0; l < m_nQn_part; l++ )

  //delete invf;
  //delete rf;
  ///
}
//--------------------------------------------------------------------------------------------------

}
#endif
