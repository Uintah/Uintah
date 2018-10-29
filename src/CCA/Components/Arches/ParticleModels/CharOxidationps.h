#ifndef Uintah_Component_Arches_CharOxidationps_h
#define Uintah_Component_Arches_CharOxidationps_h

#include <CCA/Components/Arches/Task/TaskInterface.h>
#include <CCA/Components/Arches/GridTools.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CharOxidationpsHelper.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>

#define SQUARE(x) x*x
#define CUBE(x)   x*x*x

namespace Uintah {

  template <typename T>
  class CharOxidationps : public TaskInterface {

public:

    CharOxidationps( std::string task_name, int matl_index, int Nenv );
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

      Builder( std::string task_name, int matl_index , int Nenv ) : m_task_name(task_name), m_matl_index(matl_index), _Nenv(Nenv){}
      ~Builder(){}

      CharOxidationps* build()
      { return scinew CharOxidationps<T>( m_task_name, m_matl_index, _Nenv ); }

      private:

      std::string m_task_name;
      int         m_matl_index;
      int _Nenv;
    };

private:

    int _Nenv;
    // constants
    double _R;      // [J/ (K mol) ]
    double _R_cal;  // [cal/ (K mol) ]
    double _HF_CO2; // [J/mol]
    double _HF_CO;  // [J/mol]
    double _T0;
    double _tau;    // tortuosity

    int m_nQn_part;

    // model name lists
    std::string m_modelLabel;
    std::string m_gasLabel;
    std::string m_particletemp;
    std::string m_particleSize;
    std::string m_surfacerate;
    std::string m_devolRC;
    std::string m_surfAreaF_name;

    bool m_add_rawcoal_birth{false};
    bool m_add_length_birth{false};
    bool m_add_char_birth{false};

    std::string m_char_birth_qn_name;
    std::string m_rawcoal_birth_qn_name;
    std::string m_length_birth_qn_name;

    // other models name lists
    std::string m_particle_length_qn;
    std::string m_particle_temperature;
    std::string m_particle_length;
    std::string m_particle_density;
    std::string m_rcmass;
    std::string m_char_name;
    std::string m_weight_name;
    std::string m_weightqn_name;
    std::string m_up_name;
    std::string m_vp_name;
    std::string m_wp_name;
    std::string                number_density_name;

    // RHS
    std::string m_RC_RHS;
    std::string m_ic_RHS;
    std::string m_w_RHS;
    std::string m_length_RHS;

    // scaling constant
    double m_weight_scaling_constant;
    double m_char_scaling_constant;
    double m_RC_scaling_constant;
    double m_length_scaling_constant;

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
    double m_mass_ash; //
    double m_rho_org_bulk; //
    double m_p_voidmin;

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
    std::string m_volFraction_name;
  };

//--------------------------------------------------------------------------------------------------
template<typename T>
CharOxidationps<T>::CharOxidationps( std::string task_name, int matl_index,
                                   int Nenv)
  : TaskInterface( task_name, matl_index ), _Nenv(Nenv)
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
  std::string delvol_model_name = "devol_rate";
  db->require("devol_model_name",delvol_model_name);
  m_devolRC =  ArchesCore::append_qn_env( delvol_model_name, _Nenv ) ;

  std::string surfAreaF_root    = "surfaceAreaFraction";

  // Create a label for this model
  m_modelLabel = m_task_name;

  // Create the gas phase source term associated with this model
  m_gasLabel =  m_task_name + "_gasSource";

  // Create the particle temperature source term associated with this model
  m_particletemp = m_task_name +  "_particletempSource" ;

  // Create the particle size source term associated with this model
  m_particleSize =  m_task_name + "_particleSizeSource" ;

  // Create the char oxidation surface rate term associated with this model
  m_surfacerate = m_task_name + "_surfacerate";

  // Create the char oxidation PO2 surf term associated with this model
  //std::string PO2surf_temp = modelName + "_PO2surf";
  //m_PO2surf.push_back( ArchesCore::append_env( PO2surf_temp, l ) );


  m_surfAreaF_name =  ArchesCore::append_env( surfAreaF_root, _Nenv ) ;

  _gasPressure = 101325.; // Fix this

  // gas variables
  m_density_gas_name = ArchesCore::parse_ups_for_role( ArchesCore::DENSITY,   db, "density" );

  m_cc_u_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCUVELOCITY, db, "CCUVelocity" );
  m_cc_v_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCVVELOCITY, db, "CCVVelocity" );
  m_cc_w_vel_name = ArchesCore::parse_ups_for_role( ArchesCore::CCWVELOCITY, db, "CCWVelocity" );
  m_volFraction_name = "volFraction";

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

  m_particle_temperature = ArchesCore::append_env( temperature_root, _Nenv ) ;
  m_particle_length      = ArchesCore::append_env( length_root,      _Nenv ) ;
  m_particle_density     = ArchesCore::append_env( density_root,     _Nenv ) ;
  m_rcmass               = ArchesCore::append_env( rcmass_root,      _Nenv ) ;
  m_char_name            = ArchesCore::append_env( char_root,        _Nenv ) ;
  m_weight_name          = ArchesCore::append_env( "w",              _Nenv ) ;
  m_up_name              = ArchesCore::append_env( up_root,          _Nenv ) ;
  m_vp_name              = ArchesCore::append_env( vp_root,          _Nenv ) ;
  m_wp_name              = ArchesCore::append_env( wp_root,          _Nenv ) ;

  m_RC_RHS     = ArchesCore::append_qn_env( rcmass_root, _Nenv ) + "_RHS";
  m_ic_RHS     = ArchesCore::append_qn_env( char_root, _Nenv )   + "_RHS";
  m_length_RHS = ArchesCore::append_qn_env( length_root, _Nenv ) + "_RHS";
  m_w_RHS = ArchesCore::append_qn_env( "w", _Nenv )              + "_RHS";
  m_particle_length_qn = ArchesCore::append_qn_env( length_root, _Nenv );

  const std::string rawcoal_birth_name = ArchesCore::getModelNameByType( db, rcmass_root, "BirthDeath");
  const std::string char_birth_name    = ArchesCore::getModelNameByType( db, char_root,   "BirthDeath");
  const std::string length_birth_name  = ArchesCore::getModelNameByType( db, length_root, "BirthDeath");

  if ( char_birth_name != "NULLSTRING" ) {
    m_add_char_birth = true;
  }

  if ( rawcoal_birth_name != "NULLSTRING" ) {
    m_add_rawcoal_birth = true;
  }

  if ( length_birth_name != "NULLSTRING" ) {
    m_add_length_birth = true;
  }


  if ( m_add_char_birth ) {
    m_char_birth_qn_name = ArchesCore::append_qn_env( char_birth_name, _Nenv );
  }

  if ( m_add_rawcoal_birth ) {
    m_rawcoal_birth_qn_name = ArchesCore::append_qn_env( rawcoal_birth_name, _Nenv );
  }

  if ( m_add_length_birth ) {
    m_length_birth_qn_name = ArchesCore::append_qn_env( length_birth_name, _Nenv );
  }

  // scaling constants
  m_weight_scaling_constant = ArchesCore::get_scaling_constant(db, "w", _Nenv);
  m_char_scaling_constant   = ArchesCore::get_scaling_constant(db, char_root, _Nenv);
  m_RC_scaling_constant     = ArchesCore::get_scaling_constant(db, rcmass_root, _Nenv);
  m_length_scaling_constant = ArchesCore::get_scaling_constant(db, length_root, _Nenv);

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

    double initial_diameter = ArchesCore::get_inlet_particle_size( db, _Nenv );
    double p_volume         = M_PI / 6. * CUBE( initial_diameter ); // particle volume [m^3]
    m_mass_ash              = p_volume * _init_particle_density * ash_mass_frac;
    double initial_rc       = ( M_PI / 6.0 ) * CUBE( initial_diameter ) * _init_particle_density * ( 1. - ash_mass_frac );
    m_rho_org_bulk          = initial_rc / ( p_volume * ( 1 - _p_void0 ) - m_mass_ash / _rho_ash_bulk );                            // bulk density of char [kg/m^3]
    m_p_voidmin        = 1. - ( 1 / p_volume ) * ( initial_rc * ( 1. - _v_hiT ) / m_rho_org_bulk + m_mass_ash / _rho_ash_bulk ); // bulk density of char [kg/m^3]


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

    for ( int r = 0; r < _NUM_reactions; r++ ) {
      std::string rate_name = "char_gas_reaction" + std::to_string(r) + "_qn" + std::to_string(_Nenv);
      m_reaction_rate_names.push_back( rate_name );
    }
  } // end if ( db_coal_props->findBlock( "SmithChar2016" ) )
}


//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::create_local_labels()
{
    register_new_variable< T >( m_modelLabel );
    register_new_variable< T >( m_gasLabel );
    register_new_variable< T >( m_particletemp );
    register_new_variable< T >( m_particleSize );
    register_new_variable< T >( m_surfacerate );
    //register_new_variable< T >( m_PO2surf[l] );

  for ( int l = 0; l < _NUM_reactions; l++ ) {
    register_new_variable< T >( m_reaction_rate_names[l] );
  }
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::register_initialize(       std::vector<ArchesFieldContainer::VariableInformation> & variable_registry
                                       , const bool                                                     packed_tasks
                                       )
{
  register_variable( m_modelLabel,   ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_gasLabel,     ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_particletemp, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_particleSize, ArchesFieldContainer::COMPUTES, variable_registry );
  register_variable( m_surfacerate,  ArchesFieldContainer::COMPUTES, variable_registry );
  //register_variable( m_PO2surf[l],      ArchesFieldContainer::COMPUTES, variable_registry );


  for ( int l = 0; l < _NUM_reactions; l++ ) {
    register_variable( m_reaction_rate_names[l], ArchesFieldContainer::COMPUTES, variable_registry );
  }
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::initialize( const Patch* patch
                              , ArchesTaskInfoManager* tsk_info
                              )
{

  // model variables
  T& char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel );
  T& gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel );
  T& particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp );
  T& particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize );
  T& surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate );

  char_rate.initialize         ( 0.0 );
  gas_char_rate.initialize     ( 0.0 );
  particle_temp_rate.initialize( 0.0 );
  particle_Size_rate.initialize( 0.0 );
  surface_rate.initialize      ( 0.0 );


  for ( int r = 0; r < _NUM_reactions; r++ ) {
   T& reaction_rate = tsk_info->get_uintah_field_add< T >( m_reaction_rate_names[r] );
   reaction_rate.initialize( 0.0 );
  }

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
  register_variable( m_modelLabel,   ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
  register_variable( m_gasLabel,     ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
  register_variable( m_particletemp, ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
  register_variable( m_particleSize, ArchesFieldContainer::COMPUTES, variable_registry, time_substep );
  register_variable( m_surfacerate,  ArchesFieldContainer::COMPUTES, variable_registry, time_substep );


  for ( int l = 0; l < _NUM_reactions; l++ ) {
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
  register_variable( m_volFraction_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

  // gas species
  for ( int ns = 0; ns < _NUM_species; ns++ ) {
    register_variable( _species_names[ns], ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
  }

  // other particle variables
  register_variable( number_density_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    // from devol model
    register_variable( m_devolRC,        ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_surfAreaF_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    //birth terms

    if ( m_add_char_birth ) {
      register_variable( m_char_birth_qn_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }

    if ( m_add_rawcoal_birth ) {
      register_variable( m_rawcoal_birth_qn_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }

    if ( m_add_length_birth ) {
      register_variable( m_length_birth_qn_name, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    }

    // other models
    register_variable( m_particle_temperature, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_particle_length,      ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_particle_length_qn,   ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_particle_density,     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_rcmass,               ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_char_name,            ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_weight_name,          ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_up_name,              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_vp_name,              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );
    register_variable( m_wp_name,              ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::LATEST, variable_registry, time_substep );

    // RHS
    register_variable( m_RC_RHS,     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_ic_RHS,     ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_w_RHS,      ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
    register_variable( m_length_RHS, ArchesFieldContainer::REQUIRES, 0, ArchesFieldContainer::NEWDW, variable_registry, time_substep );
}

//--------------------------------------------------------------------------------------------------
template<typename T> void
CharOxidationps<T>::eval( const Patch                 * patch
                        ,       ArchesTaskInfoManager * tsk_info
                        )
{
  // gas variables
  constCCVariable<double>& CCuVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_u_vel_name );
  constCCVariable<double>& CCvVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_v_vel_name );
  constCCVariable<double>& CCwVel = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_cc_w_vel_name );
  constCCVariable<double>& volFraction  = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_volFraction_name );

  constCCVariable<double>& den         = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_density_gas_name );
  constCCVariable<double>& temperature = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_gas_temperature_label );
  constCCVariable<double>& MWmix       = tsk_info->get_const_uintah_field_add<constCCVariable<double>>( m_MW_name ); // in kmol/kg_mix

  typedef typename ArchesCore::VariableHelper<T>::ConstType CT; // check comment from other char model

  const double dt = tsk_info->get_dt();

  Vector Dx = patch->dCell();
  const double vol = Dx.x()* Dx.y()* Dx.z();

  std::vector< CT* > species;

  for ( int ns = 0; ns < _NUM_species; ns++ ) {
    CT* species_p = tsk_info->get_const_uintah_field< CT >( _species_names[ns] );
    species.push_back( species_p );
  }

  Uintah::BlockRange range( patch->getCellLowIndex(), patch->getCellHighIndex() );

  // CT& number_density = tsk_info->get_const_uintah_field_add< CT >( number_density_name ); // total number density - unused

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

  std::vector< T* >  reaction_rate;
  std::vector< CT* > old_reaction_rate;

    // model variables
  T& char_rate          = tsk_info->get_uintah_field_add< T >( m_modelLabel );
  T& gas_char_rate      = tsk_info->get_uintah_field_add< T >( m_gasLabel );
  T& particle_temp_rate = tsk_info->get_uintah_field_add< T >( m_particletemp );
  T& particle_Size_rate = tsk_info->get_uintah_field_add< T >( m_particleSize );
  T& surface_rate       = tsk_info->get_uintah_field_add< T >( m_surfacerate );

    // reaction rate
  for ( int r = 0; r < _NUM_reactions; r++ ) {

    T*  reaction_rate_p     = tsk_info->get_uintah_field< T >       ( m_reaction_rate_names[r] );
    CT* old_reaction_rate_p = tsk_info->get_const_uintah_field< CT >( m_reaction_rate_names[r] );

    reaction_rate.push_back    ( reaction_rate_p );
    old_reaction_rate.push_back( old_reaction_rate_p );
  }

  // from devol model
  CT& devolRC = tsk_info->get_const_uintah_field_add< CT >( m_devolRC );

  // particle variables from other models
  CT& particle_temperature = tsk_info->get_const_uintah_field_add< CT >( m_particle_temperature );
  CT& length               = tsk_info->get_const_uintah_field_add< CT >( m_particle_length );
  CT& particle_density     = tsk_info->get_const_uintah_field_add< CT >( m_particle_density );
  CT& rawcoal_mass         = tsk_info->get_const_uintah_field_add< CT >( m_rcmass );
  CT& char_mass            = tsk_info->get_const_uintah_field_add< CT >( m_char_name );
  CT& weight               = tsk_info->get_const_uintah_field_add< CT >( m_weight_name );
  CT& up                   = tsk_info->get_const_uintah_field_add< CT >( m_up_name );
  CT& vp                   = tsk_info->get_const_uintah_field_add< CT >( m_vp_name );
  CT& wp                   = tsk_info->get_const_uintah_field_add< CT >( m_wp_name );

  // birth terms
  CT* rawcoal_birth_ptr = nullptr;
  CT* char_birth_ptr    = nullptr;
  CT* length_birth_ptr  = nullptr;

  if (m_add_rawcoal_birth) {
    rawcoal_birth_ptr = tsk_info->get_const_uintah_field< CT >(m_rawcoal_birth_qn_name);
  }

  if (m_add_char_birth) {
    char_birth_ptr   = tsk_info->get_const_uintah_field< CT >(m_char_birth_qn_name);
  }

  if (m_add_length_birth) {
    length_birth_ptr = tsk_info->get_const_uintah_field< CT >(m_length_birth_qn_name);
  }

  CT& rawcoal_birth = *rawcoal_birth_ptr;
  CT& char_birth    = *char_birth_ptr;
  CT& length_birth  = *length_birth_ptr;

  CT& weight_p_diam = tsk_info->get_const_uintah_field_add< CT >( m_particle_length_qn ); //check
  CT& RC_RHS_source = tsk_info->get_const_uintah_field_add< CT >( m_RC_RHS );
  CT& RHS_source    = tsk_info->get_const_uintah_field_add< CT >( m_ic_RHS );
  CT& RHS_weight    = tsk_info->get_const_uintah_field_add< CT >( m_w_RHS );
  CT& RHS_length    = tsk_info->get_const_uintah_field_add< CT >( m_length_RHS );

  CT& surfAreaF = tsk_info->get_const_uintah_field_add< CT >( m_surfAreaF_name );

  Uintah::BlockRange range_E(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Uintah::parallel_for( range_E, [&](int i, int j, int k){
    char_rate(i,j,k)          = 0.0;
    gas_char_rate(i,j,k)      = 0.0;
    particle_temp_rate(i,j,k) = 0.0;
    particle_Size_rate(i,j,k) = 0.0;
    surface_rate(i,j,k)       = 0.0;

    for ( int r = 0; r < _NUM_reactions; r++ ) {
      (*reaction_rate[r])(i,j,k) = 0.0;
    }
  });


  Uintah::parallel_for( range, [&]( int i,  int j, int k ) {

  // initialize all temporary variables which are use in the cell loop.
    //if ( weight(i,j,k) / m_weight_scaling_constant[l] < _weight_small ) {
    if (volFraction(i,j,k) > 0 ) {

      double D_oxid_mix_l     [ _NUM_reactions ];
      double phi_l            [ _NUM_reactions ];
      double hrxn_l           [ _NUM_reactions ];
      double rh_l             [ _NUM_reactions ];
      double rh_l_new         [ _NUM_reactions ];
      double species_mass_frac[ _NUM_species ];
      double oxid_mass_frac   [ _NUM_reactions ];

      double Sh            [ _NUM_reactions ];
      double co_r          [ _NUM_reactions ];
      double k_r           [ _NUM_reactions ];
      double M_T           [ _NUM_reactions ];
      double effectivenessF[ _NUM_reactions ];

      double F         [ _NUM_reactions ];
      double rh_l_delta[ _NUM_reactions ];
      double F_delta   [ _NUM_reactions ];
      double r_h_ex    [ _NUM_reactions ];
      double r_h_in    [ _NUM_reactions ];
      double dfdrh[3][3];

      for (int l=0; l<_NUM_reactions; l++) {
        for (int lm=0; lm<_NUM_reactions; lm++) {
          dfdrh[l][lm] = 0;
       }
      }
      // populate the temporary variables.
      const double gas_rho    = den(i,j,k);                  // [kg/m^3]
      const double gas_T      = temperature(i,j,k);          // [K]
      const double p_T        = particle_temperature(i,j,k); // [K]
      const double p_rho      = particle_density(i,j,k);     // [kg/m^3]
      const double p_diam     = length(i,j,k);               // [m]
      const double rc         = rawcoal_mass(i,j,k);         // [kg/#]
      const double ch         = char_mass(i,j,k);            // [kg/#]
      const double w          = weight(i,j,k);               // [#/m^3]
      const double MW         = 1. / MWmix(i,j,k);           // [kg mix / kmol mix] (MW in table is 1/MW)
      const double r_devol    = devolRC(i,j,k) * m_RC_scaling_constant * m_weight_scaling_constant; // [kg/m^3/s]
      const double r_devol_ns = -r_devol; // [kg/m^3/s]
      const double RHS_v      = RC_RHS_source(i,j,k) * m_RC_scaling_constant * m_weight_scaling_constant; // [kg/s]
      const double RHS        = RHS_source(i,j,k) * m_char_scaling_constant * m_weight_scaling_constant;  // [kg/s]

      // populate temporary variable vectors
      const double delta = 1e-6;

      for ( int r = 0; r < _NUM_reactions; r++ ) {
        rh_l_new[r] = (*old_reaction_rate[r])(i,j,k); // [kg/m^3/s]
      }

      for ( int r = 0; r < _NUM_reactions; r++ ) { // check this
        oxid_mass_frac[r] = (*species[_oxidizer_indices[r]])(i,j,k); // [mass fraction]
      }

      for ( int ns = 0; ns < _NUM_species; ns++ ) {
        species_mass_frac[ns] = (*species[ns])(i,j,k); // [mass fraction]
      }

      const double CO2onCO = 1. / ( 200. * exp( -9000. / ( _R_cal * p_T ) ) * 44.0 / 28.0 ); // [ kg CO / kg CO2] => [kmoles CO / kmoles CO2] => [kmoles CO2 / kmoles CO]

      for ( int r = 0; r < _NUM_reactions; r++ ) {

        if ( _use_co2co_l[r] ) {
          phi_l[r]  = ( CO2onCO + 1 ) / ( CO2onCO + 0.5 );
          hrxn_l[r] = ( CO2onCO * _HF_CO2 + _HF_CO ) / ( 1 + CO2onCO );
        }
        else {
          phi_l[r]  = _phi_l[r];
          hrxn_l[r] = _hrxn_l[r];
        }
      }

      const double Re_p = std::sqrt( ( CCuVel(i,j,k) - up(i,j,k) ) * ( CCuVel(i,j,k) - up(i,j,k) ) +
                        ( CCvVel(i,j,k) - vp(i,j,k) ) * ( CCvVel(i,j,k) - vp(i,j,k) ) +
                        ( CCwVel(i,j,k) - wp(i,j,k) ) * ( CCwVel(i,j,k) - wp(i,j,k) ) )*
                        p_diam / ( _dynamic_visc / gas_rho ); // Reynolds number [-]

      const double x_org    = (rc + ch) / (rc + ch + m_mass_ash );
      const double cg       = _gasPressure / (_R * gas_T * 1000.); // [kmoles/m^3] - Gas concentration
      const double p_area   = M_PI * SQUARE( p_diam );             // particle surface area [m^2]
      const double p_volume = M_PI / 6. * CUBE( p_diam );          // particle volme [m^3]
      const double p_void   = std::fmax( 1e-10, 1. - ( 1. / p_volume ) * ( ( rc + ch ) / m_rho_org_bulk + m_mass_ash / _rho_ash_bulk ) ); // current porosity. (-) required due to sign convention of char.

      const double Sj       = _init_particle_density / p_rho * ( ( 1 - p_void ) / ( 1 - _p_void0 ) ) * std::sqrt( 1 - std::fmin( 1.0, ( 1. / ( _p_void0 * ( 1. - _p_void0 ) ) ) * log( ( 1 - p_void ) / ( 1 - _p_void0 ) ) ) );
      const double rp  = 2 * p_void * (1. - p_void ) / ( p_rho * Sj * _Sg0 ); // average particle radius [m]

      // Calculate oxidizer diffusion coefficient
      // effect diffusion through stagnant gas (see "Multicomponent Mass Transfer", Taylor and Krishna equation 6.1.14)
      for ( int r = 0; r < _NUM_reactions; r++ ) {

        double sum_x_D = 0;
        double sum_x   = 0;

        for ( int ns = 0; ns < _NUM_species; ns++ ) {
          if ( _oxid_l[r] != _species_names[ns] ) {
            sum_x_D = sum_x_D + species_mass_frac[ns] / ( _MW_species[ns] * _D_mat[_oxidizer_indices[r]][ns] );
            sum_x   = sum_x   + species_mass_frac[ns] / ( _MW_species[ns] );
          }
          else {
            sum_x_D = sum_x_D;
            sum_x   = sum_x;
          }
        }

        D_oxid_mix_l[r] = sum_x / sum_x_D * std::sqrt( CUBE( gas_T / _T0 ) );
        Sh[r]             = 2.0 + 0.6 * std::sqrt( Re_p ) * std::cbrt( _dynamic_visc / ( gas_rho * D_oxid_mix_l[r] ) ); // Sherwood number [-]
        co_r[r]           = cg * ( oxid_mass_frac[r] * MW / _MW_l[r] ); // oxidizer concentration, [kmoles/m^3]
        k_r[r] = ( 10.0 * _a_l[r] * exp( - _e_l[r] / ( _R_cal * p_T)) * _R * p_T * 1000.0) / ( _Mh * phi_l[r] * 101325. ); // [m / s]
        M_T[r]            = p_diam / 2. * std::sqrt( k_r[r] * _Sg0 * Sj * p_rho /                                 // Thiele modulus, Mitchell's formulation
                              ( p_void / _tau / ( 1. / ( 97. * rp * std::sqrt( p_T / _MW_species[r] ) ) + 1. / D_oxid_mix_l[r] ) ) );
        effectivenessF[r] = ( M_T[r] < 1e-5 ) ? 1.0 : 3. / M_T[r] * ( 1. / std::tanh( M_T[r] ) - 1. / M_T[r] ); // effectiveness factor
      }

      // Newton-Raphson solve for rh_l.
      // rh_(n+1) = rh_(n) - (dF_(n)/drh_(n))^-1 * F_(n)
      double rtot    = 0.0;
      double Sfactor = 0.0;
      double Bjm     = 0.0;
      double mtc_r   = 0.0;

      int count = 0;

      for ( int it = 0; it < 100; it++ ) {

        count = count + 1;

        for ( int r = 0; r < _NUM_reactions; r++ ) {
          rh_l[r] = rh_l_new[r];
        }

        // get F and Jacobian -> dF/drh
        rtot    = ( rh_l[0] + rh_l[1] + rh_l[2] ) * x_org * ( 1. - p_void ) + r_devol_ns;
        Sfactor = 0.0;
        Bjm     = 0.0;
        mtc_r   = 0.0;

        for ( int l = 0; l < _NUM_reactions; l++ ) {

          Bjm     = std::min( 80.0, rtot * p_diam / ( D_oxid_mix_l[l] * gas_rho ) ); // [-] // this is the derived for mass flux  BSL chapter 22
          mtc_r   = ( Sh[l] * D_oxid_mix_l[l] * ( ( Bjm >= 1e-7 ) ?  Bjm / ( std::exp( Bjm ) - 1. ) : 1.0 ) ) / p_diam; // [m/s]
          Sfactor = 1 + effectivenessF[l] * p_diam * p_rho * _Sg0 * Sj / ( 6. * ( 1. - p_void ) );
          F[l]    = rh_l[l] - ( _Mh * MW * phi_l[l] * k_r[l] * mtc_r * Sfactor * co_r[l] * cg ) /
                    ( ( MW * cg * ( k_r[l] * x_org * ( 1. - p_void ) * Sfactor + mtc_r ) ) + rtot ); // [kg-char/m^3/s]
        }

        for ( int j = 0; j < _NUM_reactions; j++ ) {

          for ( int k = 0; k < _NUM_reactions; k++ ) {
            rh_l_delta[k] = rh_l[k];
          }

          rh_l_delta[j] = rh_l[j] + delta;

          rtot    = ( rh_l_delta[0] + rh_l_delta[1] + rh_l_delta[2] ) * x_org * ( 1. - p_void ) + r_devol_ns;
          Sfactor = 0.0;
          Bjm     = 0.0;
          mtc_r   = 0.0;

          for ( int l = 0; l < _NUM_reactions; l++ ) {

            Bjm        = std::min( 80.0, rtot * p_diam / ( D_oxid_mix_l[l] * gas_rho ) ); // [-] // this is the derived for mass flux  BSL chapter 22
            mtc_r      = ( Sh[l] * D_oxid_mix_l[l] * ( ( Bjm >= 1e-7 ) ?  Bjm / ( std::exp( Bjm ) - 1. ) : 1.0 ) ) / p_diam; // [m/s]
            Sfactor    = 1 + effectivenessF[l] * p_diam * p_rho * _Sg0 * Sj / ( 6. * ( 1. - p_void ) );
            F_delta[l] = rh_l_delta[l] - ( _Mh * MW * phi_l[l] * k_r[l] * mtc_r * Sfactor * co_r[l] * cg ) /
                         ( ( MW * cg * ( k_r[l] * x_org * ( 1. - p_void ) * Sfactor + mtc_r ) ) + rtot ); // [kg-char/m^3/s]
          }

          // end 07 - Root function
          for ( int r = 0; r < _NUM_reactions; r++ ) {
            dfdrh[r][j] = ( F_delta[r] - F[r] ) / delta;
          }
        }

        // invert Jacobian -> (dF_(n)/drh_(n))^-1
        invf->invert_mat( dfdrh ); // simple matrix inversion for a 2x2 matrix.

        // get rh_(n+1)
        double dominantRate = 0.0;
          //double max_F        = 1e-8;

          for ( int r = 0; r < _NUM_reactions; r++ ) {

            for ( int var = 0; var < _NUM_reactions; var++ ) {
              rh_l_new[r] -= dfdrh[r][var] * F[var];
            }

            dominantRate = std::fmax( dominantRate, std::fabs( rh_l_new[r] ) );
          }

          double residual = 0.0;

          for ( int r = 0; r < _NUM_reactions; r++ ) {
            residual += std::fabs( F[r] ) / dominantRate;
          }

          for ( int r = 0; r < _NUM_reactions; r++ ) {
            rh_l_new[r] = std::fmin( 100000., std::fmax( 0.0, rh_l_new[r] ) ); // max rate adjusted based on pressure (empirical limit)
          }

          if ( residual < 1e-3 ) {
          //if ( residual < 1e-8 ) {
            break;
          }
        } // end for ( int it = 0; it < 100; it++ )

        if ( count > 90 ) {
        //if ( count > 1 ) {
          std::cout << "warning no solution found in char ox: [env " << _Nenv << " "  << i << ", " << j << ", " << k << "] " << std::endl;
          std::cout << "F[0]: "              << F[0]              << std::endl;
          std::cout << "F[1]: "              << F[1]              << std::endl;
          std::cout << "F[2]: "              << F[2]              << std::endl;
          std::cout << "p_void: "            << p_void            << std::endl;
          std::cout << "gas_rho: "           << gas_rho           << std::endl;
          std::cout << "gas_T: "             << gas_T             << std::endl;
          std::cout << "p_T: "               << p_T               << std::endl;
          std::cout << "p_diam: "            << p_diam            << std::endl;
          std::cout << "w: "                 << w                 << std::endl;
          std::cout << "MW: "                << MW                << std::endl;
          std::cout << "r_devol_ns: "        << r_devol_ns        << std::endl;
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
        }

        double char_mass_rate      = 0.0;
        double d_mass              = 0.0;
        double d_mass2             = 0.0;
        double h_rxn               = 0.0; // this is to compute the reaction rate averaged heat of reaction. It is needed so we don't need to clip any additional rates.
        double h_rxn_factor        = 0.0; // this is to compute a multiplicative factor to correct for fp.
        double surface_rate_factor = 0.0; // this is to compute a multiplicative factor to correct for external vs interal rxn.

        const double surfaceAreaFraction = surfAreaF(i,j,k); //w*p_diam*p_diam/AreaSumF(i,j,k); // [-] this is the weighted area fraction for the current particle size.

        for ( int r = 0; r < _NUM_reactions; r++ ) {

          (*reaction_rate[r])(i,j,k) = rh_l_new[r]; // [kg/m^2/s] this is for the intial guess during the next time-step

          // check to see if reaction rate is oxidizer limited.
          const double oxi_lim = ( oxid_mass_frac[r] * gas_rho * surfaceAreaFraction ) / ( dt * w );   // [kg/s/#] // here the surfaceAreaFraction parameter is allowing us to only consume the oxidizer multiplied by the weighted area fraction for the current particle.
          const double rh_l_i  = std::fmin( rh_l_new[r] * p_area * x_org * ( 1. - p_void ), oxi_lim ); // [kg/s/#]

          char_mass_rate      += -rh_l_i; // [kg/s/#] // negative sign because we are computing the destruction rate for the particles.
          d_mass              += rh_l_i;
          r_h_ex[r]            = phi_l[r] * _Mh * k_r[r] * ( rh_l_i / ( phi_l[r] * _Mh * k_r[r] * ( 1 + effectivenessF[r] * p_diam * p_rho * _Sg0 * Sj / ( 6. * ( 1 - p_void ) ) ) ) ); // [kg/m^2/s]
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
        if ( m_add_rawcoal_birth && m_add_char_birth ) {
          char_mass_rate = std::fmax( char_mass_rate, -( ( rc + ch ) / ( dt ) + ( RHS + RHS_v ) / ( vol * w ) + r_devol / w + char_birth(i,j,k) / w + rawcoal_birth(i,j,k) / w ) ); // [kg/s/#]
        }
        else {
          char_mass_rate = std::fmax( char_mass_rate, - ( ( rc + ch ) / ( dt ) + ( RHS + RHS_v ) / ( vol * w ) + r_devol / w ) ); // [kg/s/#]
        }

        char_mass_rate = std::fmin( 0.0, char_mass_rate ); // [kg/s/#] make sure we aren't creating char.

        // organic consumption rate
        char_rate(i,j,k) = ( char_mass_rate * w ) / ( m_char_scaling_constant * m_weight_scaling_constant ); // [kg/m^3/s - scaled]

        // off-gas production rate
        gas_char_rate(i,j,k) = -char_mass_rate * w; // [kg/m^3/s] (negative sign for exchange between solid and gas)

        // heat of reaction source term for enthalpyshaddix
        particle_temp_rate(i,j,k) = h_rxn * 1000. / _Mh * h_rxn_factor * char_mass_rate * w / _ksi; // [J/s/m^4] -- the *1000 is need to convert J/mole to J/kmole. char_mass_rate was already multiplied by x_org * (1-p_void).
                                                                                                    // note: this model is designed to work with EnthalpyShaddix. The effect of ksi has already been added to Qreaction so we divide here.

        // particle shrinkage rate
        const double updated_weight = std::fmax( w / m_weight_scaling_constant + dt / vol * ( RHS_weight(i,j,k) ), 1e-15 );
        const double min_p_diam     = std::pow( m_mass_ash * 6 / _rho_ash_bulk / ( 1. - m_p_voidmin ) / M_PI, 1. / 3. );

        double max_Size_rate = 0.0;

        if ( m_add_length_birth ) {
          max_Size_rate = ( updated_weight * min_p_diam / m_length_scaling_constant - weight_p_diam(i,j,k) ) / dt - ( RHS_length(i,j,k) / vol + length_birth(i,j,k) );
        }
        else {
          max_Size_rate = ( updated_weight * min_p_diam / m_length_scaling_constant - weight_p_diam(i,j,k) ) / dt - ( RHS_length(i,j,k) / vol);
        }

        double Size_rate = ( x_org < 1e-8 ) ? 0.0 :
                    w / m_weight_scaling_constant * 2. * x_org * surface_rate_factor * char_mass_rate /
                    m_rho_org_bulk / p_area / x_org / ( 1. - p_void ) / m_length_scaling_constant; // [m/s]

        particle_Size_rate(i,j,k) = std::fmax( max_Size_rate, Size_rate ); // [m/s] -- these source terms are negative.
        surface_rate(i,j,k)       = char_mass_rate / p_area;               // in [kg/(s # m^2)]

      } // end if ( weight(i,j,k) / m_weight_scaling_constant[l] < _weight_small ) else

      //if (( i== 2) && (j== 10) && (k == 10 )) {
      //  std::cout << "warning no solution found in char ox: [env " << _Nenv << " "  << i << ", " << j << ", " << k << "] " << std::endl;
      //    std::cout << "surface_rate:                            "                  <<  surface_rate(i,j,k)              << std::endl;
      //    std::cout << "particle_Size_rate:                      "            <<  particle_Size_rate(i,j,k)            << std::endl;
      //    std::cout << "gas_char_rate:                           "                 <<  gas_char_rate(i,j,k)            << std::endl;
      //    std::cout << "particle_temp_rate:                      "            <<  particle_temp_rate(i,j,k)            << std::endl;
      //    std::cout << "char_rate:                               "                     <<  char_rate(i,j,k)            << std::endl;
          //std::cout << ": "                     <<  (i,j,k)            << std::endl;
     // }
    }); // end Uintah::parallel_for


  delete invf;
  ///
}
//--------------------------------------------------------------------------------------------------

}
#endif
