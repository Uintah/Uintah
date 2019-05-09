/* Author Information  */
// Model developed by Zhi Zhang, Zhenshan Li, Ningsheng Cai from Tsinghua University;
// Coding by Zhi Zhang under the instruction of Minmin Zhou, Ben Issac and Jeremy Thornock;
// Parameters fitted based on DTF experimental data of a Chinese bituminous coal Tsinghua.
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/SourceTerms/psNox.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>


//===========================================================================
using namespace std;
using namespace Uintah;
psNOx::psNOx( std::string src_name, ArchesLabel* field_labels,
    vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_materialManager, req_label_names, type), _field_labels(field_labels)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;
}
psNOx::~psNOx()
{
  VarLabel::destroy(NO_src_label);
  VarLabel::destroy(HCN_src_label);
  VarLabel::destroy(NH3_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------

namespace {

  auto split_lambda = []( const double& tillman_value, const double& vol_tar_split,
                       double& tar_N_split, const double& vol_fraction ){

    double split = (tillman_value - vol_tar_split*tar_N_split) / vol_fraction;
    //adjusting the tar split if bounds are exceeded
    if ( split < 0.0 ){
      split = 0.0;
      tar_N_split = tillman_value / vol_tar_split;
    }
    if ( split > 1.0 ){
      split = 1.0;
      tar_N_split = ( tillman_value - ( 1. - vol_tar_split ) ) / vol_tar_split;
    }

    //if tar goes out of bounds, throw and error
    // In this case, one needs to explore the possiblity that:
    // 1) The NOx model is wrong or
    // 2) The parameters need to be manually specificied
    // 3) More UQ works needs to be performed for the fuel of interest
    if ( tar_N_split < 0.0 || tar_N_split > 1.0 ){
      throw ProblemSetupException("Error: The splits for the volatiles cannot be resolved for this coal in the NO model.", __FILE__, __LINE__);
    }
    return split;

  };
}
  void
psNOx::problemSetup(const ProblemSpecP& inputdb)
{

  CoalHelper& coal_helper = CoalHelper::self();
  ChemHelper& chemistry_helper = ChemHelper::self();

  ProblemSpecP db = inputdb;
  const ProblemSpecP params_root = db->getRootNode();

  CoalHelper::CoalDBInfo& coal_db = coal_helper.get_coal_db();

  //Compute the nitrogen content, dry/ash-free
  _Nit = coal_db.coal.N / ( 1. - coal_db.coal.H2O - coal_db.coal.ASH );

  // A note on DEFAULT parameters:
  // Default rate/split parameters are a result of the DTF analysis performed
  // by Jeremy, Oscar, Phil and Sean. These parameters were from a best result from a Latin Hypercube
  // sampling performed for a V/UQ analysis.
  db->getWithDefault("tar_to_HCN",          _alpha2,         0.);
  db->getWithDefault("tar_to_NH3",          _alpha3,         1.);

  const double eps = 1e-8;
  if  (_alpha2+_alpha3 > 1.0 + eps || _alpha2+_alpha3 < 0.0 - eps ){

    throw ProblemSetupException("tar_to_HCN + tar_to_NH3 must be greater than 0 and less than 1.0.", __FILE__, __LINE__);

  }

  tarFrac = coal_helper.get_coal_db().Tar_fraction;

  // Here we are constraining the values of the light gas split so that the
  // total volatile split obtained matches that from
  // Combustion of Solid Fuel and Waste by Tillman
  std::string coal_rank = coal_helper.get_coal_db().coal_rank;
  double volFrac=std::max(1e-20,1.0-tarFrac);
  // Defaults (from UQ analysis)
  double F_v_hcn = 0.7;
  double F_v_nh3 = 0.3;

  double y_HCN = 0.0;
  double y_NH3 = 0.0;

  // estimated from Tillman
  if (coal_rank=="lignite"){

    y_HCN=0.1;
    y_NH3=0.9;

  } else if (coal_rank=="high_volatile_bituminous"){

    y_HCN=0.45;
    y_NH3=0.4;

  } else if (coal_rank=="subbituminous"){

    y_HCN=0.2;
    y_NH3=0.77;

  } else {

    throw ProblemSetupException("Error: In NOx model, coal type not supported: "+coal_rank, __FILE__, __LINE__);

  }

  F_v_hcn = split_lambda( y_HCN, tarFrac, _alpha2, volFrac );
  F_v_nh3 = split_lambda( y_NH3, tarFrac, _alpha3, volFrac );

  db->getWithDefault("devol_to_HCN",        _beta2,          F_v_hcn);
  db->getWithDefault("devol_to_NH3",        _beta3,          F_v_nh3);
  // Defaults from UQ analysis best fit
  db->getWithDefault("charOxy_to_HCN",      _gamma2,         0.5);
  db->getWithDefault("charOxy_to_NH3",      _gamma3,         0.5);

  //NO direct pathways
   _beta1=1.0-_beta2-_beta3;     // devol

   _alpha1=1.0-_alpha2-_alpha3;  // tar

   _gamma1=1.0- _gamma2-_gamma3; // char-oxy

  // Defaults from UQ analysis best fit
  // reburn parameters
  db->getWithDefault("PreExpReburn",_A_reburn  , 1000  );
  db->getWithDefault("ExpReburn",   _E_reburn  , 27000 );
  db->getWithDefault("PowerReburn", _m_gr      , 2.3   );
   // (Factor for A in first De soete reaction)
  db->getWithDefault("F1_De_soete", _F1_Desoete      ,1.);

  //read concentrations of species in the table
  db->getWithDefault("o2_label",             m_O2_name,            "O2");
  db->getWithDefault("n2_label",             m_N2_name,            "N2");
  db->getWithDefault("co_label",             m_CO_name,            "CO");
  db->getWithDefault("h2o_label",            m_H2O_name,           "H2O");
  db->getWithDefault("h2_label",             m_H2_name,            "H2");
  db->getWithDefault("temperature_label",    m_temperature_name,   "temperature");
  db->getWithDefault("density_label",        m_density_name,       "density");
  db->getWithDefault("mix_mol_weight_label", m_mix_mol_weight_name,"mixture_molecular_weight");
  //NOx species
  db->getWithDefault("NO_label",             NO_name,              "NO_zz");
  db->getWithDefault("HCN_label",            HCN_name,             "HCN_zz");
  db->getWithDefault("NH3_label",            NH3_name,             "NH3_zz");

  db->getWithDefault("tar_src_label",        tar_src_name,    "eta_source3");

  //read devol. & oxi. rate from coal particles
  ProblemSpecP db_source = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
  for ( ProblemSpecP db_src = db_source->findBlock( "src" ); db_src != nullptr; db_src = db_src->findNextBlock("src" ) ){
    std::string model_type;
    db_src->getAttribute("type",model_type);
    if (model_type == "coal_gas_devol"){
      db_src->getWithDefault( "devol_src_label_for_nox", devol_name, "Devol_NOx_source" );
      db_src->getWithDefault( "bd_devol_src_label", bd_devol_name, "birth_death_devol_source" );
    }
    if (model_type == "coal_gas_oxi"){
      db_src->getWithDefault( "char_src_label_for_nox", oxi_name, "Char_NOx_source" );
      db_src->getWithDefault( "bd_char_src_label", bd_oxi_name, "birth_death_char_source" );
    }
  }
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
  if (db_coal_props->findBlock("FOWYDevol")) {
    ProblemSpecP db_BT = db_coal_props->findBlock("FOWYDevol");
    db_BT->require("v_hiT", m_v_hiT); //
  } else {
    m_v_hiT = 1.0; // if fowy devol model is not used than we set devol fraction to 1 in the case of birth/death.
  }


  //source terms name,and ...
  db->findBlock("NO_src")->getAttribute( "label",  NO_src_name );
  db->findBlock("HCN_src")->getAttribute( "label", HCN_src_name );
  db->findBlock("NH3_src")->getAttribute( "label", NH3_src_name );
  _mult_srcs.push_back( NO_src_name );
  _mult_srcs.push_back( HCN_src_name );
  _mult_srcs.push_back( NH3_src_name );
  NO_src_label     = VarLabel::create( NO_src_name, CCVariable<double>::getTypeDescription() );
  HCN_src_label    = VarLabel::create( HCN_src_name, CCVariable<double>::getTypeDescription() );
  NH3_src_label    = VarLabel::create( NH3_src_name, CCVariable<double>::getTypeDescription() );
  //read concentrations of species in the table
  chemistry_helper.add_lookup_species( m_O2_name );
  chemistry_helper.add_lookup_species( m_N2_name );
  chemistry_helper.add_lookup_species( m_CO_name );
  chemistry_helper.add_lookup_species( m_H2O_name);
  chemistry_helper.add_lookup_species( m_H2_name );
  chemistry_helper.add_lookup_species( m_temperature_name);
  chemistry_helper.add_lookup_species( m_density_name);
  chemistry_helper.add_lookup_species( m_mix_mol_weight_name );
  //read DQMOM Information
  m_coal_temperature_root  = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);        // coal particle temperature
  m_weight_root            = "w";                                                                                // particle weight root name
  m_length_root            = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);               // particle diameter root name
  m_p_rho_root             = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);            // particle density root name
  m_rc_mass_root           = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);            // raw coal
  m_char_mass_root         = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR);               // char coal
  m_num_env                = ArchesCore::get_num_env(db, ArchesCore::DQMOM_METHOD);                              // qn number

  double init_particle_density = ArchesCore::get_inlet_particle_density( db );
  double ash_mass_frac = coal_helper.get_coal_db().ash_mf;
  for ( int i = 0; i < m_num_env; i++){
    double initial_diameter = ArchesCore::get_inlet_particle_size( db, i ); // [m]
    m_initial_rc.push_back( (M_PI/6.0)*initial_diameter*initial_diameter*initial_diameter*init_particle_density*(1.-ash_mass_frac) ); // [kg_i / #]
  }
  m_Fd_M = m_v_hiT/(1.0 - m_v_hiT);
  m_Fd_B = -m_v_hiT*m_v_hiT/(1.0-m_v_hiT);
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
  void
psNOx::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  ChemHelper& chemistry_helper = ChemHelper::self();
  _gasPressure=101325.;
  ChemHelper::TableConstantsMapType the_table_constants = chemistry_helper.get_table_constants();
  if (the_table_constants != nullptr){
    auto press_iter = the_table_constants->find("Pressure");
    if (press_iter !=the_table_constants->end()){
      _gasPressure=press_iter->second;
    }
  }

  std::string taskname = "psNOx::eval";
  Task* tsk = scinew Task(taskname, this, &psNOx::computeSource, timeSubStep);
  Task::WhichDW which_dw;
  if (timeSubStep == 0) {
    tsk->computes(NO_src_label);
    tsk->computes(HCN_src_label);
    tsk->computes(NH3_src_label);
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
    tsk->modifies(NO_src_label);
    tsk->modifies(HCN_src_label);
    tsk->modifies(NH3_src_label);
  }

  for ( int i = 0; i < m_num_env; i++){

    std::string coal_temperature_name = ArchesCore::append_env( m_coal_temperature_root, i );
    m_coal_temperature_label.push_back(  VarLabel::find(coal_temperature_name));
    tsk->requires( which_dw, m_coal_temperature_label[i], Ghost::None, 0 );

    std::string weight_name = ArchesCore::append_env( m_weight_root, i );
    m_weight_label.push_back(  VarLabel::find(weight_name));
    tsk->requires( which_dw, m_weight_label[i], Ghost::None, 0 );

    std::string length_name = ArchesCore::append_env( m_length_root, i );
    m_length_label.push_back(  VarLabel::find(length_name));
    tsk->requires( which_dw, m_length_label[i], Ghost::None, 0 );

    std::string p_rho_name = ArchesCore::append_env( m_p_rho_root, i );
    m_p_rho_label.push_back(  VarLabel::find(p_rho_name));
    tsk->requires( which_dw, m_p_rho_label[i], Ghost::None, 0 );

    std::string rc_mass_name = ArchesCore::append_env( m_rc_mass_root, i );
    m_rc_mass_label.push_back(  VarLabel::find(rc_mass_name));
    tsk->requires( which_dw, m_rc_mass_label[i], Ghost::None, 0 );

    std::string char_mass_name = ArchesCore::append_env( m_char_mass_root, i );
    m_char_mass_label.push_back(  VarLabel::find(char_mass_name));
    tsk->requires( which_dw, m_char_mass_label[i], Ghost::None, 0 );
  }
  // resolve some labels:
  oxi_label              = VarLabel::find( oxi_name);
  devol_label            = VarLabel::find( devol_name);
  bd_oxi_label           = VarLabel::find( bd_oxi_name);
  bd_devol_label         = VarLabel::find( bd_devol_name);
  tar_src_label          = VarLabel::find( tar_src_name);
  m_o2_label             = VarLabel::find( m_O2_name);
  m_n2_label             = VarLabel::find( m_N2_name);
  m_co_label             = VarLabel::find( m_CO_name);
  m_h2o_label            = VarLabel::find( m_H2O_name);
  m_h2_label             = VarLabel::find( m_H2_name);
  m_temperature_label    = VarLabel::find( m_temperature_name);
  m_density_label        = VarLabel::find( m_density_name);
  m_mix_mol_weight_label = VarLabel::find( m_mix_mol_weight_name);
  m_NO_label             = VarLabel::find( NO_name);
  m_HCN_label            = VarLabel::find( HCN_name);
  m_NH3_label            = VarLabel::find( NH3_name);
  m_NO_RHS_label         = VarLabel::find( NO_name+"_RHS");
  m_NH3_RHS_label        = VarLabel::find( NH3_name+"_RHS");
  m_HCN_RHS_label        = VarLabel::find( HCN_name+"_RHS");
  tsk->requires( which_dw, oxi_label,             Ghost::None, 0 );
  tsk->requires( which_dw, devol_label,           Ghost::None, 0 );
  tsk->requires( which_dw, bd_oxi_label,             Ghost::None, 0 );
  tsk->requires( which_dw, bd_devol_label,           Ghost::None, 0 );
  tsk->requires( which_dw, tar_src_label,         Ghost::None, 0 );
  tsk->requires( which_dw, m_o2_label,            Ghost::None, 0 );
  tsk->requires( which_dw, m_n2_label,            Ghost::None, 0 );
  tsk->requires( which_dw, m_co_label,            Ghost::None, 0 );
  tsk->requires( which_dw, m_h2o_label,           Ghost::None, 0 );
  tsk->requires( which_dw, m_h2_label,            Ghost::None, 0 );
  tsk->requires( which_dw, m_temperature_label,   Ghost::None, 0 );
  tsk->requires( which_dw, m_density_label,       Ghost::None, 0 );
  tsk->requires( which_dw, m_mix_mol_weight_label,Ghost::None, 0 );
  tsk->requires( which_dw, m_NO_label,            Ghost::None, 0 );
  tsk->requires( which_dw, m_HCN_label,           Ghost::None, 0 );
  tsk->requires( which_dw, m_NH3_label,           Ghost::None, 0 );
  tsk->requires( which_dw, m_NO_RHS_label,        Ghost::None, 0 );
  tsk->requires( which_dw, m_NH3_RHS_label,       Ghost::None, 0 );
  tsk->requires( which_dw, m_HCN_RHS_label,       Ghost::None, 0 );
  tsk->requires( Task::OldDW, _field_labels->d_volFractionLabel, Ghost::None, 0 );
  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
  void
psNOx::computeSource( const ProcessorGroup* pc,
    const PatchSubset*    patches,
    const MaterialSubset* matls,
    DataWarehouse*  old_dw,
    DataWarehouse*  new_dw,
    int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    Ghost::GhostType  gn  = Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> NO_src;
    CCVariable<double> HCN_src;
    CCVariable<double> NH3_src;
    constCCVariable<double> tar_src;
    constCCVariable<double> devol;
    constCCVariable<double> oxi;
    constCCVariable<double> bd_devol;
    constCCVariable<double> bd_oxi;
    constCCVariable<double> O2;
    constCCVariable<double> N2;
    constCCVariable<double> CO;
    constCCVariable<double> H2O;
    constCCVariable<double> H2;
    constCCVariable<double> temperature;
    constCCVariable<double> density;
    constCCVariable<double> mix_mol_weight;
    constCCVariable<double> tran_NO;
    constCCVariable<double> tran_HCN;
    constCCVariable<double> tran_NH3;
    constCCVariable<double> vol_fraction;
    constCCVariable<double> NO_rhs;
    constCCVariable<double> NH3_rhs;
    constCCVariable<double> HCN_rhs;

    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){

      which_dw = old_dw;
      new_dw->allocateAndPut( NO_src, NO_src_label,  matlIndex, patch );
      new_dw->allocateAndPut( HCN_src,HCN_src_label, matlIndex, patch );
      new_dw->allocateAndPut( NH3_src,NH3_src_label, matlIndex, patch );
      NO_src.initialize(0.0);
      HCN_src.initialize(0.0);
      NH3_src.initialize(0.0);

    } else {

      which_dw = new_dw;
      new_dw->getModifiable( NO_src,  NO_src_label,  matlIndex, patch );
      new_dw->getModifiable( HCN_src, HCN_src_label, matlIndex, patch );
      new_dw->getModifiable( NH3_src, NH3_src_label, matlIndex, patch );
      NO_src.initialize(0.0);
      HCN_src.initialize(0.0);
      NH3_src.initialize(0.0);

    }

    which_dw->get( tar_src,        tar_src_label,          matlIndex, patch, gn, 0 );
    which_dw->get( devol,          devol_label,            matlIndex, patch, gn, 0 );
    which_dw->get( bd_devol,       bd_devol_label,         matlIndex, patch, gn, 0 );
    which_dw->get( oxi,            oxi_label,              matlIndex, patch, gn, 0 );
    which_dw->get( bd_oxi,         bd_oxi_label,           matlIndex, patch, gn, 0 );
    which_dw->get( O2,             m_o2_label,             matlIndex, patch, gn, 0 ); //mass percentage (kg/kg)
    which_dw->get( N2,             m_n2_label,             matlIndex, patch, gn, 0 );
    which_dw->get( CO,             m_co_label,             matlIndex, patch, gn, 0 );
    which_dw->get( H2O,            m_h2o_label,            matlIndex, patch, gn, 0 );
    which_dw->get( H2,             m_h2_label,             matlIndex, patch, gn, 0 );
    which_dw->get( temperature,    m_temperature_label,    matlIndex, patch, gn, 0 );
    which_dw->get( density,        m_density_label,        matlIndex, patch, gn, 0 ); // (kg/m3)
    which_dw->get( mix_mol_weight, m_mix_mol_weight_label, matlIndex, patch, gn, 0 ); // (mol/g)
    which_dw->get( tran_NO,        m_NO_label,             matlIndex, patch, gn, 0 );
    which_dw->get( tran_HCN,       m_HCN_label,            matlIndex, patch, gn, 0 );
    which_dw->get( tran_NH3,       m_NH3_label,            matlIndex, patch, gn, 0 );
    new_dw->get( NO_rhs,         m_NO_RHS_label,         matlIndex, patch, gn, 0 );
    new_dw->get( NH3_rhs,        m_NH3_RHS_label,        matlIndex, patch, gn, 0 );
    new_dw->get( HCN_rhs,        m_HCN_RHS_label,        matlIndex, patch, gn, 0 );
    old_dw->get( vol_fraction, _field_labels->d_volFractionLabel, matlIndex, patch, gn, 0 );

    //get timestep
    delt_vartype DT;
    old_dw->get( DT, _field_labels->d_delTLabel);
    const double delta_t = DT;
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    //define constants
    const double  _MW_O2= 0.032;       // unit: (kg/mol)
    const double  _MW_CO= 0.028;
    const double  _MW_H2= 0.002;
    const double  _MW_H2O=0.018;
    const double  _MW_N  =0.014;
    const double  _MW_N2 =0.028;
    const double  _MW_NO =0.030;
    const double  _MW_NH3=0.017;
    const double  _MW_HCN=0.027;
    const double  _R     =8.314;       // unit: (kg/molK)

    //thermal-nox parameters
    const double Af1  = 1.8e8;         // unit: (m3/mols)
    const double Ef1  = 38370;
    const double Ar1  = 3.8e7;
    const double Er1  = 425;
    const double Af2  = 1.8e4;
    const double Ef2  = 4680;
    const double Ar2  = 3.81e3;
    const double Er2  = 20820;
    const double Af3  = 7.1e7;
    const double Ef3  = 450;
    const double Ar3  = 1.7e8;
    const double Er3  = 24560;

    double rate_f1,rate_r1,rate_f2,rate_r2,rate_f3,rate_r3;

    //fuel-nox De soete mechanism parameters
    const double A1   = 1.0e10*_F1_Desoete;         //unit: s-1
    const double E1   = 280451.95;      //unit: j/mol
    const double A2   = 3.0e12;
    const double E2   = 251151;
    const double A3   = 4.0e6;
    const double E3   = 133947.2;
    const double A4   = 1.8e8;
    const double E4   = 113017.95;
    double       n_O2  = 1.0;           // initiliaze oxygen reaction order

    //nox-gas-phase reduction
    const double Agr = _A_reburn;//1350*25.8
    const double Egr = _E_reburn;//2400*8.314
    double rate_1,rate_2,rate_3,rate_4,rate_gr;

    Vector Dx = patch->dCell();
    const double vol = Dx.x()*Dx.y()*Dx.z();

    //read DQMOM information
    //store sum of coal mass concentration
    std::vector< CCVariable<double> > temp_organic_conc_times_area(m_num_env);
    CCVariable<double> temp_current_organic;
    CCVariable<double> temp_initial_organic;
    std::vector< constCCVariable<double> > coal_temperature(m_num_env);
    std::vector< constCCVariable<double> > length(m_num_env);
    std::vector< constCCVariable<double> > p_rho(m_num_env);
    std::vector< constCCVariable<double> > weight(m_num_env);
    std::vector< constCCVariable<double> > rc_mass(m_num_env);
    std::vector< constCCVariable<double> > char_mass(m_num_env);

    new_dw->allocateTemporary( temp_current_organic, patch );
    temp_current_organic.initialize(0.0);

    new_dw->allocateTemporary( temp_initial_organic, patch );
    temp_initial_organic.initialize(0.0);

    for ( int i_env = 0; i_env < m_num_env; i_env++){

      new_dw->allocateTemporary( temp_organic_conc_times_area[i_env], patch );
      temp_organic_conc_times_area[i_env].initialize(0.0);
      which_dw->get( coal_temperature[i_env], m_coal_temperature_label[i_env], matlIndex, patch, gn, 0 );
      which_dw->get( weight[i_env], m_weight_label[i_env], matlIndex, patch, gn, 0 );
      which_dw->get( length[i_env], m_length_label[i_env], matlIndex, patch, gn, 0 );
      which_dw->get( p_rho[i_env], m_p_rho_label[i_env], matlIndex, patch, gn, 0 );
      which_dw->get( rc_mass[i_env], m_rc_mass_label[i_env], matlIndex, patch, gn, 0 );
      which_dw->get( char_mass[i_env], m_char_mass_label[i_env], matlIndex, patch, gn, 0 );

    }

    //store sum of coal mass concentration* coal temperature
    for ( int i_env = 0; i_env < m_num_env; i_env++){
      Uintah::parallel_for(range, [&](int i, int j, int k){
          double p_area = M_PI*length[i_env](i,j,k)*length[i_env](i,j,k);     // m^2
          double p_volume = M_PI/6.*length[i_env](i,j,k)*length[i_env](i,j,k)*length[i_env](i,j,k); // particle volme [m^3]
          double p_mass = max(1e-50,p_rho[i_env](i,j,k)*p_volume); // particle mass [kg / #]
          double organic_mass = rc_mass[i_env](i,j,k) + char_mass[i_env](i,j,k); // organic mass [kg organic / #]
          double organic_frac = min(1.0,max(0.0,organic_mass/p_mass));     // [-]
          temp_current_organic(i,j,k) += organic_mass*weight[i_env](i,j,k); //[kg/m3]
          temp_initial_organic(i,j,k) += m_initial_rc[i_env]*weight[i_env](i,j,k); //[kg/m3]
          temp_organic_conc_times_area[i_env](i,j,k) = weight[i_env](i,j,k)*organic_frac*p_area; // #/m^3 * [-] * m^2 = [m^2/m^3]
      });
    }

    //start calculation
    const int nSpecies=5;
    const int nRates=11;
    enum spc{ sNO, sHCN, sNH3, sN2, sO2};

    //  ----- RXN KEY  ----  //
    //      #0 forward extended zeldovich
    //      #1 reverse extended zeldovich
    //      #2 fuel NOx de soete r1   [HCN][O2] --> [NO]
    //      #3 fuel NOx de soete r2   [HCN][NO] --> [N2]
    //      #4 fuel NOx de soete r3   [NH3][O2] --> [NO]
    //      #5 fuel NOx de soete r4   [NH3][NO] --> [N2]
    //      #6 reburning
    //      #7 NOx-char reduction Adel sarofim
    //      #8 Devol
    //      #9 Char-oxy
    //      #10 tar

    const double prate_coef[nSpecies][nRates] // producing reactions
    // rxn# 0   1    2    3    4    5    6    7    8         9        10
         {{1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, _beta1, _gamma1, _alpha1},   // NO
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, _beta2, _gamma2, _alpha2},   // HCN
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _beta3, _gamma3, _alpha3},   // NH3
          {0.0, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,    0.0,     0.0},       // N2
          {0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    0.0,     0.0}};      // O2

    const double nrate_coef[nSpecies][nRates] // consuming reactions
    // rxn# 0   1    2    3    4    5    6    7    8    9    10
         {{0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 },    // NO
          {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },    // HCN
          {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },    // NH3
          {0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },    // N2
          {0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }};   // O2

    std::vector<std::vector<int> > rel_ind(nSpecies, std::vector<int>(0) ); // relevant indices
    std::vector<int > rel_ind_size(nSpecies,0);
    for (int ispec =0 ; ispec<nSpecies ; ispec++){
      for (int irxn =0 ; irxn<nRates ; irxn++){
        if (nrate_coef[ispec][irxn] >0) {
          rel_ind[ispec].push_back(irxn);
        }
      }
    }

        //prate_coef[sNO][0]=1.0;
        //nrate_coef[sN2][0]=0.5;
        //nrate_coef[sO2][0]=0.5;
        //nrate_coef[sNO][1]=1.0;
        //prate_coef[sN2][1]=0.5;
        //prate_coef[sO2][1]=0.5;
        //prate_coef[sNO][2]=1.0;
        //nrate_coef[sHCN][2]=1.0;
        //nrate_coef[sO2][2]=0.5;
        //prate_coef[sN2][3]=1.0;
        //nrate_coef[sHCN][3]=1.0;
        //nrate_coef[sNO][3]=1.0;
        //nrate_coef[sNH3][4]=1.0;
        //prate_coef[sNO][4]=1.0;
        //nrate_coef[sO2][4]=0.5;
        //nrate_coef[sNH3][5]=1.0;
        //nrate_coef[sNO][5]=1.0;
        //prate_coef[sN2][5]=1.0;
        //nrate_coef[sNO][6]=1.0;
        //prate_coef[sHCN][6]=1.0;
        //prate_coef[sNH3][6]=0.0;
        //nrate_coef[sNO][7]=1.0;
        //prate_coef[sNO][8]=_beta1;
        //prate_coef[sHCN][8]=_beta2;
        //prate_coef[sNH3][8]=_beta3;

        //prate_coef[sNO][9]=_gamma1;
        //prate_coef[sHCN][9]=_gamma2;
        //prate_coef[sNH3][9]=_gamma3;
        //
        //prate_coef[sNO][10]=_alpha1;
        //prate_coef[sHCN][10]=_alpha2;
        //prate_coef[sNH3][10]=_alpha3;


    //NO,HCN,NH3 source terms in continuum phase
    Uintah::parallel_for(range, [&](int i, int j, int k){
        if (vol_fraction(i,j,k) > 0.5) {
        std::vector<double> rxn_rates(nRates,0.);
        std::vector<std::vector<double> > rate_coef(nSpecies,std::vector<double > (nRates));
        for (int ispec =0 ; ispec<nSpecies ; ispec++){
          for (int irxn =0 ; irxn<nRates ; irxn++){
            rate_coef[ispec][irxn]= prate_coef[ispec][irxn] - nrate_coef[ispec][irxn];
          }
        }

        //convert mixture molecular weight
        double mix_mol_weight_r = 1.0/mix_mol_weight(i,j,k)/1000.0;   //(kg/mol)
        //convert unit:  (mol concentration: mol/m3)
        double O2_m  = max(O2(i,j,k),1e-20)   * density(i,j,k)/_MW_O2;   //(mol/m3)
        double CO_m  = CO(i,j,k)       * density(i,j,k)/_MW_CO;   //(mol/m3)
        double H2_m  = H2(i,j,k)       * density(i,j,k)/_MW_H2;   //(mol/m3)
        double N2_m  = max(N2(i,j,k), 1e-20)   * density(i,j,k)/_MW_N2;   //(mol/m3)
        double H2O_m = H2O(i,j,k)      * density(i,j,k)/_MW_H2O;  //(mol/m3)
        double NO_m  = max(tran_NO(i,j,k),1e-20 ) * density(i,j,k)/_MW_NO;   //(mol/m3)
        std::vector<double> Spec_i_m(nSpecies);
        Spec_i_m[sNO]=NO_m;
        Spec_i_m[sHCN]=max(tran_HCN(i,j,k),1e-20) * density(i,j,k)/_MW_HCN;
        Spec_i_m[sNH3]=max(tran_NH3(i,j,k),1e-20) * density(i,j,k)/_MW_NH3;
        Spec_i_m[sN2]= N2_m;
        Spec_i_m[sO2]= O2_m;
        //convert unit: (mol fraction: mol/mol)
        double O2_mp  = O2(i,j,k)      /_MW_O2 * mix_mol_weight_r; //(mol/mol)
        double NO_mp  = max(tran_NO(i,j,k) ,1e-20)/_MW_NO * mix_mol_weight_r; //(mol/mol)
        double HCN_mp = max(tran_HCN(i,j,k),1e-20)/_MW_HCN* mix_mol_weight_r; //(mol/mol)
        double NH3_mp = max(tran_NH3(i,j,k),1e-20)/_MW_NH3* mix_mol_weight_r; //(mol/mol)

        //determine reaction order of O2
        if (O2_mp<4.1e-3){
        n_O2=1.0;
        }else if(O2_mp>=4.1e-3&&O2_mp<=1.1e-2){
          n_O2=-3.95-0.9*log(O2_mp);
        }else if(O2_mp>1.1e-2&&O2_mp<0.03){
          n_O2=-0.35-0.1*log(O2_mp);
        }else if(O2_mp>=0.03){
          n_O2=0.0;
        }
        //thermal-nox,S1
        //reaction rates from r1~r3:
        rate_f1 = Af1 * std::exp(-Ef1/temperature(i,j,k));
        rate_r1 = Ar1 * std::exp(-Er1/temperature(i,j,k));
        rate_f2 = Af2 * std::exp(-Ef2/temperature(i,j,k))*temperature(i,j,k);
        rate_r2 = Ar2 * std::exp(-Er2/temperature(i,j,k))*temperature(i,j,k);
        rate_f3 = Af3 * std::exp(-Ef3/temperature(i,j,k));
        rate_r3 = Ar3 * std::exp(-Er3/temperature(i,j,k));

        //calculate [O] & [OH]
        double O_m  = 3.970e5  / std::sqrt(temperature(i,j,k))  * std::sqrt(O2_m) *  std::exp(-31090.0  /  temperature(i,j,k));                      //unit: mol/m3
        double OH_m = 2.129e2  * std::pow(temperature(i,j,k),-0.57) * std::sqrt(O2_m) *  std::sqrt(H2O_m)   *  std::exp(-4595.0/temperature(i,j,k));     //unit: mol/m3

        //calculate thermal NOx source extended zeldovich mechanism
        //double NO_S1  = 2.0*rate_f1*O_m*N2_m*(1.0-rate_r1*rate_r2*NO_m*NO_m/rate_f1/rate_f2/N2_m/O2_m)/(1.0+rate_r1*NO_m/(rate_f2*O2_m+rate_f3*OH_m));
        double NO_thermal_rate_const=2.0*rate_f1*O_m*N2_m/(1.0+rate_r1*NO_m/(rate_f2*O2_m+rate_f3*OH_m)); // mol of NO / m^3 / s
        double NO_S1_prod  = NO_thermal_rate_const;
        double NO_S1_cons  = rate_r1*rate_r2*NO_m*NO_m*NO_thermal_rate_const/(rate_f1*rate_f2*N2_m*O2_m);

        rxn_rates[0]=NO_S1_prod;

        rxn_rates[1]=NO_S1_cons;

        // de soete fuel nox rates
        //reaction rates from r1~r4:
        rate_1 = A1 * std::exp(-E1/_R/temperature(i,j,k)) * HCN_mp * std::pow(O2_mp,n_O2)/ mix_mol_weight_r*density(i,j,k);        //(mol/m^3 / s-1)
        rxn_rates[2]=rate_1;

        rate_2 = A2 * std::exp(-E2/_R/temperature(i,j,k)) * HCN_mp * NO_mp/ mix_mol_weight_r*density(i,j,k);                       //(mol/m^3 / s-1)
        rxn_rates[3]=rate_2;

        rate_3 = A3 * std::exp(-E3/_R/temperature(i,j,k)) * NH3_mp * std::pow(O2_mp,n_O2)/ mix_mol_weight_r*density(i,j,k);        //(mol/m^3 / s-1)
        rxn_rates[4]=rate_3;

        rate_4 = A4 * std::exp(-E4/_R/temperature(i,j,k)) * NH3_mp * NO_mp/ mix_mol_weight_r*density(i,j,k);                       //(mol/m^3 / s-1)
        rxn_rates[5]=rate_4;
        //nox reduction in gas phase, reburning
        rate_gr = Agr * std::exp(-Egr/_R/temperature(i,j,k)) * std::pow(NO_m,_m_gr) * (CO_m+H2_m);             //(mol/sm3)

        rxn_rates[6]=rate_gr;

        //nox reduction by particle surface, Adel sarofim char+NOx reduction
        double pNO   = NO_mp*_gasPressure/101325;                                                             //(atm);
        double NO_red_solid = 0.0;
        for ( int i_env = 0; i_env < m_num_env; i_env++){
          double NO_red_solid_rate = 4.8e4 * std::exp (-145184.8/_R/coal_temperature[i_env](i,j,k)) * pNO;                               //(mol/m2/s)
          NO_red_solid += NO_red_solid_rate*temp_organic_conc_times_area[i_env](i,j,k); // mol/m^2/s * m^2/m^3 (mol/m3/s)
        }

        rxn_rates[7]=NO_red_solid;

        // birth_death_rate = sum_i( - b/d_rc ) + sum_i( - b/d_ch ) // this is done so that the negative ch st is countered with the positive rc st.
        // devolRate = sum_i( (1-f_T)*rxn_devol_i ) + Fdevol*birth_death_rate
        // CharOxyRate = sum_i( rxn_char_i ) + Foxi*birth_death_rate
        // for TarRate the following is used to create Tar: tarSrc = sum_i( f_T*rxn_devol_i )
        // Then reactions of tar and soot are computed in the Brown soot model and "returned" to the gas phase through eta_source3 (tar_src)
        //
        //  v_hiT is the ultimate yield of devol products.
        //  f_T is the fraction of devol products that are tar (heavy gas instead of light).
        //  sum_i is the sum over all particle environments
        //  b/d_rc is the birth death term of rc from the perspective of the particles (thus a - sign for the gas)
        //  b/d_ch is the birth death term of ch from the perspective of the particles (thus a - sign for the gas)
        double V_org_fraction = temp_current_organic(i,j,k)/temp_initial_organic(i,j,k); // [-] fraction of organic to initial organic
        double Fdevol = min(1.0, max( 0.0, m_Fd_M * V_org_fraction + m_Fd_B )); // [-] fraction of b/d that goes to devol.
        double Foxi = 1.0 - Fdevol; // [-] fraction of b/d that goes to oxi.
        double birth_death_rate = bd_devol(i,j,k) + bd_oxi(i,j,k);
        double birth_death_devol_rate = Fdevol*birth_death_rate;
        double birth_death_oxi_rate = Foxi*birth_death_rate;
        double devolRate   =  _Nit*(devol(i,j,k)+birth_death_devol_rate) / _MW_N;      // mol / m^3 / s of N
        double CharOxyRate =  _Nit*(oxi(i,j,k)+birth_death_oxi_rate) /_MW_N;      // mol / m^3 / s of N
        double TarRate     =  _Nit*tar_src(i,j,k) /_MW_N;      // mol / m^3 / s of N

        rxn_rates[8]  = devolRate;
        rxn_rates[9]  = CharOxyRate;
        rxn_rates[10] = TarRate;

        for (int ix =0 ; ix<nRates ; ix++){
          for (int ispec =0 ; ispec<nSpecies ; ispec++){
            if (nrate_coef[ispec][ix] != 0.0 ) {
              rxn_rates[ix]  =  min(rxn_rates[ix], Spec_i_m[ispec]/delta_t/nrate_coef[ispec][ix]);
            }
          }
        }
        //for (int ispec =0 ; ispec<nSpecies ; ispec++){
        //  double num=0.0;
        //  double denom=0.0;
        //  double instantanousRate=0.0;
        //  for (int ix =0 ; ix<nRates ; ix++){
        //    num+=  prate_coef[ispec][ix]*rxn_rates[ix];
        //    denom+=nrate_coef[ispec][ix]*rxn_rates[ix];
        //  }
        //  instantanousRate=-denom;
        //  denom=denom/Spec_i_m[ispec];
         // double netRate;

        //  double final_species = num/denom*(1.0-std::exp(-delta_t*denom)) + Spec_i_m[ispec]*std::exp(-delta_t*denom); // quasi-analytical approach ( may need a check on denom)
          //double final_species = (num*delta_t + Spec_i_m[ispec])/(1.0 + denom*delta_t); // quasi-implicit approach
        //  netRate= (final_species-Spec_i_m[ispec])/delta_t-num;                 //(mol/sm3)

        //  for (int ispec2 =0 ; ispec2<nSpecies ; ispec2++){
       //     for (unsigned int irxn =0 ; irxn<rel_ind[ispec].size() ; irxn++){
        //      rate_coef[ispec2][rel_ind[ispec][irxn]]*= min(netRate/instantanousRate,1.000); // linear scaling of rates (negitive rates only).
       //     }
        //  }
       // }

       for (int ix =0 ; ix<nRates ; ix++){
         NO_src(i,j,k)  +=rxn_rates[ix]*rate_coef[sNO ][ix]*_MW_NO;
         HCN_src(i,j,k) +=rxn_rates[ix]*rate_coef[sHCN][ix]*_MW_HCN;
         NH3_src(i,j,k) +=rxn_rates[ix]*rate_coef[sNH3][ix]*_MW_NH3;
       }

       if (NO_src(i,j,k) < 0) {
         NO_src(i,j,k)  = - min(-NO_src(i,j,k), Spec_i_m[sNO]/delta_t*_MW_NO - NO_rhs(i,j,k)/vol );
       }

       if (HCN_src(i,j,k) < 0 ){
         HCN_src(i,j,k) = - min(-HCN_src(i,j,k), Spec_i_m[sHCN]/delta_t*_MW_HCN - HCN_rhs(i,j,k)/vol );
       }

       if (NH3_src(i,j,k) < 0) {
         NH3_src(i,j,k) = - min(-NH3_src(i,j,k), Spec_i_m[sNH3]/delta_t*_MW_NH3 - NH3_rhs(i,j,k)/vol );
       }

     } else {  // if volFraction
        NO_src(i,j,k)  = 0.0; //(kg/sm3)
        HCN_src(i,j,k) = 0.0; //(kg/sm3)
        NH3_src(i,j,k) = 0.0; //(kg/sm3)
     }
    });
  } // end for patch loop
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
  void
psNOx::sched_initialize( const LevelP& level, SchedulerP& sched )
{

  ChemHelper& chemistry_helper = ChemHelper::self();
  _gasPressure=101325.;
  ChemHelper::TableConstantsMapType the_table_constants = chemistry_helper.get_table_constants();
  if (the_table_constants != nullptr){
    auto press_iter = the_table_constants->find("Pressure");
    if (press_iter !=the_table_constants->end()){
      _gasPressure=press_iter->second;
    }
  }

  string taskname = "psNOx::initialize";

  Task* tsk = scinew Task(taskname, this, &psNOx::initialize);

  tsk->computes(NO_src_label);
  tsk->computes(HCN_src_label);
  tsk->computes(NH3_src_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
  void
psNOx::initialize( const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> NO_src;
    CCVariable<double> HCN_src;
    CCVariable<double> NH3_src;

    new_dw->allocateAndPut( NO_src, NO_src_label, matlIndex, patch );
    new_dw->allocateAndPut( HCN_src,HCN_src_label, matlIndex, patch );
    new_dw->allocateAndPut( NH3_src, NH3_src_label, matlIndex, patch );

    NO_src.initialize(0.0);
    HCN_src.initialize(0.0);
    NH3_src.initialize(0.0);
  }
}
