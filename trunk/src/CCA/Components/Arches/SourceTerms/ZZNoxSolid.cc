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
#include <CCA/Components/Arches/SourceTerms/ZZNoxSolid.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
//===========================================================================
using namespace std;
using namespace Uintah;
ZZNoxSolid::ZZNoxSolid( std::string src_name, ArchesLabel* field_labels,
    vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_materialManager, req_label_names, type), _field_labels(field_labels)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;
}
ZZNoxSolid::~ZZNoxSolid()
{
  VarLabel::destroy(NO_src_label);
  VarLabel::destroy(HCN_src_label);
  VarLabel::destroy(NH3_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
ZZNoxSolid::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;
  const ProblemSpecP params_root = db->getRootNode();
  //read pressure
  ChemHelper& helper = ChemHelper::self();

  //read and calculate nitrogen content in coal, dry and ash free basis
  if ( params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties") ){
    ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
    if ( db_coal_props->findBlock("ultimate_analysis")){
      ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis");
      db_ua->require("N",_N_ad);
      db_ua->require("ASH",_Ash_ad);
      db_ua->require("H2O",_H2O_ad);
    }
    _Nit = _N_ad/(1-_Ash_ad-_H2O_ad);
  }
  //adjustable input parameters and their default values
  db->getWithDefault("A_BET",               _A_BET,                1000000);
  db->getWithDefault("NSbeta1",             _beta1,                0.0);
  db->getWithDefault("NSbeta2",             _beta2,                0.0);
  db->getWithDefault("NSbeta3",             _beta3,                0.0);
  db->getWithDefault("NSgamma1",            _gamma1,               0.0);
  db->getWithDefault("NSgamma2",            _gamma2,               0.0);
  db->getWithDefault("NSgamma3",            _gamma3,               0.0);
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
  //read devol. & oxi. rate from coal particles
  ProblemSpecP db_source = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
  for ( ProblemSpecP db_src = db_source->findBlock( "src" ); db_src != nullptr; db_src = db_src->findNextBlock("src" ) ){
    std::string model_type;
    db_src->getAttribute("type",model_type);
    if (model_type == "coal_gas_devol"){
      db_src->getWithDefault( "devol_src_label_for_nox", devol_name, "Devol_NOx_source" ); // NOTE: this model is ignoring Tar and birth/death contributions.
    }
    if (model_type == "coal_gas_oxi"){
      db_src->getWithDefault( "char_src_label_for_nox", oxi_name, "Char_NOx_source" );
    }
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
  helper.add_lookup_species( m_O2_name );
  helper.add_lookup_species( m_N2_name );
  helper.add_lookup_species( m_CO_name );
  helper.add_lookup_species( m_H2O_name);
  helper.add_lookup_species( m_H2_name );
  helper.add_lookup_species( m_temperature_name);
  helper.add_lookup_species( m_density_name);
  helper.add_lookup_species( m_mix_mol_weight_name );
  //read DQMOM Information
  m_rcmass_root         = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL);                   //raw coal
  m_rho_coal_root       = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);                    //coal particle density
  m_coal_temperature_root       = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);        //coal particle temperature
  m_num_env             = ArchesCore::get_num_env(db, ArchesCore::DQMOM_METHOD);                     //qn number
  for ( int i = 0; i < m_num_env; i++ ){                                                            //scaling constant of raw coal
    double scaling_const = ArchesCore::get_scaling_constant( db, m_rcmass_root, i );
    m_rc_scaling_const.push_back(scaling_const);
  }
  for ( int i = 0; i < m_num_env; i++ ){                                                            //scaling constant of weight
    double scaling_const = ArchesCore::get_scaling_constant( db, "weight", i );
    m_weight_scaling_const.push_back(scaling_const);
  }
  for ( int i = 0; i < m_num_env; i++ ){                                                            //scaling constant of weight
    double scaling_const = ArchesCore::get_inlet_particle_size( db, i );
    m_particle_size.push_back(scaling_const);
  }
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
  void
ZZNoxSolid::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  ChemHelper& helper = ChemHelper::self();
  _gasPressure=101325.;
  ChemHelper::TableConstantsMapType the_table_constants = helper.get_table_constants();
  if (the_table_constants != nullptr){
    auto press_iter = the_table_constants->find("Pressure");
    if (press_iter !=the_table_constants->end()){
      _gasPressure=press_iter->second;
    }
  }

  std::string taskname = "ZZNoxSolid::eval";
  Task* tsk = scinew Task(taskname, this, &ZZNoxSolid::computeSource, timeSubStep);
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
    // weighted scaled variable = original variable/variable_scaling_constant * weight/weight_scaling_constant
    std::string rcmass_name;
    std::string rho_coalqn_name;
    std::string coal_temperatureqn_name;
    const std::string rcmassqn_name = ArchesCore::append_qn_env( m_rcmass_root, i );             //weighted scaled rcmass
    tsk->requires( which_dw, VarLabel::find(rcmassqn_name), Ghost::None, 0 );
    rcmass_name = ArchesCore::append_env( m_rcmass_root, i );                                    //unweighted unscaled rcmass, original value of rcmass of per particle
    tsk->requires( which_dw, VarLabel::find(rcmass_name), Ghost::None, 0 );
    rho_coalqn_name = ArchesCore::append_env( m_rho_coal_root, i );                              //unweighted unscaled density of coal particle, original value of coal particle density
    tsk->requires( which_dw, VarLabel::find(rho_coalqn_name), Ghost::None, 0 );
    coal_temperatureqn_name = ArchesCore::append_env( m_coal_temperature_root, i );              //unweighted unscaled coal temperature
    tsk->requires( which_dw, VarLabel::find(coal_temperatureqn_name), Ghost::None, 0 );
  }
  // resolve some labels:
  oxi_label              = VarLabel::find( oxi_name);
  devol_label            = VarLabel::find( devol_name);
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
  tsk->requires( which_dw, oxi_label,             Ghost::None, 0 );
  tsk->requires( which_dw, devol_label,           Ghost::None, 0 );
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
  tsk->requires( Task::OldDW, _field_labels->d_volFractionLabel, Ghost::None, 0 );
  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
  void
ZZNoxSolid::computeSource( const ProcessorGroup* pc,
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
    //get information from table
    CCVariable<double> NO_src;
    CCVariable<double> HCN_src;
    CCVariable<double> NH3_src;
    constCCVariable<double> devol;
    constCCVariable<double> oxi;
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
    }
    which_dw->get( devol,          devol_label,            matlIndex, patch, gn, 0 );
    which_dw->get( oxi,            oxi_label,              matlIndex, patch, gn, 0 );
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
    const double A1   = 1.0e10*0.2;         //unit: s-1
    const double E1   = 280451.95;      //unit: j/mol
    const double A2   = 3.0e12*0.2;
    const double E2   = 251151;
    const double A3   = 4.0e6*0.2;
    const double E3   = 133947.2;
    const double A4   = 1.8e8*0.2;
    const double E4   = 113017.95;
    double       n_O2  = 1.0;           // initiliaze oxygen reaction order
    //nox-gas-phase reduction
    const double Agr = 34830;//1350*25.8
    const double Egr = 19953.6;//2400*8.314
    double rate_1,rate_2,rate_3,rate_4,rate_gr;
    //read DQMOM information
    //store sum of coal mass concentration
    CCVariable<double> temp_coal_mass_concentration;
    new_dw->allocateTemporary( temp_coal_mass_concentration, patch );
    temp_coal_mass_concentration.initialize(0.0);
    //store sum of coal mass concentration* coal temperature
    CCVariable<double> temp_coal_mass_temperature;
    new_dw->allocateTemporary( temp_coal_mass_temperature, patch );
    temp_coal_mass_temperature.initialize(0.0);
    for ( int i_env = 0; i_env < m_num_env; i_env++){
      std::string rcmass_name;
      std::string rho_coalqn_name;
      std::string coal_temperatureqn_name;
      constCCVariable<double> rcmass_weighted_scaled;
      constCCVariable<double> rcmass_unweighted_unscaled;
      constCCVariable<double> rho_coal;
      constCCVariable<double> coal_temperature;
      const std::string rcmassqn_name = ArchesCore::append_qn_env( m_rcmass_root, i_env );
      which_dw->get( rcmass_weighted_scaled, VarLabel::find(rcmassqn_name), matlIndex, patch, gn, 0 );
      rcmass_name = ArchesCore::append_env( m_rcmass_root, i_env );
      which_dw->get( rcmass_unweighted_unscaled, VarLabel::find(rcmass_name), matlIndex, patch, gn, 0 );
      rho_coalqn_name = ArchesCore::append_env( m_rho_coal_root, i_env );
      which_dw->get( rho_coal, VarLabel::find(rho_coalqn_name), matlIndex, patch, gn, 0 );
      coal_temperatureqn_name = ArchesCore::append_env( m_coal_temperature_root, i_env );
      which_dw->get( coal_temperature, VarLabel::find(coal_temperatureqn_name), matlIndex, patch, gn, 0 );
      Uintah::parallel_for(range, [&](int i, int j, int k){
          double weight   =  0.0;
          double p_volume =  0.0;
          weight   = rcmass_weighted_scaled(i,j,k)/rcmass_unweighted_unscaled(i,j,k)*m_rc_scaling_const[i_env]*m_weight_scaling_const[i_env];
          p_volume = 1.0/6.0*3.1415926*m_particle_size[i_env]*m_particle_size[i_env]*m_particle_size[i_env];
          temp_coal_mass_concentration(i,j,k) +=weight * p_volume * rho_coal(i,j,k);
          temp_coal_mass_temperature(i,j,k) +=weight * p_volume * rho_coal(i,j,k) * coal_temperature(i,j,k);
          });
    }
    //start calculation
    //NO,HCN,NH3 source terms from solid phase
    auto ComputNOSource=[&](int i,int j,int k){
      return  _Nit*(devol(i,j,k)*_beta1+oxi(i,j,k)*_gamma1)*_MW_NO /_MW_N;
    };
    auto ComputHCNSource=[&](int i,int j,int k){
      return  _Nit*(devol(i,j,k)*_beta2+oxi(i,j,k)*_gamma2)*_MW_HCN/_MW_N;
    };
    auto ComputNH3Source=[&](int i,int j,int k){
      return  _Nit*(devol(i,j,k)*_beta3+oxi(i,j,k)*_gamma3)*_MW_NH3/_MW_N;
    };
    //NO,HCN,NH3 source terms in continuum phase
    Uintah::parallel_for(range, [&](int i, int j, int k){
        if (vol_fraction(i,j,k) > 0.5) {
        //convert mixture molecular weight
        double mix_mol_weight_r = 1.0/mix_mol_weight(i,j,k)/1000.0;   //(kg/mol)
        //convert unit:  (mol concentration: mol/m3)
        double O2_m  = O2(i,j,k)       * density(i,j,k)/_MW_O2;   //(mol/m3)
        double CO_m  = CO(i,j,k)       * density(i,j,k)/_MW_CO;   //(mol/m3)
        double H2_m  = H2(i,j,k)       * density(i,j,k)/_MW_H2;   //(mol/m3)
        double N2_m  = N2(i,j,k)       * density(i,j,k)/_MW_N2;   //(mol/m3)
        double H2O_m = H2O(i,j,k)      * density(i,j,k)/_MW_H2O;  //(mol/m3)
        double NO_m  = tran_NO(i,j,k)  * density(i,j,k)/_MW_NO;   //(mol/m3)
        //double HCN_m = tran_HCN(i,j,k) * density(i,j,k)/_MW_HCN;  //(mol/m3)
        //double NH3_m = tran_NH3(i,j,k) * density(i,j,k)/_MW_NH3;  //(mol/m3)
        //convert unit: (mol fraction: mol/mol)
        double O2_mp  = O2(i,j,k)      /_MW_O2 * mix_mol_weight_r; //(mol/mol)
        double NO_mp  = tran_NO(i,j,k) /_MW_NO * mix_mol_weight_r; //(mol/mol)
        double HCN_mp = tran_HCN(i,j,k)/_MW_HCN* mix_mol_weight_r; //(mol/mol)
        double NH3_mp = tran_NH3(i,j,k)/_MW_NH3* mix_mol_weight_r; //(mol/mol)
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
        double O_m  = 3.970e5  * std::pow(temperature(i,j,k),-0.5)  * std::sqrt(O2_m) *  std::exp(-31090.0  /  temperature(i,j,k));                      //unit: mol/m3
        double OH_m = 2.129e2  * std::pow(temperature(i,j,k),-0.57) * std::sqrt(O2_m) *  std::sqrt(H2O_m)   *  std::exp(-4595.0/temperature(i,j,k));     //unit: mol/m3
        //calculate thermal NOx source
        double NO_S1  = 2.0*rate_f1*O_m*N2_m*(1.0-rate_r1*rate_r2*NO_m*NO_m/rate_f1/rate_f2/N2_m/O2_m)/(1.0+rate_r1*NO_m/(rate_f2*O2_m+rate_f3*OH_m))*_MW_NO;
        double HCN_S1 = 0.0;
        double NH3_S1 = 0.0;
        //fuel-nox,S2
        //reaction rates from r1~r4:
        rate_1 = A1 * std::exp(-E1/_R/temperature(i,j,k)) * HCN_mp * std::pow(O2_mp,n_O2);        //(mol/mol*s-1)
        rate_2 = A2 * std::exp(-E2/_R/temperature(i,j,k)) * HCN_mp * NO_mp;                       //(mol/mol*s-1)
        rate_3 = A3 * std::exp(-E3/_R/temperature(i,j,k)) * NH3_mp * std::pow(O2_mp,n_O2);        //(mol/mol*s-1)
        rate_4 = A4 * std::exp(-E4/_R/temperature(i,j,k)) * NH3_mp * NO_mp;                       //(mol/mol*s-1)
        //limit the reaction rates
        double limit_NO,limit_HCN,limit_NH3;
        limit_NO  = NO_mp/delta_t  + (ComputNOSource(i,j,k)  + NO_S1) /_MW_NO  * mix_mol_weight_r/density(i,j,k);
        limit_HCN = HCN_mp/delta_t + (ComputHCNSource(i,j,k) + HCN_S1)/_MW_HCN * mix_mol_weight_r/density(i,j,k);
        limit_NH3 = NH3_mp/delta_t + (ComputNH3Source(i,j,k) + NH3_S1)/_MW_NH3 * mix_mol_weight_r/density(i,j,k);
        if (rate_2+rate_4>limit_NO+rate_1+rate_3){
          if (rate_4==0){
            rate_2=limit_NO+rate_1+rate_3;
          }
          if (rate_4>0){
            rate_2=(limit_NO+rate_1+rate_3)*rate_2/(rate_2+rate_4);
            rate_4=(limit_NO+rate_1+rate_3)*rate_4/(rate_2+rate_4);
          }
        }
        if (rate_1+rate_2>limit_HCN){
          if (rate_2==0){
            rate_1=limit_HCN;
          }
          if (rate_2>0){
            rate_1=limit_HCN*rate_1/(rate_1+rate_2);
            rate_2=limit_HCN*rate_2/(rate_1+rate_2);
          }
        }
        if (rate_3+rate_4>limit_NH3){
          if (rate_4==0){
            rate_3=limit_NH3;
          }
          if (rate_4>0){
            rate_3=limit_NH3*rate_3/(rate_3+rate_4);
            rate_4=limit_NH3*rate_4/(rate_3+rate_4);
          }
        }
        //calculate de Soete NOx source
        double NO_S2  = (rate_1+rate_3-rate_2-rate_4) *_MW_NO / mix_mol_weight_r*density(i,j,k);              //(kg/sm3)
        double HCN_S2 = (-rate_1-rate_2)              *_MW_HCN/ mix_mol_weight_r*density(i,j,k);              //(kg/sm3)
        double NH3_S2 = (-rate_3-rate_4)              *_MW_NH3/ mix_mol_weight_r*density(i,j,k);              //(kg/sm3)
        //nox reduction in gas phase,S3
        rate_gr = Agr * std::exp(-Egr/_R/temperature(i,j,k)) * std::pow(NO_m,2.25) * (CO_m+H2_m);             //(mol/sm3)
        //limit the reaction rate
        limit_NO  = NO_m/delta_t  + (ComputNOSource(i,j,k)  + NO_S1 + NO_S2)/_MW_NO;
        if (rate_gr>limit_NO){
          rate_gr=limit_NO;
        }
        double NO_S3     =(0-rate_gr) * _MW_NO;                                                               //(kg/sm3)
        double HCN_S3    = rate_gr * _MW_HCN*0.5;                                                             //(kg/sm3)
        double NH3_S3    = rate_gr * _MW_NH3*0.5;                                                             //(kg/sm3)
        //nox reduction by particle surface,S4
        double pNO   = NO_mp*_gasPressure/101325;                                                             //(atm);
        double Tav   = (temperature(i,j,k)+temp_coal_mass_temperature(i,j,k)/temp_coal_mass_concentration(i,j,k))/2.0;
        double NO_red_solid_rate = 230.0 * std::exp (-142737.485/_R/Tav) * pNO;                               //(mol/m2 BET s)
        double NO_red_solid      = NO_red_solid_rate * _A_BET * temp_coal_mass_concentration(i,j,k) * _MW_NO; //(kg/m3 s)
        //limit the reaction rate
        limit_NO  = NO_m*_MW_NO/delta_t  + (ComputNOSource(i,j,k)  + NO_S1 + NO_S2 + NO_S3);
        if (NO_red_solid>limit_NO){
          NO_red_solid=limit_NO;
        }
        double NO_S4 = -NO_red_solid;
        double HCN_S4=0.0;
        double NH3_S4=0.0;
        //calculate source terms in total
        NO_src(i,j,k)  = (NO_S1+NO_S2+NO_S3+NO_S4+ComputNOSource(i,j,k))*vol_fraction(i,j,k);                 //(kg/sm3)
        HCN_src(i,j,k) = (HCN_S1+HCN_S2+HCN_S3+HCN_S4+ComputHCNSource(i,j,k))*vol_fraction(i,j,k);            //(kg/sm3)
        NH3_src(i,j,k) = (NH3_S1+NH3_S2+NH3_S3+NH3_S4+ComputNH3Source(i,j,k))*vol_fraction(i,j,k);            //(kg/sm3)
        } else {
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
ZZNoxSolid::sched_initialize( const LevelP& level, SchedulerP& sched )
{

  ChemHelper& helper = ChemHelper::self();
  _gasPressure=101325.;
  ChemHelper::TableConstantsMapType the_table_constants = helper.get_table_constants();
  if (the_table_constants != nullptr){
    auto press_iter = the_table_constants->find("Pressure");
    if (press_iter !=the_table_constants->end()){
      _gasPressure=press_iter->second;
    }
  }

  string taskname = "ZZNoxSolid::initialize";

  Task* tsk = scinew Task(taskname, this, &ZZNoxSolid::initialize);

  tsk->computes(NO_src_label);
  tsk->computes(HCN_src_label);
  tsk->computes(NH3_src_label);

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));

}
  void
ZZNoxSolid::initialize( const ProcessorGroup* pc,
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
