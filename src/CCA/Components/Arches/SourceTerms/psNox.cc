/* Author Information  */
// Model developed by Zhi Zhang, Zhenshan Li, Ningsheng Cai from Tsinghua University;
// Coding by Zhi Zhang under the instruction of Minmin Zhou, Ben Issac and Jeremy Thornock;
// Parameters fitted based on DTF experimental data of a Chinese bituminous coal Tsinghua.
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
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
psNox::psNox( std::string src_name, ArchesLabel* field_labels,
    vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_sharedState, req_label_names, type), _field_labels(field_labels)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;
}
psNox::~psNox()
{
  VarLabel::destroy(_src_label);
  VarLabel::destroy(NO_src_label);
  VarLabel::destroy(HCN_src_label);
  VarLabel::destroy(NH3_src_label);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void
psNox::problemSetup(const ProblemSpecP& inputdb)
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


  db->getWithDefault("devol_to_HCN",        _beta2,          0.8);
  db->getWithDefault("devol_to_NH3",        _beta3,          0.2);
  db->getWithDefault("charOxy_to_HCN",      _gamma2,         0.8);
  db->getWithDefault("charOxy_to_NH3",      _gamma3,         0.2);
  db->getWithDefault("Tar_to_HCN",          _alpha2,         0.8);
  db->getWithDefault("Tar_to_NH3",          _alpha3,         0.2);


   _beta1=1.0-_beta2-_beta3;     // devol

   _alpha1=1.0-_alpha2-_alpha3;   // tar

   _gamma1=1.0- _gamma2-_gamma3; // char-oxy


  db->getWithDefault("PreExpReburn",_A_reburn  ,34830  );
  db->getWithDefault("ExpReburn",   _E_reburn  ,19953.6);


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

  db->getWithDefault("Tar_src_label",        tar_src_name,    "eta_source3");
  db->getWithDefault("Tar_fraction",         tarFrac,                  .209);

  //read devol. & oxi. rate from coal particles 
  ProblemSpecP db_source = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("TransportEqns")->findBlock("Sources");
  for ( ProblemSpecP db_src = db_source->findBlock( "src" ); db_src != nullptr; db_src = db_src->findNextBlock("src" ) ){
    std::string model_type; 
    db_src->getAttribute("type",model_type);
    if (model_type == "coal_gas_devol"){
      db_src->getAttribute("label",devol_name);
    }
    if (model_type == "coal_gas_oxi"){
      db_src->getAttribute("label",oxi_name);
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
  m_coal_temperature_root       = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE);        //coal particle temperature
  m_num_env             = ArchesCore::get_num_env(db, ArchesCore::DQMOM_METHOD);                     //qn number
  length_root=(ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE));                                 // paritcle diameter root name


  for ( int i = 0; i < m_num_env; i++ ){                                                            //scaling constant of raw coal
    double scaling_const = ArchesCore::get_scaling_constant( db, m_rcmass_root, i ); 
    m_rc_scaling_const.push_back(scaling_const); 
  }
  for ( int i = 0; i < m_num_env; i++ ){                                                            //scaling constant of weight
    double scaling_const = ArchesCore::get_scaling_constant( db, "weight", i ); 
    m_weight_scaling_const.push_back(scaling_const); 
  }
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
  void
psNox::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
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

  std::string taskname = "psNox::eval";
  Task* tsk = scinew Task(taskname, this, &psNox::computeSource, timeSubStep);
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
    std::string coal_temperatureqn_name;
    std::string weight_name;
    weight_name = ArchesCore::append_env( "w", i );                               //weight
    tsk->requires( which_dw, VarLabel::find(weight_name), Ghost::None, 0 ); 

    coal_temperatureqn_name = ArchesCore::append_env( m_coal_temperature_root, i );              //unweighted unscaled coal temperature 
    tsk->requires( which_dw, VarLabel::find(coal_temperatureqn_name), Ghost::None, 0 ); 

    std::string length_name = ArchesCore::append_env( length_root, i );
    m_length_label.push_back(  VarLabel::find(length_name));
    tsk->requires( which_dw, m_length_label[i], Ghost::None, 0 );
  }  
  // resolve some labels:
  oxi_label              = VarLabel::find( oxi_name);
  devol_label            = VarLabel::find( devol_name);
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
  tsk->requires( which_dw, oxi_label,             Ghost::None, 0 );
  tsk->requires( which_dw, devol_label,           Ghost::None, 0 );
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
  tsk->requires( Task::OldDW, _field_labels->d_volFractionLabel, Ghost::None, 0 );
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
  void
psNox::computeSource( const ProcessorGroup* pc,
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
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();
    //get information from table 
    CCVariable<double> NO_src;
    CCVariable<double> HCN_src;
    CCVariable<double> NH3_src;
    constCCVariable<double> tar_src;
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
      NO_src.initialize(0.0);
      HCN_src.initialize(0.0);
      NH3_src.initialize(0.0);
    }
    which_dw->get( tar_src,        tar_src_label,          matlIndex, patch, gn, 0 );
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
    const double A1   = 1.0e10*12;         //unit: s-1
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
    //read DQMOM information
    //store sum of coal mass concentration

    std::vector< CCVariable<double> > temp_coal_mass_concentration(m_num_env); 
    std::vector<constCCVariable<double> > coal_temperature(m_num_env);
    std::vector< constCCVariable<double> > length(m_num_env);
    for ( int i_env = 0; i_env < m_num_env; i_env++){
    std::string coal_temperatureqn_name;
    new_dw->allocateTemporary( temp_coal_mass_concentration[i_env], patch );
    temp_coal_mass_concentration[i_env].initialize(0.0); 

    coal_temperatureqn_name = ArchesCore::append_env( m_coal_temperature_root, i_env );
    which_dw->get( coal_temperature[i_env], VarLabel::find(coal_temperatureqn_name), matlIndex, patch, gn, 0 );  

    which_dw->get( length[i_env], m_length_label[i_env], matlIndex, patch, gn, 0 );
    }

    //store sum of coal mass concentration* coal temperature
    for ( int i_env = 0; i_env < m_num_env; i_env++){
      constCCVariable<double> weight;
      std::string weight_name;
      weight_name = ArchesCore::append_env( "w", i_env );
      which_dw->get( weight, VarLabel::find(weight_name), matlIndex, patch, gn, 0 );  
      Uintah::parallel_for(range, [&](int i, int j, int k){
          //double weight   =  0.0;
          double p_area =  0.0;
          //weight   = rcmass_weighted_scaled(i,j,k)/rcmass_unweighted_unscaled(i,j,k)*m_rc_scaling_const[i_env]*m_weight_scaling_const[i_env]; 
          p_area = M_PI*length[i_env](i,j,k)*length[i_env](i,j,k);     // m^2
          temp_coal_mass_concentration[i_env](i,j,k) = weight(i,j,k) * p_area; // m^2 / m^3
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
        double NO_S1_prod  = rate_f1*rate_f2*N2_m*O2_m*NO_thermal_rate_const;
        double NO_S1_cons  = rate_r1*rate_r2*NO_m*NO_m*NO_thermal_rate_const;

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
        rate_gr = Agr * std::exp(-Egr/_R/temperature(i,j,k)) * std::pow(NO_m,2.25) * (CO_m+H2_m);             //(mol/sm3)

        rxn_rates[6]=rate_gr;

        //nox reduction by particle surface, Adel sarofim char+NOx reduction
        double pNO   = NO_mp*_gasPressure/101325;                                                             //(atm);
        double NO_red_solid =0.0;
        for ( int i_env = 0; i_env < m_num_env; i_env++){
          double NO_red_solid_rate = 4.8e4 * std::exp (-145180./_R/coal_temperature[i_env](i,j,k)) * pNO;                               //(mol/m2 BET s)
          NO_red_solid += NO_red_solid_rate*temp_coal_mass_concentration[i_env](i,j,k); //(mol/m3 s)
        }

        rxn_rates[7]=NO_red_solid;
       
        double devolRate   =  _Nit*devol(i,j,k)* (1.0 - tarFrac) / _MW_N;      // mol / m^3 / s of N
        double CharOxyRate =  _Nit*oxi(i,j,k) /_MW_N;      // mol / m^3 / s of N
        double TarRate     =  _Nit*tar_src(i,j,k) /_MW_N;      // mol / m^3 / s of N

        rxn_rates[8]=devolRate;
        rxn_rates[9]=CharOxyRate;
        rxn_rates[10]=TarRate;
      
        for (int ispec =0 ; ispec<nSpecies ; ispec++){
          double num=0.0;
          double denom=0.0;
          double instantanousRate=0.0;
          for (int ix =0 ; ix<nRates ; ix++){
            num+=  prate_coef[ispec][ix]*rxn_rates[ix];
            denom+=nrate_coef[ispec][ix]*rxn_rates[ix];
          }
          instantanousRate=-denom;
          denom=denom/Spec_i_m[ispec];
          double netRate;                                

          //double final_species = num/denom*(1.0-std::exp(-delta_t*denom)) + Spec_i_m[ispec]*std::exp(-delta_t*denom); // quasi-analytical approach ( may need a check on denom)
          double final_species = (num*delta_t + Spec_i_m[ispec])/(1.0 + denom*delta_t); // quasi-implicit approach
          netRate= (final_species-Spec_i_m[ispec])/delta_t-num;                 //(mol/sm3)                          
              
          for (int ispec2 =0 ; ispec2<nSpecies ; ispec2++){
            for (unsigned int irxn =0 ; irxn<rel_ind[ispec].size() ; irxn++){
              rate_coef[ispec2][rel_ind[ispec][irxn]]*= min(netRate/instantanousRate,1.000); // linear scaling of rates (negitive rates only).
            }
          }
        }

       for (int ix =0 ; ix<nRates ; ix++){
         NO_src(i,j,k)  +=rxn_rates[ix]*rate_coef[sNO ][ix]*_MW_NO;
         HCN_src(i,j,k) +=rxn_rates[ix]*rate_coef[sHCN][ix]*_MW_HCN;
         NH3_src(i,j,k) +=rxn_rates[ix]*rate_coef[sNH3][ix]*_MW_NH3;
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
psNox::sched_initialize( const LevelP& level, SchedulerP& sched )
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

  string taskname = "psNox::initialize";

  Task* tsk = scinew Task(taskname, this, &psNox::initialize);

  tsk->computes(NO_src_label);
  tsk->computes(HCN_src_label);
  tsk->computes(NH3_src_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
  void
psNox::initialize( const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

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
