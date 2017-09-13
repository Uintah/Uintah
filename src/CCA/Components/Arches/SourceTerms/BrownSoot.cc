#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/SourceTerms/BrownSoot.h>

//===========================================================================

using namespace std;
using namespace Uintah;

BrownSoot::BrownSoot( std::string src_name, ArchesLabel* field_labels,
                                                    vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_sharedState, req_label_names, type), _field_labels(field_labels)
{

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;

}

BrownSoot::~BrownSoot()
{
      VarLabel::destroy(m_tar_src_label      );
      VarLabel::destroy(m_nd_src_label       );
      VarLabel::destroy(m_soot_mass_src_label);
      VarLabel::destroy(m_balance_src_label  );
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
BrownSoot::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->getWithDefault("mix_mol_weight_label", m_mix_mol_weight_name,   "mixture_molecular_weight");
  db->getWithDefault("tar_label",            m_tar_name,              "Tar");
  db->getWithDefault("Ysoot_label",          m_Ysoot_name,            "Ysoot");
  db->getWithDefault("Ns_label",             m_Ns_name,               "Ns");
  db->getWithDefault("o2_label",             m_O2_name,               "O2");
  db->getWithDefault("oh_label",             m_OH_name,               "OH");
  db->getWithDefault("co2_label",	         m_CO2_name,		      "CO2");
  db->getWithDefault("h2o_label",            m_H2O_name,              "H2O");
  db->getWithDefault("density_label",        m_rho_name,              "density");
  db->getWithDefault("temperature_label",    m_temperature_name,      "radiation_temperature");

  db->findBlock("tar_src")->getAttribute( "label", m_tar_src_name );
  db->findBlock("num_density_src")->getAttribute( "label", m_nd_name );
  db->findBlock("soot_mass_src")->getAttribute( "label", m_soot_mass_name );
  db->findBlock("mass_balance_src")->getAttribute( "label", m_balance_name );

  // Since we are producing multiple sources, we load each name into this vector
  // so that we can do error checking upon src term retrieval.
  _mult_srcs.push_back( m_tar_src_name );
  _mult_srcs.push_back( m_nd_name );
  _mult_srcs.push_back( m_soot_mass_name );
  _mult_srcs.push_back( m_balance_name );

  m_tar_src_label       = VarLabel::create( m_tar_src_name, CCVariable<double>::getTypeDescription() );
  m_nd_src_label        = VarLabel::create( m_nd_name, CCVariable<double>::getTypeDescription() );
  m_soot_mass_src_label = VarLabel::create( m_soot_mass_name, CCVariable<double>::getTypeDescription() );
  m_balance_src_label   = VarLabel::create( m_balance_name, CCVariable<double>::getTypeDescription() );

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( m_O2_name );
  helper.add_lookup_species( m_OH_name );
  helper.add_lookup_species( m_rho_name );
  helper.add_lookup_species( m_CO2_name );
  helper.add_lookup_species( m_H2O_name );
  helper.add_lookup_species( m_mix_mol_weight_name );
  //_field_labels->add_species( m_temperature_name );


}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
BrownSoot::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  std::string taskname = "BrownSoot::eval";
  Task* tsk = scinew Task(taskname, this, &BrownSoot::computeSource, timeSubStep);

  Task::WhichDW which_dw;
  if (timeSubStep == 0) {
    tsk->computes(m_tar_src_label);
    tsk->computes(m_nd_src_label);
    tsk->computes(m_soot_mass_src_label);
    tsk->computes(m_balance_src_label);
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
    tsk->modifies(m_tar_src_label);
    tsk->modifies(m_nd_src_label);
    tsk->modifies(m_soot_mass_src_label);
    tsk->modifies(m_balance_src_label);
  }
  // resolve some labels:
  m_mix_mol_weight_label  = VarLabel::find( m_mix_mol_weight_name);
  m_tar_label             = VarLabel::find( m_tar_name);
  m_Ysoot_label           = VarLabel::find( m_Ysoot_name);
  m_Ns_label              = VarLabel::find( m_Ns_name);
  m_o2_label              = VarLabel::find( m_O2_name);
  m_oh_label              = VarLabel::find( m_OH_name);
  m_co2_label             = VarLabel::find( m_CO2_name);
  m_h2o_label             = VarLabel::find( m_H2O_name);
  m_temperature_label     = VarLabel::find( m_temperature_name);
  m_rho_label             = VarLabel::find( m_rho_name);

  tsk->requires( which_dw, m_mix_mol_weight_label,               Ghost::None, 0 );
  tsk->requires( which_dw, m_tar_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_Ysoot_label,                        Ghost::None, 0 );
  tsk->requires( which_dw, m_Ns_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_o2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_oh_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_co2_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_h2o_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_temperature_label,                  Ghost::None, 0 );
  tsk->requires( which_dw, m_rho_label,                          Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
  
  //get the system pressure:
  ChemHelper& helper = ChemHelper::self();
  ChemHelper::TableConstantsMapType tab_constants = helper.get_table_constants();
  auto i_press = tab_constants->find("Pressure");
  if ( i_press != tab_constants->end() ){
    m_sys_pressure = i_press->second;
  } else {
    m_sys_pressure = 101325.0; //otherise assume atmospheric
  }

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
BrownSoot::computeSource( const ProcessorGroup* pc,
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

    CCVariable<double> tar_src;
    CCVariable<double> num_density_src;
    CCVariable<double> soot_mass_src;
    CCVariable<double> balance_src;

    constCCVariable<double> mix_mol_weight;
    constCCVariable<double> Tar;
    constCCVariable<double> Ysoot;
    constCCVariable<double> Ns;
    constCCVariable<double> O2;
    constCCVariable<double> OH;
    constCCVariable<double> CO2;
    constCCVariable<double> H2O;
    constCCVariable<double> rho;
    constCCVariable<double> temperature;

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
        which_dw = old_dw;
        new_dw->allocateAndPut( tar_src, m_tar_src_label, matlIndex, patch );
        new_dw->allocateAndPut( num_density_src, m_nd_src_label, matlIndex, patch );
        new_dw->allocateAndPut( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
        new_dw->allocateAndPut( balance_src, m_balance_src_label, matlIndex, patch );
        tar_src.initialize(0.0);
        num_density_src.initialize(0.0);
        soot_mass_src.initialize(0.0);
        balance_src.initialize(0.0);
    } else {
        which_dw = new_dw;
        new_dw->getModifiable( tar_src, m_tar_src_label, matlIndex, patch );
        new_dw->getModifiable( num_density_src, m_nd_src_label, matlIndex, patch );
        new_dw->getModifiable( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
        new_dw->getModifiable( balance_src, m_balance_src_label, matlIndex, patch );
    }

    which_dw->get( mix_mol_weight , m_mix_mol_weight_label , matlIndex , patch , gn, 0 );
    which_dw->get( Tar            , m_tar_label            , matlIndex , patch , gn, 0 );
    which_dw->get( Ysoot          , m_Ysoot_label          , matlIndex , patch , gn, 0 );
    which_dw->get( Ns             , m_Ns_label             , matlIndex , patch , gn, 0 );
    which_dw->get( O2             , m_o2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( OH             , m_oh_label             , matlIndex , patch , gn, 0 );
    which_dw->get( CO2            , m_co2_label            , matlIndex , patch , gn, 0 );
    which_dw->get( H2O            , m_h2o_label             , matlIndex , patch , gn, 0 );
    which_dw->get( temperature    , m_temperature_label    , matlIndex , patch , gn, 0 );
    which_dw->get( rho            , m_rho_label            , matlIndex , patch , gn, 0 );

    /// Obtain time-step length
    delt_vartype DT;
    old_dw->get( DT, _shared_state->get_delt_label());
    const double delta_t = DT;

    const double Afs = 5.02E8;          ///< preexponential: soot formation (1/s)
    const double Efs = 198.9E6;         ///< Ea: soot formation, J/kmol
    const double Agt = 9.77E10;         ///< preexponential: tar gasification (1/s)
    const double Egt = 286.9E6;         ///< Ea: soot formation, J/kmol
    const double Aot = 6.77E5;          ///< preexponential: tar oxidation (m3/kg*s)
    const double Eot = 52.3E6;          ///< Ea: soot formation, J/kmol
    //const double Aos = 108500;          ///< preexponential: soot oxidation: (K^0.5)*kg/m2/atm/s
    //const double Eos = 164.5E6;         ///< Ea: soot oxidation, J/kmol
    //const double Ags = 4.1536E9;	      ///< preexponential: soot gasification (1/s/atm^0.54)
    //const double Egs = 148E6;           ///< Ea: soot gasification, J/kmol

    const double Ao2  = 1.92E-3;                   ///< kg*K^0.5/Pa/m2/s
    const double Eo2  = 1.16E8;                    ///< J/kmol
    const double Aoh  = 2.93E-3;                   ///< kg*K^0.5/Pa/m2/s
    const double Aco2 = 1.31E-17;                  ///< kg/Pa^0.5/K2/m2/s
    const double Eco2 = 5.55E6;                    ///< J/kmol
    const double Ah2o = 1.86E6;                    ///< kg*K^0.5/Pa^n/m2/s
    const double Eh2o = 4.17E8;                    ///< J/kmol
    const double nh2o = 1.21;                      ///< unitless

    const double Rgas = 8314.46;        ///< Gas constant: J/kmol*K
    const double kb   = 1.3806488E-23;  ///< Boltzmann constant: kg*m2/s2*K
    const double Na   = 6.02201413E26;  ///< Avogadro's number: #/kmol
    const double MWc  = 12.011;         ///< molecular weight c   kg/kmol
    const double rhos = 1950.;          ///< soot density kg/m3
    const double Ca   = 3.0;            ///< collision frequency constant
    const double Cmin = 9.0E4;          ///< # carbons per incipient particle

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

        IntVector c = *iter;

        const double rhoYO2 = O2[c] * rho[c];
        const double rhoYt  = Tar[c] * rho[c];
        const double XO2    = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              O2[c] * 1.0 / (mix_mol_weight[c] * 32.0)   : 0.0;
        const double XOH    = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              OH[c] * 1.0 / (mix_mol_weight[c] * 17.01)  : 0.0;
        const double XCO2   = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              CO2[c] * 1.0 / (mix_mol_weight[c] * 44.0)  : 0.0;
        const double XH2O   = ( mix_mol_weight[c] > 1.0e-10 ) ?
                              H2O[c] * 1.0 / (mix_mol_weight[c] * 18.02) : 0.0;
        const double rhoYs  = rho[c] * Ysoot[c];
        const double nd     = Ns[c] * rho[c];

        const double T = temperature[c];
        const double P = m_sys_pressure;
        const double dt = delta_t;

        //if (c==IntVector(30,10,10){ })

        //---------------- tar

        double kfs = Afs*exp(-Efs/Rgas/abs(T));                    ///< tar to soot form.   (kg/m3*s)/rhoYt
        double kgt = Agt*exp(-Egt/Rgas/abs(T));                    ///< tar gasification rate (kg/m3*s)/rhoYt
        double kot = abs(rhoYO2)*Aot*exp(-Eot/Rgas/abs(T));        ///< tar oxidation rate (kg/m3*s)/rhoYt

        tar_src[c] = abs(rhoYt)/dt*( exp((-kfs-kgt-kot)*dt) - 1.0 );  ///< kg/m3*s

        //---------------- nd

        double rfn = Na/MWc/Cmin * kfs * abs(rhoYt);                  ///< soot nucleation rate (#/m3*s)
        double ran = 2.0*Ca*pow(6.0*MWc/M_PI/rhos, 1.0/6.0) *         ///< Aggregation rate (#/m3*s)
                     pow(abs(6.0*kb*T/rhos),1.0/2.0) *
                     pow(abs(rhoYs/MWc), 1.0/6.0) *
                     pow(abs(nd),11.0/6.0);

        num_density_src[c] = rfn - ran;                               ///< #/m3*s
        num_density_src[c] = ( num_density_src[c] < 0.0 ) ? std::max( -nd/dt, num_density_src[c] ) : num_density_src[c];

        //---------------- Ys

        double SA = M_PI*pow( abs(6./M_PI*rhoYs/rhos), 2./3. )*pow(abs(nd),1./3.);   ///< m2/m3: pi*pow() = SA/part; pi*pow()*nd = SA/part*part/m3 = SA/Vol
        double rfs = kfs * abs(rhoYt);                                               ///< soot formation rate (kg/m3*s)

        //double ros = SA*P/101325.0*abs(XO2)/sqrt(abs(T))*Aos*exp(-Eos/Rgas/abs(T));  ///< soot oxidation rate (kg/m3*s)
        //double rgs = rhos*pow(abs(XCO2),0.54)*Ags*exp(-Egs/Rgas/abs(T));	         ///< soot gasification rate kg/m3*s

        double ros = pow(abs(T), -0.5)*( Ao2*P*XO2*exp(-Eo2/Rgas/T) +           ///< soot oxidation rate (kg/m3/s)
                                         Aoh*P*XOH ) * SA;
        double rgs = ( Aco2*pow(P*XCO2,0.5)*T*T*exp(-Eco2/Rgas/T) +             ///< soot gasification rate (kg/m3/s)
                       Ah2o*pow(P*XH2O,nh2o)*pow(abs(T),-0.5)*exp(-Eh2o/Rgas/T) ) * SA;

        soot_mass_src[c] = rfs - ros - rgs;
        soot_mass_src[c] = ( soot_mass_src[c] < 0.0 ) ? std::max( -rhoYs/dt, soot_mass_src[c] ) : soot_mass_src[c];

        //---------------- Gas source

        double gas_rate_from_tar = -abs(rhoYt)/dt*( exp((-kgt-kot)*dt) - 1.0 );  ///< kg/m3*s
        balance_src[c] = rgs + ros + gas_rate_from_tar;
        balance_src[c] = std::min( rhoYs/dt + rhoYt/dt, balance_src[c] );

    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
BrownSoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
    string taskname = "BrownSoot::initialize";

    Task* tsk = scinew Task(taskname, this, &BrownSoot::initialize);

    tsk->computes(m_tar_src_label);
    tsk->computes(m_nd_src_label);
    tsk->computes(m_soot_mass_src_label);
    tsk->computes(m_balance_src_label);

    sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
BrownSoot::initialize( const ProcessorGroup* pc,
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

    CCVariable<double> tar_src;
    CCVariable<double> num_density_src;
    CCVariable<double> soot_mass_src;
    CCVariable<double> balance_src;

    new_dw->allocateAndPut( tar_src, m_tar_src_label, matlIndex, patch );
    new_dw->allocateAndPut( num_density_src, m_nd_src_label, matlIndex, patch );
    new_dw->allocateAndPut( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
    new_dw->allocateAndPut( balance_src, m_balance_src_label, matlIndex, patch );

    tar_src.initialize(0.0);
    num_density_src.initialize(0.0);
    soot_mass_src.initialize(0.0);
    balance_src.initialize(0.0);

  }
}
