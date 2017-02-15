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
  db->getWithDefault("co2_label",	     m_CO2_name,		     "CO2");
  db->getWithDefault("density_label",        m_rho_name,              "density");
  db->getWithDefault("temperature_label",    m_temperature_name,      "radiation_temperature");
  db->getWithDefault("system_pressure", m_sys_pressure, 101325.0); // [Pa]

  db->findBlock("tar_src")->getAttribute( "label", m_tar_name );
  db->findBlock("num_density_src")->getAttribute( "label", m_nd_name );
  db->findBlock("soot_mass_src")->getAttribute( "label", m_soot_mass_name );
  db->findBlock("mass_balance_src")->getAttribute( "label", m_balance_name );

  // Since we are producing multiple sources, we load each name into this vector
  // so that we can do error checking upon src term retrieval.
  _mult_srcs.push_back( m_tar_name );
  _mult_srcs.push_back( m_nd_name );
  _mult_srcs.push_back( m_soot_mass_name );
  _mult_srcs.push_back( m_balance_name );

  m_tar_src_label       = VarLabel::create( m_tar_name, CCVariable<double>::getTypeDescription() );
  m_nd_src_label        = VarLabel::create( m_nd_name, CCVariable<double>::getTypeDescription() );
  m_soot_mass_src_label = VarLabel::create( m_soot_mass_name, CCVariable<double>::getTypeDescription() );
  m_balance_src_label   = VarLabel::create( m_balance_name, CCVariable<double>::getTypeDescription() );

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( m_O2_name );
  helper.add_lookup_species( m_rho_name );
  helper.add_lookup_species( m_CO2_name );
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
  m_co2_label             = VarLabel::find( m_CO2_name);
  m_temperature_label     = VarLabel::find( m_temperature_name);
  m_rho_label             = VarLabel::find( m_rho_name);

  tsk->requires( which_dw, m_mix_mol_weight_label,               Ghost::None, 0 );
  tsk->requires( which_dw, m_tar_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_Ysoot_label,                        Ghost::None, 0 );
  tsk->requires( which_dw, m_Ns_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_o2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_co2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_temperature_label,                  Ghost::None, 0 );
  tsk->requires( which_dw, m_rho_label,                          Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

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
                                         int             timeSubStep )
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
    constCCVariable<double> CO2;
    constCCVariable<double> rho;
    constCCVariable<double> temperature;

    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
        which_dw = old_dw;
        new_dw->allocateAndPut( tar_src, m_tar_src_label, matlIndex, patch );
        new_dw->allocateAndPut( num_density_src, m_nd_src_label, matlIndex, patch );
        new_dw->allocateAndPut( soot_mass_src, m_soot_mass_src_label, matlIndex, patch );
        new_dw->allocateAndPut( balance_src, m_balance_src_label, matlIndex, patch );
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
    which_dw->get( CO2            , m_co2_label            , matlIndex , patch , gn, 0 );
    which_dw->get( temperature    , m_temperature_label    , matlIndex , patch , gn, 0 );
    which_dw->get( rho            , m_rho_label            , matlIndex , patch , gn, 0 );

    /// Obtain time-step length
    delt_vartype DT;
    old_dw->get( DT, _shared_state->get_delt_label());
    const double delta_t = DT;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      const double rhoYO2 = O2[c] * rho[c];
      const double rhoTar = Tar[c] * rho[c];
      const double XO2 = ( mix_mol_weight[c] > 1.0e-10 ) ?
                           O2[c] * 1.0 / (mix_mol_weight[c] * 32.0) : 0.0;
      const double XCO2 = ( mix_mol_weight[c] > 1.0e-10 ) ?
                            CO2[c] * 1.0 / (mix_mol_weight[c] * 44.0) : 0.0;
      const double rhoYsoot = rho[c] * Ysoot[c];
      const double nd = Ns[c] * rho[c];

      coalSootTar( m_sys_pressure,
                   temperature[c],
                   rhoYO2,
                   rhoTar,
                   delta_t,
                   tar_src[c] );

      coalSootND( m_sys_pressure,
                  temperature[c],
                  XCO2,
                  XO2,
                  rhoTar,
                  rhoYsoot,
                  nd,
                  delta_t,
                  num_density_src[c] );

      coalSootMassSrc( m_sys_pressure,
                       temperature[c],
                       XCO2,
                       XO2,
                       rhoTar,
                       rhoYsoot,
                       nd,
                       delta_t,
                       soot_mass_src[c] );

       coalGasSootSrc( m_sys_pressure,
                       temperature[c],
                       XCO2,
                       XO2,
                       rhoTar,
                       rhoYsoot,
                       nd,
                       rhoYO2,
                       delta_t,
                       balance_src[c] );


    }
  }
}

//---------------------------------------------------------------------------
// Method: Coal Soot Source Terms
//---------------------------------------------------------------------------

/** Soot source terms
*
* Alex Brown and Tom Fletcher, Energy and Fuels, Vol 12, No 4 1998, 745-757
*    Note, Alex used "c" to denote soot, here using "s" to denote soot. "t"
*        denotes tar.
*    Note, Alex's paper has a number of typos/units problems.
*    Reading Lee et al 1962 for soot oxidation, and Ma's dissertation (p 115 (102))
*    Alex's code is in his dissertation: the soot formation rate has [c_t], which is rho*Yt,
*        not concentration as mol/m3, which is what is in his notation.  Also, Ma's
*        Dissertation has Afs = 5.02E8 1/s, implying the reaction as below.
*
* @param P           \input Pressure (Pa)
* @param T           \input Temperature (K)
* @param rhoYO2      \input rho*Y_O2
* @param rhoYt       \input Tar mass fraction * rho
* @param Ytar_source \output Soot number density (kg/m3*s)
*/

void BrownSoot::coalSootTar( const double P,
                                          const double T,
                                          const double rhoYO2,
                                          const double rhoYt,
					                                const double dt,
                                          double &Ytar_source ) {

double Afs = 5.02E8;          ///< preexponential: soot formation (1/s)
double Efs = 198.9E6;         ///< Ea: soot formation, J/kmol

double Agt = 9.77E10;         ///< preexponential: tar gasification (1/s)
double Egt = 286.9E6;         ///< Ea: soot formation, J/kmol

double Aot = 6.77E6;          ///< preexponential: tar oxidation (m3/kg*s)
double Eot = 52.3E6;          ///< Ea: soot formation, J/kmol

double Rgas = 8314.46;        ///< Gas constant: J/kmol*K

//-------------------------------------

double rfs = -abs(rhoYt)*Afs*exp(-Efs/Rgas/T);                    ///< tar to soot form.   (kg/m3*s)
double rgt = -abs(rhoYt)*Agt*exp(-Egt/Rgas/T);                    ///< tar gasification rate (kg/m3*s)
double rot = -abs(rhoYt*rhoYO2)*Aot*exp(-Eot/Rgas/T);             ///< tar oxidation rate (kg/m3*s)

//-------------------------------------

Ytar_source = rfs + rgt + rot;                                      ///< kg/m3*s

/// Check if the rate is consuming all the soot in the system, and clip it if it is so the tar never goes negative in the system.
Ytar_source = std::max( -rhoYt/dt , Ytar_source);

return;

}

//---------------------------------------------------------------------------
// Method: Coal Soot Source Terms
//---------------------------------------------------------------------------

/** Soot source terms
*
* Alex Brown and Tom Fletcher, Energy and Fuels, Vol 12, No 4 1998, 745-757
*    Note, Alex used "c" to denote soot, here using "s" to denote soot. "t"
*        denotes tar.
*    Note, Alex's paper has a number of typos/units problems.
*    Reading Lee et al 1962 for soot oxidation, and Ma's dissertation (p 115 (102))
*    Alex's code is in his dissertation: the soot formation rate has [c_t], which is rho*Yt,
*        not concentration as mol/m3, which is what is in his notation.  Also, Ma's
*        Dissertation has Afs = 5.02E8 1/s, implying the reaction as below.
*
* @param P           \input Pressure (Pa)
* @param T           \input Temperature (K)
* @param Xo2         \input O2 mole fraction
* @param rhoYt       \input Tar mass fraction * rho
* @param rhoYs       \input Soot mass fraction * rho
* @param nd          \input Soot number density (#/m3)
* @param S_N         \output Soot number density (#/m3*s)
* @param S_Ys        \output Soot number density (kg/m3*s)
*/

void BrownSoot::coalSootND( const double P,
                                        const double T,
                                        const double XCO2,
                                        const double XO2,
                                        const double rhoYt,
                                        const double rhoYs,
                                        const double nd,
                                        const double dt,
                                        double &Ns_source ) {

double Afs = 5.02E8;          ///< preexponential: soot formation (1/s)
double Efs = 198.9E6;         ///< Ea: soot formation, J/kmol

double Rgas = 8314.46;        ///< Gas constant: J/kmol*K
double kb   = 1.3806488E-23;  ///< Boltzmann constant: kg*m2/s2*K
double Na   = 6.02201413E26;  ///< Avogadro's number: #/kmol

double MWc  = 12.011;         ///< molecular weight c   kg/kmol

double rhos = 1950.;          ///< soot density kg/m3

double Ca   = 3.0;            ///< collision frequency constant
double Cmin = 9.0E4;          ///< # carbons per incipient particle

//-------------------------------------

double rfs = abs(rhoYt)*Afs*exp(-Efs/Rgas/T);                 ///< soot formation rate (kg/m3*s)
double rfn = Na/MWc/Cmin*rfs;                                 ///< soot nucleation rate (#/m3*s)
double ran = 2.0*Ca*pow(6.0*MWc/M_PI/rhos, 1.0/6.0) *         ///< Aggregation rate (#/m3*s)
    pow(abs(6.0*kb*T/rhos),1.0/2.0) *
    pow(abs(rhoYs/MWc), 1.0/6.0) *
    pow(abs(nd),11.0/6.0);

//-------------------------------------

Ns_source  = rfn - ran;                                      ///< #/m3*s

/// Check if the rate is consuming all the soot in the system, and clip it if it is so the soot never goes negative in the system.
Ns_source = ( Ns_source < 0.0 ) ? std::max( -nd/dt, Ns_source ) : Ns_source;

return;

}

//---------------------------------------------------------------------------
// Method: Coal Soot Source Terms
//---------------------------------------------------------------------------

/** Soot source terms
*
* Alex Brown and Tom Fletcher, Energy and Fuels, Vol 12, No 4 1998, 745-757
*    Note, Alex used "c" to denote soot, here using "s" to denote soot. "t"
*        denotes tar.
*    Note, Alex's paper has a number of typos/units problems.
*    Reading Lee et al 1962 for soot oxidation, and Ma's dissertation (p 115 (102))
*    Alex's code is in his dissertation: the soot formation rate has [c_t], which is rho*Yt,
*        not concentration as mol/m3, which is what is in his notation.  Also, Ma's
*        Dissertation has Afs = 5.02E8 1/s, implying the reaction as below.
*
* @param P           \input Pressure (Pa)
* @param T           \input Temperature (K)
* @param Xo2         \input O2 mole fraction
* @param rhoYt       \input Tar mass fraction * rho
* @param rhoYs       \input Soot mass fraction * rho
* @param nd          \input Soot number density (#/m3)
* @param S_Ys        \output Soot number density (kg/m3*s)
*/

void BrownSoot::coalSootMassSrc( const double P,
                                                const double T,
                                                const double XCO2,
                                                const double XO2,
                                                const double rhoYt,
                                                const double rhoYs,
                                                const double nd,
                                                const double dt,
                                                double &Ysoot_source ) {

double Afs = 5.02E8;          ///< preexponential: soot formation (1/s)
double Efs = 198.9E6;         ///< Ea: soot formation, J/kmol

double Aos = 108500;           ///< preexponential: soot oxidation: (K^0.5)*kg/m2/atm/s
double Eos = 164.5E6;         ///< Ea: soot oxidation, J/kmol

double Ags = 4.1536E9;	      ///< preexponential: soot gasification (1/s/atm^0.54)
double Egs = 148E6;           ///< Ea: soot gasification, J/kmol

double Rgas = 8314.46;        ///< Gas constant: J/kmol*K
double rhos = 1950.;          ///< soot density kg/m3

//-------------------------------------

double SA   = M_PI*pow( abs(6./M_PI*rhoYs/rhos), 2./3. )*pow(abs(nd),1./3.);  ///< m2/m3: pi*pow() = SA/part; pi*pow()*nd = SA/part*part/m3 = SA/Vol

//-------------------------------------

double rgs = rhos*pow(abs(XCO2),0.54)*Ags*exp(-Egs/Rgas/abs(T));	      ///< soot gasification rate kg/m3*s
double ros = SA*P/101325.0*abs(XO2)/sqrt(abs(T))*Aos*exp(-Eos/Rgas/T);  ///< soot oxidation rate (kg/m3*s)
double rfs = abs(rhoYt)*Afs*exp(-Efs/Rgas/T);                 ///< soot formation rate (kg/m3*s)

//-------------------------------------

Ysoot_source = rfs - ros - rgs;                                      ///< kg/m3*s

/// Check if the rate is consuming all the soot in the system, and clip it if it is so the soot never goes negative in the system.
Ysoot_source = ( Ysoot_source < 0.0 ) ? std::max( -rhoYs/dt, Ysoot_source ) : Ysoot_source;

return;

}

//---------------------------------------------------------------------------
// Method: Coal Soot Source Terms
//---------------------------------------------------------------------------

/** Soot source terms
*
* Alex Brown and Tom Fletcher, Energy and Fuels, Vol 12, No 4 1998, 745-757
*    Note, Alex used "c" to denote soot, here using "s" to denote soot. "t"
*        denotes tar.
*    Note, Alex's paper has a number of typos/units problems.
*    Reading Lee et al 1962 for soot oxidation, and Ma's dissertation (p 115 (102))
*    Alex's code is in his dissertation: the soot formation rate has [c_t], which is rho*Yt,
*        not concentration as mol/m3, which is what is in his notation.  Also, Ma's
*        Dissertation has Afs = 5.02E8 1/s, implying the reaction as below.
*
* @param P           \input Pressure (Pa)
* @param T           \input Temperature (K)
* @param Xo2         \input O2 mole fraction
* @param rhoYt       \input Tar mass fraction * rho
* @param rhoYs       \input Soot mass fraction * rho
* @param nd          \input Soot number density (#/m3)
* @param S_N         \output Soot number density (#/m3*s)
* @param S_Ys        \output Soot number density (kg/m3*s)
*/

void BrownSoot::coalGasSootSrc( const double P,
                                             const double T,
                                             const double XCO2,
                                             const double XO2,
                                             const double rhoYt,
                                             const double rhoYs,
                                             const double nd,
                                             const double rhoYO2,
                                             const double dt,
                                             double &Off_Gas ) {

double Aos = 108500;           ///< preexponential: soot oxidation: (K^0.5)*kg/m2/atm/s
double Eos = 164.5E6;         ///< Ea: soot oxidation, J/kmol

double Ags = 4.1536E9;	      ///< preexponential: soot gasification (1/s/atm^0.54)
double Egs = 148E6;           ///< Ea: soot gasification, J/kmol

double Agt = 9.77E10;         ///< preexponential: tar gasification (1/s)
double Egt = 286.9E6;         ///< Ea: soot formation, J/kmol

double Aot = 6.77E6;          ///< preexponential: tar oxidation (m3/kg*s)
double Eot = 52.3E6;          ///< Ea: soot formation, J/kmol

double Rgas = 8314.46;        ///< Gas constant: J/kmol*K
double rhos = 1950.;          ///< soot density kg/m3

//-------------------------------------


double SA   = M_PI*pow( abs(6./M_PI*rhoYs/rhos), 2./3. )*pow(abs(nd),1./3.);  ///< m2/m3: pi*pow() = SA/part; pi*pow()*nd = SA/part*part/m3 = SA/Vol

//-------------------------------------

double rgs = rhos*pow(abs(XCO2),0.54)*Ags*exp(-Egs/Rgas/T);	      ///< soot gasification rate kg/m3*s
double ros = SA*P/101325.0*abs(XO2)/sqrt(abs(T))*Aos*exp(-Eos/Rgas/T);  ///< soot oxidation rate (kg/m3*s)

double rgt = abs(rhoYt)*Agt*exp(-Egt/Rgas/T);                    ///< tar gasification rate (kg/m3*s)
double rot = abs(rhoYt*rhoYO2)*Aot*exp(-Eot/Rgas/T);             ///< tar oxidation rate (kg/m3*s)
//-------------------------------------

Off_Gas = rgs + ros + rgt + rot ;                                      ///< #/m3*s

/// Check if the rate is consuming all the soot and tar in the system, and clip it if it is so the soot/tar never go negative.
Off_Gas = std::min( rhoYs/dt + rhoYt/dt , Off_Gas);

return;

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
