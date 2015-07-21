#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/BrownSootFormation_rhoYs.h>
#include <cmath>

//===========================================================================

using namespace std;
using namespace Uintah; 

BrownSootFormation_rhoYs::BrownSootFormation_rhoYs( std::string src_name, ArchesLabel* field_labels,
                                                    vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, field_labels->d_sharedState, req_label_names, type), _field_labels(field_labels)
{

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC; 

}

BrownSootFormation_rhoYs::~BrownSootFormation_rhoYs()
{
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
BrownSootFormation_rhoYs::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault("mix_mol_weight_label", _mix_mol_weight_name,   "mixture_molecular_weight");
  db->getWithDefault("tar_label",            _tar_name,              "Tar");
  db->getWithDefault("mixture_fraction_label", _mixture_fraction_name,     "mixture_fraction");
  db->getWithDefault("Ysoot_label",          _Ysoot_name,            "Ysoot");
  db->getWithDefault("Ns_label",             _Ns_name,               "Ns");
  db->getWithDefault("o2_label",             _o2_name,               "O2");
  db->getWithDefault("co2_label",	     _co2_name,		     "CO2");
  db->getWithDefault("density_label",        _rho_name,              "density");
  db->getWithDefault("temperature_label",    _temperature_name,      "radiation_temperature");

  _field_labels->add_species( _o2_name ); 
  _field_labels->add_species( _co2_name );
  _field_labels->add_species( _rho_name );
  //_field_labels->add_species( _temperature_name );

  _source_grid_type = CC_SRC; 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
BrownSootFormation_rhoYs::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BrownSootFormation_rhoYs::eval";
  Task* tsk = scinew Task(taskname, this, &BrownSootFormation_rhoYs::computeSource, timeSubStep);

  Task::WhichDW which_dw;
  if (timeSubStep == 0) {
    tsk->computes(_src_label);
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
    tsk->modifies(_src_label);
  }
  // resolve some labels:
  const VarLabel* mix_mol_weight_label  = VarLabel::find( _mix_mol_weight_name);
  const VarLabel* tar_label             = VarLabel::find( _tar_name);
  const VarLabel* mixture_fraction_label = VarLabel::find( _mixture_fraction_name);
  const VarLabel* Ysoot_label           = VarLabel::find( _Ysoot_name);
  const VarLabel* Ns_label              = VarLabel::find( _Ns_name);
  const VarLabel* o2_label              = VarLabel::find( _o2_name);
  const VarLabel* co2_label		= VarLabel::find( _co2_name);
  const VarLabel* temperature_label     = VarLabel::find( _temperature_name);
  const VarLabel* rho_label             = VarLabel::find( _rho_name);
  tsk->requires( which_dw, mix_mol_weight_label,               Ghost::None, 0 );
  tsk->requires( which_dw, tar_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, mixture_fraction_label,             Ghost::None, 0 );
  tsk->requires( which_dw, Ysoot_label,                        Ghost::None, 0 );
  tsk->requires( which_dw, Ns_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, o2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, co2_label,			       Ghost::None, 0 );
  tsk->requires( which_dw, temperature_label,                  Ghost::None, 0 );
  tsk->requires( which_dw, rho_label,                          Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BrownSootFormation_rhoYs::computeSource( const ProcessorGroup* pc, 
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

    CCVariable<double> rate; 

    constCCVariable<double> mix_mol_weight;
    constCCVariable<double> Tar;
    constCCVariable<double> mixture_fraction;
    constCCVariable<double> Ysoot;
    constCCVariable<double> Ns;
    constCCVariable<double> O2;
    constCCVariable<double> CO2; 
    constCCVariable<double> rho;
    constCCVariable<double> temperature;
    const VarLabel* mix_mol_weight_label  = VarLabel::find( _mix_mol_weight_name);
    const VarLabel* tar_label             = VarLabel::find( _tar_name);
    const VarLabel* mixture_fraction_label = VarLabel::find( _mixture_fraction_name);
    const VarLabel* Ysoot_label           = VarLabel::find( _Ysoot_name);
    const VarLabel* Ns_label              = VarLabel::find( _Ns_name);
    const VarLabel* o2_label              = VarLabel::find( _o2_name);
    const VarLabel* co2_label		  = VarLabel::find( _co2_name);
    const VarLabel* temperature_label     = VarLabel::find( _temperature_name);
    const VarLabel* rho_label             = VarLabel::find( _rho_name);    
                                                                                       
    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
        which_dw = old_dw;
        new_dw->allocateAndPut( rate, _src_label, matlIndex, patch );
        rate.initialize(0.0);
    } else {
        which_dw = new_dw;
        new_dw->getModifiable( rate, _src_label, matlIndex, patch );
    }
                                                                                       
    which_dw->get( mix_mol_weight , mix_mol_weight_label , matlIndex , patch , gn, 0 );
    which_dw->get( Tar         , tar_label            , matlIndex , patch , gn, 0 );
    which_dw->get( mixture_fraction , mixture_fraction_label , matlIndex , patch , gn, 0 );
    which_dw->get( Ysoot          , Ysoot_label          , matlIndex , patch , gn, 0 );
    which_dw->get( Ns             , Ns_label             , matlIndex , patch , gn, 0 );
    which_dw->get( O2             , o2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( CO2		  , co2_label		 , matlIndex , patch , gn, 0 );
    which_dw->get( temperature    , temperature_label    , matlIndex , patch , gn, 0 );
    which_dw->get( rho            , rho_label            , matlIndex , patch , gn, 0 );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        
      IntVector c = *iter;

      double XO2 = 0.0;
      double XCO2 = 0.0;
      if (mix_mol_weight[c] > 1e-10){
        double XO2 = O2[c] * 1.0 / (mix_mol_weight[c] * 32.0);
	double XCO2 = CO2[c] * 1.0 / (mix_mol_weight[c] * 44.0);
      }
      double rhoYsoot = rho[c] * Ysoot[c];
      double nd = Ns[c] * rho[c];
      double rhoTar = Tar[c] * rho[c];

      double throw_away;
      double sys_pressure = 101325; //Pa
      coalSootRR(sys_pressure,
                 temperature[c],
		 XCO2,
                 XO2,
                 rhoTar,
                 rhoYsoot,
                 nd,
                 throw_away,
                 rate[c]
                );
       if (c==IntVector(75,75,75)){
         cout << "Ys: " << temperature[c] << " " << rho[c] <<" " << XO2 << " " << rhoTar << " " << rhoYsoot << " " << nd << " " << rate[c] << endl; 
       }

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
* @param Xo2         \input O2 mole fraction
* @param rhoYt       \input Tar mass fraction * rho
* @param rhoYs       \input Soot mass fraction * rho
* @param nd          \input Soot number density (#/m3)
* @param S_N         \output Soot number density (#/m3*s)
* @param S_Ys        \output Soot number density (kg/m3*s)
*/
                                                                                
void BrownSootFormation_rhoYs::coalSootRR(const double P, 
                                          const double T,
					  const double XCO2,
                                          const double XO2,
                                          const double rhoYt,
                                          const double rhoYs,
                                          const double nd,
                                                double &Ns_source,
                                                double &Ysoot_source
                                         ) {
    
double Afs = 5.02E8;          ///< preexponential: soot formation (1/s)
double Efs = 198.9E6;         ///< Ea: soot formation, J/kmol
    
double Aos = 108500;           ///< preexponential: soot oxidation: (K^0.5)*kg/m2/atm/s
double Eos = 164.5E6;         ///< Ea: soot oxidation, J/kmol

double Ags = 4.1536E9;	      ///< preexponential: soot gasification (1/s/atm^0.54)
double Egs = 148E6;           ///< Ea: soot gasification, J/kmol
    
double Rgas = 8314.46;        ///< Gas constant: J/kmol*K
double kb   = 1.3806488E-23;  ///< Boltzmann constant: kg*m2/s2*K
double Na   = 6.02201413E26;  ///< Avogadro's number: #/kmol
                                                                                           
double MWo2 = 32.0;           ///< molecular weight o2  kg/kmol
double MWt  = 350.0;          ///< molecular weight tar kg/kmol
double MWc  = 12.011;         ///< molecular weight c   kg/kmol
                                                                                           
double rhos = 1950.;          ///< soot density kg/m3
                                                                                           
double Ca   = 3.0;            ///< collision frequency constant
double Cmin = 9.0E4;          ///< # carbons per incipient particle
                                                                                           
//-------------------------------------
                                                                                           
                                                                                           
double SA   = M_PI*pow( abs(6./M_PI*rhoYs/rhos), 2./3. )*pow(abs(nd),1./3.);  ///< m2/m3: pi*pow() = SA/part; pi*pow()*nd = SA/part*part/m3 = SA/Vol
                                                                                           
//-------------------------------------

double rgs = rhos*pow(abs(XCO2),0.54)*Ags*exp(-Egs/Rgas/abs(T));	      ///< soot gasification rate kg/m3*s
double ros = SA*P/101325.0*abs(XO2)/sqrt(abs(T))*Aos*exp(-Eos/Rgas/T);  ///< soot oxidation rate (kg/m3*s)
double rfs = abs(rhoYt)*Afs*exp(-Efs/Rgas/T);                 ///< soot formation rate (kg/m3*s)
double rfn = Na/MWc/Cmin*rfs;                                 ///< soot nucleation rate (#/m3*s)
double ran = 2.0*Ca*pow(6.0*MWc/M_PI/rhos, 1.0/6.0) *         ///< Aggregation rate (#/m3*s)
    pow(abs(6.0*kb*T/rhos),1.0/2.0) *
    pow(abs(rhoYs/MWc), 1.0/6.0) *
    pow(abs(nd),11.0/6.0);
                                                                                           
//-------------------------------------
                                                                                           
Ns_source  = rfn - ran;                                      ///< #/m3*s
Ysoot_source = rfs - ros - rgs;                                      ///< kg/m3*s
                                                                                           
return;
    
}
                                                                                       
                                                                                       
                                                                                       
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
BrownSootFormation_rhoYs::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "BrownSootFormation_rhoYs::initialize"; 

  Task* tsk = scinew Task(taskname, this, &BrownSootFormation_rhoYs::initialize);

  tsk->computes(_src_label);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
BrownSootFormation_rhoYs::initialize( const ProcessorGroup* pc, 
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

    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    }
  }
