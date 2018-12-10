#include <CCA/Components/Arches/CoalModels/RichardsFletcherDevol.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
RichardsFletcherDevolBuilder::RichardsFletcherDevolBuilder( const std::string         & modelName,
                                                            const vector<std::string> & reqICLabelNames,
                                                            const vector<std::string> & reqScalarLabelNames,
                                                            ArchesLabel         * fieldLabels,
                                                            MaterialManagerP          & materialManager,
                                                            int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{
}

RichardsFletcherDevolBuilder::~RichardsFletcherDevolBuilder(){}

ModelBase* RichardsFletcherDevolBuilder::build() {
  return scinew RichardsFletcherDevol( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

RichardsFletcherDevol::RichardsFletcherDevol( std::string modelName, 
                                              MaterialManagerP& materialManager,
                                              ArchesLabel* fieldLabels,
                                              vector<std::string> icLabelNames, 
                                              vector<std::string> scalarLabelNames,
                                              int qn ) 
: Devolatilization(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  pi = acos(-1.0);
}

RichardsFletcherDevol::~RichardsFletcherDevol()
{
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
RichardsFletcherDevol::problemSetup(const ProblemSpecP& params, int qn)
{

  ProblemSpecP db = params;
  const ProblemSpecP params_root = db->getRootNode();

  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  
  ProblemSpecP db_coal_props = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleProperties");
  
  // create raw coal mass var label and get scaling constant
  std::string rcmass_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_RAWCOAL); 
  std::string rcmass_name = ArchesCore::append_env( rcmass_root, d_quadNode ); 
  std::string rcmassqn_name = ArchesCore::append_qn_env( rcmass_root, d_quadNode ); 
  _rcmass_varlabel = VarLabel::find(rcmass_name);
  EqnBase& temp_rcmass_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(rcmassqn_name);
  DQMOMEqn& rcmass_eqn = dynamic_cast<DQMOMEqn&>(temp_rcmass_eqn);
   _rc_scaling_constant = rcmass_eqn.getScalingConstant(d_quadNode);
  std::string ic_RHS = rcmassqn_name+"_RHS";
  _RHS_source_varlabel = VarLabel::find(ic_RHS);

  // create char mass var label
  std::string char_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_CHAR); 
  std::string char_name = ArchesCore::append_env( char_root, d_quadNode ); 
  _char_varlabel = VarLabel::find(char_name); 
  
  // create particle temperature label
  std::string temperature_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_TEMPERATURE); 
  std::string temperature_name = ArchesCore::append_env( temperature_root, d_quadNode ); 
  _particle_temperature_varlabel = VarLabel::find(temperature_name);
 
  // Look for required scalars
  if (db_coal_props->findBlock("RichardsFletcher_coefficients")) {
    db_coal_props->require("RichardsFletcher_coefficients", RichardsFletcher_coefficients);
    // Values from Ubhayakar (1976):
    Av1=RichardsFletcher_coefficients[0];  // [=] 1/s; k1 pre-exponential factor
    Av2=RichardsFletcher_coefficients[1];  // [=] 1/s; k2 pre-exponential factor
    Ev1=RichardsFletcher_coefficients[2];  // [=] K; k1 activation energy
    Ev2=RichardsFletcher_coefficients[3];  // [=] K;  k2 activation energy
    // Y values from white book:
    Y1_=RichardsFletcher_coefficients[4];  // volatile fraction from proximate analysis
    Y2_=RichardsFletcher_coefficients[5];  // fraction devolatilized at higher temperatures
    c0_1=RichardsFletcher_coefficients[6]; 
    c1_1=RichardsFletcher_coefficients[7]; 
    c2_1=RichardsFletcher_coefficients[8]; 
    c3_1=RichardsFletcher_coefficients[9]; 
    c4_1=RichardsFletcher_coefficients[10]; 
    c5_1=RichardsFletcher_coefficients[11]; 
    c6_1=RichardsFletcher_coefficients[12]; 
    c0_2=RichardsFletcher_coefficients[13]; 
    c1_2=RichardsFletcher_coefficients[14]; 
    c2_2=RichardsFletcher_coefficients[15]; 
    c3_2=RichardsFletcher_coefficients[16]; 
    c4_2=RichardsFletcher_coefficients[17]; 
    c5_2=RichardsFletcher_coefficients[18]; 
    c5_2=RichardsFletcher_coefficients[19]; 
    c6_2=RichardsFletcher_coefficients[20]; 
  } else { 
    throw ProblemSetupException("Error: RichardsFletcher_coefficients missing in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("density")){ 
    db_coal_props->require("density", rhop);
  } else { 
    throw ProblemSetupException("Error: You must specify density in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("diameter_distribution")){ 
    db_coal_props->require("diameter_distribution", particle_sizes);
  } else { 
    throw ProblemSetupException("Error: You must specify diameter_distribution in <CoalProperties>.", __FILE__, __LINE__); 
  }
  if ( db_coal_props->findBlock("ultimate_analysis")){ 
    ProblemSpecP db_ua = db_coal_props->findBlock("ultimate_analysis"); 
    CoalAnalysis coal; 
    db_ua->require("C",coal.C);
    db_ua->require("H",coal.H);
    db_ua->require("O",coal.O);
    db_ua->require("N",coal.N);
    db_ua->require("S",coal.S);
    db_ua->require("H2O",coal.H2O);
    db_ua->require("ASH",coal.ASH);
    db_ua->require("CHAR",coal.CHAR);
    total_rc=coal.C+coal.H+coal.O+coal.N+coal.S; // (C+H+O+N+S) dry ash free total
    total_dry=coal.C+coal.H+coal.O+coal.N+coal.S+coal.ASH+coal.CHAR; // (C+H+O+N+S+char+ash)  moisture free total
    rc_mass_frac=total_rc/total_dry; // mass frac of rc (dry) 
    char_mass_frac=coal.CHAR/total_dry; // mass frac of char (dry)
    ash_mass_frac=coal.ASH/total_dry; // mass frac of ash (dry)
    int p_size=particle_sizes.size();
    for (int n=0; n<p_size; n=n+1)
      {
        vol_dry.push_back((pi/6)*pow(particle_sizes[n],3)); // m^3/particle
        mass_dry.push_back(vol_dry[n]*rhop); // kg/particle
        ash_mass_init.push_back(mass_dry[n]*ash_mass_frac); // kg_ash/particle (initial) 
        char_mass_init.push_back(mass_dry[n]*char_mass_frac); // kg_char/particle (initial)
        rc_mass_init.push_back(mass_dry[n]*rc_mass_frac); // kg_ash/particle (initial)
      }
  } else { 
    throw ProblemSetupException("Error: You must specify ultimate_analysis in <CoalProperties>.", __FILE__, __LINE__); 
  }

  // get weight scaling constant
  std::string weightqn_name = ArchesCore::append_qn_env("w", d_quadNode); 
  std::string weight_name = ArchesCore::append_env("w", d_quadNode); 
  _weight_varlabel = VarLabel::find(weight_name); 
  EqnBase& temp_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(weightqn_name);
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(temp_weight_eqn);
  _weight_small = weight_eqn.getSmallClipPlusTol();
  _weight_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
RichardsFletcherDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "RichardsFletcherDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &RichardsFletcherDevol::computeModel, timeSubStep);

  Ghost::GhostType gn = Ghost::None;

  Task::WhichDW which_dw; 

  if (timeSubStep == 0 ) { 
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->computes(d_charLabel);
    which_dw = Task::OldDW; 
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel);
    tsk->modifies(d_charLabel); 
    which_dw = Task::NewDW; 
  }

  tsk->requires( which_dw, _particle_temperature_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _rcmass_varlabel, gn, 0 ); 
  tsk->requires( which_dw, _char_varlabel, gn, 0 );
  tsk->requires( which_dw, _weight_varlabel, gn, 0 ); 
  tsk->requires( Task::OldDW, d_fieldLabels->d_delTLabel); 
  tsk->requires( Task::NewDW, _RHS_source_varlabel, gn, 0 ); 

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" )); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
RichardsFletcherDevol::computeModel( const ProcessorGroup * pc, 
                                     const PatchSubset    * patches, 
                                     const MaterialSubset * matls, 
                                     DataWarehouse        * old_dw, 
                                     DataWarehouse        * new_dw,
                                     const int timeSubStep )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    
    Vector Dx = patch->dCell(); 
    double vol = Dx.x()* Dx.y()* Dx.z(); 

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    double dt = DT;

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate; 
    CCVariable<double> char_rate;
    DataWarehouse* which_dw; 

    if ( timeSubStep == 0 ){ 
      which_dw = old_dw; 
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      devol_rate.initialize(0.0);
      new_dw->allocateAndPut( gas_devol_rate, d_gasLabel, matlIndex, patch );
      gas_devol_rate.initialize(0.0);
      new_dw->allocateAndPut( char_rate, d_charLabel, matlIndex, patch );
      char_rate.initialize(0.0);
    } else { 
      which_dw = new_dw; 
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_devol_rate, d_gasLabel, matlIndex, patch ); 
      new_dw->getModifiable( char_rate, d_charLabel, matlIndex, patch );
    }

    constCCVariable<double> temperature; 
    which_dw->get( temperature , _particle_temperature_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> rcmass; 
    which_dw->get( rcmass    , _rcmass_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> charmass; 
    which_dw->get( charmass , _char_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> weight; 
    which_dw->get( weight , _weight_varlabel , matlIndex , patch , gn , 0 );
    constCCVariable<double> RHS_source; 
    new_dw->get( RHS_source , _RHS_source_varlabel , matlIndex , patch , gn , 0 );
    
    double rcmass_init = rc_mass_init[d_quadNode];
    double Xv;
    double Fv1;
    double Fv2;
    double k1;        ///< Rate constant for devolatilization reaction 1
    double k2;        ///< Rate constant for devolatilization reaction 2
    double rateMax;
    double rate;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;
      if (weight[c]/_weight_scaling_constant > _weight_small) {

        double rcmassph=rcmass[c];
        double RHS_sourceph=RHS_source[c];
        double temperatureph=temperature[c];
        double charmassph=charmass[c];
        double weightph=weight[c];

        //VERIFICATION
        //rcmassph=4.99019e-12;
        //temperatureph=530.79;
        //charmassph=-7.1393e-18;
        //weightph=1.40781e+09;
        
        Xv = (rcmass_init-rcmassph)/rcmass_init;
        Xv = min(max(Xv,0.0),1.0);// make sure Xv is between 0 and 1
        Fv1 = c6_1*pow(Xv,6.0) + c5_1*pow(Xv,5.0) + c4_1*pow(Xv,4.0) + c3_1*pow(Xv,3.0) + c2_1*pow(Xv,2.0) + c1_1*Xv +c0_1;
        k1 = exp(Fv1)*Av1*exp(-Ev1/(temperatureph));
        Fv2 = c6_2*pow(Xv,6.0) + c5_2*pow(Xv,5.0) + c4_2*pow(Xv,4.0) + c3_2*pow(Xv,3.0) + c2_2*pow(Xv,2.0) + c1_2*Xv +c0_2;
        k2 = exp(Fv2)*Av2*exp(-Ev2/(temperatureph));

        rateMax = max( (rcmassph+min(0.0,charmassph))/(dt) + RHS_sourceph/(vol*weightph) , 0.0 );
        rate = (k1+k2)*(rcmassph+min(0.0,charmassph));
        devol_rate[c] = -rate*weightph/(_rc_scaling_constant*_weight_scaling_constant); //rate of consumption of raw coal mass
        gas_devol_rate[c] = (Y1_*k1+Y2_*k2)*(rcmassph+min(0.0,charmassph))*weightph; // rate of creation of coal off gas
        char_rate[c] = ((1.0-Y1_)*k1+(1.0-Y2_)*k2)*(rcmassph+min(0.0,charmassph))*weightph; // rate of creation of char
        if( rateMax < rate ) {
          devol_rate[c] = -rateMax*weightph/(_rc_scaling_constant*_weight_scaling_constant);
          gas_devol_rate[c] = Y1_*rateMax*weightph;
          char_rate[c] = (1.0-Y1_)*rateMax*weightph;
        }
        
        //additional check to make sure we have positive rates when we have small amounts of rc and char.. 
        if( devol_rate[c]>0.0 || (rcmassph+min(0.0,charmassph)) < 1e-20 ) {
          devol_rate[c] = 0;
          gas_devol_rate[c] = 0;
          char_rate[c] = 0;
        }
        
      } else {
        devol_rate[c] = 0;
        gas_devol_rate[c] = 0;
        char_rate[c] = 0;
      }
    }//end cell loop
  }//end patch loop
}
