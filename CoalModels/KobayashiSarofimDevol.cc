#include <CCA/Components/Arches/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
KobayashiSarofimDevolBuilder::KobayashiSarofimDevolBuilder( const std::string         & modelName,
    const vector<std::string> & reqLabelNames,
    const ArchesLabel              * fieldLabels,
    SimulationStateP          & sharedState,
    int qn ) :
  ModelBuilder( modelName, fieldLabels, reqLabelNames, sharedState, qn )
{}

KobayashiSarofimDevolBuilder::~KobayashiSarofimDevolBuilder(){}

ModelBase*
KobayashiSarofimDevolBuilder::build(){
  return scinew KobayashiSarofimDevol( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

KobayashiSarofimDevol::KobayashiSarofimDevol( std::string srcName, SimulationStateP& sharedState,
    const ArchesLabel* fieldLabels,
    vector<std::string> icLabelNames, int qn ) 
: ModelBase(srcName, sharedState, fieldLabels, icLabelNames, qn),d_fieldLabels(fieldLabels)
{
  A1  =  2.0e5;       // k1 pre-exponential factor
  A2  =  1.3e7;       // k1 activation energy
  E1  =  -25000;       // k2 pre-exponential factor
  E2  =  -40000;       // k2 activation energy

  R   =  1.987;       // ideal gas constant

  alpha_o = 0.91;      // initial mass fraction of raw coal
  c_o     = 3.90e-11;  // initial mass of raw coal

  Y1_ = 1.0;
  Y2_ = 0.4;

  d_quad_node = qn;
}

KobayashiSarofimDevol::~KobayashiSarofimDevol()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
KobayashiSarofimDevol::problemSetup(const ProblemSpecP& params, int qn)
{
  ProblemSpecP db = params; 
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
    string label_name;
    string role_name;

    variable->getAttribute("label",label_name);
    variable->getAttribute("role",role_name);

    std::string temp_label_name = label_name;
    std::string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_label_name += "_qn";
    temp_label_name += node;

    // This way restricts what "roles" the user can specify (less flexible)
    if (role_name == "temperature" || role_name == "raw_coal_mass_fraction") {
      LabelToRoleMap[temp_label_name] = role_name;
    } else {
      std::string errmsg;
      errmsg = "Invalid variable role for Kobayashi Sarofim Devolatilization model: must be \"temperature\" or \"raw_coal_mass_fraction\", you specified \"" + role_name + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }

    //This way does not restrict what "roles" the user can specify (more flexible)
    //LabelToRoleMap[label_name] = role_name;

    db->getWithDefault( "low_clip", d_lowClip, 1.e-6 );
    db->getWithDefault( "high_clip", d_highClip, 1 );  
 
  }

  // now fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); iString != d_icLabels.end(); ++iString) {
    std::string temp_ic_name        = (*iString);
    std::string temp_ic_name_full   = temp_ic_name;

    std::string node;
    std::stringstream out;
    out << qn;
    node = out.str();
    temp_ic_name_full += "_qn";
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  std::string node; 
  std::stringstream out; 
  out << qn; 
  node = out.str();
  std::string gasDevolName = "gasDevolRate_qn";
  gasDevolName += node; 
  d_gasDevolRate = VarLabel::create(gasDevolName, CCVariable<double>::getTypeDescription());


}
//---------------------------------------------------------------------------
// Method: Schedule the initialization of some variables 
//---------------------------------------------------------------------------
void 
KobayashiSarofimDevol::sched_initVars( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "KobayashiSarofimDevol::initVars";
  Task* tsk = scinew Task(taskname, this, &KobayashiSarofimDevol::initVars);

  tsk->computes(d_gasDevolRate); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}
void
KobayashiSarofimDevol::initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> gasDevolRate; 
    new_dw->allocateAndPut( gasDevolRate, d_gasDevolRate, matlIndex, patch ); 
    gasDevolRate.initialize(0.);

  }
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
  void 
KobayashiSarofimDevol::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "KobayashiSarofimDevol::computeModel";
  Task* tsk = scinew Task(taskname, this, &KobayashiSarofimDevol::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasDevolRate);
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasDevolRate);
  }

  EqnFactory& eqn_factory = EqnFactory::self();
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // construct the weight label corresponding to this quad node
  std::string temp_weight_name = "w_qn";
  std::string node;
  std::stringstream out;
  out << d_quad_node;
  node = out.str();
  temp_weight_name += node;
  EqnBase& weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  d_weight_label = weight_eqn.getTransportEqnLabel();
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);

  // For each required variable, determine if it plays the role of temperature or mass fraction;
  //  if it plays the role of mass fraction, then look for it in equation factories
  for (vector<std::string>::iterator iter = d_icLabels.begin(); 
      iter != d_icLabels.end(); iter++) { 

    map<string, string>::iterator iMap = LabelToRoleMap.find(*iter);

    if ( iMap != LabelToRoleMap.end() ) {
      if ( iMap->second == "temperature") {
        // automatically use Arches' temperature label if role="temperature"
        tsk->requires(Task::OldDW, d_fieldLabels->d_dummyTLabel, Ghost::AroundCells, 1);

        // Only require() variables found in equation factories (right now we're not tracking temperature this way)
      } else if ( iMap->second == "raw_coal_mass_fraction") {
        // if it's a normal scalar
        if ( eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& current_eqn = eqn_factory.retrieve_scalar_eqn(*iter);
          d_raw_coal_mass_fraction_label = current_eqn.getTransportEqnLabel();
          tsk->requires(Task::OldDW, d_raw_coal_mass_fraction_label, Ghost::None, 0);
          // if it's a dqmom scalar
        } else if (dqmom_eqn_factory.find_scalar_eqn(*iter) ) {
          EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(*iter);
          DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
          d_raw_coal_mass_fraction_label = current_eqn.getTransportEqnLabel();
          d_rc_scaling_factor = current_eqn.getScalingConstant();
          tsk->requires(Task::OldDW, d_raw_coal_mass_fraction_label, Ghost::None, 0);
        } else {
          std::string errmsg = "ARCHES: KobayashiSarofimDevol: Invalid variable given in <variable> tag for KobayashiSarofimDevol model";
          errmsg += "\nCould not find given coal mass fraction variable \"";
          errmsg += *iter;
          errmsg += "\" in EqnFactory or in DQMOMEqnFactory.";
          throw InvalidValue(errmsg,__FILE__,__LINE__);
        }
      } //else... we don't need that variable!!!
    } else {
      // can't find it in the labels-to-roles map!
      std::string errmsg = "ARCHES: KobayashiSarofimDevol: Could not find role for given variable \"" + *iter + "\".";
      throw InvalidValue(errmsg,__FILE__,__LINE__);
    }
  }

  // also need to track coal gas mixture fraction for visualization
  // "modify" source term for coal gas mixture fraction here...

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
  void
KobayashiSarofimDevol::computeModel( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> devol_rate;
    CCVariable<double> gas_devol_rate; 
    if (new_dw->exists( d_modelLabel, matlIndex, patch )){
      new_dw->getModifiable( devol_rate, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_devol_rate, d_gasDevolRate, matlIndex, patch ); 
      devol_rate.initialize(0.0);
      gas_devol_rate.initialize(0.0);
    } else {
      new_dw->allocateAndPut( devol_rate, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_devol_rate, d_gasDevolRate, matlIndex, patch ); 
      devol_rate.initialize(0.0);
      gas_devol_rate.initialize(0.0);
    }

    // to add:
    // - getModifiable/allocateAndPut coal gas mixture fraction source term
    // - get all weights (for number density)

    constCCVariable<double> temperature;
    old_dw->get( temperature, d_fieldLabels->d_dummyTLabel, matlIndex, patch, gac, 1 );
    constCCVariable<double> w_omegac;
    new_dw->get( w_omegac, d_raw_coal_mass_fraction_label, matlIndex, patch, gn, 0 );
    constCCVariable<double> weight;
    new_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

      double k1 = A1*exp(E1/(R*temperature[c])); // 1/s
      double k2 = A2*exp(E2/(R*temperature[c])); // 1/s

      // clip abscissa values greater than 1
      double omegac = w_omegac[c] / weight[c];
      if ( omegac > d_highClip ) {
        omegac = d_highClip;
      } else if ( omegac < d_lowClip ){
        omegac = d_lowClip;
      } else if ( weight[c] <= 0.0 ){
        omegac = 0.0;
      }

      double testVal = -1.0*(k1+k2)*(omegac);  
      if (testVal < 0.0)
        devol_rate[c] = -1.0*(k1+k2)*(omegac);  
      else 
        devol_rate[c] = 0.0;
      testVal = (Y1_*k1 + Y2_*k2)*omegac*d_rc_scaling_factor;
      if (testVal > 0.0)
        gas_devol_rate[c] = (Y1_*k1 + Y2_*k2)*omegac*d_rc_scaling_factor;
      else 
        gas_devol_rate[c] = 0.0;

    }
  }
}

// COMMENTS AND QUESTIONS:
//
//num_dens           = sum_over_omega( w_a );                       // number densiity - zeroth moment
//m_particle         = alpha[c]*c_o + (1-alpha_o)*c_o;              // mass of particle = raw coal + mineral matter (WHAT ABOUT CHAR???)
//m_total_particles  = num_dens*m_particle;                         // total mass of particles in the volume - from NDF
// QUESTION: how to deal with 2 or more model terms? (e.g. raw coal and char)
//char_model[c] = (0.622*k1)*alpha[c];                            // track char mass fraction so it is bounded from 0 to 1 (This uses Julien's proximate analysis idea - 0.388 instead of 0.3)

// QUESTION: if we're tracking coal gas mixture fraction, why have source term as TOTAL MASS source term?
//coalgas_source[c] = (0.388*k1 + k2)*alphac[c]*m_total_particles; // multiply by mass_p_total to get total amount of volatile gases

