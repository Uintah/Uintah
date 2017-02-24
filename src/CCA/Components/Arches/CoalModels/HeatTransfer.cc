#include <CCA/Components/Arches/CoalModels/HeatTransfer.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

HeatTransfer::HeatTransfer( std::string modelName, 
                            SimulationStateP& sharedState,
                            ArchesLabel* fieldLabels,
                            vector<std::string> icLabelNames, 
                            vector<std::string> scalarLabelNames,
                            int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  d_radiation = false;
  d_quadNode = qn;

  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  std::string qconvName = modelName + "_Qconv";
  d_qconvLabel = VarLabel::create( qconvName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_qconvLabel);

  std::string qradName = modelName + "_Qrad";
  d_qradLabel = VarLabel::create( qradName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_qradLabel);

  //std::string pTName = modelName + "_pT";

  std::string pTName = modelName.insert(modelName.size()-4,"_pT");

  d_pTLabel = VarLabel::create( pTName, CCVariable<double>::getTypeDescription() );
  _extra_local_labels.push_back(d_pTLabel);

}

HeatTransfer::~HeatTransfer()
{
  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++) {
    VarLabel::destroy( *iter );
  }
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
  void 
HeatTransfer::problemSetup(const ProblemSpecP& params, int qn)
{
  ProblemSpecP db = params; 

  // set model clipping (not used yet...)
  db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  // grab weight scaling factor and small value
  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();

  // check for a radiation model: 
  SourceTermFactory& srcs = SourceTermFactory::self(); 
  if ( srcs.source_type_exists("do_radiation") ){
    d_radiation = true; 
  }
  if ( srcs.source_type_exists( "rmcrt_radiation") ){
    d_radiation = true;
  }

  //user can specifically turn off radiation heat transfer
  if (db->findBlock("noRadiation"))
    d_radiation = false; 

  // set model clipping
  db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
  db->getWithDefault( "high_clip", d_highModelClip, 999999 );

  string node;
  std::stringstream out;
  out << qn; 
  node = out.str();

  string temp_weight_name = "w_qn";
  temp_weight_name += node;
  EqnBase& t_weight_eqn = dqmom_eqn_factory.retrieve_scalar_eqn( temp_weight_name );
  DQMOMEqn& weight_eqn = dynamic_cast<DQMOMEqn&>(t_weight_eqn);

  d_w_small = weight_eqn.getSmallClipPlusTol();
  d_w_scaling_constant = weight_eqn.getScalingConstant(d_quadNode);

  // Find the absorption coefficient term associated with this model
  // if it isn't there, print a warning
  std::string modelName;
  std::string baseNameAbskp;

  if (d_radiation ) {
    
    _radiateAtGasTemp=true; // this flag is arbitrary for no radiation 
    ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModels");
    for ( ProblemSpecP db_model = db_prop->findBlock("model"); db_model != nullptr; db_model = db_model->findNextBlock("model")){
      db_model->getAttribute( "type", modelName );
      if ( modelName=="radiation_properties" ) {
        if ( db_model->findBlock("calculator") == nullptr ) {
          if( qn == 0 ) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
          }
          d_radiation = false;
          break;
        }
        else if( db_model->findBlock("calculator")->findBlock("particles") == nullptr ){
          if(qn ==0) {
            proc0cout <<"\n///-------------------------------------------///\n";
            proc0cout <<"WARNING: No radiation particle properties computed!\n";
            proc0cout <<"Particles will not interact with radiation!\n";
            proc0cout <<"///-------------------------------------------///\n";
          }
          d_radiation = false;
          break;
        }
        db_model->findBlock("calculator")->findBlock("particles")->findBlock("abskp")->getAttribute("label",baseNameAbskp);
        db_model->findBlock("calculator")->findBlock("particles")->getWithDefault( "radiateAtGasTemp", _radiateAtGasTemp, true ); 
        break;
      }
      if ( db_model == nullptr ){
        if( qn == 0 ) {
          proc0cout <<"\n///-------------------------------------------///\n";
          proc0cout <<"WARNING: No radiation particle properties computed!\n";
          proc0cout <<"Particles will not interact with radiation!\n";
          proc0cout <<"///-------------------------------------------///\n";
        }
        d_radiation = false;
        break;
      }
    }
    std::stringstream out2;
    out2 <<baseNameAbskp <<"_"<< qn; 
    d_abskpLabel = VarLabel::find(out2.str());
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
/** @details
This method intentionally does nothing.

@see HeatTransfer::initVars()
*/
void 
HeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "HeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &HeatTransfer::initVars);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
/** @details
This method is left intentionally blank.  This way, if the method is
 called for a heat transfer model, and that heat transfer model
 doesn't require the initialization of any variables, the child 
 class will not need to re-define this (empty) method.

If additional variables are needed, and initVars needs to do stuff,
 the model can redefine it.
*/
void
HeatTransfer::initVars( const ProcessorGroup * pc, 
                        const PatchSubset    * patches, 
                        const MaterialSubset * matls, 
                        DataWarehouse        * old_dw, 
                        DataWarehouse        * new_dw )
{
  // This method left intentionally blank...
  // It has the form:
  /*
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> something; 
    new_dw->allocateAndPut( something, d_something_label, matlIndex, patch ); 
    something.initialize(0.0)

  }
  */
}

