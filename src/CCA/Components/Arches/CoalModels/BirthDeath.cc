#include <CCA/Components/Arches/CoalModels/BirthDeath.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ArchesLabel.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:

BirthDeathBuilder::BirthDeathBuilder( const std::string         & modelName,
                                        const vector<std::string> & reqICLabelNames,
                                        const vector<std::string> & reqScalarLabelNames,
                                        ArchesLabel         * fieldLabels,
                                        MaterialManagerP          & materialManager,
                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, materialManager, qn )
{}

BirthDeathBuilder::~BirthDeathBuilder(){}

ModelBase* BirthDeathBuilder::build(){
  return scinew BirthDeath( d_modelName, d_materialManager, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

BirthDeath::BirthDeath( std::string           modelName,
                          MaterialManagerP    & materialManager,
                          ArchesLabel   * fieldLabels,
                          vector<std::string>   icLabelNames,
                          vector<std::string>   scalarLabelNames,
                          int qn )
: ModelBase(modelName, materialManager, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );
  // Create a label for source of particle that are implacting the wall
  stringstream ss;
  ss << qn; 
  string delimiter = "qn"+ss.str()  ;

  string name_1    = modelName.substr(0, modelName.find(delimiter));

  std::string modelImpactName = name_1+"impact_loss_"+delimiter;
  d_model_impactLabel = VarLabel::create( modelImpactName, CCVariable<double>::getTypeDescription() );


  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  _is_weight = false;
  _deposition = false;

}

BirthDeath::~BirthDeath()
{
  VarLabel::destroy(d_model_impactLabel); 
}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
BirthDeath::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb;

  if ( db->findBlock("is_weight")){
    _is_weight = true;
  }

  ProblemSpecP db_partmodels =
    db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");
  for( ProblemSpecP db_model =
    db_partmodels->findBlock("model"); db_model != nullptr;
    db_model = db_model->findNextBlock("model") ) {

    std::string model_type;
    db_model->getAttribute("type", model_type);
    if ( model_type == "rate_deposition" ){
      _deposition = true;
    }
  }

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  if ( !_is_weight ){

    if ( db->findBlock("abscissa") ){
      db->findBlock("abscissa")->getAttribute( "label", _abscissa_name );
    } else {
      throw ProblemSetupException(
        "Error: Must specify an abscissa label for this model.",__FILE__,__LINE__);
    }

    std::string a_name = ArchesCore::append_qn_env( _abscissa_name, d_quadNode );
    EqnBase& a_eqn = dqmomFactory.retrieve_scalar_eqn( a_name );
    _a_scale = a_eqn.getScalingConstant(d_quadNode);

  }

  std::string w_name = ArchesCore::append_qn_env( "w", d_quadNode );
  EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn(w_name);
  _w_scale = temp_eqn.getScalingConstant(d_quadNode);
  DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(temp_eqn);
  double weight_clip = eqn.getSmallClip();

  db->require("small_weight",_small_weight);

  if ( weight_clip > _small_weight ){
    throw InvalidValue("Error: The low clip limit for the weight must be smaller than the small_weight limit for the BirthDeath model.", __FILE__, __LINE__);
  }

  _w_label = VarLabel::find(w_name);

  std::string w_rhs_name = w_name + "_RHS";
  _w_rhs_label = VarLabel::find(w_rhs_name);

  if ( _w_label == 0 ){
    throw InvalidValue("Error:Weight not found: "+w_name, __FILE__, __LINE__);
  }
  if ( _w_rhs_label == 0 ){
    throw InvalidValue("Error:Weight RHS not found: "+w_rhs_name, __FILE__, __LINE__);
  }

  if ( _deposition ){
    // create rate_deposition model base names
    std::string rate_dep_base_nameX = "RateDepositionX";
    std::string rate_dep_base_nameY = "RateDepositionY";
    std::string rate_dep_base_nameZ = "RateDepositionZ";
    std::string rate_dep_X = ArchesCore::append_env( rate_dep_base_nameX, d_quadNode );
    std::string rate_dep_Y = ArchesCore::append_env( rate_dep_base_nameY, d_quadNode );
    std::string rate_dep_Z = ArchesCore::append_env( rate_dep_base_nameZ, d_quadNode );
    _rate_depX_varlabel = VarLabel::find(rate_dep_X);
    _rate_depY_varlabel = VarLabel::find(rate_dep_Y);
    _rate_depZ_varlabel = VarLabel::find(rate_dep_Z);

    // Need a size IC:
    std::string length_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_SIZE);
    std::string length_name = ArchesCore::append_env( length_root, d_quadNode );
    _length_varlabel = VarLabel::find(length_name);

    // Need a density
    std::string density_root = ArchesCore::parse_for_particle_role_to_label(db, ArchesCore::P_DENSITY);
    std::string density_name = ArchesCore::append_env( density_root, d_quadNode );
    _particle_density_varlabel = VarLabel::find(density_name);

    std::string rate_impact_base_nameX = "RateImpactLossX";
    std::string rate_impact_base_nameY = "RateImpactLossY";
    std::string rate_impact_base_nameZ = "RateImpactLossZ";

    std::string rate_impact_X = ArchesCore::append_env( rate_impact_base_nameX, d_quadNode );
    std::string rate_impact_Y = ArchesCore::append_env( rate_impact_base_nameY, d_quadNode );
    std::string rate_impact_Z = ArchesCore::append_env( rate_impact_base_nameZ, d_quadNode );
    _rate_impactX_varlabel = VarLabel::find(rate_impact_X);
    _rate_impactY_varlabel = VarLabel::find(rate_impact_Y);
    _rate_impactZ_varlabel = VarLabel::find(rate_impact_Z);


  }
  _pi = acos(-1.0);
}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void
BirthDeath::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "BirthDeath::initVars";
  Task* tsk = scinew Task(taskname, this, &BirthDeath::initVars);

  tsk->computesVar(d_modelLabel);
  tsk->computesVar(d_model_impactLabel);
  tsk->computesVar(d_gasLabel);

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
BirthDeath::initVars( const ProcessorGroup * pc,
                            const PatchSubset    * patches,
                            const MaterialSubset * matls,
                            DataWarehouse        * old_dw,
                            DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> model;

    CCVariable<double> gas_source;

    new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
    model.initialize(0.0);
    new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
    gas_source.initialize(0.0);

    //if ( _deposition ){
    CCVariable<double> model_impact;
    new_dw->allocateAndPut( model_impact, d_model_impactLabel, matlIndex, patch );
    model_impact.initialize(0.0);
    //}

  }
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model
//---------------------------------------------------------------------------
void
BirthDeath::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BirthDeath::computeModel";
  Task* tsk = scinew Task(taskname, this, &BirthDeath::computeModel, timeSubStep );
  Ghost::GhostType gn = Ghost::None;
  Ghost::GhostType  gaf = Ghost::AroundFaces;

  if ( !_is_weight ){
    std::string abscissa_name = ArchesCore::append_env( _abscissa_name, d_quadNode );
    _abscissa_label = VarLabel::find(abscissa_name);
    if ( _abscissa_label == 0 )
      throw InvalidValue("Error: Abscissa not found: "+abscissa_name, __FILE__, __LINE__);
  }

  if (timeSubStep == 0) {
    tsk->computesVar(d_modelLabel);
    tsk->computesVar(d_model_impactLabel);
    tsk->computesVar(d_gasLabel);
    tsk->requiresVar(Task::OldDW, _w_label, Ghost::None, 0);
    if ( _deposition ){
      tsk->requiresVar(Task::OldDW, _rate_depX_varlabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, _rate_depY_varlabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, _rate_depZ_varlabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, _length_varlabel, gn, 0 );
      tsk->requiresVar(Task::OldDW, _particle_density_varlabel, gn, 0 );

      tsk->requiresVar(Task::OldDW, _rate_impactX_varlabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, _rate_impactY_varlabel, gaf, 1);
      tsk->requiresVar(Task::OldDW, _rate_impactZ_varlabel, gaf, 1);

    }
    if ( !_is_weight )
      tsk->requiresVar(Task::OldDW, _abscissa_label, Ghost::None, 0);
  } else {
    tsk->modifiesVar(d_modelLabel);
    tsk->modifiesVar(d_model_impactLabel);
    tsk->modifiesVar(d_gasLabel);
    tsk->requiresVar(Task::NewDW, _w_label, Ghost::None, 0);
    if ( _deposition ){
      tsk->requiresVar(Task::NewDW, _rate_depX_varlabel, gaf, 1);
      tsk->requiresVar(Task::NewDW, _rate_depY_varlabel, gaf, 1);
      tsk->requiresVar(Task::NewDW, _rate_depZ_varlabel, gaf, 1);
      tsk->requiresVar(Task::NewDW, _length_varlabel, gn, 0 );
      tsk->requiresVar(Task::NewDW, _particle_density_varlabel, gn, 0 );

      tsk->requiresVar(Task::NewDW, _rate_impactX_varlabel, gaf, 1);
      tsk->requiresVar(Task::NewDW, _rate_impactY_varlabel, gaf, 1);
      tsk->requiresVar(Task::NewDW, _rate_impactZ_varlabel, gaf, 1);
    }
    if ( !_is_weight )
      tsk->requiresVar(Task::NewDW, _abscissa_label, Ghost::None, 0);
  }

  tsk->requiresVar(Task::NewDW, _w_rhs_label, Ghost::None, 0);
  tsk->requiresVar(Task::OldDW, d_fieldLabels->d_delTLabel, Ghost::None, 0);
  tsk->requiresVar(Task::OldDW, VarLabel::find("volFraction"), Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), d_materialManager->allMaterials( "Arches" ));

}


//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
BirthDeath::computeModel( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gaf = Ghost::AroundFaces;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    Vector DX = patch->dCell();
    double vol = DX.x()*DX.y()*DX.z();
    double area_x = DX.y()*DX.z();
    double area_y = DX.x()*DX.z();
    double area_z = DX.x()*DX.y();

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_delTLabel);
    double dt = DT;

    CCVariable<double> model;
    CCVariable<double> model_impact;
    CCVariable<double> gas_source;
    DataWarehouse* which_dw;

    if ( timeSubStep == 0 ){
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( model_impact, d_model_impactLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
      model.initialize(0.0);
      model_impact.initialize(0.0);
      which_dw = old_dw;
    } else {
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );
      new_dw->getModifiable( model_impact, d_model_impactLabel, matlIndex, patch );
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
      model.initialize(0.0);
      model_impact.initialize(0.0);
      which_dw = new_dw;
    }

    constCCVariable<double> w;
    constCCVariable<double> w_rhs;
    constCCVariable<double> a;
    constCCVariable<double> vol_fraction;
    constCCVariable<double> diam;
    constCCVariable<double> rhop;
    constSFCXVariable<double> rate_X;
    constSFCYVariable<double> rate_Y;
    constSFCZVariable<double> rate_Z;

    constSFCXVariable<double> rate_impact_X;
    constSFCYVariable<double> rate_impact_Y;
    constSFCZVariable<double> rate_impact_Z;


    which_dw->get( w, _w_label, matlIndex, patch, Ghost::None, 0 );
    new_dw->get( w_rhs, _w_rhs_label, matlIndex, patch, Ghost::None, 0 );
    old_dw->get( vol_fraction, VarLabel::find("volFraction"), matlIndex, patch, Ghost::None, 0 );

    if ( _deposition ){
      which_dw->get( diam, _length_varlabel, matlIndex, patch, gn, 0 );
      which_dw->get( rhop, _particle_density_varlabel, matlIndex, patch, gn, 0 );
      which_dw->get( rate_X, _rate_depX_varlabel, matlIndex, patch, gaf, 1 );
      which_dw->get( rate_Y, _rate_depY_varlabel, matlIndex, patch, gaf, 1 );
      which_dw->get( rate_Z, _rate_depZ_varlabel, matlIndex, patch, gaf, 1 );

      //particles impacting the wall but not getting deposited
      which_dw->get( rate_impact_X, _rate_impactX_varlabel, matlIndex, patch, gaf, 1 );
      which_dw->get( rate_impact_Y, _rate_impactY_varlabel, matlIndex, patch, gaf, 1 );
      which_dw->get( rate_impact_Z, _rate_impactZ_varlabel, matlIndex, patch, gaf, 1 );

    }
    if ( !_is_weight ){
      which_dw->get( a, _abscissa_label, matlIndex, patch, Ghost::None, 0 );
    }

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
    if ( _deposition ){
      Uintah::parallel_for(range,  [&](int i, int j, int k) {

        double vol_p = (_pi/6.0) * diam(i,j,k) * diam(i,j,k) * diam(i,j,k);


        // particles impacting the wall but not getting deposited

        const double dimpact_src =( abs(rate_impact_X(i,j,k))*area_x
                               + abs(rate_impact_Y(i,j,k))*area_y
                               + abs(rate_impact_Z(i,j,k))*area_z
                               + abs(rate_impact_X(i+1,j,k))*area_x
                               + abs(rate_impact_Y(i,j+1,k))*area_y
                               + abs(rate_impact_Z(i,j,k+1))*area_z )
                               / ( -1.0*rhop(i,j,k)*vol_p*vol*_w_scale );// scaled #/s/m^3

        // here we add the birth rate to the destruction rate
        model_impact(i,j,k) = dimpact_src ; // scaled #/s/m^3
        // note here w and w_rhs are already scaled.
        model_impact(i,j,k) *=  _is_weight ? 1.0 : a(i,j,k)/_a_scale; // if weight
        model_impact(i,j,k) *= vol_fraction(i,j,k);

        const double dstrc_src =( abs(rate_X(i,j,k))*area_x
                               + abs(rate_Y(i,j,k))*area_y
                               + abs(rate_Z(i,j,k))*area_z
                               + abs(rate_X(i+1,j,k))*area_x
                               + abs(rate_Y(i,j+1,k))*area_y
                               + abs(rate_Z(i,j,k+1))*area_z )
                               / ( -1.0*rhop(i,j,k)*vol_p*vol*_w_scale );// scaled #/s/m^3

        // here we add the birth rate to the destruction rate
        model(i,j,k) = dstrc_src + std::max(( _small_weight - w(i,j,k) ) / dt - w_rhs(i,j,k) / vol - dstrc_src ,0.0); // scaled #/s/m^3

        // note here w and w_rhs are already scaled.
        model(i,j,k)*= _is_weight ? 1.0 : a(i,j,k)/_a_scale; // if weight

        model(i,j,k)*= vol_fraction(i,j,k);


      });
    } else {
      Uintah::parallel_for(range,  [&](int i, int j, int k) {

        // here we add the birth rate to the destruction rate
        model(i,j,k) = std::max(( _small_weight - w(i,j,k) ) / dt - w_rhs(i,j,k) / vol, 0.0); // scaled #/s/m^3

        // note here w and w_rhs are already scaled.
        model(i,j,k)*= _is_weight ? 1.0 : a(i,j,k)/_a_scale; // if weight

        model(i,j,k)*= vol_fraction(i,j,k);

      });
    }
  }
}
