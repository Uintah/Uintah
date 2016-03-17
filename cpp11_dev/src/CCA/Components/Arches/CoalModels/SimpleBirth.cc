#include <CCA/Components/Arches/CoalModels/SimpleBirth.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>
#include <CCA/Components/Arches/ArchesLabel.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>

//===========================================================================

#include <CCA/Components/Arches/FunctorSwitch.h>

#include <Core/Grid/Variables/BlockRange.h>
#ifdef UINTAH_ENABLE_KOKKOS
#  include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS
using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:

SimpleBirthBuilder::SimpleBirthBuilder( const std::string         & modelName, 
                                        const vector<std::string> & reqICLabelNames,
                                        const vector<std::string> & reqScalarLabelNames,
                                        ArchesLabel         * fieldLabels,
                                        SimulationStateP          & sharedState,
                                        int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{}

SimpleBirthBuilder::~SimpleBirthBuilder(){}

ModelBase* SimpleBirthBuilder::build(){
  return scinew SimpleBirth( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}

// End Builder
//---------------------------------------------------------------------------

SimpleBirth::SimpleBirth( std::string           modelName, 
                          SimulationStateP    & sharedState,
                          ArchesLabel   * fieldLabels,
                          vector<std::string>   icLabelNames, 
                          vector<std::string>   scalarLabelNames,
                          int qn ) 
: ModelBase(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Create a label for this model
  d_modelLabel = VarLabel::create( modelName, CCVariable<double>::getTypeDescription() );

  // Create the gas phase source term associated with this model
  std::string gasSourceName = modelName + "_gasSource";
  d_gasLabel = VarLabel::create( gasSourceName, CCVariable<double>::getTypeDescription() );

  _is_weight = false;

}

SimpleBirth::~SimpleBirth()
{}



//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SimpleBirth::problemSetup(const ProblemSpecP& inputdb, int qn)
{

  ProblemSpecP db = inputdb; 

  if ( db->findBlock("is_weight")){ 
    _is_weight = true; 
  }

  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  if ( !_is_weight ){ 

    if ( db->findBlock("abscissa") ){ 
      db->findBlock("abscissa")->getAttribute( "label", _abscissa_name );
    } else { 
      throw ProblemSetupException("Error: Must specify an abscissa label for this model.",__FILE__,__LINE__);
    }

    std::string a_name = ParticleTools::append_qn_env( _abscissa_name, d_quadNode ); 
    EqnBase& a_eqn = dqmomFactory.retrieve_scalar_eqn( a_name ); 
    _a_scale = a_eqn.getScalingConstant(d_quadNode); 

  }

  std::string w_name = ParticleTools::append_qn_env( "w", d_quadNode ); 
  EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn(w_name);
  DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(temp_eqn);
  double weight_clip = eqn.getSmallClip();

  db->require("small_weight",_small_weight);  

  if ( weight_clip > _small_weight ){ 
    throw InvalidValue("Error: The low clip limit for the weight must be smaller than the small_weight limit for the SimpleBirth model.", __FILE__, __LINE__); 
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

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
SimpleBirth::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  string taskname = "SimpleBirth::initVars"; 
  Task* tsk = scinew Task(taskname, this, &SimpleBirth::initVars);

  tsk->computes(d_modelLabel);
  tsk->computes(d_gasLabel);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
SimpleBirth::initVars( const ProcessorGroup * pc, 
                            const PatchSubset    * patches, 
                            const MaterialSubset * matls, 
                            DataWarehouse        * old_dw, 
                            DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model; 
    CCVariable<double> gas_source;
    
    new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
    model.initialize(0.0);
    new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
    gas_source.initialize(0.0);
  }
}



//---------------------------------------------------------------------------
// Method: Schedule the calculation of the model 
//---------------------------------------------------------------------------
void 
SimpleBirth::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "SimpleBirth::computeModel";
  Task* tsk = scinew Task(taskname, this, &SimpleBirth::computeModel, timeSubStep );

  if ( !_is_weight ){ 
    std::string abscissa_name = ParticleTools::append_env( _abscissa_name, d_quadNode ); 
    _abscissa_label = VarLabel::find(abscissa_name); 
    if ( _abscissa_label == 0 )
      throw InvalidValue("Error: Abscissa not found: "+abscissa_name, __FILE__, __LINE__); 
  }

  if (timeSubStep == 0) {
    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel);
    tsk->requires(Task::OldDW, _w_label, Ghost::None, 0); 
    if ( !_is_weight )
      tsk->requires(Task::OldDW, _abscissa_label, Ghost::None, 0); 
  } else {
    tsk->modifies(d_modelLabel); 
    tsk->modifies(d_gasLabel); 
    tsk->requires(Task::NewDW, _w_label, Ghost::None, 0); 
    if ( !_is_weight )
      tsk->requires(Task::NewDW, _abscissa_label, Ghost::None, 0); 
  }

  tsk->requires(Task::NewDW, _w_rhs_label, Ghost::None, 0); 
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0);
  tsk->requires(Task::OldDW, VarLabel::find("volFraction"), Ghost::None, 0 ); 

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
    
struct computeBirth{
       computeBirth(double _vol,
                    double _dt,
                    double _small_weight,
                    double _a_scale,
                    bool _is_weight,
                    constCCVariable<double> &_w, 
                    constCCVariable<double> &_w_rhs, 
                    constCCVariable<double> &_a, 
                    constCCVariable<double> &_vol_fraction,
                    CCVariable<double> &_model) :
                    vol(_vol),
                    dt(_dt),
                    small_weight(_small_weight),
                    a_scale(_a_scale),
                    is_weight(_is_weight),
                    w(_w),
                    w_rhs(_w_rhs),
                    a(_a),
                    vol_fraction(_vol_fraction),
                    model(_model) {  }

  void operator()(int i , int j, int k ) const { 

    if ( vol_fraction(i,j,k) > 0. ){ 

        model(i,j,k) = std::max(( small_weight - w(i,j,k) ) / dt - w_rhs(i,j,k) / vol ,0.0);
        model(i,j,k)*= vol_fraction(i,j,k); 
        model(i,j,k)*= is_weight ? 1.0 : a(i,j,k)/a_scale;

    }  
  }

  private:
      double vol;
      double dt;
      double small_weight;
      double a_scale;
      double is_weight;
      constCCVariable<double> &w; 
      constCCVariable<double> &w_rhs; 
      constCCVariable<double> &a; 
      constCCVariable<double> &vol_fraction;
      CCVariable<double> &model; 
};

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
SimpleBirth::computeModel( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    Vector DX = patch->dCell(); 
    double vol = DX.x()*DX.y()*DX.z();

    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT; 

    CCVariable<double> model; 
    CCVariable<double> gas_source;
    DataWarehouse* which_dw; 

    if ( timeSubStep == 0 ){ 
      new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
      new_dw->allocateAndPut( gas_source, d_gasLabel, matlIndex, patch );
      gas_source.initialize(0.0);
      model.initialize(0.0);
      which_dw = old_dw; 
    } else { 
      new_dw->getModifiable( model, d_modelLabel, matlIndex, patch ); 
      new_dw->getModifiable( gas_source, d_gasLabel, matlIndex, patch ); 
      gas_source.initialize(0.0);
      which_dw = new_dw; 
    }

    constCCVariable<double> w; 
    constCCVariable<double> w_rhs; 
    constCCVariable<double> a; 
    constCCVariable<double> vol_fraction; 

    which_dw->get( w, _w_label, matlIndex, patch, Ghost::None, 0 ); 
    new_dw->get( w_rhs, _w_rhs_label, matlIndex, patch, Ghost::None, 0 ); 
    old_dw->get( vol_fraction, VarLabel::find("volFraction"), matlIndex, patch, Ghost::None, 0 ); 
    if ( !_is_weight ){ 
      which_dw->get( a, _abscissa_label, matlIndex, patch, Ghost::None, 0 ); 
    }

#ifdef USE_FUNCTOR              
  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
  computeBirth doBirth(vol, dt, _small_weight, _a_scale, _is_weight,
                       w, 
                       w_rhs, 
                       a, 
                       vol_fraction,
                       model);
      Uintah::parallel_for( range, doBirth );
#else
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      if ( vol_fraction[c] > 0. ){ 

        double balance = ( _small_weight - w[c] ) / dt - w_rhs[c] / vol;

        balance = std::max(balance, 0.0) * vol_fraction[c]; 

        if ( _is_weight ){ 

          model[c] = balance; 

        } else { 

          model[c] = a[c]/_a_scale * balance; 

        }
      } 
    }
#endif
  }
}
