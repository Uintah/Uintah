#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/ManifoldRxn.h>
#include <Core/Exceptions/ParameterNotFound.h>

//===========================================================================

using namespace std;
using namespace Uintah;

ManifoldRxn::ManifoldRxn( std::string src_name, SimulationStateP& shared_state,
                            vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _conv_label = VarLabel::create( src_name+"_conv", CCVariable<double>::getTypeDescription() );
  _diff_label = VarLabel::create( src_name+"_diff", CCVariable<double>::getTypeDescription() );
  _disc = 0;
}

ManifoldRxn::~ManifoldRxn()
{
  delete _disc;
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
ManifoldRxn::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  _source_grid_type = CC_SRC;

  _disc = scinew Discretization_new();

  db->require("manifold_label",_manifold_var_name);

  if ( db->findBlock("conv_scheme")){
    db->findBlock("conv_scheme")->getAttribute("type",_conv_scheme);
  } else {
    throw ProblemSetupException("Error: Convection scheme not specified.",__FILE__,__LINE__);
  }

  db->getWithDefault("Pr", _prNo, 0.4);

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( _manifold_var_name, ChemHelper::OLD );
  //Note: Density old is added automagically in ClassicTable problemSetup.

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
ManifoldRxn::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "ManifoldRxn::eval";
  Task* tsk = scinew Task(taskname, this, &ManifoldRxn::computeSource, timeSubStep);

  _manifold_label = VarLabel::find( _manifold_var_name );
  if ( _manifold_label == 0 ){
    throw ProblemSetupException("Error: Cannot match the manifold variable name to a Uintah label.",__FILE__,__LINE__);
  }

  _old_manifold_label = VarLabel::find( _manifold_var_name+"_old");
  tsk->requires( Task::NewDW, _old_manifold_label, Ghost::AroundCells, 1 );

  if (timeSubStep == 0) {
    tsk->computes(_src_label);
    tsk->computes(_conv_label);
    tsk->computes(_diff_label);
  } else {
    tsk->modifies(_src_label);
    tsk->modifies(_conv_label);
    tsk->modifies(_diff_label);
  }

  tsk->requires( Task::NewDW, _manifold_label, Ghost::None, 0 );
  tsk->requires(Task::NewDW, VarLabel::find("uVelocitySPBC"), Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, VarLabel::find("vVelocitySPBC"), Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, VarLabel::find("wVelocitySPBC"), Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, VarLabel::find("areaFraction"), Ghost::AroundCells, 2);
  tsk->requires(Task::NewDW, VarLabel::find("density_old"), Ghost::AroundCells, 1);
  tsk->requires(Task::NewDW, VarLabel::find("density"), Ghost::None, 0);
  tsk->requires(Task::NewDW, VarLabel::find("turb_viscosity"), Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, _shared_state->get_delt_label());

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
ManifoldRxn::computeSource( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

    constCCVariable<double> old_var;
    constCCVariable<double> new_var;
    new_dw->get( new_var, _manifold_label, matlIndex, patch, Ghost::None, 0 );
    new_dw->get( old_var, _old_manifold_label, matlIndex, patch, Ghost::AroundCells, 1);

    CCVariable<double> src;
    CCVariable<double> conv;
    CCVariable<double> diff;
    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( src, _src_label, matlIndex, patch );
      src.initialize(0.0);
      new_dw->getModifiable( diff, _diff_label, matlIndex, patch );
      new_dw->getModifiable( conv, _conv_label, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( src, _src_label, matlIndex, patch );
      new_dw->allocateAndPut( conv, _conv_label, matlIndex, patch );
      new_dw->allocateAndPut( diff, _diff_label, matlIndex, patch );
      src.initialize(0.0);
      conv.initialize(0.0);
      diff.initialize(0.0);
    }


    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;

    new_dw->get(uVel, VarLabel::find("uVelocitySPBC"), matlIndex, patch, Ghost::AroundCells, 1);
    new_dw->get(vVel, VarLabel::find("vVelocitySPBC"), matlIndex, patch, Ghost::AroundCells, 1);
    new_dw->get(wVel, VarLabel::find("wVelocitySPBC"), matlIndex, patch, Ghost::AroundCells, 1);

    constCCVariable<Vector> areaFraction;
    old_dw->get(areaFraction, VarLabel::find("areaFraction"), matlIndex, patch, Ghost::AroundCells, 2);

    constCCVariable<double> new_den;
    constCCVariable<double> old_den;
    new_dw->get( old_den, VarLabel::find("density_old"), matlIndex, patch, Ghost::AroundCells, 1 );
    new_dw->get( new_den, VarLabel::find("density"), matlIndex, patch, Ghost::None, 0 );

    constCCVariable<double> mu_t;
    new_dw->get( mu_t, VarLabel::find("turb_viscosity"), matlIndex, patch, Ghost::AroundCells, 1);

    double const_mol_D = 0.0;
    _disc->computeConv( patch, conv, old_var, uVel, vVel, wVel,  old_den, areaFraction, _conv_scheme );
    _disc->computeDiff( patch, diff, old_var, mu_t, const_mol_D, old_den, areaFraction, _prNo );

    Vector Dx = patch->dCell();
    double vol = Dx.x()*Dx.y()*Dx.z();

    delt_vartype DT;
    old_dw->get(DT, _shared_state->get_delt_label() );
    double dt = DT;
    double voldt = vol/dt;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;

      //accumulation term:
      double accum = (new_den[c]*new_var[c] - old_den[c]*old_var[c])*voldt;

      src[c] = accum + conv[c] - diff[c];
      src[c] /= (vol);

    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
ManifoldRxn::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ManifoldRxn::initialize";

  Task* tsk = scinew Task(taskname, this, &ManifoldRxn::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
ManifoldRxn::initialize( const ProcessorGroup* pc,
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

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar;
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch );
    }
  }
}
