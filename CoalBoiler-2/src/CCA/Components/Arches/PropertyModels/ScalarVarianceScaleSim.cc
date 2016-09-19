#include <CCA/Components/Arches/PropertyModels/ScalarVarianceScaleSim.h>
#include <CCA/Components/Arches/Filter.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
ScalarVarianceScaleSim::ScalarVarianceScaleSim( std::string prop_name, SimulationStateP& shared_state ) : PropertyModelBase( prop_name, shared_state )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 

  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 

  _boundary_condition = scinew BoundaryCondition_new( shared_state->getArchesMaterial(0)->getDWIndex() ); 

}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
ScalarVarianceScaleSim::~ScalarVarianceScaleSim( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

    VarLabel::destroy( *iter ); 

  }

  delete _filter; 
  delete _boundary_condition;

}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 

  bool use_old_filter = true; 

  db->require( "mixture_fraction_label", _mf_label_name ); 
  db->require( "density_label", _density_label_name ); 
  db->require( "variance_coefficient", _Cf ); 
  std::string filter_type="moin98";
  if ( db->findBlock("filter_type")){ 
    db->getWithDefault("filter_type",filter_type, "moin98"); 
  }

  int filter_width=3; 
  db->getWithDefault("filter_width",filter_width,3);

  _filter = scinew Filter( use_old_filter, filter_type, filter_width );

}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "ScalarVarianceScaleSim::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &ScalarVarianceScaleSim::computeProp, time_substep ); 

  _density_label = 0; 
  _mf_label = 0; 
  _density_label = VarLabel::find( _density_label_name ); 
  _mf_label      = VarLabel::find( _mf_label_name ); 
  _vol_frac_label= VarLabel::find( "volFraction" ); 
  _celltype_label = VarLabel::find( "cellType" ); 
  _filter_vol_label = VarLabel::find( "filterVolume" ); 

  if ( _mf_label == 0 ){ 
    throw InvalidValue("Error: Cannot match mixture fraction name with label.",__FILE__, __LINE__);             
  } 
  if ( _density_label == 0 ){ 
    throw InvalidValue("Error: Cannot match density name with label.",__FILE__, __LINE__);             
  } 
  if ( _vol_frac_label == 0 ){ 
    throw InvalidValue("Error: Cannot match volume fraction name with label.",__FILE__, __LINE__);             
  } 

  tsk->modifies( _prop_label ); 
  if ( time_substep == 0 ){ 

    tsk->requires( Task::OldDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::OldDW, _density_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::OldDW, _vol_frac_label, Ghost::None, 0 ); 

  } else { 

    tsk->requires( Task::NewDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::NewDW, _density_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::NewDW, _vol_frac_label, Ghost::None, 0 ); 

  } 

  tsk->requires( Task::NewDW, _filter_vol_label, Ghost::None, 0); 
  tsk->requires( Task::NewDW, _celltype_label,     Ghost::AroundCells, 1); 

  sched->addTask( tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 

}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::computeProp(const ProcessorGroup* pc, 
                                    const PatchSubset* patches, 
                                    const MaterialSubset* matls, 
                                    DataWarehouse* old_dw, 
                                    DataWarehouse* new_dw, 
                                    int time_substep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    Array3<double> filterRho(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhi(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    Array3<double> filterRhoPhiSqr(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex());
    filterRho.initialize(0.0);
    filterRhoPhi.initialize(0.0);
    filterRhoPhiSqr.initialize(0.0);

    IntVector idxLo = patch->getExtraCellLowIndex(1);
    IntVector idxHi = patch->getExtraCellHighIndex(1);

    Array3<double> rhoPhi(idxLo, idxHi);
    Array3<double> rhoPhiSqr(idxLo, idxHi);

    CCVariable<double> norm_scalar_var; 
    constCCVariable<double> density; 
    constCCVariable<double> mf; 
    constCCVariable<double> vol_fraction; 
    constCCVariable<double> filter_volume; 
    constCCVariable<int>    cell_type; 

    new_dw->get(filter_volume, _filter_vol_label, matlIndex, patch, Ghost::None, 0); 
    new_dw->get(cell_type, _celltype_label, matlIndex, patch, Ghost::AroundCells, 1); 

    new_dw->getModifiable( norm_scalar_var, _prop_label, matlIndex, patch ); 
    if ( time_substep == 0 ) { 
      norm_scalar_var.initialize(0.0); 
      old_dw->get( mf,      _mf_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      old_dw->get( density, _density_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      old_dw->get( vol_fraction, _vol_frac_label, matlIndex, patch, Ghost::None, 0 ); 
    } else { 
      new_dw->get( mf,      _mf_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      new_dw->get( density, _density_label, matlIndex, patch, Ghost::AroundCells, 1 ); 
      new_dw->get( vol_fraction, _vol_frac_label, matlIndex, patch, Ghost::None, 0 ); 
    } 

    CellIterator iter = patch->getCellIterator(1); 

    //create temp fields for filtering
    for (iter.begin(); !iter.done(); iter++){
      IntVector c = *iter; 
      rhoPhi[c]    = density[c]*mf[c];
      rhoPhiSqr[c] = density[c]*mf[c]*mf[c];
    }

    //filter the fields
    _filter->applyFilter_noPetsc<constCCVariable<double> >(pc, patch, density, filter_volume, cell_type, filterRho ); 
    _filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoPhi, filter_volume, cell_type, filterRhoPhi ); 
    _filter->applyFilter_noPetsc<Array3<double> >(pc, patch, rhoPhiSqr, filter_volume, cell_type, filterRhoPhiSqr ); 

    double small = 1e-10;

    iter = patch->getCellIterator(); 

    for (iter.begin(); !iter.done(); iter++){

      IntVector c = *iter; 

      double filter_phi = 0.0; 
      if ( vol_fraction[c] > 0.0 ) {
        filter_phi = filterRhoPhi[c] / filterRho[c]; 

        norm_scalar_var[c] = _Cf * ( filterRhoPhiSqr[c]/filterRho[c] - filter_phi * filter_phi ); 

        //limits: 
        double var_limit = filter_phi * ( 1.0 - filter_phi ); 

        if ( norm_scalar_var[c] < small ){ 
          norm_scalar_var[c] = 0.0; 
        } else if ( norm_scalar_var[c] > var_limit ){ 
          norm_scalar_var[c] = var_limit; 
        } 

        norm_scalar_var[c] /= var_limit + small; //normalize it.
        norm_scalar_var[c] *= vol_fraction[c]; 
      } else { 
        norm_scalar_var[c] = 0.0;
      }

    }

    _boundary_condition->setScalarValueBC( 0, patch, norm_scalar_var, _prop_name ); 

  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "ScalarVarianceScaleSim::initialize"; 

  Task* tsk = scinew Task(taskname, this, &ScalarVarianceScaleSim::initialize);
  tsk->computes(_prop_label); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void ScalarVarianceScaleSim::initialize( const ProcessorGroup* pc, 
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

    CCVariable<double> prop; 

    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 

    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 

  }
}
