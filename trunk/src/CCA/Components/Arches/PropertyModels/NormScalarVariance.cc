#include <CCA/Components/Arches/PropertyModels/NormScalarVariance.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/BoundaryCondition.h>

using namespace Uintah; 

//---------------------------------------------------------------------------
//Method: Constructor
//---------------------------------------------------------------------------
NormScalarVariance::NormScalarVariance( std::string prop_name, MaterialManagerP& materialManager ) : PropertyModelBase( prop_name, materialManager )
{
  _prop_label = VarLabel::create( prop_name, CCVariable<double>::getTypeDescription() ); 
  
  // Evaluated before or after table lookup: 
  _before_table_lookup = true; 
  
}

//---------------------------------------------------------------------------
//Method: Destructor
//---------------------------------------------------------------------------
NormScalarVariance::~NormScalarVariance( )
{
  // Destroying all local VarLabels stored in _extra_local_labels: 
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    VarLabel::destroy( *iter ); 
  }  
}


//---------------------------------------------------------------------------
//Method: Problem Setup
//---------------------------------------------------------------------------
void NormScalarVariance::problemSetup( const ProblemSpecP& inputdb )
{
  ProblemSpecP db = inputdb; 
  db->require( "mixture_fraction_label", _mf_label_name ); 
  db->require( "second_moment_label", _mf_m2_label_name ); 

  clip = false;
  if (db->findBlock("Clip") )
    clip = true;
}

//---------------------------------------------------------------------------
//Method: Schedule Compute Property
//---------------------------------------------------------------------------
void NormScalarVariance::sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep )
{
  std::string taskname = "NormScalarVariance::computeProp"; 
  Task* tsk = scinew Task( taskname, this, &NormScalarVariance::computeProp, time_substep ); 
  
  _mf_label = 0; 
  _mf_m2_label = 0;
  _vf_label = 0;
  _mf_label    = VarLabel::find( _mf_label_name ); 
  _mf_m2_label = VarLabel::find( _mf_m2_label_name );
  _vf_label    = VarLabel::find( "volFraction" );

  if ( _mf_label == 0 )
    throw InvalidValue("Error: Cannot match mixture fraction name with label.",__FILE__, __LINE__);             

  if ( _mf_m2_label == 0 )
    throw InvalidValue("Error: Cannot match mixture fraction sencond moment name with label.",__FILE__, __LINE__);             

  tsk->modifies( _prop_label );
  if ( time_substep == 0 ){ 
    tsk->requires( Task::OldDW, _mf_label, Ghost::AroundCells, 1); 
    tsk->requires( Task::OldDW, _mf_m2_label, Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _vf_label, Ghost::AroundCells, 1); 
  } else { 
    tsk->requires( Task::NewDW, _mf_label, Ghost::AroundCells, 1 ); 
    tsk->requires( Task::NewDW, _mf_m2_label, Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _vf_label, Ghost::AroundCells, 1); 
  } 
  
  sched->addTask( tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ) ); 
}

//---------------------------------------------------------------------------
//Method: Actually Compute Property
//---------------------------------------------------------------------------
void NormScalarVariance::computeProp(const ProcessorGroup* pc, 
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
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
 
    CCVariable<double> prop; 
    CCVariable<double> mf_m2; 
    constCCVariable<double> mf; 
    constCCVariable<double> vf;
    
    if ( time_substep == 0 ) { 
      new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
      prop.initialize(0.0);
      old_dw->get( mf,       _mf_label, matlIndex, patch, Ghost::None, 0); 
      old_dw->getModifiable( mf_m2, _mf_m2_label, matlIndex, patch, Ghost::None, 0); 
      old_dw->get( vf,       _vf_label, matlIndex, patch, Ghost::None, 0);
    } else { 
      new_dw->getModifiable( prop, _prop_label, matlIndex, patch ); 
      new_dw->get( mf,       _mf_label, matlIndex, patch, Ghost::None, 0); 
      new_dw->getModifiable( mf_m2, _mf_m2_label, matlIndex, patch, Ghost::None, 0); 
      new_dw->get( vf,       _vf_label, matlIndex, patch, Ghost::None, 0);
    } 
      
    CellIterator iter = patch->getCellIterator(); 
    
    double small = 1.0e-10;
    double maxVar;
        
    for (iter.begin(); !iter.done(); iter++){
      IntVector c = *iter;
      
      if (vf[c] > 0.0) {
        maxVar = mf[c] * (1.0 - mf[c]);
        prop[c] = mf_m2[c] - mf[c]*mf[c];
        if (maxVar < 0.0)
          maxVar = 0.0;   //hard set this is 0, since some compliers 1 - 1*1 != 0
        
        if (prop[c] <= small) {
          if (clip ) {
            mf_m2[c] = mf[c]*mf[c];
          }
          prop[c] = 0.0;
        } else if (prop[c] > maxVar) {
          prop[c] = maxVar;
          if (clip) {
            mf_m2[c] = mf[c];
          } 
        }
        
        prop[c] = prop[c]/(maxVar + small);    
      } else {
        prop[c] = 0.0;
      }

    }
    
  }
}

//---------------------------------------------------------------------------
//Method: Scheduler for Initializing the Property
//---------------------------------------------------------------------------
void NormScalarVariance::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "NormScalarVariance::initialize"; 

  Task* tsk = scinew Task(taskname, this, &NormScalarVariance::initialize);
  tsk->computes(_prop_label); 
  
  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
}

//---------------------------------------------------------------------------
//Method: Actually Initialize the Property
//---------------------------------------------------------------------------
void NormScalarVariance::initialize( const ProcessorGroup* pc, 
                                     const PatchSubset* patches, 
                                     const MaterialSubset* matls, 
                                     DataWarehouse* old_dw, 
                                     DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    
    CCVariable<double> prop; 
    
    new_dw->allocateAndPut( prop, _prop_label, matlIndex, patch ); 
    prop.initialize(0.0); 
    
    PropertyModelBase::base_initialize( patch, prop ); // generic initialization functionality 
    
  }
}
