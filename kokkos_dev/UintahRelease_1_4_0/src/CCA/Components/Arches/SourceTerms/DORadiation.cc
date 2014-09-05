#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>

using namespace std;
using namespace Uintah; 

DORadiation::DORadiation( std::string src_name, ArchesLabel* labels, BoundaryCondition* bc, 
                      vector<std::string> req_label_names, const ProcessorGroup* my_world ) 
: _labels( labels ), _bc(bc), _my_world(my_world), SourceTermBase( src_name, labels->d_sharedState, req_label_names )
{

  // NOTE: This boundary condition here is bogus.  Passing it for 
  // now until the boundary condition reference can be stripped out of 
  // the radiation model. 
  
  _label_sched_init = false; 
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();

  _src_label = VarLabel::create( src_name, CC_double ); 

  // Add any other local variables here. 
  _radiationSRCLabel = VarLabel::create("new_radiationSRC",  CC_double);
  _extra_local_labels.push_back(_radiationSRCLabel);  

  _radiationFluxELabel = VarLabel::create("new_radiationFluxE",  CC_double);
  _extra_local_labels.push_back(_radiationFluxELabel); 

  _radiationFluxWLabel = VarLabel::create("new_radiationFluxW",  CC_double);
  _extra_local_labels.push_back(_radiationFluxWLabel); 

  _radiationFluxNLabel = VarLabel::create("new_radiationFluxN",  CC_double);
  _extra_local_labels.push_back(_radiationFluxNLabel); 

  _radiationFluxSLabel = VarLabel::create("new_radiationFluxS",  CC_double);
  _extra_local_labels.push_back(_radiationFluxSLabel); 

  _radiationFluxTLabel = VarLabel::create("new_radiationFluxT",  CC_double);
  _extra_local_labels.push_back(_radiationFluxTLabel); 

  _radiationFluxBLabel = VarLabel::create("new_radiationFluxB",  CC_double);
  _extra_local_labels.push_back(_radiationFluxBLabel); 

  _radiationVolqLabel = VarLabel::create("new_radiationVolq",  CC_double);
  _extra_local_labels.push_back(_radiationVolqLabel); 

  _abskgLabel    =  VarLabel::create("new_abskg",    CC_double);
  _extra_local_labels.push_back(_abskgLabel); 

  _abskpLabel = VarLabel::create("new_abskp", CC_double); 
  _extra_local_labels.push_back(_abskpLabel); 

  //Declare the source type: 
  _source_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

}

DORadiation::~DORadiation()
{
  
  // source label is destroyed in the base class 

  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++) { 

    VarLabel::destroy( *iter ); 

  }

}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DORadiation::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault( "calc_frequency",   _radiation_calc_freq, 3 ); 
  db->getWithDefault( "calc_on_all_RKsteps", _all_rk, false ); 
  db->getWithDefault( "co2_label", _co2_label_name, "CO2" ); 
  db->getWithDefault( "h2o_label", _h2o_label_name, "H2O" ); 
  db->getWithDefault( "T_label", _T_label_name, "temperature" ); 

  _DO_model = scinew DORadiationModel( _bc, _my_world ); 
  _DO_model->problemSetup( db ); 

  _labels->add_species( _co2_label_name ); 
  _labels->add_species( _h2o_label_name ); 
  _labels->add_species( _T_label_name ); 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
DORadiation::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DORadiation::computeSource";
  Task* tsk = scinew Task(taskname, this, &DORadiation::computeSource, timeSubStep);

  _perproc_patches = level->eachPatch(); 

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;

  _co2_label = VarLabel::find( _co2_label_name ); 
  _h2o_label = VarLabel::find( _h2o_label_name ); 
  _T_label   = VarLabel::find( _T_label_name ); 

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;

    tsk->computes(_src_label);

    tsk->requires( Task::OldDW, _co2_label, gac, 1 ); 
    tsk->requires( Task::OldDW, _h2o_label, gac, 1 ); 
    tsk->requires( Task::OldDW, _T_label, gac, 1 ); 
    tsk->requires( Task::OldDW, _labels->d_sootFVINLabel, gac, 1 ); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      tsk->requires( Task::OldDW, *iter, gn, 0 ); 
      tsk->computes( *iter ); 

    }

  } else {

    tsk->modifies(_src_label); 

    tsk->requires( Task::NewDW, _co2_label, gac, 1 ); 
    tsk->requires( Task::NewDW, _h2o_label, gac, 1 ); 
    tsk->requires( Task::NewDW, _T_label, gac, 1 ); 
    tsk->requires( Task::NewDW, _labels->d_sootFVINLabel, gac, 1); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      tsk->modifies( *iter ); 

    }
  }

  tsk->requires(Task::OldDW, _labels->d_cellTypeLabel, gac, 1 ); 
  tsk->requires(Task::NewDW, _labels->d_cellInfoLabel, gn);

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
DORadiation::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{

  _DO_model->d_linearSolver->matrixCreate( _perproc_patches, patches ); 

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Ghost::GhostType  gac = Ghost::AroundCells;

    int timestep = _labels->d_sharedState->getCurrentTopLevelTimeStep(); 

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, _labels->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    CCVariable<double> divQ; 

    bool do_radiation = false; 
    if ( timestep%_radiation_calc_freq == 0 ) { 
      if ( _all_rk ) { 
        do_radiation = true; 
      } else if ( timeSubStep == 0 && !_all_rk ) { 
        do_radiation = true; 
      } 
    } 

    ArchesVariables radiation_vars; 
    ArchesConstVariables const_radiation_vars; 

    if ( timeSubStep == 0 ) { 

      old_dw->get( const_radiation_vars.co2       , _co2_label               , matlIndex , patch , gac , 1 );
      old_dw->get( const_radiation_vars.h2o       , _h2o_label               , matlIndex , patch , gac , 1 );
      old_dw->get( const_radiation_vars.sootFV    , _labels->d_sootFVINLabel        , matlIndex , patch , gac , 1 ); 
      old_dw->getCopy( radiation_vars.temperature , _T_label                 , matlIndex , patch , gac , 1 );
      old_dw->get( const_radiation_vars.cellType  , _labels->d_cellTypeLabel , matlIndex , patch , gac , 1 );

      new_dw->allocateAndPut( radiation_vars.qfluxe , _radiationFluxELabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxw , _radiationFluxWLabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxn , _radiationFluxNLabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxs , _radiationFluxSLabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxt , _radiationFluxTLabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxb , _radiationFluxBLabel , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.volq   , _radiationVolqLabel  , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.src    , _radiationSRCLabel   , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.ABSKG  , _abskgLabel          , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.ABSKP  , _abskpLabel          , matlIndex , patch );

      new_dw->allocateAndPut( divQ, _src_label, matlIndex, patch ); 
      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  

      radiation_vars.src.initialize(0.0);
      radiation_vars.qfluxe.initialize(0.0);
      radiation_vars.qfluxw.initialize(0.0);
      radiation_vars.qfluxn.initialize(0.0);
      radiation_vars.qfluxs.initialize(0.0);
      radiation_vars.qfluxt.initialize(0.0);
      radiation_vars.qfluxb.initialize(0.0);
      radiation_vars.ABSKG.initialize(0.0);
      radiation_vars.ABSKP.initialize(0.0); 
      radiation_vars.ESRCG.initialize(0.0);
      divQ.initialize(0.0); 

    } else { 

      new_dw->get( const_radiation_vars.co2, _co2_label, matlIndex, patch, gac, 1 ); 
      new_dw->get( const_radiation_vars.h2o, _h2o_label, matlIndex, patch, gac, 1 ); 
      new_dw->getCopy( radiation_vars.temperature, _T_label, matlIndex, patch, gac, 1 ); 
      new_dw->get( const_radiation_vars.sootFV, _labels->d_sootFVINLabel, matlIndex, patch, gac, 1 ); 
      old_dw->get( const_radiation_vars.cellType          , _labels->d_cellTypeLabel, matlIndex, patch  , gac ,  1 ); 

      new_dw->getModifiable( radiation_vars.qfluxe , _radiationFluxELabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxw , _radiationFluxWLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxn , _radiationFluxNLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxs , _radiationFluxSLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxt , _radiationFluxTLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxb , _radiationFluxBLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.volq   , _radiationVolqLabel  , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.src    , _radiationSRCLabel   , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.ABSKG  , _abskgLabel          , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.ABSKP  , _abskpLabel          , matlIndex , patch );

      new_dw->getModifiable( divQ, _src_label, matlIndex, patch ); 
      divQ.initialize(0.0); 

      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  

    } 

    if ( do_radiation ){ 


      if ( timeSubStep == 0 ) {

        _DO_model->computeRadiationProps( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars ); 

        _DO_model->boundarycondition( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars ); 

        _DO_model->intensitysolve( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars, BoundaryCondition::WALL ); 

      }

      for ( CellIterator iter = patch->getCellIterator(); !iter.done(); iter++ ){ 

        IntVector c = *iter;
        divQ[c] = radiation_vars.src[c]; 

      }

    } 
  } // end patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
DORadiation::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DORadiation::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &DORadiation::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++){

    tsk->computes(*iter); 

  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
DORadiation::dummyInit( const ProcessorGroup* pc, 
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

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}

