#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <iomanip>

using namespace std;
using namespace Uintah; 

DORadiation::DORadiation( std::string src_name, ArchesLabel* labels, MPMArchesLabel* MAlab,
                          BoundaryCondition* bc, 
                          vector<std::string> req_label_names, const ProcessorGroup* my_world, 
                          std::string type ) 
: SourceTermBase( src_name, labels->d_sharedState, req_label_names, type ), 
  _labels( labels ),
  _MAlab(MAlab), 
  _bc(bc), 
  _my_world(my_world)
{

  // NOTE: This boundary condition here is bogus.  Passing it for 
  // now until the boundary condition reference can be stripped out of 
  // the radiation model. 
  
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();

  _src_label = VarLabel::create( src_name, CC_double ); 

  // Add any other local variables here. 
  _radiationFluxELabel = VarLabel::create("radiationFluxE",  CC_double);
  _extra_local_labels.push_back(_radiationFluxELabel); 

  _radiationFluxWLabel = VarLabel::create("radiationFluxW",  CC_double);
  _extra_local_labels.push_back(_radiationFluxWLabel); 

  _radiationFluxNLabel = VarLabel::create("radiationFluxN",  CC_double);
  _extra_local_labels.push_back(_radiationFluxNLabel); 

  _radiationFluxSLabel = VarLabel::create("radiationFluxS",  CC_double);
  _extra_local_labels.push_back(_radiationFluxSLabel); 

  _radiationFluxTLabel = VarLabel::create("radiationFluxT",  CC_double);
  _extra_local_labels.push_back(_radiationFluxTLabel); 

  _radiationFluxBLabel = VarLabel::create("radiationFluxB",  CC_double);
  _extra_local_labels.push_back(_radiationFluxBLabel); 

  _radiationVolqLabel = VarLabel::create("radiationVolq",  CC_double);
  _extra_local_labels.push_back(_radiationVolqLabel); 

  //Declare the source type: 
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  _DO_model = 0; 

}

DORadiation::~DORadiation()
{
  
  // source label is destroyed in the base class 

  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++) { 

    VarLabel::destroy( *iter ); 

  }

  delete _DO_model; 

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
  _T_label_name = "radiation_temperature"; 

  if ( db->findBlock("abskg")){ 
    db->findBlock("abskg")->getAttribute("label", _abskg_label_name); 
  } else { 
    throw ProblemSetupException("Error: DO Radiation - The absorption coefficient is not defined.",__FILE__,__LINE__);
  }
  
  proc0cout << " --- DO Radiation Model Summary: --- " << endl;
  proc0cout << "   -> calculation frequency:     " << _radiation_calc_freq << endl;
  proc0cout << "   -> temperature label:         " << _T_label_name << endl;
  proc0cout << "   -> abskg label:               " << _abskg_label_name << endl;
  proc0cout << " --- end DO Radiation Summary ------ " << endl;

  _DO_model = scinew DORadiationModel( _labels, _MAlab, _bc, _my_world ); 
  _DO_model->problemSetup( db ); 

  for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
    ostringstream my_stringstream_object;
    my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix ;
    _IntensityLabels.push_back(  VarLabel::find(my_stringstream_object.str()));
    _extra_local_labels.push_back(_IntensityLabels[ix]); 
    if(_DO_model->DOSolveInitialGuessBool()==false){
     break;  // create labels for all intensities, otherwise only create 1 label
    }
  }

  if (_DO_model->ScatteringOnBool())
    _scatktLabel =  VarLabel::find("scatkt");
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
DORadiation::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DORadiation::computeSource";
  Task* tsk = scinew Task(taskname, this, &DORadiation::computeSource, timeSubStep);

  _T_label = VarLabel::find(_T_label_name); 
  if ( _T_label == 0){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the radiation temperature label.", __FILE__, __LINE__);
  }
  _abskg_label = VarLabel::find(_abskg_label_name); 
  if ( _abskg_label == 0){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the abskg label.", __FILE__, __LINE__);
  }

  _perproc_patches = level->eachPatch(); 

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;

  tsk->requires( Task::OldDW, _src_label, gn, 0 );
  
  if (timeSubStep == 0) { 

    tsk->computes(_src_label);
    tsk->requires( Task::NewDW, _T_label, gac, 1 ); 
    tsk->requires( Task::OldDW, _abskg_label, gn, 0 ); 

  if (_DO_model->ScatteringOnBool())
    tsk->requires( Task::OldDW, _scatktLabel, gn, 0 ); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
         iter != _extra_local_labels.end(); iter++){

      tsk->requires( Task::OldDW, *iter, gn, 0 ); 
      tsk->computes( *iter ); 

    }

  } else {

    tsk->modifies(_src_label); 
    tsk->requires( Task::NewDW, _T_label, gac, 1 ); 
    tsk->requires( Task::NewDW, _abskg_label, gn, 0 ); 

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

    int timestep = _labels->d_sharedState->getCurrentTopLevelTimeStep(); 
    bool do_radiation = false; 
    if ( timestep%_radiation_calc_freq == 0 ) { 
      if ( _all_rk ) { 
        do_radiation = true; 
      } else if ( timeSubStep == 0 && !_all_rk ) { 
        do_radiation = true; 
      } 
    } 

    if (do_radiation==false){
      if ( timeSubStep == 0 ) { 
        for(unsigned int ix=0; ix< _IntensityLabels.size() ;ix++){
          new_dw->transferFrom(old_dw,_IntensityLabels[ix],  patches, matls);
        }
      }
    }
    else{  
      if(_DO_model->DOSolveInitialGuessBool()==false){
        for(unsigned int ix=0;  ix< _IntensityLabels.size();ix++){ 
          new_dw->transferFrom(old_dw,_IntensityLabels[ix],  patches, matls);
        }
      }
    }


  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, _labels->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    CCVariable<double> divQ; 


    ArchesVariables radiation_vars; 
    ArchesConstVariables const_radiation_vars;

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None; 

    if ( timeSubStep == 0 ) { 

      new_dw->get( const_radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );

      new_dw->allocateAndPut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch );
      new_dw->allocateAndPut( divQ, _src_label, matlIndex, patch ); 

      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  
      radiation_vars.ESRCG.initialize(0.0); 

      // copy old solution into new
      old_dw->copyOut( divQ, _src_label, matlIndex, patch, gn, 0 ); 
      old_dw->copyOut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch , gn , 0 );
      old_dw->get( const_radiation_vars.ABSKG  , _abskg_label , matlIndex , patch , gn , 0 );

    } else { 

      new_dw->get( const_radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );

      radiation_vars.ESRCG.allocate( patch->getExtraCellLowIndex(1), patch->getExtraCellHighIndex(1) );  
      radiation_vars.ESRCG.initialize(0.0); 

      new_dw->getModifiable( radiation_vars.qfluxe , _radiationFluxELabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxw , _radiationFluxWLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxn , _radiationFluxNLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxs , _radiationFluxSLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxt , _radiationFluxTLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxb , _radiationFluxBLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.volq   , _radiationVolqLabel  , matlIndex , patch );
      new_dw->getModifiable( divQ, _src_label, matlIndex, patch ); 

      new_dw->get( const_radiation_vars.ABSKG  , _abskg_label, matlIndex , patch, gn, 0 );

    } 

    old_dw->get( const_radiation_vars.cellType , _labels->d_cellTypeLabel, matlIndex, patch, gac, 1 ); 

    if ( do_radiation ){ 


      if ( timeSubStep == 0 ) {

      if(_DO_model->DOSolveInitialGuessBool()){
        for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
          CCVariable<double> cenint;
          new_dw->allocateAndPut(cenint,_IntensityLabels[ix] , matlIndex, patch );
        }
       }

        //Note: The final divQ is initialized (to zero) and set after the solve in the intensity solve itself.
        _DO_model->intensitysolve( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars, divQ, BoundaryCondition::WALL, matlIndex, new_dw, old_dw ); 

      }
    }
  } // end patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
DORadiation::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DORadiation::initialize"; 
  Task* tsk = scinew Task(taskname, this, &DORadiation::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); 
       iter != _extra_local_labels.end(); iter++){

    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
DORadiation::initialize( const ProcessorGroup* pc, 
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

      CCVariable<double> temp_var; 
      new_dw->allocateAndPut(temp_var, *iter, matlIndex, patch ); 
      temp_var.initialize(0.0);
      
    }

  }
}

