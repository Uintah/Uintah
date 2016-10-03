#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>
#include <ostream>
#include <fstream>
#include <stdlib.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName):
d_fieldLabels(fieldLabels), d_timeIntegrator(timeIntegrator), d_eqnName(eqnName),
b_stepUsesCellLocation(false), b_stepUsesPhysicalLocation(false),
d_constant_init(0.0), d_step_dir("x"), d_step_start(0.0), d_step_end(0.0), d_step_cellstart(0), d_step_cellend(0), d_step_value(0.0),
d_use_constant_D(false)
{
  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels->d_sharedState->getArchesMaterial(0)->getDWIndex() );
  d_disc = scinew Discretization_new();
  _using_new_intrusion = false;
  _table_init = false;
  _stage = 1;  //uses density after first table lookup

  d_X_flux_label = VarLabel::create( d_eqnName+"_fluxX", SFCXVariable<double>::getTypeDescription());
  d_Y_flux_label = VarLabel::create( d_eqnName+"_fluxY", SFCYVariable<double>::getTypeDescription());
  d_Z_flux_label = VarLabel::create( d_eqnName+"_fluxZ", SFCZVariable<double>::getTypeDescription());

  d_X_psi_label = VarLabel::create( d_eqnName+"_psiX", SFCXVariable<double>::getTypeDescription());
  d_Y_psi_label = VarLabel::create( d_eqnName+"_psiY", SFCYVariable<double>::getTypeDescription());
  d_Z_psi_label = VarLabel::create( d_eqnName+"_psiZ", SFCZVariable<double>::getTypeDescription());

}

EqnBase::~EqnBase()
{
  delete(d_boundaryCond);
  delete(d_disc);
  VarLabel::destroy(d_X_flux_label);
  VarLabel::destroy(d_Y_flux_label);
  VarLabel::destroy(d_Z_flux_label);
  VarLabel::destroy(d_X_psi_label);
  VarLabel::destroy(d_Y_psi_label);
  VarLabel::destroy(d_Z_psi_label);
}

void
EqnBase::extraProblemSetup( ProblemSpecP& db ){

  d_boundaryCond->setupTabulatedBC( db, d_eqnName, _table );

}

void
EqnBase::commonProblemSetup( ProblemSpecP& db ){

  // Clipping:
  // defaults:
  clip.activated = false;
  clip.do_low  = false;
  clip.do_high = false;
  clip.my_type = ClipInfo::STANDARD;

  ProblemSpecP db_clipping = db->findBlock("Clipping");

  if (db_clipping) {

    std::string type = "default";
    std::fstream inputFile;
    std::string clip_dep_file;
    std::string clip_dep_low_file;
    std::string clip_ind_file;

    if ( db_clipping->getAttribute( "type", type )){

      db_clipping->getAttribute( "type", type );
      if ( type == "variable_constrained" ){

        clip.my_type = ClipInfo::CONSTRAINED;
        db_clipping->findBlock("constraint")->getAttribute("label", clip.ind_var );

        db_clipping->require("clip_dep_file",clip_dep_file);
        db_clipping->require("clip_dep_low_file",clip_dep_low_file);
        db_clipping->require("clip_ind_file",clip_ind_file);

        // read clipping file
        double number;
        inputFile.open(clip_dep_file.c_str());
        while (inputFile >> number)
        {
          clip_dep_vec.push_back(number);
        }
        inputFile.close();

        inputFile.open(clip_dep_low_file.c_str());
        while (inputFile >> number)
        {
          clip_dep_low_vec.push_back(number);
        }
        inputFile.close();


        // read clipping file
        inputFile.open(clip_ind_file.c_str());
        while (inputFile >> number)
        {
          clip_ind_vec.push_back(number);
        }
        inputFile.close();

      }

    } else {

      clip.activated = true;

      db_clipping->getWithDefault("low", clip.low,  -1.e16);
      db_clipping->getWithDefault("high",clip.high, 1.e16);
      db_clipping->getWithDefault("tol", clip.tol, 1e-10);

      if ( db_clipping->findBlock("low") )
        clip.do_low = true;

      if ( db_clipping->findBlock("high") )
        clip.do_high = true;

      if ( !clip.do_low && !clip.do_high )
        throw InvalidValue("Error: A low or high clipping must be specified if the <Clipping> section is activated.", __FILE__, __LINE__);

    }

  }

  // Initialization:
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {

    db_initialValue->getAttribute("type", d_initFunction);

    if (d_initFunction == "constant") {
      db_initialValue->require("constant", d_constant_init);

    } else if (d_initFunction == "step") {
      db_initialValue->require("step_direction", d_step_dir);
      db_initialValue->require("step_value", d_step_value);

      if( db_initialValue->findBlock("step_start") ) {
        b_stepUsesPhysicalLocation = true;
        db_initialValue->require("step_start", d_step_start);
        db_initialValue->require("step_end"  , d_step_end);

      } else if ( db_initialValue->findBlock("step_cellstart") ) {
        b_stepUsesCellLocation = true;
        db_initialValue->require("step_cellstart", d_step_cellstart);
        db_initialValue->require("step_cellend", d_step_cellend);
      }

    } else if (d_initFunction == "mms1") {
      //currently nothing to do here.
    } else if (d_initFunction == "geometry_fill") {

      db_initialValue->require("constant_inside", d_constant_in_init);              //fill inside geometry
      db_initialValue->getWithDefault( "constant_outside",d_constant_out_init,0.0); //fill outside geometry

      ProblemSpecP the_geometry = db_initialValue->findBlock("geom_object");
      if (the_geometry) {
        GeometryPieceFactory::create(the_geometry, d_initGeom);
      } else {
        throw ProblemSetupException("You are missing the geometry specification (<geom_object>) for the transport eqn. initialization!", __FILE__, __LINE__);
      }
    } else if ( d_initFunction == "gaussian" ) {

      db_initialValue->require( "amplitude", d_a_gauss );
      db_initialValue->require( "center", d_b_gauss );
      db_initialValue->require( "std", d_c_gauss );
      std::string direction;
      db_initialValue->require( "direction", direction );
      if ( direction == "X" || direction == "x" ){
        d_dir_gauss = 0;
      } else if ( direction == "Y" || direction == "y" ){
        d_dir_gauss = 1;
      } else if ( direction == "Z" || direction == "z" ){
        d_dir_gauss = 2;
      }
      db_initialValue->getWithDefault( "shift", d_shift_gauss, 0.0 );

    } else if ( d_initFunction == "tabulated" ){

      db_initialValue->require( "depend_varname", d_init_dp_varname );
      _table_init = true;

    } else if ( d_initFunction == "shunn_moin"){

      ProblemSpecP db_mom = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("MomentumSolver");
      string init_type;
      db_mom->findBlock("initialization")->getAttribute("type",init_type);
      if ( init_type != "shunn_moin"){
        throw InvalidValue("Error: Trying to initialize the Shunn/Moin MMS for the mixture fraction and not matching same IC in momentum", __FILE__,__LINE__);
      }
      db_mom->findBlock("initialization")->require("k",d_k);
      db_mom->findBlock("initialization")->require("w",d_w);
      std::string plane;
      db_mom->findBlock("initialization")->require("plane",plane);

      ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow");

      db_prop->findBlock("stream_0")->getAttribute("density",d_rho0);
      db_prop->findBlock("stream_1")->getAttribute("density",d_rho1);

      if ( plane == "x-y"){
        d_dir0=0;
        d_dir1=1;
      } else if ( plane == "y-z" ){
        d_dir0=1;
        d_dir1=2;
      } else if ( plane == "z-x" ){
        d_dir0=2;
        d_dir1=0;
      } else {
        throw InvalidValue("Error: Plane not recognized. Please choose xy, yz, or xz",__FILE__,__LINE__);
      }

    }
  }

  // Molecular diffusivity:
  d_use_constant_D = false;
  if ( db->findBlock( "D_mol" ) ){
    db->findBlock("D_mol")->getAttribute("label", d_mol_D_label_name);
  } else if ( db->findBlock( "D_mol_constant" ) ){
    db->findBlock("D_mol_constant")->getAttribute("value", d_mol_diff );
    d_use_constant_D = true;
  } else {
    d_use_constant_D = true;
    d_mol_diff = 0.0;
    proc0cout << "NOTICE: For equation " << d_eqnName << " no molecular diffusivity was specified.  Assuming D=0.0. \n";
  }

  if ( db->findBlock( "D_mol" ) && db->findBlock( "D_mol_constant" ) ){
    string err_msg = "ERROR: For transport equation: "+d_eqnName+" \n Molecular diffusivity is over specified. \n";
    throw ProblemSetupException(err_msg, __FILE__, __LINE__);
  }
}

void
EqnBase::sched_checkBCs( const LevelP& level, SchedulerP& sched, bool isRegrid = 0 )
{
  string taskname = "EqnBase::checkBCs";
  Task* tsk = scinew Task(taskname, this, &EqnBase::checkBCs);

  // These dependencies may be needed for BoundaryCondition::sched_setupBCInletVelocities
  // to be executed first, order appears to be ambigous in certain circumstances.
  if(isRegrid==0){ // dependencies are here to ensure task occurs after bcs are set
    tsk->requires( Task::NewDW,VarLabel::find("densityCP") , Ghost::None, 0 );
    tsk->requires( Task::NewDW,VarLabel::find("volFraction") , Ghost::AroundCells, 0 );
  }

  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() );
}

void
EqnBase::checkBCs( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    d_boundaryCond->checkBCs( patch, d_eqnName, matlIndex );

  }
}

void
EqnBase::sched_tableInitialization( const LevelP& level, SchedulerP& sched )
{

  std::string taskname = "EqnBase::tableInitialization";
  Task* tsk = scinew Task(taskname, this, &EqnBase::tableInitialization);

  MixingRxnModel::VarMap ivVarMap = _table->getIVVars();

  // independent variables :: these must have been computed previously
  for ( MixingRxnModel::VarMap::iterator i = ivVarMap.begin(); i != ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, Ghost::None, 0 );

  }

  // for inert mixing
  MixingRxnModel::InertMasterMap inertMap = _table->getInertMap();
  for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){
    const VarLabel* label = VarLabel::find( iter->first );
    tsk->requires( Task::NewDW, label, Ghost::None, 0 );
  }

  tsk->modifies( d_transportVarLabel );

  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() );


}

void
EqnBase::tableInitialization(const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage;
    MixingRxnModel::VarMap ivVarMap = _table->getIVVars();
    std::vector<string> allIndepVarNames = _table->getAllIndepVars();

    for ( int i = 0; i < (int) allIndepVarNames.size(); i++ ){

      MixingRxnModel::VarMap::iterator ivar = ivVarMap.find( allIndepVarNames[i] );

      constCCVariable<double> the_var;
      new_dw->get( the_var, ivar->second, matlIndex, patch, Ghost::None, 0 );
      indep_storage.push_back( the_var );

    }

    MixingRxnModel::InertMasterMap inertMap = _table->getInertMap();
    MixingRxnModel::StringToCCVar inert_mixture_fractions;
    inert_mixture_fractions.clear();
    for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){
      const VarLabel* label = VarLabel::find( iter->first );
      constCCVariable<double> variable;
      new_dw->get( variable, label, matlIndex, patch, Ghost::None, 0 );
      MixingRxnModel::ConstVarContainer container;
      container.var = variable;

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) );

    }

    CCVariable<double> eqn_var;
    new_dw->getModifiable( eqn_var, d_transportVarLabel, matlIndex, patch );

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter;

      std::vector<double> iv;
      for (std::vector<constCCVariable<double> >::iterator iv_iter = indep_storage.begin();
          iv_iter != indep_storage.end(); iv_iter++ ){

        iv.push_back( (*iv_iter)[c] );

      }

      eqn_var[c] = _table->getTableValue( iv, d_init_dp_varname, inert_mixture_fractions, c );

    }

    //recompute the BCs
    computeBCsSpecial( patch, d_eqnName, eqn_var );

  }
}
