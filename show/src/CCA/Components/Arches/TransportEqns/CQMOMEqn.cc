#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/ParticleModels/CQMOMSourceWrapper.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
CQMOMEqnBuilder::CQMOMEqnBuilder( ArchesLabel* fieldLabels,
                                 ExplicitTimeInt* timeIntegrator,
                                 string eqnName ) :
CQMOMEqnBuilderBase( fieldLabels, timeIntegrator, eqnName )
{}
CQMOMEqnBuilder::~CQMOMEqnBuilder(){}

EqnBase*
CQMOMEqnBuilder::build(){
  return scinew CQMOMEqn(d_fieldLabels, d_timeIntegrator, d_eqnName);
}
// End Builder
//---------------------------------------------------------------------------

CQMOMEqn::CQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
:
EqnBase( fieldLabels, timeIntegrator, eqnName )
{
  
  string varname = eqnName+"_Fdiff";
  d_FdiffLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_Fconv";
  d_FconvLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_RHS";
  d_RHSLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_old";
  d_oldtransportVarLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName;
  d_transportVarLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = eqnName+"_src";
  d_sourceLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_tmp";
  d_tempLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_FconvX";
  d_FconvXLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_FconvY";
  d_FconvYLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  varname = eqnName+"_FconvZ";
  d_FconvZLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());
  
  uVelIndex = -1; vVelIndex = -1; wVelIndex = -1;
  d_cqmomConv = scinew Convection_CQMOM();
  
}

CQMOMEqn::~CQMOMEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel);
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_sourceLabel);
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
  VarLabel::destroy(d_tempLabel);
  VarLabel::destroy(d_FconvXLabel);
  VarLabel::destroy(d_FconvYLabel);
  VarLabel::destroy(d_FconvZLabel);
  delete d_cqmomConv;
  for (unsigned int i = 0; i < d_sources.size(); i++ ) {
    delete d_sources[i];
  }
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CQMOMEqn::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;
  
  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP cqmom_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CQMOM");
  ProblemSpecP models_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");
  std::string which_cqmom;
  cqmom_db->getAttribute( "type", which_cqmom );
  cqmom_db->getAttribute( "partvel", d_usePartVel );
  if ( which_cqmom == "normalized" )
    d_normalized= true;
  else
    d_normalized = false;
  
  db->get("m", momentIndex);
  cqmom_db->getWithDefault("turbulentPrandtlNumber",d_turbPrNo,0.4);
  // Discretization information:
  cqmom_db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  cqmom_db->getWithDefault( "doConv", d_doConv, false);
  cqmom_db->getWithDefault( "doDiff", d_doDiff, false);

  d_addSources = false;
  cqmom_db->getWithDefault( "molecular_diffusivity", d_mol_diff, 0.0);
  
  // save the moment eqn label
  CQMOMEqnFactory& cqmomFactory = CQMOMEqnFactory::self();
  cqmom_db->get("NumberInternalCoordinates",M);
  cqmom_db->get("QuadratureNodes",N_i);           //get vector of quad nodes per internal coordiante
  cqmom_db->getWithDefault("RestitutionCoefficient",epW,1.0);
  if (epW > 1.0 )
    epW = 1.0;
  if (epW < 0.0 )
    epW = 1.0e-10;
  
  cqmom_db->getWithDefault("ConvectionWeightLimit",d_convWeightLimit, 1.0e-10);
  
  nNodes = 1;
  for (unsigned int i = 0; i<N_i.size(); i++) {
    nNodes *= N_i[i];
  }
  
  //get internal coordinate indexes for each direction
  int m = 0;
  for ( ProblemSpecP db_name = cqmom_db->findBlock("InternalCoordinate");
       db_name != 0; db_name = db_name->findNextBlock("InternalCoordinate") ) {
    string varType;
    db_name->getAttribute("type",varType);
    if (varType == "uVel") {
      uVelIndex = m;
    } else if (varType == "vVel") {
      vVelIndex = m;
    } else if (varType == "wVel") {
      wVelIndex = m;
    }
    m++;
  }
  
  string name = "m_";
  for (int i = 0; i<M ; i++) {
    string node;
    std::stringstream out;
    out << momentIndex[i];
    node = out.str();
    name += node;
  }
  proc0cout << "Problem setup for " << name << endl;
  EqnBase& temp_eqn = cqmomFactory.retrieve_scalar_eqn(name);
  CQMOMEqn& eqn = dynamic_cast<CQMOMEqn&>(temp_eqn);
  d_momentLabel = eqn.getTransportEqnLabel();
  
  d_w_small = eqn.getSmallClip();
  if( d_w_small == 0.0 ) {
    d_w_small = 1e-16;
  }
  
  // Models (source terms):
  if ( models_db ) {
    d_addSources = true;
    for (ProblemSpecP m_db = models_db->findBlock("model"); m_db !=0; m_db = m_db->findNextBlock("model")){
      //parse the model blocks for var label
      std::string model_name;
      std::string source_label;
      int nIC = 0;
      std::string ic_name;
      m_db->get("IC",ic_name);
      m = 0;
      
      for ( ProblemSpecP db_name = cqmom_db->findBlock("InternalCoordinate");
           db_name != 0; db_name = db_name->findNextBlock("InternalCoordinate") ) {
        std::string var_name;
        db_name->getAttribute("name",var_name);
        if ( var_name == ic_name) {
          nIC = m;
          break;
        }
        m++;
        if ( m >= M ) { // occurs if ic not found
          string err_msg = "Error: could not find internal coordiante '" + ic_name + "' in list of internal coordinates specified by CQMOM spec";
          throw ProblemSetupException(err_msg,__FILE__,__LINE__);
        }
      }
      
      m_db->getAttribute("label",model_name);
      source_label = d_eqnName + "_" + model_name + "_src";
      proc0cout << "Creating source label: " << source_label << endl;
      CQMOMSourceWrapper * cqmomSource;
      cqmomSource =  scinew CQMOMSourceWrapper(d_fieldLabels, source_label, momentIndex, nIC) ;
      cqmomSource->problemSetup( m_db );
      d_sources.push_back(cqmomSource);
    }
  }
  // Clipping:
  clip.activated = false;
  clip.do_low  = false;
  clip.do_high = false;
  
  ProblemSpecP db_clipping = db->findBlock("Clipping");
  if (db_clipping) {
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
  
  // Scaling information:
  //Use this if the moments should be normalized later
  //  cqmom_db->getWithDefault( "scaling_const", d_scalingConstant, 0.0 );

  // Initialization (new way):
  ProblemSpecP db_initialValue = db->findBlock("initialization");
  if (db_initialValue) {
    
    db_initialValue->getAttribute("type", d_initFunction);
    
    // ---------- Constant initialization function ------------------------
    if (d_initFunction == "constant") {
      
      // Constant: normalize if specified
      db_initialValue->get("constant", d_constant_init);
//      if( d_scalingConstant != 0.0 )
//        d_constant_init /= d_scalingConstant;
      
    } else if (d_initFunction == "step" ) {
      
      // Step functions: get step direction
      db_initialValue->require("step_direction", d_step_dir);
      
      // Step functions: find start/stop location
      if( db_initialValue->findBlock("step_start") ) {
        b_stepUsesPhysicalLocation = true;
        db_initialValue->require("step_start", d_step_start);
        db_initialValue->require("step_end"  , d_step_end);
      } else if ( db_initialValue->findBlock("step_cellstart") ) {
        b_stepUsesCellLocation = true;
        db_initialValue->require("step_cellstart", d_step_cellstart);
        db_initialValue->require("step_cellend", d_step_cellend);
        // swap if out of order
        if(d_step_cellstart > d_step_cellend) {
          int temp = d_step_cellstart;
          d_step_cellstart = d_step_cellend;
          d_step_cellend = temp;
        }
      } else {
        string err_msg = "ERROR: Arches: DQMOMEqn: Could not initialize 'env_step' for equation "+d_eqnName+": You did not specify a starting or stopping point!  Add <step_cellstart> and <step_cellend>, or <step_cellstart> and <step_cellend>! \n";
        throw ProblemSetupException(err_msg,__FILE__,__LINE__);
      }//end start/stop init.
      
      // Step functions: get step values
      if (d_initFunction == "step") {
        db_initialValue->require("step_value", d_step_value);
//        d_step_value /= d_scalingConstant;
      }
      
    } else if (d_initFunction == "mms1") {
      //currently nothing to do here.
      //placeholder for now
    } else {
      //if no initialization is set, intialize to 0 everywhere
      //might be shortest way(inputfile wise) to specify large problems
      d_initFunction = "constant";
      d_constant_init = 0.0;
    }
  } else {
    //if no initialization is set, intialize to 0 everywhere
    //might be shortest way(inputfile wise) to specify large problems
    d_initFunction = "constant";
    d_constant_init = 0.0;
  }
  
}

//---------------------------------------------------------------------------
// Method: Schedule clean up.
// Probably not needed for DQMOM
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_evalTransportEqn( const LevelP& level,
                                 SchedulerP& sched, int timeSubStep )
{
  
  if (timeSubStep == 0)
    sched_initializeVariables( level, sched );

  if (d_addSources)
    sched_computeSources( level, sched, timeSubStep );
  
  sched_buildTransportEqn( level, sched, timeSubStep );
  
  sched_solveTransportEqn( level, sched, timeSubStep );
  
}
//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CQMOMEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;
  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); // for rk sub stepping
  tsk->computes(d_RHSLabel);
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);
  tsk->computes(d_tempLabel);
  
  tsk->computes(d_FconvXLabel);
  tsk->computes(d_FconvYLabel);
  tsk->computes(d_FconvZLabel);
  
  for (std::vector<CQMOMSourceWrapper* >::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
    (*iter)->sched_initializeVariables( level, sched );
  }
  
  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually initialize the variables.
//---------------------------------------------------------------------------
void CQMOMEqn::initializeVariables( const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    CCVariable<double> newVar;
    CCVariable<double> rkoldVar;
    constCCVariable<double> oldVar;
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( rkoldVar, d_oldtransportVarLabel, matlIndex, patch );
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);
    
    newVar.initialize(  0.0);
    rkoldVar.initialize(0.0);
    
    // copy old into new
    newVar.copyData(oldVar);
    rkoldVar.copyData(oldVar);
    
    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;
    CCVariable<double> phiTemp;
    CCVariable<double> FconvX;
    CCVariable<double> FconvY;
    CCVariable<double> FconvZ;
    
    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch );
    new_dw->allocateAndPut( phiTemp, d_tempLabel, matlIndex, patch);
    new_dw->allocateAndPut( FconvX, d_FconvXLabel, matlIndex, patch );
    new_dw->allocateAndPut( FconvY, d_FconvYLabel, matlIndex, patch );
    new_dw->allocateAndPut( FconvZ, d_FconvZLabel, matlIndex, patch );
    
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);
    RHS.initialize(0.0);
    phiTemp.initialize(0.0);
    FconvX.initialize(0.0);
    FconvY.initialize(0.0);
    FconvZ.initialize(0.0);
    
    curr_time = d_fieldLabels->d_sharedState->getElapsedTime();
    curr_ssp_time = curr_time;
  }
}
//---------------------------------------------------------------------------
// Method: Schedule compute the sources.
// this calls the scedulers to add up the source term from each node
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::computeSources";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::computeSources);
  
  // This scheduler only calls other schedulers
  for (vector<CQMOMSourceWrapper* >::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
    const VarLabel * temp_src = (*iter)->getSourceLabel( );
    tsk->requires(Task::NewDW, temp_src, Ghost::None, 0);
    (*iter)->sched_buildSourceTerm( level, sched, timeSubStep );
  }
  
  if (timeSubStep == 0) {
    tsk->computes(d_sourceLabel);
  } else {
    tsk->modifies(d_sourceLabel);
  }
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually compute the sources
//---------------------------------------------------------------------------
void
CQMOMEqn::computeSources( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++) {
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> totalSource;
    if (new_dw->exists(d_sourceLabel, matlIndex, patch)) {
      new_dw->getModifiable( totalSource, d_sourceLabel, matlIndex, patch);
    } else {
      new_dw->allocateAndPut( totalSource, d_sourceLabel, matlIndex, patch);
    }
    
    totalSource.initialize(0.0);
    
    for (vector<CQMOMSourceWrapper* >::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
      constCCVariable<double> thisSrc;
      const VarLabel* temp_src = (*iter)->getSourceLabel( );
      new_dw->get(thisSrc, temp_src, matlIndex, patch, gn, 0);
    
      for (CellIterator citer=patch->getCellIterator(); !citer.done(); citer++){
        IntVector c = *citer;
        totalSource[c] += thisSrc[c];
      } //cell loop
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildTransportEqn";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildTransportEqn);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  tsk->modifies(d_tempLabel);
  
  tsk->modifies(d_FconvXLabel);
  tsk->modifies(d_FconvYLabel);
  tsk->modifies(d_FconvZLabel);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cellTypeLabel, Ghost::AroundCells, 1);
  
  if (d_addSources) {
    tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0);
  }
  
  //loop over requires for weights and abscissas needed for convection term if IC=u,v,w
  if (d_usePartVel) {
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      if (timeSubStep == 0 ) {
        tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 2 );
      } else {
        tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
      }
    }
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      if (timeSubStep == 0 ) {
        tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 2 );
      } else {
        tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
      }
    }
  } else {
    tsk->requires(Task::OldDW, d_fieldLabels->d_uVelocitySPBCLabel, Ghost::AroundCells, 1);
#ifdef YDIM
    tsk->requires(Task::OldDW, d_fieldLabels->d_vVelocitySPBCLabel, Ghost::AroundCells, 1);
#endif
#ifdef ZDIM
    tsk->requires(Task::OldDW, d_fieldLabels->d_wVelocitySPBCLabel, Ghost::AroundCells, 1);
#endif
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildTransportEqn( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    Vector Dx = patch->dCell();
    
    constCCVariable<double> oldPhi;
    constCCVariable<double> mu_t;
    constCCVariable<int> cellType;
    constSFCXVariable<double> uVel;
    constSFCYVariable<double> vVel;
    constSFCZVariable<double> wVel;
    constCCVariable<double> src; //summed up source
    constCCVariable<Vector> areaFraction;
    
    CCVariable<double> phi;
    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;
    
    CCVariable<double> phiTemp;
    CCVariable<double> FconvX;
    CCVariable<double> FconvY;
    CCVariable<double> FconvZ;
    
    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    if (d_addSources) {
      new_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0);
    }
    
    old_dw->get(mu_t, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1);
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 2);
    old_dw->get(cellType, d_fieldLabels->d_cellTypeLabel, matlIndex, patch, gac, 1);
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch);
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    new_dw->getModifiable(phiTemp, d_tempLabel, matlIndex, patch);
    RHS.initialize(0.0);
    Fconv.initialize(0.0);
    Fdiff.initialize(0.0);
    
    new_dw->getModifiable(FconvX, d_FconvXLabel, matlIndex, patch);
    new_dw->getModifiable(FconvY, d_FconvYLabel, matlIndex, patch);
    new_dw->getModifiable(FconvZ, d_FconvZLabel, matlIndex, patch);
    FconvX.initialize(0.0);
    FconvY.initialize(0.0);
    FconvZ.initialize(0.0);
    
    computeBCs( patch, d_eqnName, phi );
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      phiTemp[c] = oldPhi[c];
    } //cell loop
    
    double vol = Dx.x();
#ifdef YDIM
    vol *= Dx.y();
#endif
#ifdef ZDIM
    vol *= Dx.z();
#endif
    
    vector<constCCVarWrapper> cqmomWeights;
    vector<constCCVarWrapper> cqmomAbscissas;

    //----CONVECTION
    if (d_doConv) {
      //Use the Convection_CQMOM class if using particle velocity, if not use normal convection class
      if (d_usePartVel) {
        //get weights and abscissas from dw
        for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
          const VarLabel* tempLabel = iW->second;
          constCCVarWrapper tempWrapper;
          if (new_dw->exists( tempLabel, matlIndex, patch) ) {
            new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
          } else {
            old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
          }
          cqmomWeights.push_back(tempWrapper);
        }
        
        for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
          const VarLabel* tempLabel = iA->second;
          constCCVarWrapper tempWrapper;
          if (new_dw->exists( tempLabel, matlIndex, patch) ) {
            new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
          } else {
            old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
          }
          cqmomAbscissas.push_back(tempWrapper);
        }
      } else if (!d_usePartVel) {
        old_dw->get(uVel,   d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1);
#ifdef YDIM
        old_dw->get(vVel,   d_fieldLabels->d_vVelocitySPBCLabel, matlIndex, patch, gac, 1);
#endif
#ifdef ZDIM
        old_dw->get(wVel,   d_fieldLabels->d_wVelocitySPBCLabel, matlIndex, patch, gac, 1);
#endif
        d_disc->computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, areaFraction, d_convScheme );
      }
      
      // look for and add contribution from intrusions.
      if ( _using_new_intrusion ) {
        _intrusions->addScalarRHS( patch, Dx, d_eqnName, RHS );
      }
    }
    
#ifdef cqmom_transport_dbg
    std::cout << "Transport of " << d_eqnName << std::endl;
    std::cout << "===========================" << std::endl;
#endif
    
    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, oldPhi, mu_t, d_mol_diff, areaFraction, d_turbPrNo );

    if (!d_usePartVel) {
      //----SUM UP RHS
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        RHS[c] += Fdiff[c] - Fconv[c];
      
        if (d_addSources)
          RHS[c] += src[c]*vol;
      } //cell loop
  
    } else {
      if (d_doConv) {
        //NOTE: this applys each convective direction to a temporary field, then calcuates the RHS as the
        //difference between the temp field and oldphi.  However this is not doing the full operator splitting, as
        //that requires a new CQMOM inversion step in between.  Leaving this how it is to test it out, but using the full operator
        //splitting procedure will require putting each convective flux calculation to its own scheduler task
        if (uVelIndex > -1) {
          d_cqmomConv->doConvX( patch, FconvX, d_convScheme, d_convWeightLimit, cqmomWeights,
                               cqmomAbscissas, M, nNodes, uVelIndex, momentIndex, cellType, epW );
          for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            phiTemp[c] += -FconvX[c];
          } //cell loop
        }
#ifdef YDIM
        if (vVelIndex > -1) {
          d_cqmomConv->doConvY( patch, FconvY, d_convScheme, d_convWeightLimit, cqmomWeights,
                               cqmomAbscissas, M, nNodes, vVelIndex, momentIndex, cellType, epW );
          for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            phiTemp[c] += -FconvY[c];
          } //cell loop
        }
#endif
#ifdef ZDIM
        if (wVelIndex > -1) {
          d_cqmomConv->doConvZ( patch, FconvZ, d_convScheme, d_convWeightLimit, cqmomWeights,
                               cqmomAbscissas, M, nNodes, wVelIndex, momentIndex, cellType, epW );
          for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            phiTemp[c] += -FconvZ[c];
          } //cell loop
        }
#endif
      }
      
      //The RHS of the convection for timestepping purposes is phiTemp-phi
      //NOTE: phiTemp isn't actually needed, but if further testing determines that the operator splitting
      //is needed for numerical/stability purposes, then this formulation will make that transition easier
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        Fconv[c] = phiTemp[c] - oldPhi[c];

        RHS[c] += Fconv[c];
        RHS[c] += Fdiff[c];
        if (d_addSources)
          RHS[c] += src[c]*vol;
      }
    }
    
  } //patch loop
}
//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::solveTransportEqn";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::solveTransportEqn, timeSubStep);
  
  //New
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_oldtransportVarLabel);
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);
  
  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_sharedState->get_delt_label(), Ghost::None, 0 );
  tsk->requires(Task::OldDW, d_fieldLabels->d_volFractionLabel, Ghost::None, 0 ); 
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually solve the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::solveTransportEqn( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT;
    
    CCVariable<double> phi;    // phi @ current sub-level
    CCVariable<double> oldphi; // phi @ last update for rk substeps
    constCCVariable<double> RHS;
    constCCVariable<double> rk1_phi; // phi @ n for averaging
    constCCVariable<double> vol_fraction; 
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(oldphi, d_oldtransportVarLabel, matlIndex, patch);
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);
    old_dw->get(rk1_phi, d_transportVarLabel, matlIndex, patch, gn, 0);
    old_dw->get(vol_fraction, d_fieldLabels->d_volFractionLabel, matlIndex, patch, gn, 0);
    
    d_timeIntegrator->singlePatchFEUpdate( patch, phi, RHS, dt, curr_ssp_time, d_eqnName );
    
    double factor = d_timeIntegrator->time_factor[timeSubStep];
    curr_ssp_time = curr_time + factor * dt;
    
    d_timeIntegrator->timeAvePhi( patch, phi, rk1_phi, timeSubStep, curr_ssp_time, 
        clip.tol, clip.do_low, clip.low, clip.do_high, clip.high, vol_fraction );
    
    //----BOUNDARY CONDITIONS
    // For first time step, bc's have been set in dqmomInit
    computeBCs( patch, d_eqnName, phi );
    
    // copy averaged phi into oldphi
    oldphi.copyData(phi);
    
  }
}

void
CQMOMEqn::sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
}

/*methods for using the operator splitting in covnection*/
//---------------------------------------------------------------------------
// Method: Schedule builindg the x-direction convection
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildXConvection( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildXConvection";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildXConvection);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_tempLabel);
  tsk->modifies(d_FconvXLabel);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cellTypeLabel, Ghost::AroundCells, 1);
  
  //loop over requires for weights and abscissas needed for convection term if IC=u,v,w
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildXConvection( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    constCCVariable<double> oldPhi;
    constCCVariable<int> cellType;
    
    CCVariable<double> phi;
    CCVariable<double> phiTemp;
    CCVariable<double> FconvX;

    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    old_dw->get(cellType, d_fieldLabels->d_cellTypeLabel, matlIndex, patch, gac, 1);
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(phiTemp, d_tempLabel, matlIndex, patch);
    new_dw->getModifiable(FconvX, d_FconvXLabel, matlIndex, patch);
    FconvX.initialize(0.0);
    
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      phiTemp[c] = phi[c]; //store phi in phiTemp, to reset it later after constructing all the fluxes
    } //cell loop
    
    vector<constCCVarWrapper> cqmomWeights;
    vector<constCCVarWrapper> cqmomAbscissas;
 
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomWeights.push_back(tempWrapper);
    }
        
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomAbscissas.push_back(tempWrapper);
     }
    
#ifdef cqmom_transport_dbg
    std::cout << "Transport of " << d_eqnName << " in x-dir" << std::endl;
    std::cout << "===========================" << std::endl;
#endif

    d_cqmomConv->doConvX( patch, FconvX, d_convScheme, d_convWeightLimit, cqmomWeights,
                          cqmomAbscissas, M, nNodes, uVelIndex, momentIndex, cellType, epW );
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      phi[c] += -FconvX[c]; //changing the actual phi value here
      //the updated phi is then used in the CQMOM Inversion before next convectino direction
    } //cell loop
    
  } //patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule builindg the y-direction convection
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildYConvection( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildYConvection";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildYConvection);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_tempLabel);
  tsk->modifies(d_FconvYLabel);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cellTypeLabel, Ghost::AroundCells, 1);
  
  //loop over requires for weights and abscissas needed for convection term if IC=u,v,w
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildYConvection( const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    constCCVariable<double> oldPhi;
    constCCVariable<int> cellType;
    
    CCVariable<double> phi;
    CCVariable<double> phiTemp;
    CCVariable<double> FconvY;
    
    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    old_dw->get(cellType, d_fieldLabels->d_cellTypeLabel, matlIndex, patch, gac, 1);
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(phiTemp, d_tempLabel, matlIndex, patch);
    new_dw->getModifiable(FconvY, d_FconvYLabel, matlIndex, patch);
    FconvY.initialize(0.0);
    
    vector<constCCVarWrapper> cqmomWeights;
    vector<constCCVarWrapper> cqmomAbscissas;
    
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomWeights.push_back(tempWrapper);
    }
    
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomAbscissas.push_back(tempWrapper);
    }
    
#ifdef cqmom_transport_dbg
    std::cout << "Transport of " << d_eqnName << " in y-dir" << std::endl;
    std::cout << "===========================" << std::endl;
#endif
    
    d_cqmomConv->doConvY( patch, FconvY, d_convScheme, d_convWeightLimit, cqmomWeights,
                           cqmomAbscissas, M, nNodes, vVelIndex, momentIndex, cellType, epW );
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      phi[c] += -FconvY[c];
    } //cell loop
    
  } //patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule building the z-direction convection
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildZConvection( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildZConvection";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildZConvection);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_tempLabel);
  tsk->modifies(d_FconvZLabel);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cellTypeLabel, Ghost::AroundCells, 1);
  
  //loop over requires for weights and abscissas needed for convection term if IC=u,v,w
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildZConvection( const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    constCCVariable<double> oldPhi;
    constCCVariable<int> cellType;
    
    CCVariable<double> phi;
    CCVariable<double> phiTemp;
    CCVariable<double> FconvZ;
    
    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    old_dw->get(cellType, d_fieldLabels->d_cellTypeLabel, matlIndex, patch, gac, 1);
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(phiTemp, d_tempLabel, matlIndex, patch);
    new_dw->getModifiable(FconvZ, d_FconvZLabel, matlIndex, patch);
    FconvZ.initialize(0.0);
    
    vector<constCCVarWrapper> cqmomWeights;
    vector<constCCVarWrapper> cqmomAbscissas;
    
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomWeights.push_back(tempWrapper);
    }
    
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      cqmomAbscissas.push_back(tempWrapper);
    }
    
#ifdef cqmom_transport_dbg
    std::cout << "Transport of " << d_eqnName << " in y-dir" << std::endl;
    std::cout << "===========================" << std::endl;
#endif
    
    d_cqmomConv->doConvZ( patch, FconvZ, d_convScheme, d_convWeightLimit, cqmomWeights,
                           cqmomAbscissas, M, nNodes, wVelIndex, momentIndex, cellType, epW );
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      phi[c] += -FconvZ[c];
    } //cell loop
    
  } //patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::sched_buildSplitRHS( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMEqn::buildSplitRHS";
  Task* tsk = scinew Task(taskname, this, &CQMOMEqn::buildSplitRHS);
  
  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  tsk->requires(Task::NewDW, d_tempLabel, Ghost::None, 0);
  
  tsk->requires(Task::NewDW, d_FconvXLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_FconvYLabel, Ghost::None, 0);
  tsk->requires(Task::NewDW, d_FconvZLabel, Ghost::None, 0);
  
  //-----OLD-----
  tsk->requires(Task::OldDW, d_fieldLabels->d_areaFractionLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  
  if (timeSubStep == 0) {
    tsk->requires(Task::OldDW, d_sourceLabel, Ghost::None, 0);
  } else {
    tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0);
  }
  
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 1 );
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 1 );
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOMEqn::buildSplitRHS( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    Vector Dx = patch->dCell();
    
    constCCVariable<double> oldPhi;
    constCCVariable<double> mu_t;
    constCCVariable<double> src; //summed up source
    constCCVariable<Vector> areaFraction;
    
    CCVariable<double> phi;
    CCVariable<double> Fdiff;
    CCVariable<double> Fconv;
    CCVariable<double> RHS;
    
    constCCVariable<double> phiTemp;
    constCCVariable<double> FconvX;
    constCCVariable<double> FconvY;
    constCCVariable<double> FconvZ;
    
    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    if (new_dw->exists(d_sourceLabel, matlIndex, patch)) {
      new_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0); // only get new_dw value on rkstep > 0
    } else {
      old_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0);
    }
    
    old_dw->get(mu_t, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1);
    old_dw->get(areaFraction, d_fieldLabels->d_areaFractionLabel, matlIndex, patch, gac, 2);
    
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch);
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);
    RHS.initialize(0.0);
    Fconv.initialize(0.0);
    Fdiff.initialize(0.0);
    
    new_dw->get(phiTemp, d_tempLabel, matlIndex, patch, gn, 0);
    new_dw->get(FconvX, d_FconvXLabel, matlIndex, patch, gn, 0);
    new_dw->get(FconvY, d_FconvYLabel, matlIndex, patch, gn, 0);
    new_dw->get(FconvZ, d_FconvZLabel, matlIndex, patch, gn, 0);
    
    computeBCs( patch, d_eqnName, phi );
    
    double vol = Dx.x();
#ifdef YDIM
    vol *= Dx.y();
#endif
#ifdef ZDIM
    vol *= Dx.z();
#endif

    // look for and add contribution from intrusions.
    if ( _using_new_intrusion ) {
      _intrusions->addScalarRHS( patch, Dx, d_eqnName, RHS );
    }
    
    //----DIFFUSION
    if (d_doDiff)
      d_disc->computeDiff( patch, Fdiff, oldPhi, mu_t, d_mol_diff, areaFraction, d_turbPrNo );
   
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
        
      Fconv[c] = phi[c] - phiTemp[c];
      //reset phi so it doesn't mess with time integrator
      phi[c] = phiTemp[c];
        
      RHS[c] += Fconv[c];
      RHS[c] += Fdiff[c];
      if (d_addSources)
        RHS[c] += src[c]*vol;
    }
    
  } //patch loop
}

