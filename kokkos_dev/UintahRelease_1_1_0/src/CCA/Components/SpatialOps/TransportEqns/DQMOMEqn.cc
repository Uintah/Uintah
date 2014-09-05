#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/SpatialOps/BoundaryCond.h>
//#include <CCA/Components/SpatialOps/ExplicitTimeInt.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
DQMOMEqnBuilder::DQMOMEqnBuilder( Fields* fieldLabels, 
                                  ExplicitTimeInt* timeIntegrator,
                                  string eqnName ) : 
DQMOMEqnBuilderBase( fieldLabels, timeIntegrator, eqnName )
{}
DQMOMEqnBuilder::~DQMOMEqnBuilder(){}

EqnBase*
DQMOMEqnBuilder::build(){
  return scinew DQMOMEqn(d_fieldLabels, d_timeIntegrator, d_eqnName);
}
// End Builder
//---------------------------------------------------------------------------

DQMOMEqn::DQMOMEqn( Fields* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
: 
EqnBase( fieldLabels, timeIntegrator, eqnName )
{
  
  std::string varname = eqnName+"Fdiff"; 
  d_FdiffLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"Fconv"; 
  d_FconvLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"RHS";
  d_RHSLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());
  varname = eqnName+"old";
  d_oldtransportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());
  varname = eqnName;
  d_transportVarLabel = VarLabel::create(varname,
            CCVariable<double>::getTypeDescription());

  // This is the source term:
  // I just tagged "_src" to the end of the transport variable name
  varname = eqnName+"_src";
  d_sourceLabel = VarLabel::create(varname, 
            CCVariable<double>::getTypeDescription());

  d_weight = false; 
}

DQMOMEqn::~DQMOMEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel); 
  VarLabel::destroy(d_RHSLabel);    
  VarLabel::destroy(d_sourceLabel); 
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void
DQMOMEqn::problemSetup(const ProblemSpecP& inputdb, int qn)
{
  ProblemSpecP db = inputdb; 
  d_quadNode = qn; 
  
  // Now look for other things:
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);

  // convection scheme
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");

  // initial value
  db->getWithDefault( "initial_value", d_initValue, 0.0);
  if (d_weight) 
    d_initValue = 1;
  else
    d_initValue = (qn + 1)/3.5;

  // Set some things:
  d_addSources = true; 

  // Get the list of models:
  for (ProblemSpecP m_db = db->findBlock("model"); m_db !=0; m_db = m_db->findNextBlock("model")){
    std::string model_name; 
    m_db->getAttribute("label", model_name); 

    // now tag on the internal coordinate
    std::string node;  
    std::stringstream out; 
    out << d_quadNode; 
    node = out.str(); 
    model_name += "_qn";
    model_name += node; 
    // put it in the list
    d_models.push_back(model_name); 
  }  
}
//---------------------------------------------------------------------------
// Method: Schedule clean up. 
// Probably not needed for DQMOM
//---------------------------------------------------------------------------
void 
DQMOMEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::cleanUp";
  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::cleanUp);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually clean up. 
//---------------------------------------------------------------------------
void DQMOMEqn::cleanUp( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{
}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_evalTransportEqn( const LevelP& level, 
                                   SchedulerP& sched, int timeSubStep )
{

  if (timeSubStep == 0) 
    sched_initializeVariables( level, sched );

  sched_buildTransportEqn( level, sched, timeSubStep );

  sched_solveTransportEqn( level, sched, timeSubStep );
}
//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables. 
//---------------------------------------------------------------------------
void 
DQMOMEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::initializeVariables);
  Ghost::GhostType gn = Ghost::None;
  //New
  tsk->computes(d_transportVarLabel);
  tsk->computes(d_oldtransportVarLabel); // for rk sub stepping 
  tsk->computes(d_RHSLabel); 
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually initialize the variables. 
//---------------------------------------------------------------------------
void DQMOMEqn::initializeVariables( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> newVar;
    CCVariable<double> rkoldVar; 
    constCCVariable<double> oldVar; 
    new_dw->allocateAndPut( newVar, d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( rkoldVar, d_oldtransportVarLabel, matlIndex, patch ); 
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);

    newVar.initialize(0.0);
    rkoldVar.initialize(0.0);
    // copy old into new
    newVar.copyData(oldVar);
    rkoldVar.copyData(oldVar); 

    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->allocateAndPut( Fdiff, d_FdiffLabel, matlIndex, patch );
    new_dw->allocateAndPut( Fconv, d_FconvLabel, matlIndex, patch );
    new_dw->allocateAndPut( RHS, d_RHSLabel, matlIndex, patch ); 
    
    Fdiff.initialize(0.0);
    Fconv.initialize(0.0);
    RHS.initialize(0.0);

  }
}
//---------------------------------------------------------------------------
// Method: Schedule compute the sources. 
// Probably not needed for DQMOM EQN
//--------------------------------------------------------------------------- 
void 
DQMOMEqn::sched_computeSources( const LevelP& level, SchedulerP& sched )
{
}
//---------------------------------------------------------------------------
// Method: Schedule build the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "DQMOMEqn::buildTransportEqn"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::buildTransportEqn);

  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  Fields::PartVelMap::iterator iter = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires(Task::NewDW, iter->second, Ghost::AroundCells, 1); 
 
  //-----OLD-----
  //tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->propLabels.lambda, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->velocityLabels.uVelocity, Ghost::AroundCells, 1);   
#ifdef YDIM
  tsk->requires(Task::OldDW, d_fieldLabels->velocityLabels.vVelocity, Ghost::AroundCells, 1); 
#endif
#ifdef ZDIM
  tsk->requires(Task::OldDW, d_fieldLabels->velocityLabels.wVelocity, Ghost::AroundCells, 1); 
#endif

  if (timeSubStep == 0) {
    tsk->requires(Task::OldDW, d_sourceLabel, Ghost::None, 0);
  } else {
    tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0); 
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation. 
//---------------------------------------------------------------------------
void 
DQMOMEqn::buildTransportEqn( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;
    Vector Dx = patch->dCell(); 

    constCCVariable<double> oldPhi;
    constCCVariable<double> lambda;
    constSFCXVariable<double> uVel; 
    constSFCYVariable<double> vVel; 
    constSFCZVariable<double> wVel; 
    constCCVariable<double> src; 
    constCCVariable<Vector> partVel; 

    CCVariable<double> phi;
    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    if (new_dw->exists(d_sourceLabel, matlIndex, patch)) { 
      new_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0); // only get new_dw value on rkstep > 0
    } else {
      old_dw->get(src, d_sourceLabel, matlIndex, patch, gn, 0); 
    }
    old_dw->get(lambda, d_fieldLabels->propLabels.lambda, matlIndex, patch, gac, 1);
    old_dw->get(uVel,   d_fieldLabels->velocityLabels.uVelocity, matlIndex, patch, gac, 1); 
    double vol = Dx.x();
#ifdef YDIM
    old_dw->get(vVel,   d_fieldLabels->velocityLabels.vVelocity, matlIndex, patch, gac, 1); 
    vol *= Dx.y();
#endif
#ifdef ZDIM
    old_dw->get(wVel,   d_fieldLabels->velocityLabels.wVelocity, matlIndex, patch, gac, 1); 
    vol *= Dx.z();
#endif
    Fields::PartVelMap::iterator iter = d_fieldLabels->partVel.find(d_quadNode);
    new_dw->get( partVel, iter->second, matlIndex, patch, gac, 1 ); 
 
    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch); 
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);

    //----BOUNDARY CONDITIONS
    computeBCs( patch, d_eqnName, phi );

    //----CONVECTION
    if (d_doConv)
      computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel, partVel );
  
    //----DIFFUSION
    if (d_doDiff)
      computeDiff( patch, Fdiff, oldPhi, lambda );
 
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

      RHS[c] += Fdiff[c] - Fconv[c];

      if (d_addSources) {
        RHS[c] += src[c]*vol;           
      }
    } 
  }
}
//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation. 
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "DQMOMEqn::solveTransportEqn";

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::solveTransportEqn, timeSubStep);

  //NEW
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_oldtransportVarLabel); 
  tsk->requires(Task::NewDW, d_RHSLabel, Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually solve the transport equation. 
//---------------------------------------------------------------------------
void 
DQMOMEqn::solveTransportEqn( const ProcessorGroup* pc, 
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
    int matlIndex = 0;
    delt_vartype DT;
    old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
    double dt = DT; 

    CCVariable<double> phi;
    CCVariable<double> oldphi; 
    constCCVariable<double> RHS; 

    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(oldphi, d_oldtransportVarLabel, matlIndex, patch); 
    new_dw->get(RHS, d_RHSLabel, matlIndex, patch, gn, 0);

    d_timeIntegrator->singlePatchFEUpdate( patch, phi, RHS, dt );
    Vector alpha = d_timeIntegrator->d_alpha; 
    Vector beta  = d_timeIntegrator->d_beta; 
    d_timeIntegrator->timeAvePhi( patch, phi, oldphi, timeSubStep, alpha, beta ); 
    
    // copy averaged phi into oldphi
    oldphi.copyData(phi); 

  }
}
//---------------------------------------------------------------------------
// Method: Compute the convection term. 
//---------------------------------------------------------------------------
template <class fT, class oldPhiT> void
DQMOMEqn::computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
                       constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
                       constSFCZVariable<double>& wVel, constCCVariable<Vector>& partVel ) 
{
  Vector Dx = p->dCell(); 
  FaceData<double> F;

  if (d_convScheme == "upwind") {

  // UPWIND
  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

    IntVector c = *iter;
    IntVector cxp = *iter + IntVector(1,0,0);
    IntVector cxm = *iter - IntVector(1,0,0);
    IntVector cyp = *iter + IntVector(0,1,0);
    IntVector cym = *iter - IntVector(0,1,0);
    IntVector czp = *iter + IntVector(0,0,1);
    IntVector czm = *iter - IntVector(0,0,1);

    double xmpVel = ( partVel[c].x() + partVel[cxm].x() )/2.0;
    double xppVel = ( partVel[c].x() + partVel[cxp].x() )/2.0;

    if ( xmpVel >= 0.0 )
      F.w = oldPhi[cxm];
    else
      F.w = oldPhi[c]; 

    if ( xppVel >= 0.0 )
      F.e = oldPhi[c];
    else
      F.e = oldPhi[cxp];

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xppVel - F.w * xmpVel );

#ifdef YDIM
    double ympVel = ( partVel[c].y() + partVel[cym].y() )/2.0;
    double yppVel = ( partVel[c].y() + partVel[cyp].y() )/2.0;
 
    if ( ympVel >= 0.0 )
      F.s = oldPhi[cym];
    else
      F.s = oldPhi[c]; 

    if ( yppVel >= 0.0 )
      F.n = oldPhi[c];
    else
      F.n = oldPhi[cyp];

    Fconv[c] += Dx.x()*Dx.z()*( F.n * yppVel - F.s * ympVel ); 
#endif
#ifdef ZDIM
    double zmpVel = ( partVel[c].z() + partVel[czm].z() )/2.0;
    double zppVel = ( partVel[c].z() + partVel[czp].z() )/2.0;
 
    if ( zmpVel >= 0.0 )
      F.b = oldPhi[czm];
    else
      F.b = oldPhi[c]; 

    if ( zppVel >= 0.0 )
      F.t = oldPhi[c];
    else
      F.t = oldPhi[czp];

    Fconv[c] += Dx.x()*Dx.y()*( F.t * zppVel - F.b * zmpVel ); 
#endif 
  }
  } else if (d_convScheme == "super_bee") { 

  // SUPERBEE
  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){
    IntVector c = *iter;
    IntVector cxp = *iter + IntVector(1,0,0);
    IntVector cxpp= *iter + IntVector(2,0,0);
    IntVector cxm = *iter - IntVector(1,0,0);
    IntVector cxmm= *iter - IntVector(2,0,0);
    IntVector cyp = *iter + IntVector(0,1,0);
    IntVector cypp= *iter + IntVector(0,2,0);
    IntVector cym = *iter - IntVector(0,1,0);
    IntVector cymm= *iter - IntVector(0,2,0);
    IntVector czp = *iter + IntVector(0,0,1);
    IntVector czpp= *iter + IntVector(0,0,2);
    IntVector czm = *iter - IntVector(0,0,1);
    IntVector czmm= *iter - IntVector(0,0,2);

    interpPtoF( oldPhi, c, F ); 

    double xmpVel = ( partVel[c].x() + partVel[cxm].x() )/2.0;
    double xppVel = ( partVel[c].x() + partVel[cxp].x() )/2.0;

    double r; 
    double psi; 
    double Sup;
    double Sdn;

    // EAST
    if ( xppVel >= 0.0 ) {
      r = ( oldPhi[c] - oldPhi[cxm] ) / ( oldPhi[cxp] - oldPhi[c] );
      Sup = oldPhi[c];
      Sdn = oldPhi[cxp];
    } else {
      r = ( oldPhi[cxpp] - oldPhi[cxp] ) / ( oldPhi[cxp] - oldPhi[c] );
      Sup = oldPhi[cxp];
      Sdn = oldPhi[c]; 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.e = Sup + 0.5*psi*( Sdn - Sup ); 

    // WEST 
    if ( xmpVel >= 0.0 ) {
      Sup = oldPhi[cxm];
      Sdn = oldPhi[c];
      r = ( oldPhi[cxm] - oldPhi[cxmm] ) / ( oldPhi[c] - oldPhi[cxm] ); 
    } else {
      Sup = oldPhi[c];
      Sdn = oldPhi[cxm];
      r = ( oldPhi[cxp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cxm] ); 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.w = Sup + 0.5*psi*( Sdn - Sup ); 

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xppVel - F.w * xmpVel );
#ifdef YDIM
    double ympVel = ( partVel[c].y() + partVel[cym].y() )/2.0;
    double yppVel = ( partVel[c].y() + partVel[cyp].y() )/2.0;
    // NORTH
    if ( yppVel >= 0.0 ) {
      r = ( oldPhi[c] - oldPhi[cym] ) / ( oldPhi[cyp] - oldPhi[c] );
      Sup = oldPhi[c];
      Sdn = oldPhi[cyp];
    } else {
      r = ( oldPhi[cypp] - oldPhi[cyp] ) / ( oldPhi[cyp] - oldPhi[c] );
      Sup = oldPhi[cyp];
      Sdn = oldPhi[c]; 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.n = Sup + 0.5*psi*( Sdn - Sup ); 

    // SOUTH 
    if ( ympVel >= 0.0 ) {
      Sup = oldPhi[cym];
      Sdn = oldPhi[c];
      r = ( oldPhi[cym] - oldPhi[cymm] ) / ( oldPhi[c] - oldPhi[cym] ); 
    } else {
      Sup = oldPhi[c];
      Sdn = oldPhi[cym];
      r = ( oldPhi[cyp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cym] ); 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.s = Sup + 0.5*psi*( Sdn - Sup ); 

    Fconv[c] += Dx.x()*Dx.z()*( F.n * yppVel - F.s * ympVel ); 
#endif
#ifdef ZDIM
    double zmpVel = ( partVel[c].z() + partVel[czm].z() )/2.0;
    double zppVel = ( partVel[c].z() + partVel[czp].z() )/2.0;

    // TOP
    if ( zppVel >= 0.0 ) {
      r = ( oldPhi[c] - oldPhi[czm] ) / ( oldPhi[czp] - oldPhi[c] );
      Sup = oldPhi[c];
      Sdn = oldPhi[czp];
    } else {
      r = ( oldPhi[czpp] - oldPhi[czp] ) / ( oldPhi[czp] - oldPhi[c] );
      Sup = oldPhi[czp];
      Sdn = oldPhi[c]; 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.t = Sup + 0.5*psi*( Sdn - Sup ); 

    // BOTTOM 
    if ( zmpVel >= 0.0 ) {
      Sup = oldPhi[czm];
      Sdn = oldPhi[c];
      r = ( oldPhi[czm] - oldPhi[czmm] ) / ( oldPhi[c] - oldPhi[czm] ); 
    } else {
      Sup = oldPhi[c];
      Sdn = oldPhi[czm];
      r = ( oldPhi[czp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[czm] ); 
    }

    psi = max( min(2.0*r, 1.0), min(r, 2.0) );
    psi = max( 0.0, psi );
    F.b = Sup + 0.5*psi*( Sdn - Sup ); 

    Fconv[c] += Dx.x()*Dx.y()*( F.t * zppVel - F.b * zmpVel ); 
#endif
  }

  } else {

    cout << "Convection scheme not supported! " << endl;

  }
}
//---------------------------------------------------------------------------
// Method: Compute the diffusion term. 
// I was templating this to see if I could produce a computeDiff that
// worked for all data types. Note sure if it works yet. But it does at least work for 
// a cc scalar. 
//---------------------------------------------------------------------------
template <class fT, class oldPhiT, class lambdaT > void
DQMOMEqn::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, lambdaT& lambda )
{
  // --- compute diffusion term ---
  Vector Dx = p->dCell();

  FaceData<double> F;
  FaceData<double> G;

  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

    IntVector c = *iter; 

    // Get gradient and interpolated diffusion coef.     
    interpPtoF( lambda, c, F ); 
    gradPtoF( oldPhi, c, p, G ); 

    Fdiff[c] = Dx.y()*Dx.z()*(F.e*G.e - F.w*G.w);           
#ifdef YDIM
    Fdiff[c] += Dx.x()*Dx.z()*(F.n*G.n - F.s*G.s);      
#endif  
#ifdef ZDIM 
    Fdiff[c] += Dx.y()*Dx.z()*(F.t*G.t - F.b*G.b);      
#endif
  } 

}
//---------------------------------------------------------------------------
// Method: Compute the boundary conditions. 
//---------------------------------------------------------------------------
template<class phiType> void
DQMOMEqn::computeBCs( const Patch* patch, 
                       string varName,
                       phiType& phi )
{
  //d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
}
//---------------------------------------------------------------------------
// Method: Interpolate a variable to the face of its respective cv
//---------------------------------------------------------------------------
template <class phiT, class interpT > void
DQMOMEqn::interpPtoF( phiT& phi, const IntVector c, interpT& F )
{
  IntVector xd(1,0,0);
  IntVector yd(0,1,0);
  IntVector zd(0,0,1);
  
  F.p = phi[c]; 

  F.e = 0.5 * ( phi[c] + phi[c + xd] );
  F.w = 0.5 * ( phi[c] + phi[c - xd] );
#ifdef YDIM
  F.n = 0.5 * ( phi[c] + phi[c + yd] ); 
  F.s = 0.5 * ( phi[c] + phi[c - yd] ); 
#endif
#ifdef ZDIM
  F.t = 0.5 * ( phi[c] + phi[c + zd] ); 
  F.b = 0.5 * ( phi[c] - phi[c - zd] ); 
#endif
} 
//---------------------------------------------------------------------------
// Method: Gradient a variable to the face of its respective cv
//---------------------------------------------------------------------------
template <class phiT, class gradT> void
DQMOMEqn::gradPtoF( phiT& phi, const IntVector c, const Patch* p, gradT& G )
{
  IntVector xd(1,0,0);
  IntVector yd(0,1,0);
  IntVector zd(0,0,1);
  
  Vector Dx = p->dCell();
  
  G.p = phi[c]; 

  G.e =  ( phi[c + xd] - phi[c] ) / Dx.x();
  G.w =  ( phi[c] - phi[c - xd] ) / Dx.y();
#ifdef YDIM
  G.n =  ( phi[c + yd] - phi[c] ) / Dx.y(); 
  G.s =  ( phi[c] - phi[c - yd] ) / Dx.y(); 
#endif
#ifdef ZDIM
  G.t =  ( phi[c + zd] - phi[c] ) / Dx.z(); 
  G.b =  ( phi[c] - phi[c - zd] ) / Dx.z(); 
#endif
} 

