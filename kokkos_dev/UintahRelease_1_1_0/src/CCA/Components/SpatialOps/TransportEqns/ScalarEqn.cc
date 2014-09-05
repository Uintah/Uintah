#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/SpatialOps/BoundaryCond.h>
//#include <CCA/Components/SpatialOps/ExplicitTimeInt.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/TransportEqns/ScalarEqn.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermBase.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
CCScalarEqnBuilder::CCScalarEqnBuilder( Fields* fieldLabels, 
                                        ExplicitTimeInt* timeIntegrator,
                                        string eqnName ) : 
EqnBuilder( fieldLabels, timeIntegrator, eqnName )
{}
CCScalarEqnBuilder::~CCScalarEqnBuilder(){}

EqnBase*
CCScalarEqnBuilder::build(){
  return scinew ScalarEqn(d_fieldLabels, d_timeIntegrator, d_eqnName);
}
// End Builder
//---------------------------------------------------------------------------

ScalarEqn::ScalarEqn( Fields* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
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


}

ScalarEqn::~ScalarEqn()
{
  VarLabel::destroy(d_FdiffLabel);
  VarLabel::destroy(d_FconvLabel); 
  VarLabel::destroy(d_RHSLabel);
  VarLabel::destroy(d_transportVarLabel);
  VarLabel::destroy(d_oldtransportVarLabel);
}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void
ScalarEqn::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 
  
  if (db->findBlock("src")){
    string srcname; 
    for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){
      src_db->getAttribute("label", srcname);
      //which sources are turned on for this equation
      d_sources.push_back( srcname ); 

    }
  }

  // Now look for other things:
  // convection scheme
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);
  db->getWithDefault( "addSources", d_addSources, true); 

}
//---------------------------------------------------------------------------
// Method: Schedule clean up. 
//---------------------------------------------------------------------------
void 
ScalarEqn::sched_cleanUp( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ScalarEqn::cleanUp";
  Task* tsk = scinew Task(taskname, this, &ScalarEqn::cleanUp);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually clean up. 
//---------------------------------------------------------------------------
void ScalarEqn::cleanUp( const ProcessorGroup* pc, 
                         const PatchSubset* patches, 
                         const MaterialSubset* matls, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw )
{

  //Set the initialization flag for the source label to false.
  SourceTermFactory& factory = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
 
    SourceTermBase& temp_src = factory.retrieve_source_term( *iter ); 
  
    temp_src.reinitializeLabel(); 

  }
}
//---------------------------------------------------------------------------
// Method: Schedule the evaluation of the transport equation. 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_evalTransportEqn( const LevelP& level, 
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
ScalarEqn::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "ScalarEqn::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &ScalarEqn::initializeVariables);
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
void ScalarEqn::initializeVariables( const ProcessorGroup* pc, 
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
//--------------------------------------------------------------------------- 
void 
ScalarEqn::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  // This scheduler only calls other schedulers
  SourceTermFactory& factory = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_sources.begin(); iter != d_sources.end(); iter++){
 
    SourceTermBase& temp_src = factory.retrieve_source_term( *iter ); 
    cout << "source name  = " << *iter << endl;
   
    temp_src.sched_computeSource( level, sched, timeSubStep ); 

  }

}
//---------------------------------------------------------------------------
// Method: Schedule build the transport equation. 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_buildTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ScalarEqn::buildTransportEqn"; 

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::buildTransportEqn);

  //----NEW----
  tsk->modifies(d_transportVarLabel);
  tsk->modifies(d_FdiffLabel);
  tsk->modifies(d_FconvLabel);
  tsk->modifies(d_RHSLabel);
  tsk->requires(Task::NewDW, d_oldtransportVarLabel, Ghost::AroundCells, 2);

  // srcs
  SourceTermFactory& src_factory = SourceTermFactory::self(); 
  for (vector<std::string>::iterator iter = d_sources.begin(); 
       iter != d_sources.end(); iter++){
    SourceTermBase& temp_src = src_factory.retrieve_source_term( *iter ); 
    const VarLabel* temp_varLabel; 
    temp_varLabel = temp_src.getSrcLabel(); 
    cout << "The var label = " << *temp_varLabel << endl;
    tsk->modifies(temp_src.getSrcLabel()); 
  }
  
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

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the transport equation. 
//---------------------------------------------------------------------------
void 
ScalarEqn::buildTransportEqn( const ProcessorGroup* pc, 
                              const PatchSubset* patches, 
                              const MaterialSubset* matls, 
                              DataWarehouse* old_dw, 
                              DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    constCCVariable<double> oldPhi;
    constCCVariable<double> lambda;
    constSFCXVariable<double> uVel; 
    constSFCYVariable<double> vVel; 
    constSFCZVariable<double> wVel; 

    CCVariable<double> phi;
    CCVariable<double> Fdiff; 
    CCVariable<double> Fconv; 
    CCVariable<double> RHS; 

    new_dw->get(oldPhi, d_oldtransportVarLabel, matlIndex, patch, gac, 2);
    old_dw->get(lambda, d_fieldLabels->propLabels.lambda, matlIndex, patch, gac, 1);
    old_dw->get(uVel,   d_fieldLabels->velocityLabels.uVelocity, matlIndex, patch, gac, 1); 
#ifdef YDIM
    old_dw->get(vVel,   d_fieldLabels->velocityLabels.vVelocity, matlIndex, patch, gac, 1); 
#endif
#ifdef ZDIM
    old_dw->get(wVel,   d_fieldLabels->velocityLabels.wVelocity, matlIndex, patch, gac, 1); 
#endif

    new_dw->getModifiable(phi, d_transportVarLabel, matlIndex, patch);
    new_dw->getModifiable(Fdiff, d_FdiffLabel, matlIndex, patch);
    new_dw->getModifiable(Fconv, d_FconvLabel, matlIndex, patch); 
    new_dw->getModifiable(RHS, d_RHSLabel, matlIndex, patch);

    //----BOUNDARY CONDITIONS
    computeBCs( patch, d_eqnName, phi );

    //----CONVECTION
    if (d_doConv)
      computeConv( patch, Fconv, oldPhi, uVel, vVel, wVel );
  
    //----DIFFUSION
    if (d_doDiff)
      computeDiff( patch, Fdiff, oldPhi, lambda );
 
    //----SUM UP RHS
    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 

      RHS[c] += Fdiff[c] - Fconv[c];

      if (d_addSources) {
        // Get the factory of source terms
        SourceTermFactory& src_factory = SourceTermFactory::self(); 
        for (vector<std::string>::iterator src_iter = d_sources.begin(); src_iter != d_sources.end(); src_iter++){
          CCVariable<double> src;  // Outside of this scope src is no longer available 
          SourceTermBase& temp_src = src_factory.retrieve_source_term( *src_iter ); 
          new_dw->getModifiable(src, temp_src.getSrcLabel(), matlIndex, patch);
          // Add to the RHS
          RHS[c] += src[c]; 
        }            
      }
    } 
  }
}
//---------------------------------------------------------------------------
// Method: Schedule solve the transport equation. 
//---------------------------------------------------------------------------
void
ScalarEqn::sched_solveTransportEqn( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "ScalarEqn::solveTransportEqn";

  Task* tsk = scinew Task(taskname, this, &ScalarEqn::solveTransportEqn, timeSubStep);

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
ScalarEqn::solveTransportEqn( const ProcessorGroup* pc, 
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
ScalarEqn::computeConv(const Patch* p, fT& Fconv, oldPhiT& oldPhi, 
                       constSFCXVariable<double>& uVel, constSFCYVariable<double>& vVel, 
                       constSFCZVariable<double>& wVel) 
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

    double xmVel = uVel[c];
    double xpVel = uVel[cxp];

    if ( xmVel >= 0.0 )
      F.w = oldPhi[cxm];
    else
      F.w = oldPhi[c]; 

    if ( xpVel >= 0.0 )
      F.e = oldPhi[c];
    else
      F.e = oldPhi[cxp];

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xpVel - F.w * xmVel );

#ifdef YDIM
    double ymVel = vVel[c];
    double ypVel = vVel[cyp];
 
    if ( ymVel >= 0.0 )
      F.s = oldPhi[cym];
    else
      F.s = oldPhi[c]; 

    if ( ypVel >= 0.0 )
      F.n = oldPhi[c];
    else
      F.n = oldPhi[cyp];

    Fconv[c] += Dx.x()*Dx.z()*( F.n * ypVel - F.s * ymVel ); 
#endif
#ifdef ZDIM
    double zmVel = wVel[c];
    double zpVel = wVel[czp];
 
    if ( zmpVel >= 0.0 )
      F.b = oldPhi[czm];
    else
      F.b = oldPhi[c]; 

    if ( zppVel >= 0.0 )
      F.t = oldPhi[c];
    else
      F.t = oldPhi[czp];

    Fconv[c] += Dx.x()*Dx.y()*( F.t * zpVel - F.b * zmVel ); 
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

    double xmVel = uVel[c];
    double xpVel = uVel[cxp];

    double r; 
    double psi; 
    double Sup;
    double Sdn;

    // EAST
    if ( xpVel >= 0.0 ) {
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
    if ( xmVel >= 0.0 ) {
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

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xpVel - F.w * xmVel );
#ifdef YDIM
    double ymVel = vVel[c];
    double ypVel = vVel[cyp];
    // NORTH
    if ( ypVel >= 0.0 ) {
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
    if ( ymVel >= 0.0 ) {
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

    Fconv[c] += Dx.x()*Dx.z()*( F.n * ypVel - F.s * ymVel ); 
#endif
#ifdef ZDIM
    double zmVel = wVel[c];
    double zpVel = wVel[czp];

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
    if ( zmVel >= 0.0 ) {
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

    Fconv[c] += Dx.x()*Dx.y()*( F.t * zpVel - F.b * zmVel ); 
#endif
  }

  } else if ( d_convScheme == "central") {

  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

    IntVector c = *iter;
    IntVector cxp = *iter + IntVector(1,0,0);
    IntVector cyp = *iter + IntVector(0,1,0);
    IntVector czp = *iter + IntVector(0,0,1);

    interpPtoF( oldPhi, c, F ); 

    Fconv[c] = Dx.y()*Dx.z()*( F.e * uVel[cxp] - F.w * uVel[c] );
#ifdef YDIM
    Fconv[c] += Dx.x()*Dx.z()*( F.n * vVel[cyp] - F.s * vVel[c] );
#endif
#ifdef ZDIM
    Fconv[c] += Dx.x()*Dx.y()*( F.t * wVel[czp] - F.b * wVel[c] ); 
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
ScalarEqn::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, lambdaT& lambda )
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
ScalarEqn::computeBCs( const Patch* patch, 
                       string varName,
                       phiType& phi )
{
  //d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
}
//---------------------------------------------------------------------------
// Method: Interpolate a variable to the face of its respective cv
//---------------------------------------------------------------------------
template <class phiT, class interpT > void
ScalarEqn::interpPtoF( phiT& phi, const IntVector c, interpT& F )
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
ScalarEqn::gradPtoF( phiT& phi, const IntVector c, const Patch* p, gradT& G )
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

