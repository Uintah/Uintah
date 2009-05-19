#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Builder:
DQMOMEqnBuilder::DQMOMEqnBuilder( ArchesLabel* fieldLabels, 
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

DQMOMEqn::DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName )
: 
EqnBase( fieldLabels, timeIntegrator, eqnName )
{
  
  string varname = eqnName+"Fdiff"; 
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
  varname = eqnName+"_icv"; // icv = internal coordinate value
  d_icLabel = VarLabel::create(varname, 
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
  VarLabel::destroy(d_icLabel); 
}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void
DQMOMEqn::problemSetup(const ProblemSpecP& inputdb, int qn)
{
  ProblemSpecP db = inputdb; 
  d_quadNode = qn; 

  d_turbPrNo = 0.4; //for the turb diff model.  Need to set as input


  // Now look for other things:
  db->getWithDefault( "doConv", d_doConv, false);
  db->getWithDefault( "doDiff", d_doDiff, false);

  // convection scheme
  db->getWithDefault( "conv_scheme", d_convScheme, "upwind");

  // get normalization constant
  db->require( "scaling_const", d_scalingConstant ); 

  // initial value blocks
  for (ProblemSpecP db_initvals = db->findBlock("initial_value");
       db_initvals != 0; db_initvals = db_initvals->findNextBlock("initial_value") ) {
    
    string tempQuadNode;
    db_initvals->getAttribute("qn", tempQuadNode);

    int temp_qn = atoi(tempQuadNode.c_str());

    string s_myValue; 
    db_initvals->getAttribute("value", s_myValue); 
    double myValue = atof(s_myValue.c_str());
    if (temp_qn == d_quadNode) 
      d_initValue = myValue/d_scalingConstant;

    // if this is an IC, then multiply by the weight.
    if (!d_weight) {
      DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
      string name = "w_qn"; 
      string node; 
      std::stringstream out; 
      out << d_quadNode; 
      node = out.str(); 
      name += node;
      EqnBase& t_w = dqmomFactory.retrieve_scalar_eqn( name );
      DQMOMEqn& w = dynamic_cast<DQMOMEqn&>(t_w);

      d_initValue *= w.getInitValue(); 

    } 
  } 

  // Set some things:
  d_addSources = true; 

  // Get the list of models:
  for (ProblemSpecP m_db = db->findBlock("model"); m_db !=0; m_db = m_db->findNextBlock("model")){
    string model_name; 
    m_db->getAttribute("label", model_name); 

    // now tag on the internal coordinate
    string node;  
    std::stringstream out; 
    out << d_quadNode; 
    node = out.str(); 
    model_name += "_qn";
    model_name += node; 
    // put it in the list
    d_models.push_back(model_name); 
  }  

  // Check for clipping
  d_doClipping = false; 
  ProblemSpecP db_clipping = db->findBlock("Clipping");
  d_lowClip = 0.0; // initializing this to zero for getAbscissa values
  double clip_default = -9999999999.0;
  if (db_clipping) {
    //This seems like a "safe" number to assume 
    d_doLowClip = false; 
    d_doHighClip = false; 
    d_doClipping = true;
    
    db_clipping->getWithDefault("low", d_lowClip,  clip_default);
    db_clipping->getWithDefault("high",d_highClip, clip_default);

    if ( d_lowClip != clip_default ) 
      d_doLowClip = true; 

    if ( d_highClip != clip_default ) 
      d_doHighClip = true; 

    if ( !d_doHighClip && !d_doLowClip ) 
      throw InvalidValue("A low or high clipping must be specified if the <Clipping> section is activated!", __FILE__, __LINE__);
   } 
  if (d_weight) { 
    if (!d_doClipping) { 
      //By default, set the low value for this weight to 0 and run on low clipping
      d_lowClip = 0;
      d_doClipping = true; 
      d_doLowClip = true;  
    } else { 
      if (!d_doLowClip){ 
        //weights always have low clip values!  ie, negative weights not allowed
        d_lowClip = 0;
        d_doLowClip = true; 
      } 
    }
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

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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
  tsk->computes(d_icLabel); 
  tsk->computes(d_RHSLabel); 
  tsk->computes(d_FconvLabel);
  tsk->computes(d_FdiffLabel);

  //Old
  tsk->requires(Task::OldDW, d_transportVarLabel, gn, 0);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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
    CCVariable<double> icValue; 
    constCCVariable<double> oldVar; 
    new_dw->allocateAndPut( newVar  , d_transportVarLabel, matlIndex, patch );
    new_dw->allocateAndPut( rkoldVar, d_oldtransportVarLabel, matlIndex, patch ); 
    new_dw->allocateAndPut( icValue , d_icLabel, matlIndex, patch );
    old_dw->get(oldVar, d_transportVarLabel, matlIndex, patch, gn, 0);

    newVar.initialize(0.0);
    rkoldVar.initialize(0.0);
    icValue.initialize(0.0);
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
  ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(d_quadNode);
  tsk->requires(Task::NewDW, iter->second, Ghost::AroundCells, 1); 
 
  //-----OLD-----
  //tsk->requires(Task::OldDW, d_transportVarLabel, Ghost::AroundCells, 2);
  tsk->requires(Task::OldDW, d_fieldLabels->d_viscosityCTSLabel, Ghost::AroundCells, 1);
  tsk->requires(Task::OldDW, d_fieldLabels->d_uVelocitySPBCLabel, Ghost::AroundCells, 1);   
#ifdef YDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_vVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif
#ifdef ZDIM
  tsk->requires(Task::OldDW, d_fieldLabels->d_wVelocitySPBCLabel, Ghost::AroundCells, 1); 
#endif

  if (timeSubStep == 0) {
    tsk->requires(Task::OldDW, d_sourceLabel, Ghost::None, 0);
  } else {
    tsk->requires(Task::NewDW, d_sourceLabel, Ghost::None, 0); 
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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
    constCCVariable<double> mu_t;
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

    old_dw->get(mu_t, d_fieldLabels->d_viscosityCTSLabel, matlIndex, patch, gac, 1); 
    old_dw->get(uVel,   d_fieldLabels->d_uVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    double vol = Dx.x();
#ifdef YDIM
    old_dw->get(vVel,   d_fieldLabels->d_vVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.y(); 
#endif
#ifdef ZDIM
    old_dw->get(wVel,   d_fieldLabels->d_wVelocitySPBCLabel, matlIndex, patch, gac, 1); 
    vol *= Dx.z(); 
#endif

    ArchesLabel::PartVelMap::iterator iter = d_fieldLabels->partVel.find(d_quadNode);
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
      computeDiff( patch, Fdiff, oldPhi, mu_t );
 
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

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
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

    if (d_doClipping) 
      clipPhi( patch, phi ); 
    
    // copy averaged phi into oldphi
    oldphi.copyData(phi); 

  }
}
//---------------------------------------------------------------------------
// Method: Schedule the compute of the IC values
//---------------------------------------------------------------------------
void
DQMOMEqn::sched_getAbscissaValues( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOMEqn::getAbscissaValues"; 

  Task* tsk = scinew Task(taskname, this, &DQMOMEqn::getAbscissaValues);
  
  Ghost::GhostType  gn  = Ghost::None;
  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 

  //NEW
  tsk->modifies(d_icLabel);
  tsk->modifies(d_transportVarLabel); 

  string name = "w_qn"; 
  string node; 
  std::stringstream out; 
  out << d_quadNode; 
  node = out.str(); 
  name += node; 

  EqnBase& eqn = dqmomFactory.retrieve_scalar_eqn( name ); 
  const VarLabel* weightLabel = eqn.getTransportEqnLabel(); 
  
  tsk->requires( Task::NewDW, weightLabel, gn, 0 ); 
 
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}
//---------------------------------------------------------------------------
// Method: Compute the IC vaues by dividing by the weights
//---------------------------------------------------------------------------
void 
DQMOMEqn::getAbscissaValues( const ProcessorGroup* pc, 
                             const PatchSubset* patches, 
                             const MaterialSubset* matls, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    CCVariable<double> wa;
    constCCVariable<double> w;  
    CCVariable<double> ic; 

    new_dw->getModifiable(ic, d_icLabel, matlIndex, patch);

    new_dw->getModifiable(wa, d_transportVarLabel, matlIndex, patch); 
    DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
    string name = "w_qn"; 
    string node;
    std::stringstream out;
    out << d_quadNode;
    node = out.str();
    name += node;

    EqnBase& temp_eqn = dqmomFactory.retrieve_scalar_eqn( name );
    DQMOMEqn& eqn = dynamic_cast<DQMOMEqn&>(temp_eqn);
    const VarLabel* mywLabel = eqn.getTransportEqnLabel();  
    double smallWeight = eqn.getLowClip(); 

    new_dw->get(w, mywLabel, matlIndex, patch, gn, 0); 

    // now loop over all cells
    for (CellIterator iter=patch->getCellIterator__New(0); !iter.done(); iter++){
  
      IntVector c = *iter;

      if (w[c] > smallWeight)
        ic[c] = wa[c]/w[c]*d_scalingConstant;
      else {
        ic[c] = 0.0;
        wa[c] = 0.0; // if the weight is small (near zero) , then the product must also be small (near zero)
      }
    }
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
  IntVector cLow  = p->getCellLowIndex__New(); 
  IntVector cHigh = p->getCellHighIndex__New();  

  if (d_convScheme == "upwind") {

    //bool xminus = p->getBCType(Patch::xminus) != Patch::Neighbor;
    //bool xplus =  p->getBCType(Patch::xplus) != Patch::Neighbor;
    //bool yminus = p->getBCType(Patch::yminus) != Patch::Neighbor;
    //bool yplus =  p->getBCType(Patch::yplus) != Patch::Neighbor;
    //bool zminus = p->getBCType(Patch::zminus) != Patch::Neighbor;
    //bool zplus =  p->getBCType(Patch::zplus) != Patch::Neighbor;

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

    if ( xmpVel > 0.0 )
      F.w = oldPhi[cxm];
    else if (xmpVel < 0.0 )
      F.w = oldPhi[c]; 
    else 
      F.w = 0.0;

    if ( xppVel > 0.0 )
      F.e = oldPhi[c];
    else if ( xppVel < 0.0 )
      F.e = oldPhi[cxp];
    else 
      F.e = 0.0;  

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xppVel - F.w * xmpVel );

#ifdef YDIM
    double ympVel = ( partVel[c].y() + partVel[cym].y() )/2.0;
    double yppVel = ( partVel[c].y() + partVel[cyp].y() )/2.0;
 
    if ( ympVel > 0.0 )
      F.s = oldPhi[cym];
    else if ( ympVel < 0.0 )
      F.s = oldPhi[c]; 
    else
      F.s = 0.0;  


    if ( yppVel > 0.0 )
      F.n = oldPhi[c];
    else if ( yppVel < 0.0 )
      F.n = oldPhi[cyp];
    else  
      F.n = 0.0; 

    Fconv[c] += Dx.x()*Dx.z()*( F.n * yppVel - F.s * ympVel ); 
#endif
#ifdef ZDIM
    double zmpVel = ( partVel[c].z() + partVel[czm].z() )/2.0;
    double zppVel = ( partVel[c].z() + partVel[czp].z() )/2.0;
 
    if ( zmpVel > 0.0 )
      F.b = oldPhi[czm];
    else if ( zmpVel < 0.0 )
      F.b = oldPhi[c]; 
    else 
      F.b = 0.0;   

    if ( zppVel > 0.0 )
      F.t = oldPhi[c];
    else if ( zppVel < 0.0 )
      F.t = oldPhi[czp];
    else 
      F.t = 0.0;  

    Fconv[c] += Dx.x()*Dx.y()*( F.t * zppVel - F.b * zmpVel ); 
#endif 
  }
  } else if (d_convScheme == "super_bee") { 

  // SUPERBEE
  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

    bool xminus = p->getBCType(Patch::xminus) != Patch::Neighbor;
    bool xplus =  p->getBCType(Patch::xplus) != Patch::Neighbor;
    bool yminus = p->getBCType(Patch::yminus) != Patch::Neighbor;
    bool yplus =  p->getBCType(Patch::yplus) != Patch::Neighbor;
    bool zminus = p->getBCType(Patch::zminus) != Patch::Neighbor;
    bool zplus =  p->getBCType(Patch::zplus) != Patch::Neighbor;

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

    //interpPtoF( oldPhi, c, F ); 

    double xmpVel = ( partVel[c].x() + partVel[cxm].x() )/2.0;
    double xppVel = ( partVel[c].x() + partVel[cxp].x() )/2.0;

    double r; 
    double psi; 
    double Sup;
    double Sdn;

    // EAST
    if ( xplus && c.x() == cHigh.x() ) {
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( xppVel > 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( xppVel < 0.0 ) { 
        Sup = oldPhi[cxp]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
    } else { 
      if ( xppVel > 0.0 ) {
        r = ( oldPhi[c] - oldPhi[cxm] ) / ( oldPhi[cxp] - oldPhi[c] );
        Sup = oldPhi[c];
        Sdn = oldPhi[cxp];
      } else if ( xppVel < 0.0 ) {
        r = ( oldPhi[cxpp] - oldPhi[cxp] ) / ( oldPhi[cxp] - oldPhi[c] );
        Sup = oldPhi[cxp];
        Sdn = oldPhi[c]; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      }
      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }

    F.e = Sup + 0.5*psi*( Sdn - Sup ); 

    // WEST 
    if ( xminus && c.x() == cLow.x() ) { 
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( xmpVel > 0.0 ) { 
        Sup = oldPhi[cxm]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( xmpVel < 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
    } else {
      if ( xmpVel > 0.0 ) {
        Sup = oldPhi[cxm];
        Sdn = oldPhi[c];
        r = ( oldPhi[cxm] - oldPhi[cxmm] ) / ( oldPhi[c] - oldPhi[cxm] ); 
      } else if ( xmpVel < 0.0 ) {
        Sup = oldPhi[c];
        Sdn = oldPhi[cxm];
        r = ( oldPhi[cxp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cxm] );
      } else { 
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      }
      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }
  
    F.w = Sup + 0.5*psi*( Sdn - Sup ); 

    Fconv[c] = Dx.y()*Dx.z()*( F.e * xppVel - F.w * xmpVel );
#ifdef YDIM
    double ympVel = ( partVel[c].y() + partVel[cym].y() )/2.0;
    double yppVel = ( partVel[c].y() + partVel[cyp].y() )/2.0;
    // NORTH
    if ( yplus && c.y() == cHigh.y() ) {
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( yppVel > 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( yppVel < 0.0 ) { 
        Sup = oldPhi[cyp]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      }
    } else { 
      if ( yppVel > 0.0 ) {
        r = ( oldPhi[c] - oldPhi[cym] ) / ( oldPhi[cyp] - oldPhi[c] );
        Sup = oldPhi[c];
        Sdn = oldPhi[cyp];
      } else if ( yppVel < 0.0 ) {
        r = ( oldPhi[cypp] - oldPhi[cyp] ) / ( oldPhi[cyp] - oldPhi[c] );
        Sup = oldPhi[cyp];
        Sdn = oldPhi[c]; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }

    F.n = Sup + 0.5*psi*( Sdn - Sup ); 

    // SOUTH 
    if ( yminus && c.y() == cLow.y() ) {
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( ympVel > 0.0 ) { 
        Sup = oldPhi[cym]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( ympVel < 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
    } else { 
      if ( ympVel > 0.0 ) {
        Sup = oldPhi[cym];
        Sdn = oldPhi[c];
        r = ( oldPhi[cym] - oldPhi[cymm] ) / ( oldPhi[c] - oldPhi[cym] ); 
      } else if ( ympVel > 0.0 ) {
        Sup = oldPhi[c];
        Sdn = oldPhi[cym];
        r = ( oldPhi[cyp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[cym] ); 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }

    F.s = Sup + 0.5*psi*( Sdn - Sup ); 

    Fconv[c] += Dx.x()*Dx.z()*( F.n * yppVel - F.s * ympVel ); 
#endif
#ifdef ZDIM
    double zmpVel = ( partVel[c].z() + partVel[czm].z() )/2.0;
    double zppVel = ( partVel[c].z() + partVel[czp].z() )/2.0;

    // TOP
    if ( zplus && c.z() == cHigh.z() ) { 
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( zppVel > 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( zppVel < 0.0 ) { 
        Sup = oldPhi[czp]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
    } else { 
      if ( zppVel > 0.0 ) {
        r = ( oldPhi[c] - oldPhi[czm] ) / ( oldPhi[czp] - oldPhi[c] );
        Sup = oldPhi[c];
        Sdn = oldPhi[czp];
      } else if ( zppVel < 0.0 ) {
        r = ( oldPhi[czpp] - oldPhi[czp] ) / ( oldPhi[czp] - oldPhi[c] );
        Sup = oldPhi[czp];
        Sdn = oldPhi[c]; 
      } else { 
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      }

      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }

    F.t = Sup + 0.5*psi*( Sdn - Sup ); 

    // BOTTOM 
    if ( zminus && c.z() == cLow.z() ) { 
      // Boundary condition 
      // resorting to upwind @ boundary 
      if ( zmpVel > 0.0 ) { 
        Sup = oldPhi[czm]; 
        Sdn = 0.0; 
        psi = 0.0;  
      } else if ( zmpVel < 0.0 ) { 
        Sup = oldPhi[c]; 
        Sdn = 0.0; 
        psi = 0.0; 
      } else {
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      } 
    } else { 
      if ( zmpVel > 0.0 ) {
        Sup = oldPhi[czm];
        Sdn = oldPhi[c];
        r = ( oldPhi[czm] - oldPhi[czmm] ) / ( oldPhi[c] - oldPhi[czm] ); 
      } else if ( zmpVel > 0.0 ) {
        Sup = oldPhi[c];
        Sdn = oldPhi[czm];
        r = ( oldPhi[czp] - oldPhi[c] ) / ( oldPhi[c] - oldPhi[czm] );
      } else {  
        Sup = 0.0;
        Sdn = 0.0; 
        psi = 0.0;
      }
      psi = max( min(2.0*r, 1.0), min(r, 2.0) );
      psi = max( 0.0, psi );
    }

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
template <class fT, class oldPhiT, class gammaT > void
DQMOMEqn::computeDiff( const Patch* p, fT& Fdiff, oldPhiT& oldPhi, gammaT& gamma )
{
  // --- compute diffusion term ---
  Vector Dx = p->dCell();

  FaceData<double> F;
  FaceData<double> G;

  for (CellIterator iter=p->getCellIterator__New(); !iter.done(); iter++){

    IntVector c = *iter; 

    // Get gradient and interpolated diffusion coef.     
    interpPtoF( gamma, c, F ); 
    gradPtoF( oldPhi, c, p, G ); 

    Fdiff[c] = Dx.y()*Dx.z()*(F.e*G.e - F.w*G.w)/d_turbPrNo;           
#ifdef YDIM
    Fdiff[c] += Dx.x()*Dx.z()*(F.n*G.n - F.s*G.s)/d_turbPrNo;      
#endif  
#ifdef ZDIM 
    Fdiff[c] += Dx.y()*Dx.z()*(F.t*G.t - F.b*G.b)/d_turbPrNo;      
#endif
  } 

}
//---------------------------------------------------------------------------
// Method: Compute the boundary conditions. 
//---------------------------------------------------------------------------
template<class phiType> void
DQMOMEqn::computeBCs( const Patch* p, 
                       string varName,
                       phiType& phi )
{
  //d_boundaryCond->setScalarValueBC( 0, patch, phi, varName ); 
  // Going to hard code these until I figure out a better way to handle
  // the boundary conditions
  bool xminus = p->getBCType(Patch::xminus) != Patch::Neighbor;
  //bool xplus =  p->getBCType(Patch::xplus) != Patch::Neighbor;
  //bool yminus = p->getBCType(Patch::yminus) != Patch::Neighbor;
  //bool yplus =  p->getBCType(Patch::yplus) != Patch::Neighbor;
  //bool zminus = p->getBCType(Patch::zminus) != Patch::Neighbor;
  //bool zplus =  p->getBCType(Patch::zplus) != Patch::Neighbor;

  double inletR = 0.007112;
  double boundaryValue = d_initValue;
  Vector cent(0,.1556,.1556); 

  IntVector cLow  = p->getCellLowIndex__New(); 
  IntVector cHigh = p->getCellHighIndex__New();  

  if (xminus) {
    int cx = cLow.x() - 1; 
    for(  int cy = cLow.y(); cy < cHigh.y(); cy += 1)
    {
      for(  int cz = cLow.z(); cz < cHigh.z(); cz += 1)
      {

        IntVector c(cx,cy,cz);
        Point mypoint = p->cellPosition(c); 
        double myradius = pow( mypoint.y() - cent.y(), 2.0 );
        myradius += pow( mypoint.z() - cent.z(), 2.0 ); 
        myradius = pow( myradius, 1.0/2.0 ); 

        if ( myradius < inletR ) 
          phi[c] = boundaryValue; 

      }
    }
  }
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
//---------------------------------------------------------------------------
// Method: Clip the scalar 
//---------------------------------------------------------------------------
template<class phiType> void
DQMOMEqn::clipPhi( const Patch* p, 
                       phiType& phi )
{
  // probably should put these "if"s outside the loop   
  for (CellIterator iter=p->getCellIterator__New(0); !iter.done(); iter++){

    IntVector c = *iter; 

    if (d_doLowClip) {
      if (phi[c] < d_lowClip) 
        phi[c] = d_lowClip; 
    }

    if (d_doHighClip) { 
      if (phi[c] > d_highClip) 
        phi[c] = d_highClip; 
    } 
  }
}

