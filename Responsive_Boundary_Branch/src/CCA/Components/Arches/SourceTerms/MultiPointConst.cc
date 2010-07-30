#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/MultiPointConst.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
MultiPointConstBuilder::MultiPointConstBuilder(std::string srcName, 
                                         vector<std::string> reqLabelNames, 
                                         SimulationStateP& sharedState)
: SourceTermBuilder(srcName, reqLabelNames, sharedState)
{}

MultiPointConstBuilder::~MultiPointConstBuilder(){}

SourceTermBase*
MultiPointConstBuilder::build(){
  return scinew MultiPointConst( d_srcName, d_sharedState, d_requiredLabels );
}
// End Builder
//---------------------------------------------------------------------------

MultiPointConst::MultiPointConst( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{}

MultiPointConst::~MultiPointConst()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
MultiPointConst::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  for (ProblemSpecP inject_db = db->findBlock("injector"); inject_db != 0; inject_db = inject_db->findNextBlock("injector")){

    ProblemSpecP geomObj = inject_db->findBlock("geom_object");
    GeometryPieceFactory::create(geomObj, d_geomPieces); 

  }
  db->getWithDefault("constant",d_constant, 0.); 

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
MultiPointConst::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "MultiPointConst::eval";
  Task* tsk = scinew Task(taskname, this, &MultiPointConst::computeSource, timeSubStep);

  if (timeSubStep == 0 && !d_labelSchedInit) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_srcLabel);
  } else {
    tsk->modifies(d_srcLabel); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
MultiPointConst::computeSource( const ProcessorGroup* pc, 
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
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Box patchInteriorBox = patch->getBox(); 

    CCVariable<double> constSrc; 
    if ( new_dw->exists(d_srcLabel, matlIndex, patch ) ){
      new_dw->getModifiable( constSrc, d_srcLabel, matlIndex, patch ); 
      constSrc.initialize(0.0);
    } else {
      new_dw->allocateAndPut( constSrc, d_srcLabel, matlIndex, patch );
      constSrc.initialize(0.0);
    } 

    // not sure which logic is best...
    // currently assuming that the # of geometry pieces is a small # so checking for patch/geometry piece 
    // intersection first rather than putting the cell iterator loop first. That way, we won't loop over every 
    // patch. 

    // loop over all geometry pieces
    for (int gp = 0; gp < d_geomPieces.size(); gp++){

      GeometryPieceP piece = d_geomPieces[gp];
      Box geomBox          = piece->getBoundingBox(); 
      Box b                = geomBox.intersect(patchInteriorBox); 
      
      // patch and geometry intersect
      if ( !( b.degenerate() ) ){

        // loop over all cells
        for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter; 
          
          Point p = patch->cellPosition( *iter );
          if ( piece->inside(p) ) {

            // add constant source if cell is inside geometry piece 
            constSrc[c] += d_constant; 
          }
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
MultiPointConst::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MultiPointConst::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &MultiPointConst::dummyInit);

  tsk->computes(d_srcLabel);

  for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());

}
void 
MultiPointConst::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 


    CCVariable<double> src;

    new_dw->allocateAndPut( src, d_srcLabel, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = d_extraLocalLabels.begin(); iter != d_extraLocalLabels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}

