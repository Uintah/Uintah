/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/OnTheFlyAnalysis/planeAverage.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>

#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#define ALL_LEVELS 99
#define FINEST_LEVEL -1
using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "planeAverage:+"
Dout dbg_OTF_PA("planeAverage", "OnTheFlyAnalysis", "planeAverage debug stream", false);

MPI_Comm planeAverage::d_my_MPI_COMM_WORLD;

//______________________________________________________________________
/*
     This module computes the spatial average of a variable over a plane
TO DO:
    - add delT to now.

Optimization:
    - only define CC_pos once.
    - only compute sum of the weight once (planarSum_weight)
______________________________________________________________________*/

planeAverage::planeAverage( const ProcessorGroup    * myworld,
                            const MaterialManagerP    materialManager,
                            const ProblemSpecP      & module_spec,
                            const bool                parse_ups_vars,
                            const bool                writeOutput,
                            const int                 ID )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_className ="planeAverage_" + to_string(ID);

  d_matl_set  = nullptr;
  d_zero_matl = nullptr;
  d_lb        = scinew planeAverageLabel();

  d_parse_ups_variables     = parse_ups_vars;
  d_writeOutput             = writeOutput;

  d_lb->lastCompTimeName    = "lastCompTime_planeAvg" + to_string(ID);
  d_lb->lastCompTimeLabel   =  VarLabel::create( d_lb->lastCompTimeName, max_vartype::getTypeDescription() );

  d_lb->fileVarsStructName   = "FileInfo_planeAvg" + to_string(ID);
  d_lb->fileVarsStructLabel = VarLabel::create( d_lb->fileVarsStructName, PerPatch<FileInfoP>::getTypeDescription() );

  d_allLevels_planarVars.resize( d_MAXLEVELS );

  d_progressVar.resize( N_TASKS );
  for (auto i =0;i<N_TASKS; i++){
    d_progressVar[i].resize( d_MAXLEVELS, false );
  }
}

//__________________________________
planeAverage::~planeAverage()
{
  DOUT(dbg_OTF_PA, " Doing: destorying "<< d_className );
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
   if(d_zero_matl && d_zero_matl->removeReference()) {
    delete d_zero_matl;
  }

  VarLabel::destroy(d_lb->lastCompTimeLabel);
  VarLabel::destroy(d_lb->fileVarsStructLabel);

  delete d_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void planeAverage::problemSetup(const ProblemSpecP&,
                                const ProblemSpecP&,
                                GridP& grid,
                                std::vector<std::vector<const VarLabel* > > &PState,
                                std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  DOUT(dbg_OTF_PA , "Doing problemSetup \t\t\t\t" << d_className );
  
  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", m_analysisFreq);
  m_module_spec->require("timeStart",         d_startTime);
  m_module_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("planeAverage: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  //__________________________________
  // find the material to extract data from.  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  if(m_module_spec->findBlock("material") ){
    d_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  } else {
    d_matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = d_matl->getDWIndex();

  vector<int> m;
  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);

  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  //Construct the matl_set
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // for fileInfo variable
  d_zero_matl = scinew MaterialSubset();
  d_zero_matl->add(0);
  d_zero_matl->addReference();


  //__________________________________
  //  Plane orientation
  string orient;
  m_module_spec->require("planeOrientation", orient);
  if ( orient == "XY" ){
    d_planeOrientation = XY;
  } else if ( orient == "XZ" ) {
    d_planeOrientation = XZ;
  } else if ( orient == "YZ" ) {
    d_planeOrientation = YZ;
  }

  //__________________________________
  //  OPTIONAL weighting label
  map<string,string> attribute;
  ProblemSpecP w_ps = m_module_spec->findBlock( "weight" );

  if( w_ps ) {
    w_ps->getAttributes( attribute );
    string labelName  = attribute["label"];
    d_lb->weightLabel = VarLabel::find( labelName );

    if( d_lb->weightLabel == nullptr ){
      throw ProblemSetupException("planeAverage: weight label not found: " + labelName , __FILE__, __LINE__);
    }
  }

  if ( d_parse_ups_variables ) {                        // the MeanTurbFluxes module defines the planarVars
    //__________________________________
    //  Now loop over all the variables to be analyzed

    std::vector< std::shared_ptr< planarVarBase >  >planarVars;

    for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {

      var_spec->getAttributes( attribute );

      //__________________________________
      // Read in the variable name
      string labelName = attribute["label"];
      VarLabel* label = VarLabel::find(labelName);
      if( label == nullptr ){
        throw ProblemSetupException("planeAverage: analyze label not found: " + labelName , __FILE__, __LINE__);
      }

      //__________________________________
      //  read in the weighting type for this variable
      weightingType weight = NONE;

      string w = attribute["weighting"];
      if(w == "nCells" ){
        weight = NCELLS;
      }else if( w == "mass" ){
        weight = MASS;
      }

      // Bulletproofing - The user must specify the matl for single matl
      // variables
      if ( labelName == "press_CC" && attribute["matl"].empty() ){
        throw ProblemSetupException("planeAverage: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
      }

      // Read in the optional level index
      int level = ALL_LEVELS;
      if (attribute["level"].empty() == false){
        level = atoi(attribute["level"].c_str());
      }

      //  Read in the optional material index
      int matl = defaultMatl;
      if (attribute["matl"].empty() == false){
        matl = atoi(attribute["matl"].c_str());
      }

      //__________________________________
      bool throwException = false;

      const TypeDescription* td = label->typeDescription();
      const TypeDescription* subtype = td->getSubType();

      const TypeDescription::Type baseType = td->getType();
      const TypeDescription::Type subType  = subtype->getType();

      // only CC, SFCX, SFCY, SFCZ variables
      if(baseType != TypeDescription::CCVariable &&
         baseType != TypeDescription::SFCXVariable &&
         baseType != TypeDescription::SFCYVariable &&
         baseType != TypeDescription::SFCZVariable ){
         throwException = true;
      }
      // CC Variables, only Doubles and Vectors
      if(baseType != TypeDescription::CCVariable &&
         subType  != TypeDescription::double_type &&
         subType  != TypeDescription::Vector  ){
        throwException = true;
      }
      // Face Centered Vars, only Doubles
      if( (baseType == TypeDescription::SFCXVariable ||
           baseType == TypeDescription::SFCYVariable ||
           baseType == TypeDescription::SFCZVariable) &&
           subType != TypeDescription::double_type) {
        throwException = true;
      }

      if(throwException){
        ostringstream warn;
        warn << "ERROR:AnalysisModule:planeAverage: ("<<label->getName() << " "
             << td->getName() << " ) has not been implemented" << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }

      //__________________________________
      //  populate the vector of averages
      // double
      if( subType == TypeDescription::double_type ) {
        planarVar_double* me = new planarVar_double();
        me->label      = label;
        me->matl       = matl;
        me->level      = level;
        me->baseType   = baseType;
        me->subType    = subType;
        me->weightType = weight;

        planarVars.push_back( std::shared_ptr<planarVarBase>(me) );

      }
      // Vectors
      if( subType == TypeDescription::Vector ) {
        planarVar_Vector* me = new planarVar_Vector();
        me->label      = label;
        me->matl       = matl;
        me->level      = level;
        me->baseType   = baseType;
        me->subType    = subType;
        me->weightType = weight;

        planarVars.push_back( std::shared_ptr<planarVarBase>(me) );
      }
    }
    d_allLevels_planarVars.at(0) = planarVars;
  }
}

//______________________________________________________________________
void planeAverage::scheduleInitialize(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,dbg_OTF_PA, d_className + "::scheduleInitialize");

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior(d_lb->fileVarsStructName, false, false, false, true, true);

  // no checkpointing
  sched->overrideVariableBehavior(d_lb->lastCompTimeName, false, false, false, false, true);
  
  //__________________________________
  //
  Task* t = scinew Task("planeAverage::initialize",
                  this, &planeAverage::initialize);

  t->setType( Task::OncePerProc );
  t->computes(d_lb->lastCompTimeLabel );

  if( d_writeOutput ){
    t->computes(d_lb->fileVarsStructLabel, d_zero_matl);
  }

  const PatchSet * perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  sched->addTask(t, perProcPatches,  d_matl_set);

}

//______________________________________________________________________
void planeAverage::initialize(const ProcessorGroup  *,
                              const PatchSubset     * patches,
                              const MaterialSubset  *,
                              DataWarehouse         *,
                              DataWarehouse         * new_dw)
{

  // With multiple levels a rank may not own any patches
  if(patches->size() == 0 ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask( patch, dbg_OTF_PA,"Doing "+ d_className + "::initialize 1/2");

    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put(max_vartype(tminus), d_lb->lastCompTimeLabel );


    //__________________________________
    //  initialize fileInfo struct
    if( d_writeOutput && patch->getGridIndex() == 0 ){
      PerPatch<FileInfoP> fileInfo;
      FileInfo* myFileInfo = scinew FileInfo();
      fileInfo.get() = myFileInfo;

      new_dw->put(fileInfo,    d_lb->fileVarsStructLabel, 0, patch);

      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:planeAverage  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }

  //__________________________________
  //  reserve space for each of the VarLabels and weighting vars
  const LevelP level = getLevelP(patches);
  int L_indx = level->getIndex();

  if ( d_progressVar[INITIALIZE][L_indx] == true ){
    return;
  }

   printTask( patches, dbg_OTF_PA,"Doing "+ d_className + "::initialize 2/2");

  //__________________________________
  // on this level and rank create a deep copy of level 0 planarVars vector
  // This task is called by all levels
  std::vector< std::shared_ptr< planarVarBase > > L0_vars = d_allLevels_planarVars.at(0);
  std::vector< std::shared_ptr< planarVarBase > > planarVars;

  for( unsigned int i= 0; i<L0_vars.size(); i++){
    planarVarBase* me = L0_vars[i].get();
    planarVars.push_back( me->clone() );
  }

  //  Loop over variables */
  for (unsigned int i =0 ; i < planarVars.size(); i++) {

    const TypeDescription::Type td = planarVars[i]->baseType;

    IntVector EC  = IntVector( 2,2,2 ) * level->getExtraCells();    
    IntVector L_lo_EC;      // includes extraCells
    IntVector L_hi_EC;

    level->computeVariableExtents( td, L_lo_EC, L_hi_EC );

    int nPlanes = 0;
    switch( d_planeOrientation ){
      case XY:{
        nPlanes = L_hi_EC.z() - L_lo_EC.z() - EC.z();   // subtract EC for interior cells
        break;
      }
      case XZ:{
        nPlanes = L_hi_EC.y() - L_lo_EC.y() - EC.y();
        break;
      }
      case YZ:{
        nPlanes = L_hi_EC.x() - L_lo_EC.x() - EC.x();
        break;
      }
      default:
        break;
    }

    planarVars[i]->level = L_indx;
    planarVars[i]->set_nPlanes(nPlanes);     // number of planes that will be averaged
    planarVars[i]->reserve();                // reserve space for the planar variables
  }

  d_allLevels_planarVars.at(L_indx) = planarVars;
  d_progressVar[INITIALIZE][L_indx] = true;
}

//______________________________________________________________________
void planeAverage::scheduleRestartInitialize(SchedulerP   & sched,
                                             const LevelP & level)
{
  scheduleInitialize( sched, level);
}


//______________________________________________________________________
void planeAverage::sched_computePlanarAve(SchedulerP   & sched,
                                          const LevelP & level)
{
  printSchedule(level,dbg_OTF_PA, d_className + "::sched_computePlanarAve");

  sched_zeroPlanarVars(    sched, level);

  sched_computePlanarSums( sched, level);

  sched_sumOverAllProcs(   sched, level);

}

//______________________________________________________________________
void planeAverage::scheduleDoAnalysis(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,dbg_OTF_PA, d_className + "::scheduleDoAnalysis");

  // schedule tasks that calculate the planarAve
  sched_computePlanarAve( sched, level );

  sched_writeToFiles(     sched, level, "planeAverage" );

  sched_resetProgressVar( sched, level );

  //__________________________________
  //  Not all ranks own patches, need custom MPI communicator
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  createMPICommunicator( perProcPatches );
}

//______________________________________________________________________
//  This task is a set the variables sum = 0 for each variable type
//
void planeAverage::sched_zeroPlanarVars(SchedulerP   & sched,
                                        const LevelP & level)
{
  //__________________________________
  //  Task to zero the summed variables;
  printSchedule(level,dbg_OTF_PA, d_className + "::sched_zeroPlanarVars");

  Task* t = scinew Task( "planeAverage::zeroPlanarVars",
                     this,&planeAverage::zeroPlanarVars );

  t->setType( Task::OncePerProc );
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);

  sched_TimeVars( t, level, d_lb->lastCompTimeLabel, false );
  
  sched->addTask( t, perProcPatches, d_matl_set );
}

//______________________________________________________________________
//
void planeAverage::zeroPlanarVars(const ProcessorGroup * ,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        * old_dw,
                                  DataWarehouse        *)
{
  // With multiple levels a rank may not own any patches
  if( patches->empty() ){
    return;
  }

  //__________________________________
  // zero variables if it's time to write and if this MPIRank hasn't excuted this
  // task on this level
  const Level* level = getLevel( patches );
  const int L_indx = level->getIndex();

  if (d_progressVar[ZERO][L_indx] == true || isItTime( old_dw, level, d_lb->lastCompTimeLabel) == false){
    return;
  }

  printTask( patches, dbg_OTF_PA,"Doing " + d_className + "::zeroPlanarVars" );

  //__________________________________
  // Loop over variables
  std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

  for (unsigned int i =0 ; i < planarVars.size(); i++) {
    std::shared_ptr<planarVarBase> analyzeVar = planarVars[i];
    analyzeVar->zero_all_vars();
  }
  d_progressVar[ZERO][L_indx] = true;
}

//______________________________________________________________________
//  Computes planar sum of each variable type
//
void planeAverage::sched_computePlanarSums(SchedulerP   & sched,
                                           const LevelP & level)
{
  //__________________________________
  //  compute the planar sums task;
  printSchedule( level, dbg_OTF_PA, d_className +"::sched_computePlanarSums" );

  Task* t = scinew Task( "planeAverage::computePlanarSums",
                     this,&planeAverage::computePlanarSums );

  t->setType( Task::OncePerProc );


  sched_TimeVars( t, level, d_lb->lastCompTimeLabel, false );
  
  Ghost::GhostType gn = Ghost::None;
  if( d_lb->weightLabel != nullptr ){
    t->requires( Task::NewDW, d_lb->weightLabel, gn, 0);
  }

  const int L_indx = level->getIndex();
  std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

  for ( unsigned int i =0 ; i < planarVars.size(); i++ ) {
    VarLabel* label   = planarVars[i]->label;

    // bulletproofing
    if( label == nullptr ){
      string name = label->getName();
      throw InternalError("planeAverage: scheduleDoAnalysis label not found: "
                         + name , __FILE__, __LINE__);
    }

    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( planarVars[i]->matl );
    matSubSet->addReference();

    t->requires( Task::NewDW, label, matSubSet, gn, 0 );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }

  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);

  sched->addTask( t, perProcPatches, d_matl_set );
}

//______________________________________________________________________
//
void planeAverage::computePlanarSums(const ProcessorGroup * pg,
                                     const PatchSubset    * patches,
                                     const MaterialSubset *,
                                     DataWarehouse        * old_dw,
                                     DataWarehouse        * new_dw)
{
  // With multiple levels a rank may not own any patches
  if( patches->empty() ){
    return;
  }
  
  const Level* level = getLevel( patches );
  const int L_indx = level->getIndex();

  // is it time to execute
  if( isItTime( old_dw, level, d_lb->lastCompTimeLabel) == false ){
    return;
  }

  //__________________________________
  // Loop over patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, dbg_OTF_PA, "Doing " + d_className + "::computePlanarSums" );

    //__________________________________
    // Loop over variables
    std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

    for (unsigned int i =0 ; i < planarVars.size(); i++) {
      std::shared_ptr<planarVarBase> analyzeVar = planarVars[i];

      VarLabel* label = planarVars[i]->label;
      string labelName = label->getName();

      // bulletproofing
      if( label == nullptr ){
        throw InternalError("planeAverage: analyze label not found: "
                             + labelName , __FILE__, __LINE__);
      }

      const TypeDescription::Type type    = analyzeVar->baseType;
      const TypeDescription::Type subType = analyzeVar->subType;

      // compute the planar sum of the weighting variable, CC Variable for now
      planarSum_weight <constCCVariable<double>, double > ( new_dw, analyzeVar, patch );

      //__________________________________
      //  compute planarSum for each variable type
      switch( type ){

        // CC Variables
        case TypeDescription::CCVariable: {

          GridIterator iter=patch->getCellIterator();
          switch( subType ) {
            case TypeDescription::double_type:{         // CC double
              planarSum_Q <constCCVariable<double>, double > ( new_dw, analyzeVar, patch, iter );
              break;
            }
            case TypeDescription::Vector: {             // CC Vector
              planarSum_Q< constCCVariable<Vector>, Vector > ( new_dw, analyzeVar, patch, iter );
              break;
            }
            default:
              throw InternalError("planeAverage: invalid data type", __FILE__, __LINE__);
          }
          break;
        }

        case TypeDescription::SFCXVariable: {         // SFCX double
          GridIterator iter=patch->getSFCXIterator();
          planarSum_Q <constSFCXVariable<double>, double > ( new_dw, analyzeVar, patch, iter );
          break;
        }
        case TypeDescription::SFCYVariable: {         // SFCY double
          GridIterator iter=patch->getSFCYIterator();
          planarSum_Q <constSFCYVariable<double>, double > ( new_dw, analyzeVar, patch, iter );
          break;
        }
        case TypeDescription::SFCZVariable: {         // SFCZ double
          GridIterator iter=patch->getSFCZIterator();
          planarSum_Q <constSFCZVariable<double>, double > ( new_dw, analyzeVar, patch, iter );
          break;
        }
        default:
          ostringstream mesg;
          mesg << "ERROR:AnalysisModule:planeAverage: ("<< label->getName() << " "
               << label->typeDescription()->getName() << " ) has not been implemented" << endl;
          throw InternalError(mesg.str(), __FILE__, __LINE__);
      }
    }  // VarLabel loop
  }  // patches
}


//______________________________________________________________________
//  Find the sum of the weight for this rank  CCVariables for now
template <class Tvar, class Ttype>
void planeAverage::planarSum_weight( DataWarehouse * new_dw,
                                     std::shared_ptr< planarVarBase > analyzeVar,
                                     const Patch   * patch )
{

  Tvar weight;
  if ( analyzeVar->weightType == MASS ){
    int indx = analyzeVar->matl;
    new_dw->get(weight, d_lb->weightLabel, indx, patch, Ghost::None, 0);
  }

  std::vector<Ttype> local_weight_sum;    // local planar sum of the weight
  std::vector<Ttype> proc_weight_sum;     // planar sum already computed on this proc
  std::vector<int>   local_nCells_sum;    // With AMR this could change for each plane
  std::vector<int>   proc_nCells_sum;

  analyzeVar->getPlanarWeight( proc_weight_sum, proc_nCells_sum );

  const int nPlanes = analyzeVar->get_nPlanes();
  local_weight_sum.resize( nPlanes, 0 );
  local_nCells_sum.resize( nPlanes, 0 );

  IntVector lo;
  IntVector hi;
  GridIterator iter=patch->getCellIterator();
  planeIterator( iter, lo, hi );

  for ( auto z = lo.z(); z<hi.z(); z++ ) {          // This is the loop over all planes for this patch

    Ttype sum( 0 );                                 // initial values on this plane/patch
    int nCells = 0;

    for ( auto y = lo.y(); y<hi.y(); y++ ) {        // cells in the plane
      for ( auto x = lo.x(); x<hi.x(); x++ ) {
        nCells += 1;

        if ( analyzeVar->weightType == MASS ){
          IntVector c(x,y,z);
          c = transformCellIndex(x, y, z);
          sum = sum + weight[c];
        }

      }
    }
    local_weight_sum[z] = sum;
    local_nCells_sum[z] = nCells;
  }

  //__________________________________
  //  Add to the existing sums
  //  A proc could have more than 1 patch
  for ( auto z = lo.z(); z<hi.z(); z++ ) {
    proc_weight_sum[z] += local_weight_sum[z];
    proc_nCells_sum[z] += local_nCells_sum[z];
  }

  analyzeVar->setPlanarWeight( proc_weight_sum, proc_nCells_sum );
}


//______________________________________________________________________
//  Find the sum of Q for each plane on this rank
template <class Tvar, class Ttype>
void planeAverage::planarSum_Q( DataWarehouse * new_dw,
                                std::shared_ptr< planarVarBase > analyzeVar,
                                const Patch   * patch,
                                GridIterator    iter )
{
  int indx = analyzeVar->matl;

  const VarLabel* varLabel = analyzeVar->label;

  Ttype zero = Ttype(0.);

  Tvar Q_var;
  new_dw->get(Q_var, varLabel, indx, patch, Ghost::None, 0);

  std::vector<Ttype> local_Q_sum;    // Q_sum over all cells in the plane
  std::vector<Ttype> proc_Q_sum;     // planar Q_sum already computed on this proc
  std::vector<Point> local_CC_pos;   // cell centered position

  analyzeVar->getPlanarSum( proc_Q_sum );

  const int nPlanes = analyzeVar->get_nPlanes();
  local_Q_sum.resize( nPlanes, zero );
  local_CC_pos.resize( nPlanes, Point(-DBL_MAX, -DBL_MAX, -DBL_MAX) );

  //__________________________________
  // compute mid point of the level
  const Level* level = patch->getLevel();
  IntVector L_lo;
  IntVector L_hi;
  level->findInteriorCellIndexRange( L_lo, L_hi );
  IntVector L_midPt = Uintah::roundNearest( ( L_hi - L_lo ).asVector()/2.0 );

  IntVector plane_midPt = transformCellIndex( L_midPt.x(), L_midPt.y(), L_midPt.z() );

  IntVector lo;
  IntVector hi;
  planeIterator( iter, lo, hi );

  for ( auto z = lo.z(); z<hi.z(); z++ ) {          // This is the loop over all planes for this patch

    Ttype Q_sum( 0 );  // initial value

    for ( auto y = lo.y(); y<hi.y(); y++ ) {        // cells in the plane
      for ( auto x = lo.x(); x<hi.x(); x++ ) {
        IntVector c(x,y,z);

        c = transformCellIndex(x, y, z);
        Q_sum = Q_sum + Q_var[c];
      }
    }

    // cell-centered position
    IntVector here = transformCellIndex( plane_midPt.x(), plane_midPt.y(), z );

    local_CC_pos[z] = patch->cellPosition( here );
    local_Q_sum[z]  = Q_sum;
  }

  //__________________________________
  //  Add this patch's contribution of Q_sum to existing Q_sum
  //  A proc could have more than 1 patch
  for ( auto z = lo.z(); z<hi.z(); z++ ) {
    proc_Q_sum[z] += local_Q_sum[z];
  }

  // CC_positions and planar sum on this rank
  analyzeVar->setCC_pos( local_CC_pos, lo.z(), hi.z() );
  analyzeVar->setPlanarSum( proc_Q_sum );

}

//______________________________________________________________________
//  This task performs a reduction (sum) over all ranks
void planeAverage::sched_sumOverAllProcs( SchedulerP   & sched,
                                          const LevelP & level)
{
  //__________________________________
  //  Call MPI reduce on all variables;
  printSchedule( level, dbg_OTF_PA, d_className + "::sched_sumOverAllProcs" );

  Task* t = scinew Task( "planeAverage::sumOverAllProcs",
                     this,&planeAverage::sumOverAllProcs );

  t->setType( Task::OncePerProc );

  sched_TimeVars( t, level, d_lb->lastCompTimeLabel, false );
    
  // only compute task on 1 patch in this proc
  const PatchSet* perProcPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);

  sched->addTask( t, perProcPatches, d_matl_set );
}

//______________________________________________________________________
//
void planeAverage::sumOverAllProcs(const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset *,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw)
{
  // With multiple levels a rank may not own any patches
  if( patches->empty() ){
    return;
  }
  
  const Level * level = getLevel( patches );
  const int L_indx    = level->getIndex();
  
  // is it time to execute
  if( isItTime( old_dw, level, d_lb->lastCompTimeLabel) == false){
    return;
  }

  // Has this rank performed the reduction
  if( d_progressVar[SUM][L_indx] == true){
    return;
  }

  printTask( patches, dbg_OTF_PA,"Doing " + d_className + "::sumOverAllProcs");

  //__________________________________
  // Loop over variables
  std::vector< std::shared_ptr< planarVarBase > >planarVars = d_allLevels_planarVars[L_indx];

  for (unsigned int i =0 ; i < planarVars.size(); i++) {

    std::shared_ptr<planarVarBase> analyzeVar = planarVars[i];

    int rank = pg->myRank();
    analyzeVar->ReduceCC_pos( rank );
    analyzeVar->ReduceBcastWeight( rank );
    analyzeVar->ReduceBcastVar( rank );

  }  // loop over planarVars
  d_progressVar[SUM][L_indx] = true;
}

//______________________________________________________________________
//  This task writes out the plane average of each VarLabel to a separate file.

void planeAverage::sched_writeToFiles(SchedulerP   &    sched,
                                      const LevelP &    level,
                                      const std::string  dirName)
{
  //__________________________________
  //  Task that writes averages to files
  // Only write data on patch 0 on each level
  printSchedule(level,dbg_OTF_PA, d_className +"::writeToFiles");

  Task* t = scinew Task( "planeAverage::writeToFiles",
                     this,&planeAverage::writeToFiles,
                     dirName );

  sched_TimeVars( t, level, d_lb->lastCompTimeLabel, true );
    
  t->requires( Task::OldDW, d_lb->fileVarsStructLabel, d_zero_matl, Ghost::None, 0 );
  t->computes( d_lb->fileVarsStructLabel, d_zero_matl );

  // first patch on this level
  const Patch* p = level->getPatch(0);
  PatchSet* zeroPatch = new PatchSet();
  zeroPatch->add(p);
  zeroPatch->addReference();

  sched->addTask( t, zeroPatch , d_matl_set );

  if (zeroPatch && zeroPatch->removeReference()) {
    delete zeroPatch;
  }
}

//______________________________________________________________________
//
void planeAverage::writeToFiles(const ProcessorGroup* pg,
                                const PatchSubset   * patches,
                                const MaterialSubset*,
                                DataWarehouse       * old_dw,
                                DataWarehouse       * new_dw,
                                const std::string     dirName )
{
  //const LevelP level = getLevelP( patches );
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  timeVars tv;  
  getTimeVars( old_dw, level, d_lb->lastCompTimeLabel, tv );
  putTimeVars( new_dw, d_lb->lastCompTimeLabel, tv );
  
  if( tv.isItTime == false ){
    return;
  }
  
  //__________________________________
  //
  for(int p=0;p<patches->size();p++){                  // IS THIS LOOP NEEDED?? Todd
    const Patch* patch = patches->get(p);

    // open the struct that contains a map of the file pointers
    // Note: after regridding this may not exist for this patch in the old_dw
    PerPatch<FileInfoP> fileInfo;

    if( old_dw->exists( d_lb->fileVarsStructLabel, 0, patch ) ){
      old_dw->get( fileInfo, d_lb->fileVarsStructLabel, 0, patch );
    }else{
      FileInfo* myFileInfo = scinew FileInfo();
      fileInfo.get() = myFileInfo;
    }

    std::map<string, FILE *> myFiles;

    if( fileInfo.get().get_rep() ){
      myFiles = fileInfo.get().get_rep()->files;
    }

    int proc = m_scheduler->getLoadBalancer()->getPatchwiseProcessorAssignment(patch);

    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write.  With AMR data the proc
    // may not own the patch
    if( proc == pg->myRank() ){

      printTask( patch, dbg_OTF_PA,"Doing " + d_className + "::writeToFiles" );


      std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

      for (unsigned int i =0 ; i < planarVars.size(); i++) {
        VarLabel* label = planarVars[i]->label;
        string labelName = label->getName();

        //__________________________________
        // create the directory structure including sub directories
        string udaDir = m_output->getOutputLocation();

        timeStep_vartype timeStep_var;
        old_dw->get( timeStep_var, m_timeStepLabel );
        int ts = timeStep_var;

        ostringstream tname;
        tname << "t" << std::setw(5) << std::setfill('0') << ts;
        string timestep = tname.str();

        ostringstream li;
        li<<"L-"<<level->getIndex();
        string levelIndex = li.str();

        string path = dirName + "/" + timestep + "/" + levelIndex;

        if( d_isDirCreated.count( path ) == 0 ){
          createDirectory( 0777, udaDir,  path );
          d_isDirCreated.insert( path );
        }

        ostringstream fname;
        fname << udaDir << "/" << path << "/" << labelName << "_" << planarVars[i]->matl;
        string filename = fname.str();

        //__________________________________
        //  Open the file pointer
        //  if it's not in the fileInfo struct then create it
        FILE *fp;
        if( myFiles.count(filename) == 0 ){
          createFile( filename, fp, levelIndex );
          myFiles[filename] = fp;

        } else {
          fp = myFiles[filename];
        }
        if (!fp){
          throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed opening file: "+filename,__FILE__, __LINE__);
        }

        //__________________________________
        //  Write to the file
        planarVars[i]->printAverage( fp, L_indx, tv.now );

        fflush(fp);
      }  // planarVars loop
    }  // time to write data

    // Put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo, d_lb->fileVarsStructLabel, 0, patch);
  }  // patches

#if 0
  //__________________________________
  //  reset the progress variable to false
  std::vector<bool> zero( d_MAXLEVELS, false );
  for (auto i =0;i<N_TASKS; i++){
    d_progressVar[i]=zero;
  }
#endif
}

//______________________________________________________________________
//  resetProgressVar;
void planeAverage::sched_resetProgressVar( SchedulerP   & sched,
                                           const LevelP & level )
{
  //__________________________________
  //
  printSchedule(level,dbg_OTF_PA, d_className + "::resetProgressVar");

  Task* t = scinew Task( "planeAverage::resetProgressVar",
                     this,&planeAverage::resetProgressVar );

  t->setType( Task::OncePerProc );

  // only compute task on 1 patch in this proc
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);

  sched->addTask( t, perProcPatches, d_matl_set );
}

//______________________________________________________________________
//  This task is a set the variables sum = 0 for each variable type
//
void planeAverage::resetProgressVar(const ProcessorGroup * ,
                                  const PatchSubset    * patches,
                                  const MaterialSubset *,
                                  DataWarehouse        * old_dw,
                                  DataWarehouse        *)
{
  // With multiple levels a rank may not own any patches
  if( patches->empty() ){
    return;
  }

  const LevelP level = getLevelP( patches );
  const int L_indx = level->getIndex();

  printTask( patches, dbg_OTF_PA,"Doing " + d_className + "::resetProgressVar" );

  for (unsigned int i =0 ; i < d_progressVar.size(); i++) {
    d_progressVar[i][L_indx] = false;
  }
}

//______________________________________________________________________
//  In the multi-level grid not every rank owns a PatchSet.
void planeAverage::createMPICommunicator(const PatchSet* perProcPatches)
{
  int rank = d_myworld->myRank();
  const PatchSubset* myPatches = perProcPatches->getSubset( rank );

  printTask( myPatches, dbg_OTF_PA,"Doing " + d_className + "::createMPICommunicator");

  int color = 1;

  if ( myPatches->empty() ){
    color = 0;
    ostringstream msg;
    msg<< "    PlaneAverage:  No patches on rank " << rank << " removing rank from MPI communicator";
    DOUT(true, msg.str() );
  }

  MPI_Comm_split( d_myworld->getComm(), color, rank, &d_my_MPI_COMM_WORLD );
}


//______________________________________________________________________
//  Open the file if it doesn't exist
void planeAverage::createFile(string  & filename,
                              FILE*   & fp,
                              string  & levelIndex)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }

  fp = fopen(filename.c_str(), "w");

  if (!fp){
    perror("Error opening file:");
    throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed opening file: " + filename,__FILE__, __LINE__);
  }

  cout << "OnTheFlyAnalysis planeAverage results are located in " << filename << endl;
}

//______________________________________________________________________
// create a series of sub directories below the rootpath.
int
planeAverage::createDirectory( mode_t mode,
                               const std::string & rootPath,
                               std::string       & subDirs )
{
  struct stat st;

  DOUT( dbg_OTF_PA, d_myworld->myRank() << " planeAverage:Making directory " << subDirs << "\n" );

  for( std::string::iterator iter = subDirs.begin(); iter != subDirs.end(); ){

    string::iterator newIter = std::find( iter, subDirs.end(), '/' );
    std::string newPath = rootPath + "/" + std::string( subDirs.begin(), newIter);

    // does path exist
    if( stat( newPath.c_str(), &st) != 0 ){

      int rc = mkdir( newPath.c_str(), mode);

      // bulletproofing
      if(  rc != 0 && errno != EEXIST ){
        cout << "cannot create folder [" << newPath << "] : " << strerror(errno) << endl;
        throw InternalError("\nERROR:dataAnalysisModule:planeAverage:  failed creating dir: "+newPath,__FILE__, __LINE__);
      }
    }
    else {
      if( !S_ISDIR( st.st_mode ) ){
        errno = ENOTDIR;
        cout << "path [" << newPath << "] not a dir " << endl;
        return -1;
      } else {
        //cout << "path [" << newPath << "] already exists " << endl;
      }
    }

    iter = newIter;
    if( newIter != subDirs.end() ){
      ++ iter;
    }
  }
  return 0;
}

//______________________________________________________________________
// transform cell index back to original orientation
IntVector planeAverage::transformCellIndex(const int i,
                                           const int j,
                                           const int k)
{
  IntVector c(-9,-9,-9);
  switch( d_planeOrientation ){
    case XY:{                   // z is constant
      c = IntVector( i,j,k );
      break;
    }
    case XZ:{                   // y is constant
      c = IntVector( i,k,j );
      break;
    }
    case YZ:{                   // x is constant
      c = IntVector( k,i,j );
      break;
    }
    default:
      break;
  }
  return c;
}


//______________________________________________________________________
//  Returns an index range for a plane
void planeAverage::planeIterator( const GridIterator& patchIter,
                                  IntVector & lo,
                                  IntVector & hi )
{
  IntVector patchLo = patchIter.begin();
  IntVector patchHi = patchIter.end();

  switch( d_planeOrientation ){
    case XY:{                 // z is constant
      lo = patchLo;           // Iterate over x, y cells
      hi = patchHi;
      break;
    }
    case XZ:{                 // y is constant
      lo.x( patchLo.x() );    // iterate over x and z cells
      lo.y( patchLo.z() );
      lo.z( patchLo.y() );

      hi.x( patchHi.x() );
      hi.y( patchHi.z() );
      hi.z( patchHi.y() );
      break;
    }
    case YZ:{                 // x is constant
      lo.x( patchLo.y() );    // iterate over y and z cells
      lo.y( patchLo.z() );
      lo.z( patchLo.x() );

      hi.x( patchHi.y() );
      hi.y( patchHi.z() );
      hi.z( patchHi.x() );
      break;
    }
    default:
      break;
  }
}
