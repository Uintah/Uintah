/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
static DebugStream do_cout("planeAverage", "OnTheFlyAnalysis", "planeAverage debug stream", false);

//______________________________________________________________________
/*
     This module computes the spatial average of a variable over a plane
TO DO:
    - allow user to control the number of planes
    - add delT to now.

Optimization:
    - only define CC_pos once.
    - only compute sum of the weight once (planarSum_weight)
______________________________________________________________________*/

planeAverage::planeAverage( const ProcessorGroup    * myworld,
                            const MaterialManagerP    materialManager,
                            const ProblemSpecP      & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set  = nullptr;
  d_zero_matl = nullptr;
  d_lb        = scinew planeAverageLabel();
  
  d_lb->lastCompTimeLabel =  VarLabel::create("lastCompTime_planeAvg",
                                              max_vartype::getTypeDescription() );
  d_lb->fileVarsStructLabel = VarLabel::create("FileInfo_planeAvg",
                                               PerPatch<FileInfoP>::getTypeDescription() );

  d_allLevels_planarVars.resize( d_MAXLEVELS );
  
  d_progressVar.resize( N_TASKS );
  for (auto i =0;i<N_TASKS; i++){
    d_progressVar[i].resize( d_MAXLEVELS, false );
  } 
}

//__________________________________
planeAverage::~planeAverage()
{
  do_cout << " Doing: destorying planeAverage " << endl;
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
  do_cout << "Doing problemSetup \t\t\t\tplaneAverage" << endl;

  int numMatls  = m_materialManager->getNumMatls();

  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", d_writeFreq);
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
  } else if (m_module_spec->findBlock("materialIndex") ){
    int indx;
    m_module_spec->get("materialIndex", indx);
    d_matl = m_materialManager->getMaterial(indx);
  } else {
    d_matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = d_matl->getDWIndex();

  //__________________________________
  vector<int> m;
  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);
  map<string,string> attribute;


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
  ProblemSpecP w_ps = m_module_spec->findBlock( "weight" );
  
  if( w_ps ) {
    w_ps->getAttributes( attribute );
    string labelName  = attribute["label"];
    d_lb->weightLabel = VarLabel::find( labelName );
    
    if( d_lb->weightLabel == nullptr ){
      throw ProblemSetupException("planeAverage: weight label not found: " + labelName , __FILE__, __LINE__);
    }
  }
  
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

    //  Read in the optional material index from the variables that
    //  may be different from the default index and construct the
    //  material set
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }

    // Bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("planeAverage: Invalid material index specified for a variable", __FILE__, __LINE__);
    }

    m.push_back(matl);

    //__________________________________
    bool throwException = false;

    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    const TypeDescription::Type baseType = td->getType();
    const TypeDescription::Type subType  = subtype->getType();

    // only CC, SFCX, SFCY, SFCZ variables
    if(baseType != TypeDescription::CCVariable &&
       baseType != TypeDescription::NCVariable &&
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
    // NC Variables, only Doubles and Vectors
    if(baseType != TypeDescription::NCVariable &&
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
  
  // fill with d_MAXLEVELS copies of planarVars
  std::fill_n( d_allLevels_planarVars.begin(), d_MAXLEVELS, planarVars );
  
  //__________________________________
  //
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
}

//______________________________________________________________________
void planeAverage::scheduleInitialize(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,do_cout,"planeAverage::scheduleInitialize");

  Task* t = scinew Task("planeAverage::initialize",
                  this, &planeAverage::initialize);


  t->setType( Task::OncePerProc );
  t->computes(d_lb->lastCompTimeLabel );
  t->computes(d_lb->fileVarsStructLabel, d_zero_matl);
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}

//______________________________________________________________________
void planeAverage::initialize(const ProcessorGroup  *,
                              const PatchSubset     * patches,
                              const MaterialSubset  *,
                              DataWarehouse         *,
                              DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask( patch, do_cout,"Doing planeAverage::initialize 1/2");

    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), d_lb->lastCompTimeLabel );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo,    d_lb->fileVarsStructLabel, 0, patch);

    if( patch->getGridIndex() == 0 ){   // only need to do this once
    
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
   
   printTask( patches, do_cout,"Doing planeAverage::initialize 2/2");

  //  Loop over variables */
  std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];
  
  for (unsigned int i =0 ; i < planarVars.size(); i++) {
    
    const TypeDescription::Type td = planarVars[i]->baseType;

    IntVector L_lo_EC;      // includes extraCells
    IntVector L_hi_EC;

    level->computeVariableExtents( td, L_lo_EC, L_hi_EC );

    int nPlanes = 0;
    switch( d_planeOrientation ){
      case XY:{
        nPlanes = L_hi_EC.z() - L_lo_EC.z() - 2 ;   // subtract 2 for interior cells
        break;
      }
      case XZ:{
        nPlanes = L_hi_EC.y() - L_lo_EC.y() - 2 ;
        break;
      }
      case YZ:{
        nPlanes = L_hi_EC.x() - L_lo_EC.x() - 2 ;
        break;
      }
      default:
        break;
    }
    planarVars[i]->level = L_indx;
    planarVars[i]->set_nPlanes(nPlanes);     // number of planes that will be averaged
    planarVars[i]->reserve();                // reserve space for the planar variables
  }
  d_progressVar[INITIALIZE][L_indx] = true;
}

//______________________________________________________________________
void planeAverage::scheduleRestartInitialize(SchedulerP   & sched,
                                             const LevelP & level)
{
  printSchedule(level,do_cout,"planeAverage::scheduleRestartInitialize");

  Task* t = scinew Task("planeAverage::initialize",
                  this, &planeAverage::initialize);
  
  t->setType( Task::OncePerProc );
  t->computes(d_lb->lastCompTimeLabel );
  t->computes(d_lb->fileVarsStructLabel, d_zero_matl);
  sched->addTask(t, level->eachPatch(),  d_matl_set);
}

//______________________________________________________________________
void
planeAverage::restartInitialize()
{
}
//______________________________________________________________________
void planeAverage::scheduleDoAnalysis(SchedulerP   & sched,
                                      const LevelP & level)
{
  printSchedule(level,do_cout,"planeAverage::scheduleDoAnalysis");

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_planeAvg", false, false, false, true, true);

  Ghost::GhostType gn = Ghost::None;
  const int L_indx = level->getIndex();

  std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];

  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  


  cout << " perProcPatches: " << *perProcPatches << endl;
  //__________________________________
  //  Task to zero the summed variables;
  Task* t0 = scinew Task( "planeAverage::zeroPlanarVars",
                     this,&planeAverage::zeroPlanarVars );

  t0->setType( Task::OncePerProc );
  t0->requires( Task::OldDW, m_simulationTimeLabel );
  t0->requires( Task::OldDW, d_lb->lastCompTimeLabel );
  
  sched->addTask( t0, perProcPatches, d_matl_set );

  //__________________________________
  //  compute the planar sums task;
  Task* t1 = scinew Task( "planeAverage::computePlanarSums",
                     this,&planeAverage::computePlanarSums );

  t0->setType( Task::OncePerProc );
  t1->requires( Task::OldDW, m_simulationTimeLabel );
  t1->requires( Task::OldDW, d_lb->lastCompTimeLabel );

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

    t1->requires( Task::NewDW, label, matSubSet, gn, 0 );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }

  sched->addTask( t1, perProcPatches, d_matl_set );

  //__________________________________
  //  Call MPI reduce on all variables;
  Task* t2 = scinew Task( "planeAverage::sumOverAllProcs",
                     this,&planeAverage::sumOverAllProcs );

  t2->setType( Task::OncePerProc );
  t2->requires( Task::OldDW, m_simulationTimeLabel );
  t2->requires( Task::OldDW, d_lb->lastCompTimeLabel );
  
  // only compute task on 1 patch in this proc
  sched->addTask( t2, perProcPatches, d_matl_set );

  //__________________________________
  //  Task that writes averages to files
  // Only write data on patch 0 on each level

  Task* t3 = scinew Task( "planeAverage::writeToFiles",
                     this,&planeAverage::writeToFiles );

  t3->requires( Task::OldDW, m_simulationTimeLabel );
  t3->requires( Task::OldDW, d_lb->lastCompTimeLabel );
  t3->requires( Task::OldDW, d_lb->fileVarsStructLabel, d_zero_matl, gn, 0 );

  for ( unsigned int i =0 ; i < planarVars.size(); i++ ) {
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( planarVars[i]->matl );
    matSubSet->addReference();
  }

  t3->computes( d_lb->lastCompTimeLabel );
  t3->computes( d_lb->fileVarsStructLabel, d_zero_matl );

  // first patch on this level
  const Patch* p = level->getPatch(0);
  PatchSet* zeroPatch = scinew PatchSet();
  zeroPatch->add(p);
  zeroPatch->addReference();
  
  sched->addTask( t3, zeroPatch , d_matl_set );
}


//______________________________________________________________________
//  This task is a set the variables sum = 0 for each variable type
//
void planeAverage::zeroPlanarVars(const ProcessorGroup * pg,
                                  const PatchSubset    * patches,   
                                  const MaterialSubset *,           
                                  DataWarehouse        * old_dw,    
                                  DataWarehouse        * new_dw)    
{
  //__________________________________
  // zero variables if it's time to write if this MPIRank hasn't excuted this
  // task on this level
  const LevelP level = getLevelP( patches );
  const int L_indx = level->getIndex();
      
  if( isItTime( old_dw ) == false || d_progressVar[ZERO][L_indx] == true){
    return;
  }
  
  printTask( patches, do_cout,"Doing planeAverage::zeroPlanarVars" );

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
void planeAverage::computePlanarSums(const ProcessorGroup * pg,
                                     const PatchSubset    * patches,
                                     const MaterialSubset *,
                                     DataWarehouse        * old_dw,
                                     DataWarehouse        * new_dw)
{

  // Is it the right time and has this MPIRank executed this task on this level?
  const LevelP level = getLevelP( patches );
  const int L_indx = level->getIndex();
  
  if( isItTime( old_dw ) == false || d_progressVar[COMPUTE][L_indx] == true){
    return;
  }
  
  //__________________________________
  // Loop over patches 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, do_cout, "Doing planeAverage::computePlanarSums" );

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
  
  d_progressVar[COMPUTE][L_indx] = true;
}


//______________________________________________________________________
//  Find the sum of the weight for this rank  CCVariables for now
template <class Tvar, class Ttype>
void planeAverage::planarSum_weight( DataWarehouse * new_dw,
                                     std::shared_ptr< planarVarBase > analyzeVar,     
                                     const Patch   * patch )                           
{
  int indx = analyzeVar->matl;

  Tvar weight;
  new_dw->get(weight, d_lb->weightLabel, indx, patch, Ghost::None, 0);
  
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
        IntVector c(x,y,z); 
        
        c = findCellIndex(x, y, z);
        sum = sum + weight[c];
        nCells += 1;
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
  local_CC_pos.resize( nPlanes, Point(0,0,0) );
  
  IntVector lo;
  IntVector hi;
  planeIterator( iter, lo, hi );
  
  for ( auto z = lo.z(); z<hi.z(); z++ ) {          // This is the loop over all planes for this patch
  
    Ttype Q_sum( 0 );  // initial value
      
    for ( auto y = lo.y(); y<hi.y(); y++ ) {        // cells in the plane
      for ( auto x = lo.x(); x<hi.x(); x++ ) {
        IntVector c(x,y,z);
        
        c = findCellIndex(x, y, z);
        Q_sum = Q_sum + Q_var[c];
      }
    }
    
    // cdll-centered position
    IntVector here = findCellIndex( Uintah::Round( ( hi.x() - lo.x() )/2 ), 
                                    Uintah::Round( ( hi.y() - lo.y() )/2 ), 
                                    z );

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
void planeAverage::sumOverAllProcs(const ProcessorGroup * pg,
                                   const PatchSubset    * patch,
                                   const MaterialSubset *,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw)
{

  printTask( patch, do_cout,"Doing planeAverage::sumOverAllProcs");

  const LevelP level = getLevelP( patch );
  const int L_indx = level->getIndex();
  
  // Is it the right time and has this rank performed the reduction
  if( isItTime( old_dw ) == false || d_progressVar[SUM][L_indx] == true){
    return;
  }

  //__________________________________
  // Loop over variables
  std::vector< std::shared_ptr< planarVarBase > >planarVars = d_allLevels_planarVars[L_indx];
  
  for (unsigned int i =0 ; i < planarVars.size(); i++) {

    std::shared_ptr<planarVarBase> analyzeVar = planarVars[i];
    
    int rank = pg->myRank();
    analyzeVar->ReduceWeight( rank );
    analyzeVar->ReduceCC_pos( rank );
    analyzeVar->ReduceVar( rank );

  }  // loop over planarVars
  d_progressVar[SUM][L_indx] = true;
}


//______________________________________________________________________
//  This task writes out the plane average of each VarLabel to a separate file.
void planeAverage::writeToFiles(const ProcessorGroup* pg,
                                const PatchSubset   * patches,
                                const MaterialSubset*,
                                DataWarehouse       * old_dw,
                                DataWarehouse       * new_dw)
{
  const LevelP level = getLevelP( patches );
  int L_indx = level->getIndex();

  max_vartype writeTime;
  simTime_vartype simTimeVar;
  old_dw->get(writeTime, d_lb->lastCompTimeLabel);
  old_dw->get(simTimeVar, m_simulationTimeLabel);
  double lastWriteTime = writeTime;
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime){
    new_dw->put(max_vartype(lastWriteTime), d_lb->lastCompTimeLabel);
    return;
  }

  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;

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
    if( proc == pg->myRank() && now >= nextWriteTime){

      printTask( patch, do_cout,"Doing planeAverage::writeToFiles" );


      std::vector< std::shared_ptr< planarVarBase > > planarVars = d_allLevels_planarVars[L_indx];
      
      for (unsigned int i =0 ; i < planarVars.size(); i++) {
        VarLabel* label = planarVars[i]->label;
        string labelName = label->getName();

        // bulletproofing
        if(label == nullptr){
          throw InternalError("planeAverage: analyze label not found: "
                          + labelName , __FILE__, __LINE__);
        }

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
        
        string path = "planeAverage/" + timestep + "/" + levelIndex;
        
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
        planarVars[i]->printAverage( fp, L_indx, now );
        
        fflush(fp);
      }  // planarVars loop

      lastWriteTime = now;
    }  // time to write data

    // Put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;

    new_dw->put(fileInfo,                   d_lb->fileVarsStructLabel, 0, patch);
    new_dw->put(max_vartype(lastWriteTime), d_lb->lastCompTimeLabel );
  }  // patches
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

  do_cout << d_myworld->myRank() << " planeAverage:Making directory " << subDirs << endl;
  
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
        cout << "path [" << newPath << "] already exists " << endl;
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
//
IntVector planeAverage::findCellIndex(const int i,
                                      const int j,
                                      const int k)
{
  IntVector c(-9,-9,-9);
  switch( d_planeOrientation ){        
    case XY:{                          
      c = IntVector( i,j,k );            
      break;                           
    }                                  
    case XZ:{                          
      c = IntVector( i,k,j );            
      break;                           
    }                                  
    case YZ:{                          
      c = IntVector( j,k,i );            
      break;                           
    }                                  
    default:                           
      break;                           
  }
  return c;                                    
}

//______________________________________________________________________
//
bool planeAverage::isItTime( DataWarehouse * old_dw)
{
  max_vartype writeTime;
  simTime_vartype simTimeVar;
  
  old_dw->get( writeTime,  d_lb->lastCompTimeLabel );
  old_dw->get( simTimeVar, m_simulationTimeLabel );
  
  double lastWriteTime = writeTime;
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;
  double now = simTimeVar;

  if(now < d_startTime || now > d_stopTime || now < nextWriteTime ){
    return false;
  }
  return true;
}

//______________________________________________________________________
//
bool planeAverage::isRightLevel(const int myLevel, 
                                const int L_indx, 
                                const LevelP& level)
{
  if( myLevel == ALL_LEVELS || myLevel == L_indx )
    return true;

  int numLevels = level->getGrid()->numLevels();
  if( myLevel == FINEST_LEVEL && L_indx == numLevels -1 ){
    return true;
  }else{
    return false;
  }
}
//______________________________________________________________________
//
void planeAverage::planeIterator( const GridIterator& patchIter,
                                  IntVector & lo,
                                  IntVector & hi )
{
  IntVector patchLo = patchIter.begin();
  IntVector patchHi = patchIter.end();

  switch( d_planeOrientation ){
    case XY:{
      lo = patchLo;
      hi = patchHi;
      break;
    }
    case XZ:{
      lo.x( patchLo.x() );
      lo.y( patchLo.z() );
      lo.z( patchLo.y() );

      hi.x( patchHi.x() );
      hi.y( patchHi.z() );
      hi.z( patchHi.y() );
      break;
    }
    case YZ:{
      lo.x( patchLo.y() );
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
