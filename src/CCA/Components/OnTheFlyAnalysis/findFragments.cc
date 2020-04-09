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

#include <CCA/Components/OnTheFlyAnalysis/findFragments.h>

#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <set>

using namespace Uintah;
using namespace std;
//______________________________________________________________________
//    Reference:  P. Yang, Y. Liu, X. Zhang, X. Zhou, Y. Zhao, 
//    Simulation of Fragmentation with Material Point Method Based on 
//    Gurson Model and Random Failure, CMES, Vol. 85, no.3, pp207-236, 2012
//______________________________________________________________________
//    To Do
//    - Read variables (Q) from ups to compute mean quantities
//    - Identify fragments.  What is the criteria???
//    - Find data structure for mean quantities (Q), doubles and vectors and matrix3??
//    - MPI code for summing each Q for each fragment
//    - compute mass weighted average for each Q and each fragment
//    - output mean quantites for each fragment
//
//______________________________________________________________________
//
static DebugStream dbg("findFragments", false);

//______________________________________________________________________
findFragments::findFragments(ProblemSpecP     & module_spec,
                             SimulationStateP & sharedState,
                             Output           * dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl         = nullptr;
  d_matl_set     = nullptr;
  d_lb           = scinew findFragmentsLabel();
  
  d_lb->prevAnalysisTimeLabel = VarLabel::create( "prevAnalysisTime", max_vartype::getTypeDescription() );
  d_lb->fragmentIDLabel       = VarLabel::create( "fragmentID",       CCVariable<int>::getTypeDescription() );
  d_lb->nTouchedLabel         = VarLabel::create( "nTouched",         CCVariable<int>::getTypeDescription() );
  d_lb->numLocalized_CCLabel  = VarLabel::create( "sumLocalized_CC",  CCVariable<int>::getTypeDescription() );
  d_lb->maxFragmentIDLabel    = VarLabel::create( "nFragments",       max_vartype::getTypeDescription() );

  d_lb->gMassLabel            = VarLabel::find( "g.mass" );
  d_lb->pMassLabel            = VarLabel::find( "p.mass" );
  d_lb->pLocalizedLabel       = VarLabel::find( "p.localizedMPM+" );
  d_lb->pXLabel               = VarLabel::find( "p.x+" );
}

//__________________________________
findFragments::~findFragments()
{
  dbg << " Doing: destorying findFragments \n";
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy( d_lb->prevAnalysisTimeLabel );
  VarLabel::destroy( d_lb->fragmentIDLabel );
  VarLabel::destroy( d_lb->nTouchedLabel );
  VarLabel::destroy( d_lb->gMassLabel );
  VarLabel::destroy( d_lb->numLocalized_CCLabel );
  
  VarLabel::destroy( d_lb->pLocalizedLabel );
  VarLabel::destroy( d_lb->pMassLabel );
  VarLabel::destroy( d_lb->pXLabel );
  delete d_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void findFragments::problemSetup(const ProblemSpecP & prob_spec,
                                 const ProblemSpecP & ,
                                 GridP              & grid,
                                 SimulationStateP   & sharedState)
{

  cout << "HERE \n";
  dbg << "Doing problemSetup \t\t\t\tfindFragments\n";

  if(!d_dataArchiver){
    throw InternalError("findFragments:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_analysisFreq);
  d_prob_spec->require("timeStart",         d_startTime);
  d_prob_spec->require("timeStop",          d_stopTime);

  ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("findFragments: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  // find the material.
  //  <material>   atmosphere </material>
  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");

  int indx   = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->add( indx );
  d_matl_set->addReference();

  d_matl_subSet = d_matl_set->getUnion();

  //__________________________________
  //  Read in variables label names
  for( ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != nullptr; var_spec = var_spec->findNextBlock("analyze") ) {

    map<string,string> attribute;
    var_spec->getAttributes(attribute);

    string name     = attribute["label"];
    VarLabel* label = VarLabel::find(name);

    if(label == nullptr){
      throw ProblemSetupException("findFragments: analyze label not found: "+ name , __FILE__, __LINE__);
    }

    const Uintah::TypeDescription* td      = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    //__________________________________
    // bulletproofing

    // Only NC Variables, Doubles and Vectors
    if(td->getType()      != TypeDescription::NCVariable  &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::Vector  ){

      ostringstream warn;
      warn << "ERROR:AnalysisModule:findFragments: ("<<label->getName() << " "
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    d_varLabels.push_back(label);
  }
}

//______________________________________________________________________
//
void findFragments::scheduleInitialize( SchedulerP   & sched,
                                        const LevelP & level)
{
  printSchedule(level,dbg, "findFragments::scheduleInitialize " );

  Task* t = scinew Task("findFragments::initialize",
                  this, &findFragments::initialize);

  t->computes(d_lb->prevAnalysisTimeLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void findFragments::initialize(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        *,
                               DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, dbg,"Doing  findFragments::initialize" );

    double tminus = -1.0/d_analysisFreq;
    new_dw->put(max_vartype(tminus), d_lb->prevAnalysisTimeLabel);


    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:findFragments  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}

//______________________________________________________________________
//
void findFragments::scheduleRestartInitialize(SchedulerP   & sched,
                                              const LevelP & level)
{
  scheduleInitialize( sched, level);
}

//______________________________________________________________________
//
void findFragments::scheduleDoAnalysis(SchedulerP   & sched,
                                       const LevelP & level)
{
  printSchedule(level,dbg, "findFragments::scheduleDoAnalysis");

  Task* t = scinew Task("findFragments::doAnalysis",
                   this,&findFragments::doAnalysis);
                   
  sched_sumLocalizedParticles( sched, level);
  
  sched_identifyFragments(     sched, level);
  
  sched_sumQ_inFragments(      sched, level);
  

  t->requires(Task::OldDW, d_lb->prevAnalysisTimeLabel);
  Ghost::GhostType gac = Ghost::AroundCells;

  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    t->requires(Task::NewDW, d_varLabels[i], d_matl_subSet, gac, 1);
  }

  t->computes(d_lb->prevAnalysisTimeLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void findFragments::doAnalysis(const ProcessorGroup * pg,
                               const PatchSubset    * patches,
                               const MaterialSubset *,
                               DataWarehouse        * old_dw,
                               DataWarehouse        * new_dw)
{
  UintahParallelComponent * DA = dynamic_cast<UintahParallelComponent*>( d_dataArchiver );
  LoadBalancerPort        * lb = dynamic_cast<LoadBalancerPort*>( DA->getPort("load balancer") );

  // the user may want to restart from an uda that wasn't using the DA module
  // This logic allows that.
  max_vartype writeTime;
  double prevAnalysisTime = 0;
  if( old_dw->exists( d_lb->prevAnalysisTimeLabel ) ){
    old_dw->get(writeTime, d_lb->prevAnalysisTimeLabel);
    prevAnalysisTime = writeTime;
  }

  double now = d_sharedState->getElapsedTime();

  if(now < d_startTime || now > d_stopTime){
    new_dw->put(max_vartype(prevAnalysisTime), d_lb->prevAnalysisTimeLabel);
    return;
  }

  double nextWriteTime = prevAnalysisTime + 1.0/d_analysisFreq;

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask( patch, dbg,"Doing findFragments::doAnalysis" );

    int proc = lb->getPatchwiseProcessorAssignment(patch);

    prevAnalysisTime = now;

    new_dw->put(max_vartype(prevAnalysisTime), d_lb->prevAnalysisTimeLabel);
  }  // patches
}

//______________________________________________________________________
//
void findFragments::sched_sumLocalizedParticles(SchedulerP   & sched,
                                                const LevelP & level)
{
  Task* t = scinew Task("findFragments::sumLocalizedParticles",      
                    this,&findFragments::sumLocalizedParticles); 
  
  printSchedule(level,dbg, "findFragments::sched_sumLocalizedParticles " );    
                                                                     
  Ghost::GhostType  gn = Ghost::None;                        
                                               
  t->requires(Task::NewDW, d_lb->pLocalizedLabel, d_matl_subSet, gn,0);
  t->requires(Task::NewDW, d_lb->pXLabel,         d_matl_subSet, gn,0); 
  t->computes( d_lb->numLocalized_CCLabel,        d_matl_subSet );        
                                                                     
  sched->addTask(t, level->eachPatch(), d_matl_set);                 
}

//______________________________________________________________________
//  Loop over all cells and count the number of particles that are localized
// in a cell
void findFragments::sumLocalizedParticles(const ProcessorGroup * pg,
                                          const PatchSubset    * patches,
                                          const MaterialSubset *,
                                          DataWarehouse        * old_dw,
                                          DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask( patch, dbg,"Doing findFragments::sumLocalizedParticles" );
    
    int dwi = d_matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset( dwi, patch );

    constParticleVariable<int>   pLocalized;
    constParticleVariable<Point> px;
    CCVariable<int>  numLoc_CC;
    
    new_dw->get(pLocalized, d_lb->pLocalizedLabel, pset);
    new_dw->get(px,         d_lb->pXLabel,         pset);
    new_dw->allocateAndPut( numLoc_CC, d_lb->numLocalized_CCLabel, dwi, patch );

    numLoc_CC.initialize(0);

    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      
      IntVector c = patch->getLevel()->getCellIndex( px[idx] );
      
      // disable for testing
      // if( pLocalized[idx] > 0 )
        numLoc_CC[c] += 1;
      // }
    }
  }
}

//______________________________________________________________________
//
void findFragments::sched_identifyFragments(SchedulerP   & sched,
                                            const LevelP & level)
{
    Task* t = scinew Task("findFragments::identifyFragments",
                      this,&findFragments::identifyFragments);
                 
    printSchedule(level,dbg, "findFragments::sched_identifyFragments " );
                      
    Ghost::GhostType  gac = Ghost::AroundCells;
    t->requires(Task::NewDW, d_lb->gMassLabel,           d_matl_subSet, gac,1);
    t->requires(Task::NewDW, d_lb->numLocalized_CCLabel, d_matl_subSet, gac,1);
    
    t->computes( d_lb->fragmentIDLabel, d_matl_subSet);
    t->computes( d_lb->nTouchedLabel,   d_matl_subSet);
    t->computes( d_lb->maxFragmentIDLabel );
    
    sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//  See Reference, pg 224 for pseudo code
void findFragments::identifyFragments(const ProcessorGroup * pg,
                                      const PatchSubset    * patches,
                                      const MaterialSubset *,
                                      DataWarehouse        *,
                                      DataWarehouse        * new_dw)
{
  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);


    printTask( patch, dbg,"Doing findFragments::identifyFragments" );

    Ghost::GhostType  gac = Ghost::AroundCells;
    int indx = d_matl->getDWIndex();
    
    CCVariable<int>         fragmentID;
    CCVariable<int>         nTouched;           // how many time has this cell been interogated
    constNCVariable<double> gmass;
    constCCVariable<int>    numLocalized;

    new_dw->get( numLocalized, d_lb->numLocalized_CCLabel,indx,patch, gac,1);
    
    new_dw->allocateAndPut( fragmentID, d_lb->fragmentIDLabel, indx, patch);
    new_dw->allocateAndPut( nTouched,   d_lb->nTouchedLabel,   indx, patch);
    fragmentID.initialize( 0 );
    nTouched.initialize( 0 );
    
    int fragID = 0;
    
    // This linked list contains all the cells that have been searched
    std::list<IntVector> patchCellList;
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      patchCellList.push_back( c );
    } 

    std::cout << "patchCellList contains:\n";
    for (auto& x: patchCellList){
      std::cout << "  " << x << endl;;
    }
    
    //__________________________________
    //  Loop over all cells
    // Optimization is to search through the patchCellList
    // that is dynamically updated
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
  
      cout << "  topLevel: " << c << endl;
      nTouched[c] += 1;
      
      if (fragmentID[c] != 0 ){
        continue;
      }
      
      // cell is fragmented
      if ( numLocalized[c] != 0 ){
        
        fragID ++;
        
        fragmentID[c] = fragID;
        patchCellList.remove( c );
        
        cout << "Top Level c: " << c << " is a fragment: " << fragmentID[c] << ".  Patch limits: " << patch->getExtraCellLowIndex() << " " << patch->getExtraCellHighIndex() << endl;
        cout << "cellList.size() " <<patchCellList.size() << endl;
        
        checkNeighborCells( "cell-neighbor", patchCellList, c, patch, fragID, numLocalized, fragmentID, nTouched );
        

      }  // if: is cellfragmented
      
      patchCellList.remove( c );
      
    }  // cellIterator
    
    new_dw->put( max_vartype(fragID), d_lb->maxFragmentIDLabel); 
  }  // patches
}


//______________________________________________________________________
//
std::list<IntVector> 
findFragments::intersection_of(const std::list<IntVector>& a, 
                               const std::list<IntVector>& b)
{
  std::list<IntVector> intersect;
  std::set<IntVector> tmpSet;

  auto func1 = [](int i = 6) { return i + 4; };


  std::for_each( a.begin(), a.end(), 
    [&tmpSet](const IntVector& k)         // lamda that populates the set
    { 
      tmpSet.insert(k); 
    } 
  );

  std::for_each( b.begin(), b.end(),
      [&tmpSet, &intersect](const IntVector& k)
      {
        auto iter = tmpSet.find(k);

        if(iter != tmpSet.end()){
          intersect.push_front(k);
          tmpSet.erase(iter);
        }
      }
  );
  return intersect;
}
//______________________________________________________________________
//  Loop over all neighboring cells  mark cells that are fragmented

void findFragments::checkNeighborCells( const std::string    & desc,
                                        std::list<IntVector> & patchCellList,
                                        const IntVector      & cell, 
                                        const Patch          * patch,
                                        const int              fragID,
                                        constCCVariable<int> & numLocalized,
                                        CCVariable<int>      & fragmentID,
                                        CCVariable<int>      & nTouched )
{
  cout << "__________________________________" << desc << " cell: " << cell << endl;
  // create a list of adjacent cells to "cell"
  std::list< IntVector > adjCellList;
  
  for (int k=-1; k<= 1; k++){
    for (int j=-1; j<= 1; j++){
      for (int i=-1; i<= 1; i++){
        const IntVector n = (cell + IntVector(i,j, k) );
        adjCellList.push_back( n );
      }
    }
  }
 
  //__________________________________
  // Find the intersection between the patchCellsList and adjacent list
  // This creates a list of cells that hasn't been searched.
  std::list<IntVector> nbrCellList;
  nbrCellList = intersection_of( patchCellList, adjCellList );
  nbrCellList.sort();
  
  
#if 0
  std::cout << "  adjCellList contains:" << endl;
  for (auto& x: adjCellList){
    std::cout << "    " << x << endl;;
  }
#endif

  if ( desc == "cell-neighbor-neighbor" || desc == "cell-neighbor"){
    std::cout << "  nbrCellList contains:" << endl;
    for (auto& x: nbrCellList){
      std::cout << "    " << x << endl;;
    }
  }

  cout << "  nbrCellList.size() " << nbrCellList.size() << endl;

  //__________________________________
  //  Search neighboring cells
  for (auto& nc: nbrCellList){

    nTouched[nc] += 1;
    patchCellList.remove(nc);
    
    cout << "  removing from patchCellList: " << nc << endl;
    
    if ( !patch->containsCell(nc) ){
      continue;
    }

    if ( fragmentID[nc] != 0 ){
      cout << "        BBB cell: "<< nc << " is a fragment: "<< fragmentID[nc] << endl;
      continue;
    }

    // cell is fragmented
    if ( numLocalized[nc] != 0 ){
      fragmentID[nc] = fragID;
      cout << desc << " " << nc << " is a fragment: "<< fragmentID[nc] << endl;

      checkNeighborCells( "cell-neighbor-neighbor",patchCellList, nc, patch, fragID, numLocalized, fragmentID, nTouched );

    }
  }  // neighborCell loop
  
  
  cout << "   patchCellList.size() " << patchCellList.size() << endl;
#if 0  
  std::cout << "patchCellList contains:" << endl;
  for (auto& x: patchCellList){
    std::cout << "    " << x << endl;;
  }  
#endif  
  cout << "__________________________________exit" << endl;
}




//______________________________________________________________________
//
void findFragments::sched_sumQ_inFragments(SchedulerP   & sched,
                                           const LevelP & level)
{
    Task* t = scinew Task( "findFragments::sumQ_inFragments", this,
                           &findFragments::sumQ_inFragments);
                 
    printSchedule(level,dbg, "findFragments::sched_sumQ_inFragments" );
                      
    Ghost::GhostType  gn = Ghost::None;
    t->requires( Task::NewDW, d_lb->fragmentIDLabel, d_matl_subSet, gn, 0);
    t->requires( Task::NewDW, d_lb->pXLabel,         d_matl_subSet, gn, 0); 
    t->requires( Task::NewDW, d_lb->pMassLabel,      d_matl_subSet, gn, 0); 
    t->requires( Task::NewDW, d_lb->maxFragmentIDLabel );
    
    sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
// 
void findFragments::sumQ_inFragments(const ProcessorGroup      * pg,
                                          const PatchSubset    * patches,
                                          const MaterialSubset *,
                                          DataWarehouse        * old_dw,
                                          DataWarehouse        * new_dw)
{
  const Level* level = getLevel(patches);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask( patch, dbg,"Doing findFragments::sumQ_inFragments" );
    
    max_vartype nFragments;
    new_dw->get( nFragments, d_lb->maxFragmentIDLabel );
    
    vector<double> fragMass( nFragments+1, 0.0 );
    
    int dwi = d_matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset( dwi, patch );

    constCCVariable<int> fragID;
    constParticleVariable<Point>  px;
    constParticleVariable<double> pMass;
    
    new_dw->get( fragID, d_lb->fragmentIDLabel, dwi, patch,  Ghost::None, 0 );
    new_dw->get( px,     d_lb->pXLabel,     pset);
    new_dw->get( pMass,  d_lb->pMassLabel,  pset);

    for(ParticleSubset::iterator iter = pset->begin();iter != pset->end(); iter++) {
      particleIndex idx = *iter;
      
      IntVector c = level->getCellIndex( px[idx] );

      const int ID = fragID[c];
      if( ID != 0 ){
        fragMass[ID] += pMass[idx];
        
        cout << " c " << c << " ID: " << ID << " fragMass: " << fragMass[ID] << endl;  
      }
    }
    
    for ( size_t i=0;i<fragMass.size(); i++){
      cout << " patch: " << patch->getID() << " fragment ID: " << i << " mass: " << fragMass[i] << endl;
    }
  }
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void findFragments::createFile( const string   & filename,
                                const VarLabel * varLabel,
                                const int matl,
                                FILE           *& fp )
{
  // if the file already exists then exit.
  ifstream doExists( filename.c_str() );
  if( doExists ){
    fp = fopen(filename.c_str(), "a");
    return;
  }

  fp = fopen(filename.c_str(), "w");

  if (!fp){
    perror("Error opening file:");
    throw InternalError("\nERROR:dataAnalysisModule:findFragments:  failed opening file"+filename,__FILE__, __LINE__);
  }

  //__________________________________
  //Write out the header
  fprintf(fp,"# X      Y      Z ");


  const Uintah::TypeDescription* td = varLabel->typeDescription();
  const Uintah::TypeDescription* subtype = td->getSubType();
  string labelName = varLabel->getName();

  switch( subtype->getType( )) {

    case Uintah::TypeDescription::double_type:
      fprintf(fp,"     %s(%i)", labelName.c_str(), matl);
      break;

    case Uintah::TypeDescription::Vector:
      fprintf(fp,"     %s(%i).x      %s(%i).y      %s(%i).z", labelName.c_str(), matl, labelName.c_str(), matl, labelName.c_str(), matl);
      break;

    default:
      throw InternalError("findFragments: invalid data type", __FILE__, __LINE__);
  }

  fprintf(fp,"\n");
  fflush(fp);

  cout << Parallel::getMPIRank() << " findFragments:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure
//
void
findFragments::createDirectory(string      & planeName,
                               string      & timestep,
                               const double  now,
                               string      & levelIndex)
{
  DIR *check = opendir(planeName.c_str());
  if ( check == nullptr ) {
    cout << Parallel::getMPIRank() << " findFragments:Making directory " << planeName << endl;
    MKDIR( planeName.c_str(), 0777 );
  } else {
    closedir(check);
  }

  // timestep
  string path = planeName + "/" + timestep;
  check = opendir( path.c_str() );

  if ( check == nullptr ) {
    cout << Parallel::getMPIRank() << " findFragments:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );

    // write out physical time
    string filename = planeName + "/" + timestep + "/physicalTime";
    FILE *fp;
    fp = fopen( filename.c_str(), "w" );
    fprintf( fp, "%16.15E\n",now);
    fclose(fp);

  } else {
    closedir(check);
  }

  // level index
  path = planeName + "/" + timestep + "/" + levelIndex;
  check = opendir( path.c_str() );

  if ( check == nullptr ) {
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
