/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/statistics.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace Uintah;
using namespace std;
static DebugStream cout_doing("STATISTICS_DOING_COUT", false);
static DebugStream cout_dbg("STATISTICS_DBG_COUT", false);
//______________________________________________________________________              
statistics::statistics(ProblemSpecP& module_spec,
                       SimulationStateP& sharedState,
                       Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set     = 0;
}

//__________________________________
statistics::~statistics()
{
  cout_doing << " Doing: destorying statistics " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  // delete each Qstats label
  for (unsigned int i =0 ; i < d_Qstats.size(); i++) {
    VarLabel::destroy( d_Qstats[i]->Qsum_Label );
    VarLabel::destroy( d_Qstats[i]->QsumSqr_Label );
    VarLabel::destroy( d_Qstats[i]->Qmean_Label );
    VarLabel::destroy( d_Qstats[i]->Qmean_Label );
    VarLabel::destroy( d_Qstats[i]->Qvariance_Label );
    delete d_Qstats[i];
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void statistics::problemSetup(const ProblemSpecP& prob_spec,
                             const ProblemSpecP& ,
                             GridP& grid,
                             SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tstatistics" << endl;
  
  int numMatls  = d_sharedState->getNumMatls();
  if(!d_dataArchiver){
    throw InternalError("statistics:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  // find the material to extract data from.  Default is matl 0.
  // The user can use either 
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>

  Material* matl = NULL;

  if(d_prob_spec->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  } else if (d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }
  
  int defaultMatl = matl->getDWIndex();

  vector<int> m;
  m.push_back( defaultMatl );

  
  //__________________________________
  //  Read in variables label names       
  
 ProblemSpecP vars_ps = d_prob_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("statistics: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }
           
  for (ProblemSpecP var_spec = vars_ps->findBlock("analyze"); var_spec != 0; 
                    var_spec = var_spec->findNextBlock("analyze")) {
    map<string,string> attribute;
    var_spec->getAttributes(attribute);
    
    
    //__________________________________
    //  Read in the optional material index from the variables that may be different
    //  from the default index and construct the material set
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }
    
    // bulletproofing
    if(matl < 0 || matl > numMatls){
      throw ProblemSetupException("MinMax: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);
    
    
    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("statistics label not found: "
                           + name , __FILE__, __LINE__);
    }
    
    //__________________________________
    // Only CCVariable Doubles and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();
    
    if(td->getType() != TypeDescription::CCVariable  &&
       subtype->getType() != TypeDescription::double_type &&
       subtype->getType() != TypeDescription::Vector  ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:statisticst: ("<<label->getName() << " " 
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    } 
    
    //__________________________________
    // create the labels for this variable
    Qstats* Q = scinew Qstats;
    Q->Qsum_Label      = VarLabel::create( "sum_" + name,      td);
    Q->QsumSqr_Label   = VarLabel::create( "sumSqr_" + name,   td);
    Q->Qmean_Label     = VarLabel::create( "mean_" + name,     td);
    Q->QmeanSqr_Label  = VarLabel::create( "meanSqr_" + name,  td);
    Q->Qvariance_Label = VarLabel::create( "variance_" + name, td);

    std::string variance = "variance_"+ name;
    //__________________________________
    //  bulletproofing
    if(!d_dataArchiver->isLabelSaved( variance ) ){
      ostringstream warn;
      warn << "\nERROR:  You've activated the DataAnalysis:statistics module but your not saving the variable ("
           << variance << ")\n";
      throw ProblemSetupException( warn.str(),__FILE__, __LINE__ );
    }
  }
  
  //__________________________________
  //  create the matl set
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());
  
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_sub = d_matl_set->getUnion();
}

//______________________________________________________________________
void statistics::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level)
{
  return;  // do nothing
}

void statistics::initialize(const ProcessorGroup*, 
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw)
{  
}

void statistics::restartInitialize()
{
}

//______________________________________________________________________
void statistics::scheduleDoAnalysis(SchedulerP& sched,
                                   const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleDoAnalysis" );
#if 0  
  Task* t = scinew Task("statistics::doAnalysis", 
                   this,&statistics::doAnalysis);

  Ghost::GhostType  gn  = Ghost::None;
  
  t->computes(lb->statisticsLabel, d_matl_sub);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
#endif
}

//______________________________________________________________________
// Compute the statistics field.
void statistics::doAnalysis(const ProcessorGroup* pg,
                           const PatchSubset* patches,
                           const MaterialSubset* matl_sub ,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{       
  const Level* level = getLevel(patches);
#if 0  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch,cout_doing,"Doing statistics::doAnalysis");
                
    Ghost::GhostType  gn  = Ghost::None;
    
    CCVariable<Vector> statistics;
    constCCVariable<Vector> vel_CC;
    
    int indx = d_matl->getDWIndex();
    new_dw->allocateAndPut(statistics, lb->statisticsLabel, indx,patch);
    
    statistics.initialize(Vector(0.0));
    
    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      
    }         
  }  // patches
#endif
}
