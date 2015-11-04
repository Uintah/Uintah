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
#include <iostream>
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
  d_matlSet     = 0;
}

//__________________________________
statistics::~statistics()
{
  cout_doing << " Doing: destorying statistics " << endl;
  if(d_matlSet && d_matlSet->removeReference()) {
    delete d_matlSet;
  }

  // delete each Qstats label
  for (unsigned int i =0 ; i < d_Qstats.size(); i++) {
    Qstats Q = d_Qstats[i];
    VarLabel::destroy( Q.Qsum_Label );
    VarLabel::destroy( Q.QsumSqr_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qmean_Label );
    VarLabel::destroy( Q.Qvariance_Label );
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
      throw ProblemSetupException("statistics: problemSetup: analyze: Invalid material index specified for a variable", __FILE__, __LINE__);
    }
    m.push_back(matl);


    // What is the label name and does it exist?
    string name = attribute["label"];
    VarLabel* label = VarLabel::find(name);
    if(label == NULL){
      throw ProblemSetupException("statistics label not found: " + name , __FILE__, __LINE__);
    }

    //__________________________________
    // Only CCVariable Doubles and Vectors for now
    const Uintah::TypeDescription* td = label->typeDescription();
    const Uintah::TypeDescription* subtype = td->getSubType();

    if(td->getType() != TypeDescription::CCVariable  ||
       ( subtype->getType() != TypeDescription::double_type &&
         subtype->getType() != TypeDescription::Vector)  ) {
      ostringstream warn;
      warn << "ERROR:AnalysisModule:statisticst: ("<<label->getName() << " "
           << td->getName() << " ) has not been implemented" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // create the labels for this variable
    Qstats Q;
    Q.name    = name;
    Q.matl    = matl;
    Q.Q_Label = label;
    Q.subtype = subtype;
    Q.Qsum_Label      = VarLabel::create( "sum_" + name,      td);
    Q.QsumSqr_Label   = VarLabel::create( "sumSqr_" + name,   td);
    Q.Qmean_Label     = VarLabel::create( "mean_" + name,     td);
    Q.QmeanSqr_Label  = VarLabel::create( "meanSqr_" + name,  td);
    Q.Qvariance_Label = VarLabel::create( "variance_" + name, td);

    d_Qstats.push_back( Q );
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

  d_matlSet = scinew MaterialSet();
  d_matlSet->addAll(m);
  d_matlSet->addReference();
  d_matSubSet = d_matlSet->getUnion();
}

//______________________________________________________________________
void statistics::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleInitialize" );

  Task* t = scinew Task("statistics::initialize",
                   this,&statistics::initialize);

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    t->computes ( d_Qstats[i].Qsum_Label );
    t->computes ( d_Qstats[i].QsumSqr_Label );
  }
  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
//
void statistics::initialize(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing statistics::initialize");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          allocateAndZero<double>( new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          allocateAndZero<Vector>( new_dw, patch, Q);
          break;
        }
        default: {
          throw InternalError("statistics: invalid data type", __FILE__, __LINE__);
        }
      }
    }  // loop over Qstat
  }  // pathes
}

void statistics::restartInitialize()
{
}

//______________________________________________________________________
void statistics::scheduleDoAnalysis(SchedulerP& sched,
                                    const LevelP& level)
{
  printSchedule( level,cout_doing,"statistics::scheduleDoAnalysis" );

  Task* t = scinew Task("statistics::doAnalysis",
                   this,&statistics::doAnalysis);

  Ghost::GhostType  gn  = Ghost::None;

  for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
    Qstats Q = d_Qstats[i];
    //Q.print();

    // define the matl subset for this variable
    MaterialSubset* matSubSet = scinew MaterialSubset();
    matSubSet->add( Q.matl );
    matSubSet->addReference();

    t->requires( Task::NewDW, Q.Q_Label,       matSubSet, gn, 0 );
    t->requires( Task::OldDW, Q.Qsum_Label,    matSubSet, gn, 0 );
    t->requires( Task::OldDW, Q.QsumSqr_Label, matSubSet, gn, 0 );

    t->computes ( Q.Qsum_Label,       matSubSet );
    t->computes ( Q.QsumSqr_Label,    matSubSet );
    t->computes ( Q.Qmean_Label,      matSubSet );
    t->computes ( Q.QmeanSqr_Label,   matSubSet );
    t->computes ( Q.Qvariance_Label,  matSubSet );

    if(matSubSet && matSubSet->removeReference()){
      delete matSubSet;
    }
  }

  sched->addTask(t, level->eachPatch(), d_matlSet);
}

//______________________________________________________________________
// Compute the statistics for each variable the user requested
void statistics::doAnalysis(const ProcessorGroup* pg,
                            const PatchSubset* patches,
                            const MaterialSubset* matl_sub ,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing statistics::doAnalysis");

    for ( unsigned int i =0 ; i < d_Qstats.size(); i++ ) {
      Qstats Q = d_Qstats[i];

      switch(Q.subtype->getType()) {

        case TypeDescription::double_type:{         // double
          computeStats< double >(old_dw, new_dw, patch, Q);
          break;
        }
        case TypeDescription::Vector: {             // Vector
          computeStats< Vector >(old_dw, new_dw, patch, Q);

          break;
        }
        default: {
          throw InternalError("statistics: invalid data type", __FILE__, __LINE__);
        }
      }
    }
  }  // patches
}

//______________________________________________________________________
//
template <class T>
void statistics::computeStats( DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               const Patch*    patch,
                               Qstats Q)
{
  const int matl = Q.matl;

  constCCVariable<T> Qvar;
  constCCVariable<T> Qsum_old;
  constCCVariable<T> QsumSqr_old;

  Ghost::GhostType  gn  = Ghost::None;
  new_dw->get ( Qvar,        Q.Q_Label,       matl, patch, gn, 0 );
  old_dw->get ( Qsum_old,    Q.Qsum_Label,    matl, patch, gn, 0 );
  old_dw->get ( QsumSqr_old, Q.QsumSqr_Label, matl, patch, gn, 0 );

  CCVariable< T > Qsum;
  CCVariable< T > QsumSqr;
  CCVariable< T > Qmean;
  CCVariable< T > QmeanSqr;
  CCVariable< T > Qvariance;
  new_dw->allocateAndPut( Qsum,      Q.Qsum_Label,      matl, patch );
  new_dw->allocateAndPut( QsumSqr,   Q.QsumSqr_Label,   matl, patch );
  new_dw->allocateAndPut( Qmean,     Q.Qmean_Label,     matl, patch );
  new_dw->allocateAndPut( QmeanSqr,  Q.QmeanSqr_Label,  matl, patch );
  new_dw->allocateAndPut( Qvariance, Q.Qvariance_Label, matl, patch );
  int timestep = d_sharedState->getCurrentTopLevelTimeStep();

  T nTimesteps(timestep);

  for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
    IntVector c = *iter;

    T me = Qvar[c];     // for readability
    Qsum[c]    = me + Qsum_old[c];
    Qmean[c]   = Qsum[c]/nTimesteps;

    QsumSqr[c]  = me * me + QsumSqr_old[c];
    QmeanSqr[c] = QsumSqr[c]/nTimesteps;

    Qvariance[c] = QmeanSqr[c] - Qmean[c] * Qmean[c];
  }
}

//______________________________________________________________________
//  allocateAndZero
template <class T>
void statistics::allocateAndZero( DataWarehouse* new_dw,
                                  const Patch*    patch,
                                  Qstats Q )
{
  int matl = Q.matl;
  CCVariable<T> Qsum;
  CCVariable<T> QsumSqr;

  new_dw->allocateAndPut( Qsum,    Q.Qsum_Label,    matl, patch );
  new_dw->allocateAndPut( QsumSqr, Q.QsumSqr_Label, matl, patch );

  T zero(0.0);
  Qsum.initialize( zero );
  QsumSqr.initialize( zero );
}
