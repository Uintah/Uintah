//NOTE: I just included all DQMOM.cc includes here.  Some can likely be removed later.
#include <CCA/Components/Arches/CQMOM.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/LU.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/FileNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Time.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <CCA/Components/Arches/CQMOMInversion.h>
//===========================================================================

using namespace std;
using namespace Uintah;


CQMOM::CQMOM(ArchesLabel* fieldLabels, std::string which_cqmom):
d_fieldLabels(fieldLabels), d_which_cqmom(which_cqmom)
{
//  string varname;
  nMoments = 0;
  uVelIndex = -1;
  vVelIndex = -1;
  wVelIndex = -1;
}

CQMOM::~CQMOM()
{
  //NOTE:destory extra var labels if needed
}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void CQMOM::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params;
  
  //leave a boolean in for a normalized pdf = 1 option later?
  if ( d_which_cqmom == "normalized" )
    d_normalized = true;
  else
    d_normalized = false;
  
  int index_length = 0;
  db->get("NumberInternalCoordinates",M);   //get number of coordiantes
  db->get("QuadratureNodes",N_i);           //get vector of quad nodes per internal coordiante
  db->get("HighestOrder",maxInd);           //vector of maxium moment order NOTE: this could be made automatic from indexes later
  db->getWithDefault("Adaptive",d_adaptive,false);   //use adaptive quadrature or not - NYI
  db->getWithDefault("CutOff",d_small,1.0e-10);      //value of moment 0 to fix weights and abscissas to 0
  db->getWithDefault("UseLapack",d_useLapack,false); //pick which linear solve to use
  db->getWithDefault("WeightRatio",weightRatio,1.0e-5);     //maximum ratio for min to max weight
  db->getWithDefault("AbscissaRatio",abscissaRatio,1.0e-5); //maximum ratio for min to max spacing of abscissas
  db->getWithDefault("OperatorSplitting",d_doOperatorSplitting,false); //boolean for operator splitting
  
  //NOTE: redo this to only have one xml tag here?
  int m = 0;
  for ( ProblemSpecP db_name = db->findBlock("InternalCoordinate");
       db_name != 0; db_name = db_name->findNextBlock("InternalCoordinate") ) {
    string coordName;
    string varType;
    db_name->getAttribute("name",coordName);
    coordinateNames.push_back(coordName);
    db_name->getAttribute("type",varType);
    varTypes.push_back(varType);
    proc0cout << "Internal Coordinate Found: " << coordName << endl;
    if (varType == "uVel") {
      uVelIndex = m;
    } else if (varType == "vVel") {
      vVelIndex = m;
    } else if (varType == "wVel") {
      wVelIndex = m;
    }
    m++;
  }
  
  proc0cout << "Internal Coordinates M: " << M << endl;
  proc0cout << "Operator Splitting is " << d_doOperatorSplitting << endl;
  
  nNodes = 1;
  momentSize = 1;
  for (unsigned int i = 0; i<N_i.size(); i++) {
    nNodes *= N_i[i];
    momentSize *= (maxInd[i]+1);
    maxInd[i]++; //increase maxindex by one to account for 0th index
  }
  proc0cout << "Nodes: " << nNodes << " momentSize: " << momentSize << endl;
  
  CQMOMEqnFactory & eqn_factory = CQMOMEqnFactory::self();
  
  nMoments = 0;
  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment");
       db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
    temp_moment_index.resize(0);
    db_moments->get("m", temp_moment_index);
    
    // put moment index into vector of moment indices:
    momentIndexes.push_back(temp_moment_index);
    
    index_length = temp_moment_index.size();
    if (index_length != M) {
      proc0cout << "Putting eqns in CQMOM.h, index:" << temp_moment_index << " does not have same number of indexes as internal coordiante #" << M << endl;
      throw InvalidValue("All specified moment must have same number of internal coordinates", __FILE__, __LINE__);
    }
    
    //register eqns
    std::string moment_name = "m_";
    std::string mIndex;
    std::stringstream out;
    for (int i = 0; i<M ; i++) {
      out << temp_moment_index[i];
      mIndex = out.str();
    }
    moment_name += mIndex;
    //store eqn
    EqnBase& temp_momentEqnE = eqn_factory.retrieve_scalar_eqn( moment_name );
    CQMOMEqn& temp_momentEqnD = dynamic_cast<CQMOMEqn&> (temp_momentEqnE);
    momentEqns.push_back( &temp_momentEqnD );
    ++nMoments; // keep track of total number of moments
  }
  
  //populate names for weights with just numberign them 1,2,3...
  //NOTE: should weights be numbered 1,2,3,4... or 11,12,21,22...
  for (int i = 1; i <= nNodes; i++) {
    string weight_name = "w_";
    string node;
    stringstream out;
    out << i;
    node = out.str();
    weight_name += node;
    weightNames.push_back(weight_name);
    
    //make varLabel
    const VarLabel* tempVarLabel = VarLabel::create(weight_name, CCVariable<double>::getTypeDescription());
    proc0cout << "Creating var label for " << weight_name << endl;
    d_fieldLabels->CQMOMWeights[i-1] = tempVarLabel;
  }
  
  //repeat for the abscissas, but include coordinate names
  int ii = 0;
  for (int m = 0; m < M; m++) {
    string coordName = coordinateNames[m];
    for (int i = 1; i <= nNodes; i++) {
      string abscissa_name = "a_" + coordName + "_";
      string node;
      stringstream out;
      out << i;
      node = out.str();
      abscissa_name += node;
      abscissaNames.push_back(abscissa_name);
      
      //make varLabel
      const VarLabel* tempVarLabel = VarLabel::create(abscissa_name, CCVariable<double>::getTypeDescription());
      proc0cout << "Creating var label for " << abscissa_name << endl;
      d_fieldLabels->CQMOMAbscissas[ii] = tempVarLabel;
      ii++;
      
      //keep track of any abscissa corresponding to velocities
      if (varTypes[m] == "uVel") {
        uVelAbscissas.push_back(abscissa_name);
      } else if (varTypes[m] == "vVel") {
        vVelAbscissas.push_back(abscissa_name);
      } else if (varTypes[m] == "wVel") {
        wVelAbscissas.push_back(abscissa_name);
      }
        
    }
  }
  
  
  // Check to make sure number of total moments specified in input file is correct
  int reqMoments;
  if (M == 2) {
    reqMoments = 2*N_i[1]*N_i[0] + N_i[0];
  } else {
    //NOTE: fix this for other #'s later
    reqMoments = nMoments;
  }
//Comment this out for now as number of required moments change based on splitting
//  if ( nMoments != reqMoments ) {
//    proc0cout << "ERROR:CQMOM:ProblemSetup: You specified " << nMoments << " moments, but you need " << reqMoments << " moments." << endl;
//    throw InvalidValue( "ERROR:CQMOM:ProblemSetup: The number of moments specified was incorrect!",__FILE__,__LINE__);
//  }

//set up more than one linear solver type in future?
//  ProblemSpecP db_linear_solver = db->findBlock("LinearSolver");
  
}


// **********************************************
// sched_solveCQMOMInversion
// **********************************************
void
CQMOM::sched_solveCQMOMInversion( const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  string taskname = "CQMOM:solveCQMOMInversion";
  Task* tsk = scinew Task(taskname, this, &CQMOM::solveCQMOMInversion);

  //tsk requires on moment eqns
  for (vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0);
  }

  //tsk computs on weights
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    if( timeSubStep == 0) {
      tsk->computes(tempLabel);
    } else {
      tsk->modifies(tempLabel);
    }
  }

  //tsk computes on abscissas
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    if( timeSubStep == 0) {
      tsk->computes(tempLabel);
    } else {
      tsk->modifies(tempLabel);
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// solveCQMOMInversion
// **********************************************
void CQMOM::solveCQMOMInversion( const ProcessorGroup* pc,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  //time how long the CQMOM solve takes in total
  double start_SolveTime = Time::currentSeconds();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
   
    
    // get moments from data warehouse and put into CCVariable
    vector<constCCVarWrapper> momentCCVars;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      
      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );
      
      // put the wrapper into a vector
      momentCCVars.push_back(tempWrapper);
    }
    
    //get/allocate the weights
    vector<CCVariable<double>* > cqmomWeights;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* weight_label = iW->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if( new_dw->exists(weight_label, matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, weight_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, weight_label, matlIndex, patch);
      }
      cqmomWeights.push_back(tempCCVar);
    }
    
    //get/allocate the abscissas
    vector<CCVariable<double>* > cqmomAbscissas;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* abscissa_label = iA->second;
      
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if( new_dw->exists(abscissa_label, matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, abscissa_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, abscissa_label, matlIndex, patch);
      }
      cqmomAbscissas.push_back(tempCCVar);
    }
    
    for ( CellIterator iter = patch->getExtraCellIterator();
         !iter.done(); ++iter) {
      IntVector c = *iter;
      vector<double> temp_weights (nNodes, 0.0);
      vector<vector<double> > temp_abscissas (M, vector<double> (nNodes, 0.0));
      vector<double> temp_moments (momentSize, 0.0);
      
      //loop over moments and put in vector
      //these are numbered into a flatindex based on moment index
      int ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = momentCCVars.begin(); iter!= momentCCVars.end(); ++iter) {
        double temp_value = (iter->data)[c];
        int flatIndex;
        
        vector<int> temp_index = momentIndexes[ii];
        
        if (M == 2) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0];
        } else if (M == 3) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0] + temp_index[2]*maxInd[0]*maxInd[1];
        }
        
        temp_moments[flatIndex] = temp_value;
        ii++;
      }
      
      //actually do the cqmom inversion step
      if (temp_moments[0] < d_small) {
        //if m0 is very small, leave all weights/absciassa equal to 0 (as intialized)
        if (temp_moments[0] < 0.0 )
          cout << "WARNING: Negative Moment " << temp_moments[0] <<  " in cell " << c << " settign all wegiths and abscissas to 0" << endl;
      } else {
#ifdef cqmom_dbg
        cout << "Cell Location " << c << endl;
        cout << "______________" << endl;
#endif
        //actually compute inversion
        CQMOMInversion( temp_moments, M, N_i, maxInd,
                        temp_weights, temp_abscissas, d_adaptive, d_useLapack, weightRatio, abscissaRatio);
      }
      
      //Now actually assign the new weights and abscissas
      int jj = 0;
      for (int m = 0; m < M; m++) {
        for (int z = 0; z < nNodes; z++) {
          (*(cqmomAbscissas[jj]))[c] = temp_abscissas[m][z];
          jj++;
        }
      }
      for (int z = 0; z < nNodes; z++) {
        (*(cqmomWeights[z]))[c] = temp_weights[z];
      }
      
    } //end cell loop
    
    //delete pointers
    for(unsigned int i=0; i<cqmomAbscissas.size(); ++i ){
      delete cqmomAbscissas[i];
    }
    
    for(unsigned int i=0; i<cqmomWeights.size(); ++i ){
      delete cqmomWeights[i];
    }
  } //end patch loop
  double total_SolveTime = (Time::currentSeconds() - start_SolveTime);
  proc0cout << "CQMOM Solve time: " << total_SolveTime << endl;
}


// **********************************************
// schedule the re-calculation of moments
// **********************************************
//void
//CQMOM::sched_momentCorrection( const LevelP& level, SchedulerP& sched, int timeSubStep )
//{
//  //placeholder for now
//  string taskname = "CQMOM::momentCorrection";
//  Task* tsk = scinew Task(taskname, this, &CQMOM::momentCorrection);
//
//}
//
//
//// **********************************************
//// actualyl do the re-calculation of moments
//// **********************************************
//void
//CQMOM::momentCorrection( const ProcessorGroup* pc,
//                        const PatchSubset* patches,
//                        const MaterialSubset* matls,
//                        DataWarehouse* old_dw,
//                        DataWarehouse* new_dw )
//{
//
//  //place holder for now
//}

// **********************************************
// sched_solveCQMOMInversion 3|2|1
// **********************************************
void
CQMOM::sched_solveCQMOMInversion321( const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  string taskname = "CQMOM:solveCQMOMInversion321";
  Task* tsk = scinew Task(taskname, this, &CQMOM::solveCQMOMInversion321);
  
  //tsk requires on moment eqns
  for (vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0);
  }
  
  //tsk computs on weights
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    if( timeSubStep == 0) {
      tsk->computes(tempLabel);
    } else {
      tsk->modifies(tempLabel);
    }
  }
  
  //tsk computes on abscissas
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    if( timeSubStep == 0) {
      tsk->computes(tempLabel);
    } else {
      tsk->modifies(tempLabel);
    }
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// solveCQMOMInversion 3|2|1
// **********************************************
void CQMOM::solveCQMOMInversion321( const ProcessorGroup* pc,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  //time how long the CQMOM solve takes in total
  double start_SolveTime = Time::currentSeconds();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    
    // get moments from data warehouse and put into CCVariable
    vector<constCCVarWrapper> momentCCVars;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      
      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper tempWrapper;
      old_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );
      
      // put the wrapper into a vector
      momentCCVars.push_back(tempWrapper);
    }
    
    //get/allocate the weights
    vector<CCVariable<double>* > cqmomWeights;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* weight_label = iW->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if( new_dw->exists(weight_label, matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, weight_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, weight_label, matlIndex, patch);
      }
      cqmomWeights.push_back(tempCCVar);
    }
    
    //get/allocate the abscissas
    vector<CCVariable<double>* > cqmomAbscissas;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* abscissa_label = iA->second;
      
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if( new_dw->exists(abscissa_label, matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, abscissa_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, abscissa_label, matlIndex, patch);
      }
      cqmomAbscissas.push_back(tempCCVar);
    }
    
    for ( CellIterator iter = patch->getExtraCellIterator();
         !iter.done(); ++iter) {
      IntVector c = *iter;
      vector<double> temp_weights (nNodes, 0.0);
      vector<vector<double> > temp_abscissas (M, vector<double> (nNodes, 0.0));
      vector<double> temp_moments (momentSize, 0.0);
      
      //loop over moments and put in vector
      //these are numbered into a flatindex based on moment index
      int ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = momentCCVars.begin(); iter!= momentCCVars.end(); ++iter) {
        double temp_value = (iter->data)[c];
        int flatIndex;
        
        vector<int> temp_index = momentIndexes[ii];
        
        if (M == 2) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0];
        } else if (M == 3) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0] + temp_index[2]*maxInd[0]*maxInd[1];
        }
        
        temp_moments[flatIndex] = temp_value;
        ii++;
      }
      
      //actually do the cqmom inversion step
      if (temp_moments[0] < d_small) {
        //if m0 is very small, leave all weights/absciassa equal to 0 (as intialized)
        if (temp_moments[0] < 0.0 )
          cout << "WARNING: Negative Moment " << temp_moments[0] <<  " in cell " << c << " settign all wegiths and abscissas to 0" << endl;
      } else {
#ifdef cqmom_dbg
        cout << "Permutation 3|2|1 Cell Location " << c << endl;
        cout << "______________" << endl;
#endif
        //actually compute inversion
        CQMOMInversion( temp_moments, M, N_i, maxInd,
                       temp_weights, temp_abscissas, d_adaptive, d_useLapack, weightRatio, abscissaRatio);
      }
      
      //Now actually assign the new weights and abscissas
      int jj = 0;
      for (int m = 0; m < M; m++) {
        for (int z = 0; z < nNodes; z++) {
          (*(cqmomAbscissas[jj]))[c] = temp_abscissas[m][z];
          jj++;
        }
      }
      for (int z = 0; z < nNodes; z++) {
        (*(cqmomWeights[z]))[c] = temp_weights[z];
      }
      
    } //end cell loop
    
    //delete pointers
    for(unsigned int i=0; i<cqmomAbscissas.size(); ++i ){
      delete cqmomAbscissas[i];
    }
    
    for(unsigned int i=0; i<cqmomWeights.size(); ++i ){
      delete cqmomWeights[i];
    }
  } //end patch loop
  double total_SolveTime = (Time::currentSeconds() - start_SolveTime);
  proc0cout << "CQMOM Solve time: " << total_SolveTime << endl;
}


// **********************************************
// sched_solveCQMOMInversion 3|1|2
// **********************************************
void
CQMOM::sched_solveCQMOMInversion312( const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  string taskname = "CQMOM:solveCQMOMInversion312";
  Task* tsk = scinew Task(taskname, this, &CQMOM::solveCQMOMInversion312);
  
  //tsk requires on moment eqns
  for (vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0);
  }
  
  //tsk computs on weights
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->modifies(tempLabel);
  }
  
  //tsk computes on abscissas
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->modifies(tempLabel);
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// solveCQMOMInversion 3|1|2
// **********************************************
void CQMOM::solveCQMOMInversion312( const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  //time how long the CQMOM solve takes in total
  double start_SolveTime = Time::currentSeconds();
  
  //change Ni and maxInd to match new varaible order
  vector<int> maxInd_tmp (3);
  vector<int> N_i_tmp (3);
  N_i_tmp[0] = N_i[1]; N_i_tmp[1] = N_i[0]; N_i_tmp[2] = N_i[2];
  maxInd_tmp[0] = maxInd[1]; maxInd_tmp[1] = maxInd[0]; maxInd_tmp[2] = maxInd[2];
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    
    // get moments from data warehouse and put into CCVariable
    vector<constCCVarWrapper> momentCCVars;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      
      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );
      
      // put the wrapper into a vector
      momentCCVars.push_back(tempWrapper);
    }
    
    //get/allocate the weights
    vector<CCVariable<double>* > cqmomWeights;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* weight_label = iW->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      new_dw->getModifiable(*tempCCVar, weight_label, matlIndex, patch);
      cqmomWeights.push_back(tempCCVar);
    }
    
    //get/allocate the abscissas
    vector<CCVariable<double>* > cqmomAbscissas;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* abscissa_label = iA->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      new_dw->getModifiable(*tempCCVar, abscissa_label, matlIndex, patch);
      cqmomAbscissas.push_back(tempCCVar);
    }
    
    for ( CellIterator iter = patch->getExtraCellIterator();
         !iter.done(); ++iter) {
      IntVector c = *iter;
      vector<double> temp_weights (nNodes, 0.0);
      vector<vector<double> > temp_abscissas (M, vector<double> (nNodes, 0.0));
      vector<double> temp_moments (momentSize, 0.0);
      
      //loop over moments and put in vector
      //these are numbered into a flatindex based on moment index
      int ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = momentCCVars.begin(); iter!= momentCCVars.end(); ++iter) {
        double temp_value = (iter->data)[c];
        int flatIndex;
        
        vector<int> temp_index = momentIndexes[ii];
        
        //change flatindex value here, now 2 = i, 1 = j, 3 = k
        if (M == 2) {
          flatIndex = temp_index[1] + temp_index[0]*maxInd[1];
        } else if (M == 3) {
          flatIndex = temp_index[1] + temp_index[0] * maxInd[1] + temp_index[2]*maxInd[1]*maxInd[0];
        }
        
        temp_moments[flatIndex] = temp_value;
        ii++;
      }
      
      //actually do the cqmom inversion step
      if (temp_moments[0] < d_small) {
        //if m0 is very small, leave all weights/absciassa equal to 0 (as intialized)
        if (temp_moments[0] < 0.0 )
          cout << "WARNING: Negative Moment " << temp_moments[0] <<  " in cell " << c << " settign all wegiths and abscissas to 0" << endl;
      } else {
#ifdef cqmom_dbg
        cout << "Permutation 3|1|2 Cell Location " << c << endl;
        cout << "______________" << endl;
#endif

        
        //actually compute inversion
        CQMOMInversion( temp_moments, M, N_i_tmp, maxInd_tmp,
                       temp_weights, temp_abscissas, d_adaptive, d_useLapack, weightRatio, abscissaRatio);
      }
      
      //Now actually assign the new weights and abscissas
      // need to fill these in correct varlabel order as 123, but absicssas are 213
      // make temp vectors to rearrange absciassas
      std::vector<double> aTemp1 (nNodes);
      std::vector<double> aTemp2 (nNodes);
      
      for (int z = 0; z < nNodes; z++ ) {
        aTemp1[z] = temp_abscissas[1][z];
        aTemp2[z] = temp_abscissas[0][z];
      }
      
      for (int z = 0; z < nNodes; z++ ) {
        temp_abscissas[0][z] = aTemp1[z];
        temp_abscissas[1][z] = aTemp2[z];
      }
      
      int jj = 0;
      for (int m = 0; m < M; m++) {
        for (int z = 0; z < nNodes; z++) {
          (*(cqmomAbscissas[jj]))[c] = temp_abscissas[m][z];
          jj++;
        }
      }
      for (int z = 0; z < nNodes; z++) {
        (*(cqmomWeights[z]))[c] = temp_weights[z];
      }
      
    } //end cell loop
    
    //delete pointers
    for(unsigned int i=0; i<cqmomAbscissas.size(); ++i ){
      delete cqmomAbscissas[i];
    }
    
    for(unsigned int i=0; i<cqmomWeights.size(); ++i ){
      delete cqmomWeights[i];
    }
  } //end patch loop
  double total_SolveTime = (Time::currentSeconds() - start_SolveTime);
  proc0cout << "CQMOM Solve time: " << total_SolveTime << endl;
}

// **********************************************
// sched_solveCQMOMInversion 2|1|3
// **********************************************
void
CQMOM::sched_solveCQMOMInversion213( const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  string taskname = "CQMOM:solveCQMOMInversion213";
  Task* tsk = scinew Task(taskname, this, &CQMOM::solveCQMOMInversion213);
  
  //tsk requires on moment eqns
  for (vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0);
  }
  
  //tsk computes on weights
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->modifies(tempLabel);
  }
  
  //tsk computes on abscissas
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->modifies(tempLabel);
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// solveCQMOMInversion 2|1|3
// **********************************************
void CQMOM::solveCQMOMInversion213( const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  //time how long the CQMOM solve takes in total
  double start_SolveTime = Time::currentSeconds();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    // get moments from data warehouse and put into CCVariable
    vector<constCCVarWrapper> momentCCVars;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      
      // instead of using a CCVariable, use a constCCVarWrapper struct
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, equation_label, matlIndex, patch, Ghost::None, 0 );
      
      // put the wrapper into a vector
      momentCCVars.push_back(tempWrapper);
    }
    
    //get/allocate the weights
    vector<CCVariable<double>* > cqmomWeights;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* weight_label = iW->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      new_dw->getModifiable(*tempCCVar, weight_label, matlIndex, patch);
      cqmomWeights.push_back(tempCCVar);
    }
    
    //get/allocate the abscissas
    vector<CCVariable<double>* > cqmomAbscissas;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* abscissa_label = iA->second;
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      new_dw->getModifiable(*tempCCVar, abscissa_label, matlIndex, patch);
      cqmomAbscissas.push_back(tempCCVar);
    }
    
    for ( CellIterator iter = patch->getExtraCellIterator();
         !iter.done(); ++iter) {
      IntVector c = *iter;
      vector<double> temp_weights (nNodes, 0.0);
      vector<vector<double> > temp_abscissas (M, vector<double> (nNodes, 0.0));
      vector<double> temp_moments (momentSize, 0.0);
      
      //loop over moments and put in vector
      //these are numbered into a flatindex based on moment index
      int ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = momentCCVars.begin(); iter!= momentCCVars.end(); ++iter) {
        double temp_value = (iter->data)[c];
        int flatIndex;
        
        vector<int> temp_index = momentIndexes[ii];
        
        //change flatindex value here, now 2 = i, 1 = j, 3 = k
        if (M == 2) {
          flatIndex = temp_index[2] + temp_index[0]*maxInd[2];
        } else if (M == 3) {
          flatIndex = temp_index[2] + temp_index[0] * maxInd[2] + temp_index[1]*maxInd[2]*maxInd[0];
        }
        
        temp_moments[flatIndex] = temp_value;
        ii++;
      }
      
      //actually do the cqmom inversion step
      if (temp_moments[0] < d_small) {
        //if m0 is very small, leave all weights/absciassa equal to 0 (as intialized)
        if (temp_moments[0] < 0.0 )
          cout << "WARNING: Negative Moment " << temp_moments[0] <<  " in cell " << c << " settign all wegiths and abscissas to 0" << endl;
      } else {
#ifdef cqmom_dbg
        cout << "Permutation 2|1|3 Cell Location " << c << endl;
        cout << "______________" << endl;
#endif
        //redo Ni/maxind
        vector<int> maxInd_tmp (3);
        vector<int> N_i_tmp (3);
        N_i_tmp[0] = N_i[2]; N_i_tmp[1] = N_i[0]; N_i_tmp[2] = N_i[1];
        maxInd_tmp[0] = maxInd[2]; maxInd_tmp[1] = maxInd[0]; maxInd_tmp[2] = maxInd[1];
        //actually compute inversion
        CQMOMInversion( temp_moments, M, N_i, maxInd,
                       temp_weights, temp_abscissas, d_adaptive, d_useLapack, weightRatio, abscissaRatio);
      }
      
      //Now actually assign the new weights and abscissas
      // need to fill these in correct varlabel order as 123, but absicssas are 312
      // make temp vectors to rearrange abscissas
      std::vector<double> aTemp1 (nNodes);
      std::vector<double> aTemp2 (nNodes);
      std::vector<double> aTemp3 (nNodes);
      
      for (int z = 0; z < nNodes; z++ ) {
        aTemp1[z] = temp_abscissas[1][z];
        aTemp2[z] = temp_abscissas[2][z];
        aTemp3[z] = temp_abscissas[0][z];
      }
      
      for (int z = 0; z < nNodes; z++ ) {
        temp_abscissas[0][z] = aTemp1[z];
        temp_abscissas[1][z] = aTemp2[z];
        temp_abscissas[2][z] = aTemp3[z];
      }
      
      //Now actually assign the new weights and abscissas
      int jj = 0;
      for (int m = 0; m < M; m++) {
        for (int z = 0; z < nNodes; z++) {
          (*(cqmomAbscissas[jj]))[c] = temp_abscissas[m][z];
          jj++;
        }
      }
      for (int z = 0; z < nNodes; z++) {
        (*(cqmomWeights[z]))[c] = temp_weights[z];
      }
      
    } //end cell loop
    
    //delete pointers
    for(unsigned int i=0; i<cqmomAbscissas.size(); ++i ){
      delete cqmomAbscissas[i];
    }
    
    for(unsigned int i=0; i<cqmomWeights.size(); ++i ){
      delete cqmomWeights[i];
    }
  } //end patch loop
  double total_SolveTime = (Time::currentSeconds() - start_SolveTime);
  proc0cout << "CQMOM Solve time: " << total_SolveTime << endl;
}