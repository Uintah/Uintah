//NOTE: I just included all DQMOM.cc includes here.  Some can likely be removed later.
#include <CCA/Components/Arches/CQMOM.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/CQMOMEqn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/FileNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/Timers/Timers.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

#include <CCA/Components/Arches/CQMOMInversion.h>

//#define output_warning
//===========================================================================

using namespace std;
using namespace Uintah;


CQMOM::CQMOM(ArchesLabel* fieldLabels, bool usePartVel):
d_fieldLabels(fieldLabels), d_usePartVel(usePartVel)
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
  
  int index_length = 0;
  db->get("NumberInternalCoordinates",M);   //get number of coordiantes
  db->get("QuadratureNodes",N_i);           //get vector of quad nodes per internal coordiante
  maxInd.resize(M);
  for (int i = 0; i < M; i++) {
    maxInd[i] = N_i[i]*2 - 1;
  }
  db->getWithDefault("Adaptive",d_adaptive,false);   //use adaptive quadrature or not - NYI
  db->getWithDefault("CutOff",d_small,1.0e-10);      //value of moment 0 to fix weights and abscissas to 0
  db->getWithDefault("UseLapack",d_useLapack,false); //pick which linear solve to use
  db->getWithDefault("WeightRatio",weightRatio,1.0e-5);     //maximum ratio for min to max weight
  db->getWithDefault("AbscissaRatio",abscissaRatio,1.0e-5); //maximum ratio for min to max spacing of abscissas
  db->getWithDefault("OperatorSplitting",d_doOperatorSplitting,false); //boolean for operator splitting
  
  //NOTE: redo this to only have one xml tag here?
  int m = 0;
  for ( ProblemSpecP db_name = db->findBlock("InternalCoordinate"); db_name != nullptr; db_name = db_name->findNextBlock("InternalCoordinate") ) {
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
    
    ClipInfo clip;
    clip.activated = false;
    ProblemSpecP db_clipping = db_name->findBlock("Clipping");
    if (db_clipping) {
      clip.activated = true;
      
      if ( db_clipping->findBlock("low") )
        clip.do_low = true;
      
      if ( db_clipping->findBlock("high") )
        clip.do_high = true;
      
      db_clipping->getWithDefault("low", clip.low,  -1.e16);
      db_clipping->getWithDefault("high",clip.high, 1.e16);
      db_clipping->getWithDefault("tol", clip.tol, 1e-10);
      db_clipping->getWithDefault("clip_zero",clip.clip_to_zero,false);
      db_clipping->getWithDefault("min_weight",clip.weight_clip,1.0e-5);
      
      if ( !clip.do_low && !clip.do_high )
        throw InvalidValue("Error: A low or high clipping must be specified if the <Clipping> section is activated.", __FILE__, __LINE__);
    }
    clipNodes.push_back(clip);
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
  for ( ProblemSpecP db_moments = db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
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
  
  //populate names for weights with just numberign them 0,1,2,3...
  for (int i = 0; i < nNodes; i++) {
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
    d_fieldLabels->CQMOMWeights[i] = tempVarLabel;
  }
  
  //repeat for the abscissas, but include coordinate names
  int ii = 0;
  for (int m = 0; m < M; m++) {
    string coordName = coordinateNames[m];
    for (int i = 0; i < nNodes; i++) {
      string abscissa_name = coordName + "_";
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
        
    }
  }
  
  // Check to make sure number of total moments specified in input file is correct
  if (!d_doOperatorSplitting) {
    int reqMoments;
    reqMoments = 2*N_i[0];
    for ( int i = 1; i < M; i++ ) {
      int product = N_i[0];
      for ( int j = 1; j < i; j ++) {
        product *= N_i[j];
      }
      product*= (2*N_i[i]-1);
      reqMoments += product;
    }
    if ( nMoments != reqMoments ) {
      proc0cout << "ERROR:CQMOM:ProblemSetup: You specified " << nMoments << " moments, but you need " << reqMoments << " moments." << endl;
      throw InvalidValue( "ERROR:CQMOM:ProblemSetup: The number of moments specified was incorrect!",__FILE__,__LINE__);
    }
  }
  
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

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
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
  Timers::Simple timer;
  timer.start();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
   
    // get moments from data warehouse and put into CCVariable
    std::vector <constCCVariable<double> > momentCCVars ( nMoments );
    int i = 0;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      new_dw->get( momentCCVars[i], equation_label, matlIndex, patch, Ghost::None, 0);
      i++;
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
      for ( int ii = 0; ii < nMoments; ii++ ) {
        double temp_value = momentCCVars[ii][c];
        vector<int> temp_index = momentIndexes[ii];
        
        int flatIndex = temp_index[0];
        for (int i = 1; i < M; i++ ) {
          int product = temp_index[i];
          for (int j = 0; j<i; j++) {
            product *= maxInd[j];
          }
          flatIndex += product;
        }
        
        temp_moments[flatIndex] = temp_value;
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

  proc0cout << "CQMOM Solve time: " << timer().seconds() << endl;
}


 /**********************************************
 schedule the re-calculation of moments
 **********************************************/
void
CQMOM::sched_momentCorrection( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOM::momentCorrection";
  Task* tsk = scinew Task(taskname, this, &CQMOM::momentCorrection);

  //tsk modifies on moment eqns
  for (vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->modifies(tempLabel);
  }
  
  //tsk requires on weights
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    tsk->modifies(tempLabel);
  }
  
  //tsk modifies on abscissas
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    tsk->modifies(tempLabel);
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
}


// **********************************************
// actualyl do the re-calculation of moments
// **********************************************
void
CQMOM::momentCorrection( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw )
{
  /*this will loop over all absciassa of the system and determine if any unphysical abscissa
   (as determined by the user though the inputfile) were calculated from the CQMOM inversion,
   if these are found, the abscissa are first clipped then the moments of that cell are all recalculated
   with the new clipped value of the abscissa */
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    
    // get moments from data warehouse and put into CCVariable
    vector<CCVariable<double>* > ccMoments;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      new_dw->getModifiable( *tempCCVar, equation_label, matlIndex, patch );
      ccMoments.push_back(tempCCVar);
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
      
      bool correctMoments = false;
      //check all abscissa
      for ( int m = 0; m < M; m++ ) {
        if (clipNodes[m].activated ) { //don't check values if clipping is off
          for (int z = 0 ; z < nNodes; z++) {
            if (clipNodes[m].do_high ) {
              if ( (*cqmomAbscissas[z + m*nNodes])[c] > clipNodes[m].high - clipNodes[m].tol ) {
                if ( !clipNodes[m].clip_to_zero ) {
#ifdef output_warning
                  cout << "fix cell " << c << " IC# " << m << " a[" << z << "]= " << (*cqmomAbscissas[z + m*nNodes])[c] << " to " << clipNodes[m].high << " w= " << (*cqmomWeights[z])[c] << endl;
#endif
                  (*cqmomAbscissas[z + m*nNodes])[c] = clipNodes[m].high;
                } else  {
#ifdef output_warning
                  cout << "fix cell " << c << " IC# " << m << " a[" << z << "]= " << (*cqmomAbscissas[z + m*nNodes])[c] << " to " << 0.0 << " w= " << (*cqmomWeights[z])[c] << endl;
                  cout << "M0 " << (*ccMoments[0])[c] << endl;
#endif
                  (*cqmomAbscissas[z + m*nNodes])[c] = 0.0;
                  if ((*cqmomWeights[z])[c] < clipNodes[m].weight_clip) {
                    (*cqmomWeights[z])[c] = 0.0;
                  }
                }
                correctMoments = true;
              }
            }
            
            if (clipNodes[m].do_low ) {
              if ( (*cqmomAbscissas[z + m*nNodes])[c] < clipNodes[m].low + clipNodes[m].tol ) {
                if (!clipNodes[m].clip_to_zero ) {
#ifdef output_warning
                  cout << "fix cell " << c << " IC# " << m << " a[" << z << "]= " << (*cqmomAbscissas[z + m*nNodes])[c] << " to " << clipNodes[m].low  << " w= " << (*cqmomWeights[z])[c] << endl;
#endif
                  (*cqmomAbscissas[z + m*nNodes])[c] = clipNodes[m].low;
                } else {
#ifdef output_warning
                  cout << "fix cell " << c << " IC# " << m << " a[" << z << "]= " << (*cqmomAbscissas[z + m*nNodes])[c] << " to " << 0.0 << " w= " << (*cqmomWeights[z])[c] << endl;
                  cout << "M0 " << (*ccMoments[0])[c] << endl;
#endif
                  (*cqmomAbscissas[z + m*nNodes])[c] = 0.0;
                  if ((*cqmomWeights[z])[c] < clipNodes[m].weight_clip) {
                    (*cqmomWeights[z])[c] = 0.0;
                  }
                }
                correctMoments = true;
              }
            }
          }
        }
      }
      
      //fix moments in this cell if needed
      if ( correctMoments ) {
        int i = 0;
        for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
          vector<int> tempIndex = momentIndexes[i];
          
          double summation = 0.0;
          for ( int z = 0; z < nNodes; z++ ) {
            double product = 1.0;
            for (int m = 0; m < M; m++) {
              
              product *= pow( (*cqmomAbscissas[z + m*nNodes])[c] , tempIndex[m] );
            }
            summation += product * (*cqmomWeights[z])[c];
          }
          (*ccMoments[i])[c] = summation;
          i++;
        }
      }
      
    } //cell loop
    
    //delete pointers
    for(unsigned int i=0; i<cqmomAbscissas.size(); ++i ){
      delete cqmomAbscissas[i];
    }
    
    for(unsigned int i=0; i<cqmomWeights.size(); ++i ){
      delete cqmomWeights[i];
    }
    
    for(unsigned int i=0; i<ccMoments.size(); ++i ){
      delete ccMoments[i];
    }
  } //patch loop
}

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
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
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
  Timers::Simple timer;
  timer.start();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    
    // get moments from data warehouse and put into CCVariable
    std::vector <constCCVariable<double> > momentCCVars ( nMoments );
    int i = 0;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      new_dw->get( momentCCVars[i], equation_label, matlIndex, patch, Ghost::None, 0);
      i++;
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
      for ( int ii = 0; ii < nMoments; ii++ ) {
        double temp_value = momentCCVars[ii][c];
        int flatIndex = 0;
        
        vector<int> temp_index = momentIndexes[ii];
        
        if (M == 2) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0];
        } else if (M == 3) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0] + temp_index[2]*maxInd[0]*maxInd[1];
        } else if ( M == 4 ) {
          flatIndex = temp_index[0] + temp_index[1]*maxInd[0] + temp_index[2]*maxInd[0]*maxInd[1] + temp_index[3]*maxInd[0]*maxInd[1]*maxInd[2];
        }
        
        temp_moments[flatIndex] = temp_value;
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
  
  proc0cout << "CQMOM Solve time: " << timer().seconds() << endl;
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
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
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
  Timers::Simple timer;
  timer.start();
  
  //change Ni and maxInd to match new varaible order
  vector<int> maxInd_tmp (3);
  vector<int> N_i_tmp (3);
  N_i_tmp[0] = N_i[1]; N_i_tmp[1] = N_i[0]; N_i_tmp[2] = N_i[2];
  maxInd_tmp[0] = maxInd[1]; maxInd_tmp[1] = maxInd[0]; maxInd_tmp[2] = maxInd[2];
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    
    // get moments from data warehouse and put into CCVariable
    std::vector <constCCVariable<double> > momentCCVars ( nMoments );
    int i = 0;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      new_dw->get( momentCCVars[i], equation_label, matlIndex, patch, Ghost::None, 0);
      i++;
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
      for ( int ii = 0; ii < nMoments; ii++ ) {
        double temp_value = momentCCVars[ii][c];
        int flatIndex = 0;
        
        vector<int> temp_index = momentIndexes[ii];
        
        //change flatindex value here, now 2 = i, 1 = j, 3 = k
        if (M == 2) {
          flatIndex = temp_index[1] + temp_index[0]*maxInd[1];
        } else if (M == 3) {
          flatIndex = temp_index[1] + temp_index[0] * maxInd[1] + temp_index[2]*maxInd[1]*maxInd[0];
        } else if ( M == 4 ) {
          flatIndex = temp_index[1] + temp_index[0]*maxInd[1] + temp_index[2]*maxInd[1]*maxInd[0] + temp_index[3]*maxInd[0]*maxInd[1]*maxInd[2];
        }
        
        temp_moments[flatIndex] = temp_value;
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

  proc0cout << "CQMOM Solve time: " << timer().seconds() << endl;
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
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_materialManager->allMaterials( "Arches" ));
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
  Timers::Simple timer;
  timer.start();
  
  for (int p = 0; p< patches->size(); ++p) {
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex();
    
    // get moments from data warehouse and put into CCVariable
    std::vector <constCCVariable<double> > momentCCVars ( nMoments );
    int i = 0;
    for( vector<CQMOMEqn*>::iterator iEqn = momentEqns.begin(); iEqn != momentEqns.end(); ++iEqn ) {
      const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
      new_dw->get( momentCCVars[i], equation_label, matlIndex, patch, Ghost::None, 0);
      i++;
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
      for ( int ii = 0; ii < nMoments; ii++ ) {
        double temp_value = momentCCVars[ii][c];
        int flatIndex = 0;
        
        vector<int> temp_index = momentIndexes[ii];
        
        //change flatindex value here, now 2 = i, 1 = j, 3 = k
        if (M == 2) {
          flatIndex = temp_index[2] + temp_index[0]*maxInd[2];
        } else if (M == 3) {
          flatIndex = temp_index[2] + temp_index[0] * maxInd[2] + temp_index[1]*maxInd[2]*maxInd[0];
        } else if ( M == 4 ) {
          flatIndex = temp_index[2] + temp_index[0]*maxInd[2] + temp_index[1]*maxInd[2]*maxInd[0] + temp_index[3]*maxInd[0]*maxInd[1]*maxInd[2];
        }
        
        temp_moments[flatIndex] = temp_value;
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

  proc0cout << "CQMOM Solve time: " << timer().seconds() << endl;
}
