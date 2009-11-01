#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/LU.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Thread/Time.h>

#include <fstream>

// Output matrices once per timestep (will cause a big slowdown)
//#define DEBUG_MATRICES 1

//===========================================================================

using namespace Uintah;

DQMOM::DQMOM(ArchesLabel* fieldLabels):
d_fieldLabels(fieldLabels)
{

  string varname;
  
  varname = "normB";
  d_normBLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normX";
  d_normXLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normRes";
  d_normResLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "normResNormalized";
  d_normResNormalizedLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

  varname = "determinant";
  d_determinantLabel = VarLabel::create(varname, CCVariable<double>::getTypeDescription());

}

DQMOM::~DQMOM()
{
  VarLabel::destroy(d_normBLabel); 
  VarLabel::destroy(d_normXLabel); 
  VarLabel::destroy(d_normResLabel);
  VarLabel::destroy(d_normResNormalizedLabel);
  VarLabel::destroy(d_determinantLabel);
}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void DQMOM::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params; 
  unsigned int moments = 0;
  unsigned int index_length = 0;
  moments = 0;

  d_small_B = 1e-10;

  db->getWithDefault("LU_solver_tolerance", d_solver_tolerance, 1.0e-5);

  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment");
        db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
    temp_moment_index.resize(0);
    db_moments->get("m", temp_moment_index);
    
    // put moment index into vector of moment indexes:
    momentIndexes.push_back(temp_moment_index);
    
    // keep track of total number of moments
    ++moments;

    index_length = temp_moment_index.size();
  }

  db->getWithDefault("save_moments", b_save_moments, true);
  if( b_save_moments ) {
    DQMOM::populateMomentsMap(momentIndexes);
  }

  // This block puts the labels in the same order as the input file, so the moment indexes match up OK
  
  DQMOMEqnFactory & eqn_factory = DQMOMEqnFactory::self();
  //CoalModelFactory & model_factory = CoalModelFactory::self();
  N_ = eqn_factory.get_quad_nodes();
 
  for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
    string weight_name = "w_qn";
    string node;
    stringstream out;
    out << alpha;
    node = out.str();
    weight_name += node;
    // store equation:
    EqnBase& temp_weightEqnE = eqn_factory.retrieve_scalar_eqn( weight_name );
    DQMOMEqn& temp_weightEqnD = dynamic_cast<DQMOMEqn&>(temp_weightEqnE);
    weightEqns.push_back( &temp_weightEqnD );
    d_w_small = temp_weightEqnD.getSmallClip();
    d_weight_scaling_constant = temp_weightEqnD.getScalingConstant();
  }

  N_xi = 0;
  for (ProblemSpecP db_ic = db->findBlock("Ic"); db_ic != 0; db_ic = db_ic->findNextBlock("Ic") ) {
    string ic_name;
    vector<string> modelsList;
    db_ic->getAttribute("label", ic_name);
    for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
      string final_name = ic_name + "_qn";
      string node;
      stringstream out;
      out << alpha;
      node = out.str();
      final_name += node;
      // store equation:
      EqnBase& temp_weightedAbscissaEqnE = eqn_factory.retrieve_scalar_eqn( final_name );
      DQMOMEqn& temp_weightedAbscissaEqnD = dynamic_cast<DQMOMEqn&>(temp_weightedAbscissaEqnE);
      weightedAbscissaEqns.push_back( &temp_weightedAbscissaEqnD );
      d_weighted_abscissa_scaling_constants.push_back( temp_weightedAbscissaEqnD.getScalingConstant() );
    }
    N_xi = N_xi + 1;
  }
  
  // Check to make sure number of total moments specified in input file is correct
  if ( moments != (N_xi+1)*N_ ) {
    proc0cout << "ERROR:DQMOM:ProblemSetup: You specified " << moments << " moments, but you need " << (N_xi+1)*N_ << " moments." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moments specified was incorrect!",__FILE__,__LINE__);
  }

  // Check to make sure number of moment indices matches the number of internal coordinates
  if ( index_length != N_xi ) {
    proc0cout << "ERROR:DQMOM:ProblemSetup: You specified " << index_length << " moment indices, but there are " << N_xi << " internal coordinates." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moment indices specified was incorrect! Need ",__FILE__,__LINE__);
  }

}

// *************************************************************
// Populate the moments map (contains a label for each moment)
// *************************************************************
/** @details  This method populates a map containing moment labels; the map
  *           is contained in the ArchesLabel class and consists of:
  *           first, vector<int> (the moment index, e.g. <0,0,1>);
  *           second, VarLabel (with name e.g. "moment_001").
  *           This allows you to save out and visualize each moment
  *           being given to the DQMOM class; but only if you specify
  *           <save_moments>true</save_moments>.
  */
void
DQMOM::populateMomentsMap( std::vector<MomentVector> allMoments )
{ 
  proc0cout << endl;
  proc0cout << "******* Moment Registration ********" << endl;

  for( vector<MomentVector>::iterator iAllMoments = allMoments.begin();
       iAllMoments != allMoments.end(); ++iAllMoments ) {
    string name = "moment_";
    std::stringstream out;
    for( MomentVector::iterator iMomentIndex = (*iAllMoments).begin();
         iMomentIndex != (*iAllMoments).end(); ++iMomentIndex ) {
      out << (*iMomentIndex);
    }
    name += out.str();
    //vector<int> tempMomentVector = (*iAllMoments);
    const VarLabel* tempVarLabel = VarLabel::create(name, CCVariable<double>::getTypeDescription());

    proc0cout << "Creating label for " << name << endl;

    // actually populate the DQMOMMoments map with moment names
    // e.g. moment_001 &c.
    //d_fieldLabels->DQMOMMoments[tempMomentVector] = tempVarLabel;
    d_fieldLabels->DQMOMMoments[(*iAllMoments)] = tempVarLabel;
  }
}

// **********************************************
// sched_solveLinearSystem
// **********************************************
void
DQMOM::sched_solveLinearSystem( const LevelP& level, SchedulerP& sched, int timeSubStep )
{ 
  string taskname = "DQMOM::solveLinearSystem";
  Task* tsk = scinew Task(taskname, this, &DQMOM::solveLinearSystem);

  CoalModelFactory& model_factory = CoalModelFactory::self();

  if (timeSubStep == 0) {
    proc0cout << "Asking for norm labels" << endl; 
    tsk->computes(d_normBLabel); 
    tsk->computes(d_normXLabel);
    tsk->computes(d_normResLabel);
    tsk->computes(d_normResNormalizedLabel);
    tsk->computes(d_determinantLabel);
  } else {
    tsk->modifies(d_normBLabel); 
    tsk->modifies(d_normXLabel);
    tsk->modifies(d_normResLabel);
    tsk->modifies(d_normResNormalizedLabel);
    tsk->modifies(d_determinantLabel);
  }

  for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    
    // require weights
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );

    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      tsk->computes(sourceterm_label);
    } else {
      tsk->modifies(sourceterm_label);
    }
 
  }
  
  for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    
    // require weighted abscissas
    tsk->requires(Task::NewDW, tempLabel, Ghost::None, 0);

    // compute or modify source terms
    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      tsk->computes(sourceterm_label);
    } else {
      tsk->modifies(sourceterm_label);
    }
    
    // require model terms
    vector<string> modelsList = (*iEqn)->getModelsList();
    for ( vector<string>::iterator iModels = modelsList.begin(); iModels != modelsList.end(); ++iModels ) {
      ModelBase& model_base = model_factory.retrieve_model(*iModels);
      const VarLabel* model_label = model_base.getModelLabel();
      tsk->requires( Task::NewDW, model_label, Ghost::AroundCells, 1 );
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}

// **********************************************
// Actually solve system
// **********************************************
void
DQMOM::solveLinearSystem( const ProcessorGroup* pc,  
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw )
{
  double start_solveLinearSystemTime = Time::currentSeconds();
  double total_CroutSolveTime = 0.0;
  double total_IRSolveTime = 0.0;
  double total_AXBConstructionTime = 0.0;
#ifdef DEBUG_MATRICES
  double total_FileWriteTime = 0.0;
  bool b_writeFile = true;
#endif

  // patch loop
  for (int p=0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gn  = Ghost::None; 
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CoalModelFactory& model_factory = CoalModelFactory::self();

    // get/allocate normB label
    CCVariable<double> normB; 
    if (new_dw->exists(d_normBLabel, matlIndex, patch)) { 
      new_dw->getModifiable(normB, d_normBLabel, matlIndex, patch);
    } else {
      new_dw->allocateAndPut(normB, d_normBLabel, matlIndex, patch);
    }
    normB.initialize(0.0);

    // get/allocate normX label
    CCVariable<double> normX; 
    if (new_dw->exists(d_normXLabel, matlIndex, patch)) { 
      new_dw->getModifiable(normX, d_normXLabel, matlIndex, patch);
    } else {
      new_dw->allocateAndPut(normX, d_normXLabel, matlIndex, patch);
    }
    normX.initialize(0.0);

    // get/allocate normRes label
    CCVariable<double> normRes;
    if( new_dw->exists(d_normResLabel, matlIndex, patch) ) {
      new_dw->getModifiable( normRes, d_normResLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( normRes, d_normResLabel, matlIndex, patch );
    }
    normRes.initialize(0.0);

    // get/allocate normResNormalized label
    CCVariable<double> normResNormalized;
    if( new_dw->exists(d_normResNormalizedLabel, matlIndex, patch) ) {
      new_dw->getModifiable( normResNormalized, d_normResNormalizedLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( normResNormalized, d_normResNormalizedLabel, matlIndex, patch );
    }
    normResNormalized.initialize(0.0);

    // get/allocate determinant label
    CCVariable<double> determinant;
    if( new_dw->exists(d_determinantLabel, matlIndex, patch) ) {
      new_dw->getModifiable( determinant, d_determinantLabel, matlIndex, patch );
    } else {
      new_dw->allocateAndPut( determinant, d_determinantLabel, matlIndex, patch );
    }

    // get/allocate weight source term labels
    for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
         iEqn != weightEqns.end(); iEqn++) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double> tempCCVar;
      if (new_dw->exists(source_label, matlIndex, patch)) {
        new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
      }
      tempCCVar.initialize(0.0);
    }
  
    // get/allocate weighted abscissa source term labels
    for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
         iEqn != weightedAbscissaEqns.end(); ++iEqn) {
      const VarLabel* source_label = (*iEqn)->getSourceLabel();
      CCVariable<double> tempCCVar;
      if (new_dw->exists(source_label, matlIndex, patch)) {
        new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
      } else {
        new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
      }
      tempCCVar.initialize(0.0);
    }

    // Cell iterator
    for ( CellIterator iter = patch->getCellIterator__New();
          !iter.done(); ++iter) {

      LU A ( (N_xi+1)*N_ );
      vector<double> B( A.getDimension(), 0.0 );
      vector<double> Xdoub( A.getDimension(), 0.0 );
      vector<long double> Xlong( A.getDimension(), 0.0 );
      vector<double> Resid( A.getDimension(), 0.0 );
      IntVector c = *iter;

      vector<double> weights;
      vector<double> weightedAbscissas;
      vector<double> models;

      // get weights from data warehouse
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); 
           iEqn != weightEqns.end(); ++iEqn) {
        constCCVariable<double> temp;
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        new_dw->get( temp, equation_label, matlIndex, patch, gn, 0);
        weights.push_back(temp[c]);
      }

      // get weighted abscissas from data warehouse 
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        constCCVariable<double> temp;
        new_dw->get(temp, equation_label, matlIndex, patch, gn, 0);
        weightedAbscissas.push_back(temp[c]);

        double runningsum = 0;
        vector<string> modelsList = (*iEqn)->getModelsList();
        for ( vector<string>::iterator iModels = modelsList.begin();
              iModels != modelsList.end(); ++iModels) {
          ModelBase& model_base = model_factory.retrieve_model(*iModels);
          const VarLabel* model_label = model_base.getModelLabel();
          constCCVariable<double> tempCCVar;
          new_dw->get(tempCCVar, model_label, matlIndex, patch, gn, 0);
          runningsum = runningsum + tempCCVar[c];
        }
        
        models.push_back(runningsum);
      }


      // FIXME:
      // This construction process needs to be using d_w_small to check for small weights!

      double start_AXBConstructionTime = Time::currentSeconds();

      // construct AX=B
      for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
        MomentVector thisMoment = momentIndexes[k];
        
        // weights
        for ( unsigned int alpha = 0; alpha < N_; ++alpha) {
          double prefixA = 1;
          double productA = 1;
          for ( unsigned int i = 0; i < thisMoment.size(); ++i) {
            if (weights[alpha] != 0) {
              // Appendix C, C.9 (A1 matrix)
              prefixA = prefixA - (thisMoment[i]);
              productA = productA*( pow((weightedAbscissas[i*(N_)+alpha]/weights[alpha]),thisMoment[i]) );
            } else {
              prefixA = 0;
              productA = 0;
            }
          }
          A(k,alpha)=prefixA*productA;
        } //end weights sub-matrix

        // weighted abscissas
        double totalsumS = 0;
        for( unsigned int j = 0; j < N_xi; ++j ) {
          double prefixA    = 1;
          double productA   = 1;
          
          double prefixS    = 1;
          double productS   = 1;
          double modelsumS  = 0;
          
          double quadsumS = 0;
          for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
            if (weights[alpha] == 0) {
              prefixA = 0;
              productA = 0;
              productS = 0;
            } else if ( weightedAbscissas[j*(N_)+alpha] == 0 && thisMoment[j] == 0) {
              //FIXME:
              // do something
            } else {
              // Appendix C, C.11 (A_j+1 matrix)
              prefixA = (thisMoment[j])*( pow((weightedAbscissas[j*(N_)+alpha]/weights[alpha]),(thisMoment[j]-1)) );
              productA = 1;

              // Appendix C, C.16 (S matrix)
              prefixS = -(thisMoment[j])*( pow((weightedAbscissas[j*(N_)+alpha]/weights[alpha]),(thisMoment[j]-1)));
              productS = 1;

              for (unsigned int n = 0; n < N_xi; ++n) {
                if (n != j) {
                  // if statements needed b/c only checking int coord j above
                  if (weights[alpha] == 0) {
                    productA = 0;
                    productS = 0;
                  } else if (weightedAbscissas[n*(N_)+alpha] == 0 && thisMoment[n] == 0) {
                    productA = 0;
                    productS = 0;
                  } else {
                    productA = productA*( pow( (weightedAbscissas[n*(N_)+alpha]/weights[alpha]), thisMoment[n] ));
                    productS = productS*( pow( (weightedAbscissas[n*(N_)+alpha]/weights[alpha]), thisMoment[n] ));
                  }//end divide by zero conditionals
                }
              }//end int coord n
            }//end divide by zero conditionals
            

            modelsumS = - models[j*(N_)+alpha];

            A(k,(j+1)*N_ + alpha)=prefixA*productA;
            
            quadsumS = quadsumS + weights[alpha]*modelsumS*prefixS*productS;
          }//end quad nodes
          totalsumS = totalsumS + quadsumS;
        }//end int coords j sub-matrix
        
        B[k] = totalsumS;
      } // end moments

      total_AXBConstructionTime += Time::currentSeconds() - start_AXBConstructionTime;

      double start_CroutSolveTime = Time::currentSeconds();
 
      // save original A before decomposition into LU
      LU Aorig( A );

      // decompose into LU
      A.decompose();

      // forward- then back-substitution
      A.back_subs( &B[0], &Xdoub[0] );

      total_CroutSolveTime += (Time::currentSeconds() - start_CroutSolveTime);

      // copy Xdoub into Xlong to begin with 
      for( unsigned int j=0; j < Xdoub.size(); ++j ) {
        Xlong[j] = Xdoub[j];
      }

      determinant[c] = A.getDeterminant();

      if( A.isSingular() ) {
        //proc0cout << "WARNING: Arches: DQMOM: matrix A is singular at cell = " << c << ";" << endl;
        //proc0cout << "\t\tDeterminant = " << A.getDeterminant() << endl;
        //for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
        //  proc0cout << "\t\tWeight, qn" << alpha << " = " << weights[alpha] << endl;
        //}

        // Set solution vector = 0.0
        for( vector<long double>::iterator iX = Xlong.begin(); iX != Xlong.end(); ++iX ) {
          (*iX) = 0.0;
        }
        // Set residual vector = 0.0
        for( vector<double>::iterator iR = Resid.begin(); iR != Resid.end(); ++iR ) {
          (*iR) = 0.0;
        }
        total_IRSolveTime = 0.0;

      } else {

        double start_IRSolveTime = Time::currentSeconds();

        // iterative refinement
        bool do_iterative_refinement = false;
        if( do_iterative_refinement )
          A.iterative_refinement( Aorig, &B[0], &Xdoub[0], &Xlong[0] );

        total_IRSolveTime += Time::currentSeconds() - start_IRSolveTime;

        // Find the norm of the residual vector (B-AX)
        Aorig.getResidual( &B[0], &Xlong[0], &Resid[0] );
        double temp = A.getNorm( &Resid[0], 0 );
        normRes[c] = temp;

        // Find the norm of the (normalized) residual vector (B-AX)/(B)
        for( int ii = 0; ii < A.getDimension(); ++ii ) {
          // Normalize by B (not by X)
          if (abs(B[ii]) > d_small_B) 
            Resid[ii] = Resid[ii] / B[ii]; //try normalizing componentwise error
          // else B is zero so b - Ax should also be close to zero
          // so keep residual = Ax 
        }
        normResNormalized[c] = A.getNorm( &Resid[0], 0);
      }
      
      // Find the norm of the RHS and solution vectors
      normB[c] = A.getNorm( &B[0], 0 );
      normX[c] = A.getNorm( &Xlong[0], 0 );



      // set weight transport eqn source terms equal to results
      unsigned int z = 0;
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
           iEqn != weightEqns.end(); iEqn++) {
        const VarLabel* source_label = (*iEqn)->getSourceLabel();
        CCVariable<double> tempCCVar;
        if (new_dw->exists(source_label, matlIndex, patch)) {
          new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
        } else {
          new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
        }

        // Make sure several critera are met for an acceptable solution
        if(  fabs(normResNormalized[c]) > d_solver_tolerance ) {
            tempCCVar[c] = 0;
        } else if( std::isnan(Xlong[z]) ) {
            tempCCVar[c] = 0;
        } else {
          tempCCVar[c] = Xlong[z];
        }
        ++z;
      }
  
      // set weighted abscissa transport eqn source terms equal to results
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        const VarLabel* source_label = (*iEqn)->getSourceLabel();
        CCVariable<double> tempCCVar;
        if (new_dw->exists(source_label, matlIndex, patch)) {
          new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
        } else {
          new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
        }

        // Make sure several critera are met for an acceptable solution
        if(  fabs(normResNormalized[c]) > d_solver_tolerance ) {
            tempCCVar[c] = 0;
        } else if( std::isnan(Xlong[z]) ){
            tempCCVar[c] = 0;
        } else {
          tempCCVar[c] = Xlong[z];
        }
        ++z;
      }

#ifdef DEBUG_MATRICES
      if(b_writeFile) {
        char filename[28];
        int currentTimeStep = d_fieldLabels->d_sharedState->getCurrentTopLevelTimeStep();
        int dimension = A.getDimension();
        int sizeofit;

        double start_FileWriteTime = Time::currentSeconds();

        ofstream oStream;

        // write A matrix to file:
        sizeofit = sprintf( filename, "A_%.2d.mat", currentTimeStep );
        oStream.open(filename);
        for( int iRow = 0; iRow < dimension; ++iRow ) {
          for( int iCol = 0; iCol < dimension; ++iCol ) {
            oStream << " " << Aorig(iRow,iCol); 
          }
          oStream << endl;
        }
        oStream.close();

        // Write X (solution vector) to file:
        sizeofit = sprintf( filename, "X_%.2d.mat", currentTimeStep );
        oStream.open(filename);
        for( vector<long double>::iterator iX = Xlong.begin();
             iX != Xlong.end(); ++iX) {
          oStream << (*iX) << endl;
        }
        oStream.close();

        // write B matrix to file:
        sizeofit = sprintf( filename, "B_%.2d.mat", currentTimeStep );
        oStream.open(filename);
        for( vector<double>::iterator iB = B.begin(); iB != B.end(); ++iB ) {
          oStream << (*iB) << endl;
        }
        oStream.close();

        // write Resid vector to file:
        sizeofit = sprintf( filename, "R_%.2d.mat", currentTimeStep );
        oStream.open(filename);
        for( vector<double>::iterator iR = Resid.begin(); iR != Resid.end(); ++iR ) {
          oStream << (*iR) << endl;
        }
        oStream.close();

        // write determinant of A to file:
        sizeofit = sprintf( filename, "D_%.2d.mat", currentTimeStep );
        oStream.open(filename);
        oStream << A.getDeterminant() << endl;
        oStream.close();

        total_FileWriteTime += Time::currentSeconds() - start_FileWriteTime;
      }
      b_writeFile = false;
#endif

    }//end for cells

  }//end per patch

#ifdef DEBUG_MATRICES
  proc0cout << "Time for file write: " << total_FileWriteTime << " seconds\n";
#endif
  proc0cout << "Time for AX=B construction: " << total_AXBConstructionTime << " seconds\n";
  proc0cout << "Time for LU solve: " << total_CroutSolveTime + total_IRSolveTime << " seconds\n";
    proc0cout << "\t" << "Time for Crout's Method: " << total_CroutSolveTime << " seconds\n";
    proc0cout << "\t" << "Time for iterative refinement: " << total_IRSolveTime << " seconds\n";
  proc0cout << "Time in DQMOM::solveLinearSystem: " << Time::currentSeconds()-start_solveLinearSystemTime << " seconds \n";
}

// **********************************************
// schedule the calculation of moments
// **********************************************
void
DQMOM::sched_calculateMoments( const LevelP& level, SchedulerP& sched, int timeSubStep )
{ 
  string taskname = "DQMOM::calculateMoments";
  Task* tsk = scinew Task(taskname, this, &DQMOM::calculateMoments);

  if( timeSubStep == 0 )
    proc0cout << "Requesting DQMOM moment labels" << endl;
  
  // computing/modifying the actual moments
  for( ArchesLabel::MomentMap::iterator iMoment = d_fieldLabels->DQMOMMoments.begin();
       iMoment != d_fieldLabels->DQMOMMoments.end(); ++iMoment ) {
    if( timeSubStep == 0 )
      tsk->computes( iMoment->second );
    else
      tsk->modifies( iMoment->second );
  }

  // require the weights and weighted abscissas
  for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    // require weights
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );
  }
  for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    // require weighted abscissas
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    tsk->requires(Task::NewDW, tempLabel, Ghost::None, 0);
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

// **********************************************
// actually calculate the moments
// **********************************************
/** @details  This calculates the value of each of the $\f(N_{\xi}+1)N$\f moments
  *           given to the DQMOM class. For a given moment index $\f k = k_1, k_2, \dots $\f, 
  *           the moment is calculated as:
  *           \f[ m_{k} = \sum_{\alpha=1}^{N} w_{\alpha} \prod_{j=1}^{N_{\xi}} \xi_{j}^{k_j} \f]
  */
void
DQMOM::calculateMoments( const ProcessorGroup* pc,  
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{
  // patch loop
  for( int p=0; p < patches->size(); ++p ) {
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gn  = Ghost::None; 
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    // Cell iterator
    for ( CellIterator iter = patch->getCellIterator__New();
          !iter.done(); ++iter) {
      IntVector c = *iter;
      
      vector<double> weights;
      vector<double> weightedAbscissas;

      // get weights from data warehouse
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); 
           iEqn != weightEqns.end(); ++iEqn) {
        constCCVariable<double> temp;
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        new_dw->get( temp, equation_label, matlIndex, patch, gn, 0);
        weights.push_back(temp[c]);
      }
      // get weighted abscissas from data warehouse 
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        constCCVariable<double> temp;
        new_dw->get(temp, equation_label, matlIndex, patch, gn, 0);
        weightedAbscissas.push_back(temp[c]);
      }

      // moment index k = {k1, k2, k3, ...}
      // moment k = \displaystyle{ sum_{alpha=1}^{N}{ w_{\alpha} \prod_{j=1}^{N_xi}{ \langle \xi_{j} \rangle_{\alpha}^{k_j} } } }
      for( vector<MomentVector>::iterator iAllMoments = momentIndexes.begin();
           iAllMoments != momentIndexes.end(); ++iAllMoments ) {

        // Grab the corresponding moment from the DQMOMMoment map (to get the VarLabel associated with this moment)
        const VarLabel* moment_label;
        ArchesLabel::MomentMap::iterator iMoment = d_fieldLabels->DQMOMMoments.find( (*iAllMoments) );
        if( iMoment != d_fieldLabels->DQMOMMoments.end() ) {
          // grab the corresponding label
          moment_label = iMoment->second;
        } else {
          string index;
          std::stringstream out;
          for( MomentVector::iterator iMomentIndex = (*iAllMoments).begin();
               iMomentIndex != (*iAllMoments).end(); ++iMomentIndex ) {
            out << (*iMomentIndex);
            index += out.str();
          }
          string errmsg = "ERROR: DQMOM: calculateMoments: could not find moment index " + index + " in DQMOMMoment map!\n";
          throw InvalidValue( errmsg,__FILE__,__LINE__);
          // FIXME:
          // could not find the moment in the moment map!  
        }

        // Associate a CCVariable<double> with this moment 
        CCVariable<double> moment_k;
        if( new_dw->exists( moment_label, matlIndex, patch ) ) {
          // running getModifiable once for each cell
          new_dw->getModifiable( moment_k, moment_label, matlIndex, patch );
        } else {
          // only run for first cell - so this doesn't wipe out previous calculations
          new_dw->allocateAndPut( moment_k, moment_label, matlIndex, patch );
          moment_k.initialize(0.0);
        }

        // Calculate the value of the moment
        double temp_moment_k = 0.0;
        double running_weights_sum = 0.0; // this is the denominator of p_alpha
        double running_product = 0.0; // this is w_alpha * xi_1_alpha^k1 * xi_2_alpha^k2 * ...

        MomentVector thisMoment = (*iAllMoments);

        for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
          if( weights[alpha] < d_w_small ) {
            temp_moment_k = 0.0;
          } else {
            double weight = weights[alpha]*d_weight_scaling_constant;
            running_weights_sum += weight;
            running_product = weight;
            for( unsigned int j = 0; j < N_xi; ++j ) {
              running_product *= pow( (weightedAbscissas[j*(N_)+alpha]/weights[alpha])*(d_weighted_abscissa_scaling_constants[j]/d_weight_scaling_constant), thisMoment[j] );
            }
          }
          temp_moment_k += running_product;
          running_product = 0.0;
        }

        moment_k[c] = temp_moment_k/running_weights_sum; // normalize environment weight to get environment probability

      }//end all moments

    }//end cells
  }//end patches
}


