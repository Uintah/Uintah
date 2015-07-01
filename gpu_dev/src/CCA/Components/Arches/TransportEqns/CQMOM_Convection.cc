
#include <CCA/Components/Arches/TransportEqns/CQMOM_Convection.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Geometry/IntVector.h>
#include <Core/ProblemSpec/ProblemSpec.h>


//===========================================================================

using namespace std;
using namespace Uintah;


CQMOM_Convection::CQMOM_Convection(ArchesLabel* fieldLabels):
d_fieldLabels(fieldLabels)
{
  uVelIndex = -1;
  vVelIndex = -1;
  wVelIndex = -1;
  partVel = false;
}

CQMOM_Convection::~CQMOM_Convection()
{
  //NOTE:destory extra var labels if needed
  if ( partVel )
    delete _opr;
}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void
CQMOM_Convection::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP db = params;
  partVel = true;    //only call problem setup if Part vel is on, this boolean then deletes interpolant object if needed
  db->get("NumberInternalCoordinates",M);   //get number of coordiantes
  db->get("QuadratureNodes",N_i);           //get vector of quad nodes per internal coordiante
  db->get( "conv_scheme", d_convScheme);
  db->getWithDefault("RestitutionCoefficient",epW,1.0);
  if (epW > 1.0 )
    epW = 1.0;
  if (epW <= 0.0 )
    epW = 1.0e-10;
  db->getWithDefault("ConvectionWeightLimit",convWeightLimit, 1.0e-10);
  nNodes = 1;
  for (unsigned int i = 0; i<N_i.size(); i++) {
    nNodes *= N_i[i];
  }
  
  //get internal coordinate indexes for each velocity direction
  int m = 0;
  for ( ProblemSpecP db_name = db->findBlock("InternalCoordinate");
       db_name != 0; db_name = db_name->findNextBlock("InternalCoordinate") ) {
    string varType;
    db_name->getAttribute("type",varType);
    if (varType == "uVel") {
      uVelIndex = m;
    } else if (varType == "vVel") {
      vVelIndex = m;
    } else if (varType == "wVel") {
      wVelIndex = m;
    }
    m++;
  }
  
  nMoments = 0;
  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment");
       db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
    temp_moment_index.resize(0);
    db_moments->get("m", temp_moment_index);
    
    // put moment index into vector of moment indices:
    momentIndexes.push_back(temp_moment_index);
    
    //base moment name
    string name = "m_";
    for (int i = 0; i<M ; i++) {
      string node;
      std::stringstream out;
      out << temp_moment_index[i];
      node = out.str();
      name += node;
    }
    
    //push back convection varlabels here
    string conv = name + "_Fconv";
    const VarLabel * tempConvLabel;
    tempConvLabel = VarLabel::find( conv );
    convLabels.push_back( tempConvLabel );
    
    string xConv = name + "_FconvX";
    const VarLabel * xtempConvLabel;
    xtempConvLabel = VarLabel::find( xConv );
    xConvLabels.push_back( xtempConvLabel );
    
    string yConv = name + "_FconvY";
    const VarLabel * ytempConvLabel;
    ytempConvLabel = VarLabel::find( yConv );
    yConvLabels.push_back( ytempConvLabel );
    
    string zConv = name + "_FconvZ";
    const VarLabel * ztempConvLabel;
    ztempConvLabel = VarLabel::find( zConv );
    zConvLabels.push_back( ztempConvLabel );
    
    nMoments++; // keep track of total number of moments
  }
  
  //make interpolant object
  if (d_convScheme == "first" ) {
    _opr = scinew FirstOrderInterpolant();
  } else if (d_convScheme == "second" ) {
    _opr = scinew SecondOrderInterpolant();
  } else {
    throw InvalidValue("Error: Convection scheme not recognized. Check UPS file and try again.", __FILE__, __LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
CQMOM_Convection::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CQMOM_Convection::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &CQMOM_Convection::initializeVariables);
  
  for ( int i = 0; i < nMoments; i++ ) {
    tsk->computes( convLabels[i] );
    tsk->computes( xConvLabels[i] );
    tsk->computes( yConvLabels[i] );
    tsk->computes( zConvLabels[i] );
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually initialize the variables.
//---------------------------------------------------------------------------
void
CQMOM_Convection::initializeVariables( const ProcessorGroup* pc,
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    for ( int i = 0; i <nMoments; i++ ) {
      CCVariable<double> Fconv;
      new_dw->allocateAndPut( Fconv, convLabels[i], matlIndex, patch );
      Fconv.initialize(0.0);
  //    cout << "intializing fconv... " << endl;
      
      CCVariable<double> FconvX;
      new_dw->allocateAndPut( FconvX, xConvLabels[i], matlIndex, patch );
      FconvX.initialize(0.0);
      
      CCVariable<double> FconvY;
      new_dw->allocateAndPut( FconvY, yConvLabels[i], matlIndex, patch );
      FconvY.initialize(0.0);
      
      CCVariable<double> FconvZ;
      new_dw->allocateAndPut( FconvZ, zConvLabels[i], matlIndex, patch );
      FconvZ.initialize(0.0);
    }
  }
}


//---------------------------------------------------------------------------
// Method: Schedule the calcualting the convective terms
//---------------------------------------------------------------------------
void
CQMOM_Convection::sched_solveCQMOMConvection( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOM_Convection::solveCQMOMConvection";
  Task* tsk = scinew Task(taskname, this, &CQMOM_Convection::solveCQMOMConvection);
  
  
  tsk->requires(Task::OldDW, d_fieldLabels->d_cellTypeLabel, Ghost::AroundCells, 1);
  
  //requires updated weights and abscissas
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    if (timeSubStep == 0 ) {
      tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 2 );
    } else {
      tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
    }
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    if (timeSubStep == 0 ) {
      tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 2 );
    } else {
      tsk->requires( Task::NewDW, tempLabel, Ghost::AroundCells, 2 );
    }
  }
  
  //computes convection terms
  for ( int i = 0; i < nMoments; i++ ) {
//    if (timeSubStep == 0 ) {
//      tsk->computes( convLabels[i] );
//      tsk->computes( xConvLabels[i] );
//      tsk->computes( yConvLabels[i] );
//      tsk->computes( zConvLabels[i] );
//    } else {
      tsk->modifies( convLabels[i] );
      tsk->modifies( xConvLabels[i] );
      tsk->modifies( yConvLabels[i] );
      tsk->modifies( zConvLabels[i] );
//    }
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}


//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOM_Convection::solveCQMOMConvection( const ProcessorGroup* pc,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    Vector Dx = patch->dCell();
    
    constCCVariable<int> cellType;
    old_dw->get(cellType, d_fieldLabels->d_cellTypeLabel, matlIndex, patch, gac, 1);
    
    //allocate convective terms
    vector<CCVariable<double>* > Fconv;
    vector<CCVariable<double>* > FconvX;
    vector<CCVariable<double>* > FconvY;
    vector<CCVariable<double>* > FconvZ;

    for (int i = 0; i < nMoments; i++ ) {
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      CCVariable<double>* xTempCCVar = scinew CCVariable<double>;
      CCVariable<double>* yTempCCVar = scinew CCVariable<double>;
      CCVariable<double>* zTempCCVar = scinew CCVariable<double>;
      
      if( new_dw->exists(convLabels[i], matlIndex, patch) ) {
        new_dw->getModifiable(*tempCCVar, convLabels[i], matlIndex, patch);
        new_dw->getModifiable(*xTempCCVar, xConvLabels[i], matlIndex, patch);
        new_dw->getModifiable(*yTempCCVar, yConvLabels[i], matlIndex, patch);
        new_dw->getModifiable(*zTempCCVar, zConvLabels[i], matlIndex, patch);
      } else {
        new_dw->allocateAndPut(*tempCCVar, convLabels[i], matlIndex, patch);
        new_dw->allocateAndPut(*xTempCCVar, xConvLabels[i], matlIndex, patch);
        new_dw->allocateAndPut(*yTempCCVar, yConvLabels[i], matlIndex, patch);
        new_dw->allocateAndPut(*zTempCCVar, zConvLabels[i], matlIndex, patch);
      }
      Fconv.push_back(tempCCVar);
      FconvX.push_back(xTempCCVar);
      FconvY.push_back(yTempCCVar);
      FconvZ.push_back(zTempCCVar);
    }
    
    for ( int i = 0; i <nMoments; i++ ) {
      Fconv[i]->initialize(0.0);
      FconvX[i]->initialize(0.0);
      FconvY[i]->initialize(0.0);
      FconvZ[i]->initialize(0.0);
    }

    //get weights and abscissas
    vector<constCCVarWrapper> weights;
    vector<constCCVarWrapper> abscissas;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      constCCVarWrapper tempWrapper;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      } else {
        old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      }
      weights.push_back(tempWrapper);
    }
    
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      constCCVarWrapper tempWrapper;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      } else {
        old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gac, 2 );
      }
      abscissas.push_back(tempWrapper);
    }
    
    //-------------------- Interior cell loop
    CellIterator iIter  = getInteriorCellIterator( patch );
    for (iIter.begin(); !iIter.done(); iIter++){
      
      IntVector c   = *iIter;
      double area;
      int currVelIndex;
      
      cqFaceData1D gPhi;
      
      if ( uVelIndex > -1 ) {
      // do X convection:
      //--------------------------------------------------
        IntVector coord = IntVector(1,0,0);
        area = Dx.y() * Dx.z();
        currVelIndex = uVelIndex;
      
        int aSize = M*nNodes;
        std::vector<cqFaceData1D> faceAbscissas (aSize);
        std::vector<cqFaceData1D> faceWeights (nNodes);
      
        int ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
          faceWeights[ii] = _opr->no_bc_weight( c, coord, (iter->data), cellType, epW );
          ii++;
        }
      
        ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
          if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
            faceAbscissas[ii] = _opr->no_bc_normVel( c, coord, (iter->data), cellType, epW );
          } else {
            faceAbscissas[ii] = _opr->no_bc( c, coord, (iter->data), cellType, epW );
          }
          ii++;
        }
#ifdef cqmom_transport_dbg
      std::cout << "Cell location: " << c << " in dimension x" << std::endl;
      std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          (*(FconvX[i]))[c] = getFlux( area, gPhi, c, cellType );
        }
      }
      
      if (vVelIndex > -1 ) {
      // do Y convection
      //----------------------------------------
        IntVector coord = IntVector(0,1,0);
        area = Dx.x() * Dx.z();
        currVelIndex = vVelIndex;
        
        int aSize = M*nNodes;
        std::vector<cqFaceData1D> faceAbscissas (aSize);
        std::vector<cqFaceData1D> faceWeights (nNodes);
        
        int ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
          faceWeights[ii] = _opr->no_bc_weight( c, coord, (iter->data), cellType, epW );
          ii++;
        }
        
        ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
          if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
            faceAbscissas[ii] = _opr->no_bc_normVel( c, coord, (iter->data), cellType, epW );
          } else {
            faceAbscissas[ii] = _opr->no_bc( c, coord, (iter->data), cellType, epW );
          }
          ii++;
        }
#ifdef cqmom_transport_dbg
        std::cout << "Cell location: " << c << " in dimension y" << std::endl;
        std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          (*(FconvY[i]))[c] = getFlux( area, gPhi, c, cellType );
        }
      }
     
      if (wVelIndex > -1 ) {
        // do Y convection
        //----------------------------------------
        IntVector coord = IntVector(0,0,1);
        area = Dx.x() * Dx.y();
        currVelIndex = wVelIndex;
        
        int aSize = M*nNodes;
        std::vector<cqFaceData1D> faceAbscissas (aSize);
        std::vector<cqFaceData1D> faceWeights (nNodes);
        
        int ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
          faceWeights[ii] = _opr->no_bc_weight( c, coord, (iter->data), cellType, epW );
          ii++;
        }

        ii = 0;
        for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
          if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
            faceAbscissas[ii] = _opr->no_bc_normVel( c, coord, (iter->data), cellType, epW );
          } else {
            faceAbscissas[ii] = _opr->no_bc( c, coord, (iter->data), cellType, epW );
          }
          ii++;
        }
#ifdef cqmom_transport_dbg
        std::cout << "Cell location: " << c << " in dimension z" << std::endl;
        std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          (*(FconvZ[i]))[c] = getFlux( area, gPhi, c, cellType );
        }
      }

      for ( int i = 0; i<nMoments; i++ ) {
        (*(Fconv[i]))[c] = (*(FconvX[i]))[c] + (*(FconvY[i]))[c] + (*(FconvZ[i]))[c];
      }
      
    }
    
    //-------------------- Boundary cell loop
    std::vector<Patch::FaceType> bf;
    std::vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);
    
    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){
      
      Patch::FaceType face = *bf_iter;
      IntVector inside = patch->faceDirection(face);
      CellIterator c_iter = getInteriorBoundaryCellIterator( patch, bf_iter );
      cqFaceBoundaryBool faceIsBoundary;
      
      for (c_iter.begin(); !c_iter.done(); c_iter++){
        
        IntVector c = *c_iter - inside;
        cqFaceData1D gPhi;
        double area;
        int currVelIndex;
        
        // do X convection
        // --------------------------------
        if ( uVelIndex > -1 ) {
          IntVector coord = IntVector(1,0,0);
          area = Dx.y() * Dx.z();
          currVelIndex = uVelIndex;
          
          faceIsBoundary = checkFacesForBoundaries( patch, c, coord );
          int aSize = M*nNodes;
          std::vector<cqFaceData1D> faceAbscissas (aSize);
          std::vector<cqFaceData1D> faceWeights (nNodes);
        
          int ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
            faceWeights[ii] = _opr->with_bc_weight( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            ii++;
          }
        
          ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
            if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[ii] = _opr->with_bc_normVel( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            } else {
              faceAbscissas[ii] = _opr->with_bc( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            }
            ii++;
          }

          for ( int i = 0; i < nMoments; i++ ) {
            gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            (*(FconvX[i]))[c] = getFlux( area,  gPhi, c, cellType );
          }
        }
        
        // do Y convection
        // ----------------------------------
        if ( vVelIndex > -1 ) {
          IntVector coord = IntVector(0,1,0);
          area = Dx.x() * Dx.z();
          currVelIndex = vVelIndex;
          
          faceIsBoundary = checkFacesForBoundaries( patch, c, coord );
          int aSize = M*nNodes;
          std::vector<cqFaceData1D> faceAbscissas (aSize);
          std::vector<cqFaceData1D> faceWeights (nNodes);
          
          int ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
            faceWeights[ii] = _opr->with_bc_weight( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            ii++;
          }
          
          ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
            if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[ii] = _opr->with_bc_normVel( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            } else {
              faceAbscissas[ii] = _opr->with_bc( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            }
            ii++;
          }

          for ( int i = 0; i < nMoments; i++ ) {
            gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            (*(FconvY[i]))[c] = getFlux( area,  gPhi, c, cellType );
          }
        }
        
        // do Z convection
        // ----------------------------------
        if ( wVelIndex > -1 ) {
          IntVector coord = IntVector(0,0,1);
          area = Dx.x() * Dx.y();
          currVelIndex = wVelIndex;
          
          faceIsBoundary = checkFacesForBoundaries( patch, c, coord );
          int aSize = M*nNodes;
          std::vector<cqFaceData1D> faceAbscissas (aSize);
          std::vector<cqFaceData1D> faceWeights (nNodes);
          
          int ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
            faceWeights[ii] = _opr->with_bc_weight( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            ii++;
          }
          
          ii = 0;
          for (std::vector<constCCVarWrapper>::iterator iter = abscissas.begin(); iter != abscissas.end(); ++iter) {
            if ( ii >= (currVelIndex*nNodes) && ii < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[ii] = _opr->with_bc_normVel( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            } else {
              faceAbscissas[ii] = _opr->with_bc( c, coord, (iter->data), cellType, epW, faceIsBoundary );
            }
            ii++;
          }

          for ( int i = 0; i < nMoments; i++ ) {
            gPhi     = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            (*(FconvZ[i]))[c] = getFlux( area,  gPhi, c, cellType );
          }
        }
        
        for ( int i = 0; i<nMoments; i++ ) {
          (*(Fconv[i]))[c] = (*(FconvX[i]))[c] + (*(FconvY[i]))[c] + (*(FconvZ[i]))[c];
        }
      }
    }
    
    //delete pointers
    for ( int i = 0; i < nMoments; i++ ) {
      delete Fconv[i];
      delete FconvX[i];
      delete FconvY[i];
      delete FconvZ[i];
    }
    
  } //patch loop
}

