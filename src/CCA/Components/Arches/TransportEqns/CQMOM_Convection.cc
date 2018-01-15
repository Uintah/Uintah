
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
  d_deposition = false;
}

CQMOM_Convection::~CQMOM_Convection()
{
  if ( partVel ) {
    delete _opr;
    VarLabel::destroy(d_wallIntegerLabel);
  }
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
  for ( ProblemSpecP db_name = db->findBlock("InternalCoordinate"); db_name != nullptr; db_name = db_name->findNextBlock("InternalCoordinate") ) {
    string varType;
    db_name->getAttribute("type",varType);
    if (varType == "uVel") {
      uVelIndex = m;
    }
    else if (varType == "vVel") {
      vVelIndex = m;
    }
    else if (varType == "wVel") {
      wVelIndex = m;
    }
    m++;
  }

  nMoments = 0;
  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
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

  //var label for handling walls
  string varname = "wallInteger";
  d_wallIntegerLabel = VarLabel::create(varname, CCVariable<int>::getTypeDescription() );

  //create list of variables for deposition, if label is found
  if ( db->findBlock("DepositionLabel") ) {
    string base_deposition_name;
    db->get("DepositionLabel",base_deposition_name);
    d_deposition = true;

    for ( int i = 0; i < nNodes; i++ ) {
      string node;
      stringstream out;
      out << i;
      node = out.str();

      const VarLabel * tempDepositionLabel;
      string deposition_name = base_deposition_name + "_" + node;
      tempDepositionLabel = VarLabel::find( deposition_name );
      fStickLabels.push_back( tempDepositionLabel );
    }
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


  tsk->requires(Task::OldDW, d_fieldLabels->d_volFractionLabel, Ghost::AroundCells, 1);

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

  if ( d_deposition ) {
    for ( int i = 0; i < nNodes; i++ ) {
      tsk->requires( Task::NewDW, fStickLabels[i], Ghost::None, 0 );
    }
  }

  tsk->requires( Task::NewDW, d_wallIntegerLabel, Ghost::AroundCells, 1);

  //computes convection terms
  for ( int i = 0; i < nMoments; i++ ) {
    tsk->modifies( convLabels[i] );
    tsk->modifies( xConvLabels[i] );
    tsk->modifies( yConvLabels[i] );
    tsk->modifies( zConvLabels[i] );
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
    Ghost::GhostType  gn = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    Vector Dx = patch->dCell();

    constCCVariable<double> volFrac;
    old_dw->get(volFrac, d_fieldLabels->d_volFractionLabel, matlIndex, patch, gac, 1);
    constCCVariable<int> wallInt;
    new_dw->get( wallInt, d_wallIntegerLabel, matlIndex, patch, gac, 1);

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
    std::vector <constCCVariable<double> > weights (nNodes);
    std::vector <constCCVariable<double> > abscissas (nNodes * M);

    int j = 0;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( weights[j], tempLabel, matlIndex, patch, gac, 2 );
      } else {
        old_dw->get( weights[j], tempLabel, matlIndex, patch, gac, 2 );
      }
      j++;
    }

    j = 0;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( abscissas[j], tempLabel, matlIndex, patch, gac, 2 );
      } else {
        old_dw->get( abscissas[j], tempLabel, matlIndex, patch, gac, 2 );
      }
      j++;
    }

    //deal with deposition, use a placeholder value = 0.0 if deposition is off
    std::vector <constCCVariable<double> > fStickCC (nNodes);
    if (d_deposition) {
      for ( int i = 0; i < nNodes; i++ ) {
        const VarLabel* tempLabel = fStickLabels[i];
        new_dw->get( fStickCC[i], tempLabel, matlIndex, patch, gn, 0);
      }
    }

    //-------------------- Interior cell loop
    CellIterator iIter  = getInteriorCellIterator( patch );
    for (iIter.begin(); !iIter.done(); iIter++){

      IntVector c   = *iIter;
      double area;
      int currVelIndex;

      //Base on the WallInt variable set the wall normal vector
      std::vector<double> wallNorm (3,0.0);

      if ( wallInt[c] > 30 && wallInt[c] < 40) { //3 wall cells
        if ( wallInt[c] == 31 ) {
          wallNorm[0] = 1.0; wallNorm[1] = 1.0; wallNorm[2] = 1.0;
        } else if ( wallInt[c] == 32 ) {
          wallNorm[0] = 1.0; wallNorm[1] = 1.0; wallNorm[2] = -1.0;
        } else if ( wallInt[c] == 33 ) {
          wallNorm[0] = 1.0; wallNorm[1] = -1.0; wallNorm[2] = 1.0;
        } else if ( wallInt[c] == 34 ) {
          wallNorm[0] = -1.0; wallNorm[1] = 1.0; wallNorm[2] = 1.0;
        }
      }

      if (wallInt[c] > 20 && wallInt[c] < 30 ) { //2 cells
        if ( wallInt[c] == 21 ) {
          wallNorm[0] = 1.0; wallNorm[1] = 1.0;
        } else if ( wallInt[c] == 22 ) {
          wallNorm[0] = 1.0; wallNorm[1] = -1.0;
        } else if ( wallInt[c] == 23 ) {
          wallNorm[0] = 1.0; wallNorm[2] = 1.0;
        } else if ( wallInt[c] == 24 ) {
          wallNorm[0] = 1.0; wallNorm[2] = -1.0;
        } else if ( wallInt[c] == 25 ) {
          wallNorm[1] = 1.0; wallNorm[2] = 1.0;
        } else if ( wallInt[c] == 26 ) {
          wallNorm[1] = -1.0; wallNorm[2] = 1.0;
        }
      }

      cqFaceData1D gPhi;

      //set fstick values for deposition, if no depostion leave at 0
      std::vector<double> fStick (nNodes, 0.0);
      if ( d_deposition ) {
        for (int i = 0; i < nNodes; i++ ) {
          fStick[i] = fStickCC[i][c];
        }
      }

      if ( uVelIndex > -1 ) {
      // do X convection:
      //--------------------------------------------------
        IntVector coord = IntVector(1,0,0);
        area = Dx.y() * Dx.z();
        currVelIndex = uVelIndex;

        int aSize = M*nNodes;
        std::vector<cqFaceData1D> faceAbscissas (aSize);
        std::vector<cqFaceData1D> faceWeights (nNodes);

        //Use a different function for each type of near-wall cell based on number of wall cells touching it
        if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
          for ( int i = 0; i < nNodes; i++ ) {
            _opr->wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                          volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], fStick[i]);
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] >= 21 && wallInt[c] <= 24 ) { //2D wall with x-component
          std::vector<double> epVec (3);
          if (wallInt[c] == 21 || wallInt[c] == 22 ) {
            epVec[0] = epW; epVec[1] = epW; epVec[2] = 1.0;
          } else {
            epVec[0] = epW; epVec[1] = 1.0; epVec[2] = epW;
          }
          for ( int i = 0; i < nNodes; i++ ) {
            int tempIndex;
            tempIndex = (wVelIndex > -1 ) ? wVelIndex : 0;
            _opr->wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*tempIndex],
                          volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, fStick[i]);
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] == 11 || wallInt[c] == 12 ) { //flat x-wall
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_bc_weight( c, coord, weights[i], volFrac, epW, fStick[i] );
          }

          for ( int i = 0; i < nNodes * M; i++ ) {
            if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[i] = _opr->no_bc_normVel( c, coord, abscissas[i], volFrac, epW );
            } else {
              faceAbscissas[i] = _opr->no_bc( c, coord, abscissas[i], volFrac, epW );
            }
          }

        } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_wall(c, coord, weights[i], wallInt );
            for ( int m = 0; m < M; m++ ) {
              faceAbscissas[i + m*nNodes] = _opr->no_wall(c, coord, abscissas[i + m*nNodes], wallInt );
            }
          }
        }

#ifdef cqmom_transport_dbg
      std::cout << "Cell location: " << c << " in dimension x" << std::endl;
      std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          if (volFrac[c] == 1.0 ) {
            gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          } else {
            gPhi.plus = 0.0; gPhi.minus = 0.0;
          }
          (*(FconvX[i]))[c] = getFlux( area, gPhi, c, volFrac );
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

        if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
          for ( int i = 0; i < nNodes; i++ ) {
            _opr->wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                           volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], fStick[i]);
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] == 21 || wallInt[c] == 22 || wallInt[c] == 25 || wallInt[c] == 26 ) { //2D wall with y-component
          std::vector<double> epVec (3);
          if (wallInt[c] == 21 || wallInt[c] == 22 ) {
            epVec[0] = epW; epVec[1] = epW; epVec[2] = 1.0;
          } else {
            epVec[0] = 1.0; epVec[1] = epW; epVec[2] = epW;
          }
          for ( int i = 0; i < nNodes; i++ ) {
            int tempIndex;
            tempIndex = (wVelIndex > -1 ) ? wVelIndex : 0;
            _opr->wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*tempIndex],
                          volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, fStick[i] );
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] == 13 || wallInt[c] == 14 ) { //flat y-wall
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_bc_weight( c, coord, weights[i], volFrac, epW, fStick[i] );
          }

          for ( int i = 0; i < nNodes * M; i++ ) {
            if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[i] = _opr->no_bc_normVel( c, coord, abscissas[i], volFrac, epW );
            } else {
              faceAbscissas[i] = _opr->no_bc( c, coord, abscissas[i], volFrac, epW );
            }
          }

        } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_wall(c, coord, weights[i], wallInt );
            for ( int m = 0; m < M; m++ ) {
              faceAbscissas[i + m*nNodes] = _opr->no_wall(c, coord, abscissas[i + m*nNodes], wallInt);
            }
          }
        }
#ifdef cqmom_transport_dbg
        std::cout << "Cell location: " << c << " in dimension y" << std::endl;
        std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          if (volFrac[c] == 1.0 ) {
            gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          } else {
            gPhi.plus = 0.0; gPhi.minus = 0.0;
          }
          (*(FconvY[i]))[c] = getFlux( area, gPhi, c, volFrac );
        }
      }

      if (wVelIndex > -1 ) {
        // do Z convection
        //----------------------------------------
        IntVector coord = IntVector(0,0,1);
        area = Dx.x() * Dx.y();
        currVelIndex = wVelIndex;

        int aSize = M*nNodes;
        std::vector<cqFaceData1D> faceAbscissas (aSize);
        std::vector<cqFaceData1D> faceWeights (nNodes);

        if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
          for ( int i = 0; i < nNodes; i++ ) {
            _opr->wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                          volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], fStick[i]);
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] >= 23 && wallInt[c] <= 26 ) { //2D wall with z-component
          for ( int i = 0; i < nNodes; i++ ) {
            std::vector<double> epVec (3);
            if (wallInt[c] == 23 || wallInt[c] == 24 ) {
              epVec[0] = epW; epVec[1] = 1.0; epVec[2] = epW;
            } else {
              epVec[0] = 1.0; epVec[1] = epW; epVec[2] = epW;
            }
            _opr->wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                          volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                          faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, fStick[i]);
            for (int m = 0; m < M; m++ ) {
              if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                faceAbscissas[i + nNodes*m] = _opr->no_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW );
              }
            }
          }

        } else if ( wallInt[c] == 15 || wallInt[c] == 16 ) { //flat z-wall
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_bc_weight( c, coord, weights[i], volFrac, epW, fStick[i] );
          }

          for ( int i = 0; i < nNodes * M; i++ ) {
            if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
              faceAbscissas[i] = _opr->no_bc_normVel( c, coord, abscissas[i], volFrac, epW );
            } else {
              faceAbscissas[i] = _opr->no_bc( c, coord, abscissas[i], volFrac, epW );
            }
          }

        } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
          for ( int i = 0; i < nNodes; i++ ) {
            faceWeights[i] = _opr->no_wall(c, coord, weights[i], wallInt );
            for ( int m = 0; m < M; m++ ) {
              faceAbscissas[i + m*nNodes] = _opr->no_wall(c, coord, abscissas[i + m*nNodes], wallInt );
            }
          }
        }
#ifdef cqmom_transport_dbg
        std::cout << "Cell location: " << c << " in dimension z" << std::endl;
        std::cout << "____________________________" << std::endl;
#endif
        for ( int i = 0; i < nMoments; i++ ) {
          if (volFrac[c] == 1.0 ) {
            gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
          } else {
            gPhi.plus = 0.0; gPhi.minus = 0.0;
          }
          (*(FconvZ[i]))[c] = getFlux( area, gPhi, c, volFrac );
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
        std::vector<double> wallNorm (3,0.0);

        if ( wallInt[c] > 30 && wallInt[c] < 40) { //3 wall cells
          if ( wallInt[c] == 31 ) {
            wallNorm[0] = 1.0; wallNorm[1] = 1.0; wallNorm[2] = 1.0;
          } else if ( wallInt[c] == 32 ) {
            wallNorm[0] = 1.0; wallNorm[1] = 1.0; wallNorm[2] = -1.0;
          } else if ( wallInt[c] == 33 ) {
            wallNorm[0] = 1.0; wallNorm[1] = -1.0; wallNorm[2] = 1.0;
          } else if ( wallInt[c] == 34 ) {
            wallNorm[0] = -1.0; wallNorm[1] = 1.0; wallNorm[2] = 1.0;
          }
        }

        if (wallInt[c] > 20 && wallInt[c] < 30 ) { //2 cells
          if ( wallInt[c] == 21 ) {
            wallNorm[0] = 1.0; wallNorm[1] = 1.0;
          } else if ( wallInt[c] == 22 ) {
            wallNorm[0] = 1.0; wallNorm[1] = -1.0;
          } else if ( wallInt[c] == 23 ) {
            wallNorm[0] = 1.0; wallNorm[2] = 1.0;
          } else if ( wallInt[c] == 24 ) {
            wallNorm[0] = 1.0; wallNorm[2] = -1.0;
          } else if ( wallInt[c] == 25 ) {
            wallNorm[1] = 1.0; wallNorm[2] = 1.0;
          } else if ( wallInt[c] == 26 ) {
            wallNorm[1] = -1.0; wallNorm[2] = 1.0;
          }
        }

        //set fstick values for deposition, if no depostion leave at 0
        std::vector<double> fStick (nNodes, 0.0);
        if ( d_deposition ) {
          for (int i = 0; i < nNodes; i++ ) {
            fStick[i] = fStickCC[i][c];
          }
        }

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

          if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
            for ( int i = 0; i < nNodes; i++ ) {
              _opr->bc_wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                               volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                               faceAbscissas[i + nNodes*wVelIndex], faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] >= 21 && wallInt[c] <= 24 ) { //2D wall with x-component
            std::vector<double> epVec (3);
            if (wallInt[c] == 21 || wallInt[c] == 22 ) {
              epVec[0] = epW; epVec[1] = epW; epVec[2] = 1.0;
            } else {
              epVec[0] = epW; epVec[1] = 1.0; epVec[2] = epW;
            }
            for ( int i = 0; i < nNodes; i++ ) {
              int tempIndex;
              tempIndex = (wVelIndex > -1 ) ? wVelIndex : 0;
              _opr->bc_wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*tempIndex],
                               volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                               faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] == 11 || wallInt[c] == 12 ) { //flat x-wall
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->with_bc_weight( c, coord, weights[i], volFrac, epW, faceIsBoundary, fStick[i] );
            }

            for ( int i = 0; i < nNodes * M; i++ ) {
              if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
                faceAbscissas[i] = _opr->with_bc_normVel( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              } else {
                faceAbscissas[i] = _opr->with_bc( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              }
            }

          } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->bc_no_wall(c, coord, weights[i], faceIsBoundary );
              for ( int m = 0; m < M; m++ ) {
                faceAbscissas[i + m*nNodes] = _opr->bc_no_wall(c, coord, abscissas[i + m*nNodes], faceIsBoundary);
              }
            }
          }

          for ( int i = 0; i < nMoments; i++ ) {
            if (volFrac[c] == 1.0 ) {
              gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            } else {
              gPhi.plus = 0.0; gPhi.minus = 0.0;
            }
            (*(FconvX[i]))[c] = getFlux( area,  gPhi, c, volFrac );
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

          if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
            for ( int i = 0; i < nNodes; i++ ) {
              _opr->bc_wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                              volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                              faceAbscissas[i + nNodes*wVelIndex], faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] == 21 || wallInt[c] == 22 || wallInt[c] == 25 || wallInt[c] == 26) { //2D wall with y-component
            std::vector<double> epVec (3);
            if (wallInt[c] == 21 || wallInt[c] == 22 ) {
              epVec[0] = epW; epVec[1] = epW; epVec[2] = 1.0;
            } else {
              epVec[0] = 1.0; epVec[1] = epW; epVec[2] = epW;
            }
            for ( int i = 0; i < nNodes; i++ ) {
              int tempIndex;
              tempIndex = (wVelIndex > -1 ) ? wVelIndex : 0;
              _opr->bc_wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*tempIndex],
                              volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                              faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] == 13 || wallInt[c] == 14 ) { //flat y-wall
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->with_bc_weight( c, coord, weights[i], volFrac, epW, faceIsBoundary, fStick[i] );
            }

            for ( int i = 0; i < nNodes * M; i++ ) {
              if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
                faceAbscissas[i] = _opr->with_bc_normVel( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              } else {
                faceAbscissas[i] = _opr->with_bc( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              }
            }

          } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->bc_no_wall(c, coord, weights[i], faceIsBoundary );
              for ( int m = 0; m < M; m++ ) {
                faceAbscissas[i + m*nNodes] = _opr->bc_no_wall(c, coord, abscissas[i + m*nNodes], faceIsBoundary);
              }
            }
          }

          for ( int i = 0; i < nMoments; i++ ) {
            if (volFrac[c] == 1.0 ) {
              gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            } else {
              gPhi.plus = 0.0; gPhi.minus = 0.0;
            }
            (*(FconvY[i]))[c] = getFlux( area,  gPhi, c, volFrac );
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

          if ( wallInt[c] >= 31 && wallInt[c] <= 34 ) { //3D wall
            for ( int i = 0; i < nNodes; i++ ) {
              _opr->bc_wall3D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                              volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                              faceAbscissas[i + nNodes*wVelIndex], faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] >= 23 && wallInt[c] <= 26 ) { //2D wall with z-component
            std::vector<double> epVec (3);
            if (wallInt[c] == 21 || wallInt[c] == 22 ) {
              epVec[0] = epW; epVec[1] = 1.0; epVec[2] = epW;
            } else {
              epVec[0] = 1.0; epVec[1] = epW; epVec[2] = epW;
            }
            for ( int i = 0; i < nNodes; i++ ) {
              _opr->bc_wall2D( c, coord, weights[i], abscissas[i + nNodes*uVelIndex], abscissas[i + nNodes*vVelIndex], abscissas[i + nNodes*wVelIndex],
                              volFrac, epW, wallNorm, faceWeights[i], faceAbscissas[i + nNodes*uVelIndex], faceAbscissas[i +nNodes*vVelIndex],
                              faceAbscissas[i + nNodes*wVelIndex], wVelIndex, epVec, faceIsBoundary, fStick[i]);
              for (int m = 0; m < M; m++ ) {
                if ( m != uVelIndex && m != vVelIndex && m != wVelIndex ) { //scalar IC
                  faceAbscissas[i + nNodes*m] = _opr->with_bc( c, coord, abscissas[i + nNodes*m], volFrac, epW, faceIsBoundary );
                }
              }
            }

          } else if ( wallInt[c] == 15 || wallInt[c] == 16 ) { //flat z-wall
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->with_bc_weight( c, coord, weights[i], volFrac, epW, faceIsBoundary, fStick[i] );
            }

            for ( int i = 0; i < nNodes * M; i++ ) {
              if ( i >= (currVelIndex*nNodes) && i < (currVelIndex+1)*nNodes ) { //check if wall is in this direction of velocity convection
                faceAbscissas[i] = _opr->with_bc_normVel( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              } else {
                faceAbscissas[i] = _opr->with_bc( c, coord, abscissas[i], volFrac, epW, faceIsBoundary );
              }
            }

          } else { // wallint = 0 (no wall) or 99 (possible bad case to handle later)
            for ( int i = 0; i < nNodes; i++ ) {
              faceWeights[i] = _opr->bc_no_wall(c, coord, weights[i], faceIsBoundary );
              for ( int m = 0; m < M; m++ ) {
                faceAbscissas[i + m*nNodes] = _opr->bc_no_wall(c, coord, abscissas[i + m*nNodes], faceIsBoundary);
              }
            }
          }

          for ( int i = 0; i < nMoments; i++ ) {
            if (volFrac[c] == 1.0 ) {
              gPhi = sumNodes( faceWeights, faceAbscissas, nNodes, M, currVelIndex, momentIndexes[i], convWeightLimit );
            } else {
              gPhi.plus = 0.0; gPhi.minus = 0.0;
            }
            (*(FconvZ[i]))[c] = getFlux( area,  gPhi, c, volFrac );
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

//---------------------------------------------------------------------------
// Method: Schedule calculating the number of wall cells touching each flow cell
//---------------------------------------------------------------------------
void
CQMOM_Convection::sched_initializeWalls( const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  string taskname = "CQMOM_Convection::intializeWalls";
  Task* tsk = scinew Task(taskname, this, &CQMOM_Convection::initializeWalls);

  tsk->requires(Task::OldDW, d_fieldLabels->d_volFractionLabel, Ghost::AroundCells, 1);

  if (timeSubStep == 0) {
    tsk->computes(d_wallIntegerLabel);
  } else {
    tsk->modifies(d_wallIntegerLabel);
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

//---------------------------------------------------------------------------
// Method: Actually build the transport equation.
//---------------------------------------------------------------------------
void
CQMOM_Convection::initializeWalls( const ProcessorGroup* pc,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++) {

    Ghost::GhostType  gac = Ghost::AroundCells;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    //Vector Dx = patch->dCell();

    constCCVariable<double> vf;
    old_dw->get(vf, d_fieldLabels->d_volFractionLabel, matlIndex, patch, gac, 1);
    CCVariable<int> wallInt;

    if ( new_dw->exists(d_wallIntegerLabel, matlIndex, patch) ) {
      new_dw->getModifiable(wallInt, d_wallIntegerLabel, matlIndex, patch);
    } else {
      new_dw->allocateAndPut(wallInt, d_wallIntegerLabel, matlIndex, patch);
    }

    //set a CC variable to denote number of wall cells touching a flow cell near a wall
    //first integer digit denotes number of cells touchign, and the 2nd digit is the case
    //i.e 11-16 flat 1D wall, 21-26 2D angle wall, 31-34 3D angle wall
    for (CellIterator citer=patch->getCellIterator(); !citer.done(); citer++){
      IntVector c = *citer;
      IntVector cxm = c - IntVector(1, 0, 0);
      IntVector cxp = c + IntVector(1, 0, 0);
      IntVector cym = c - IntVector(0, 1, 0);
      IntVector cyp = c + IntVector(0, 1, 0);
      IntVector czm = c - IntVector(0, 0, 1);
      IntVector czp = c + IntVector(0, 0, 1);
      if ( vf[c] == 0.0 ) {
        wallInt[c] = 0;
      } else {
        if ( vf[cxm] == 1.0 && vf[cxp] == 1.0 && vf[cym] == 1.0 && vf[cyp] == 1.0 && vf[czm] == 1.0 && vf[czp] == 1.0 ) {
          wallInt[c] = 0; //no wall touching this flow cell

          //PlaceHolder if these "teeth" type cells cause issues later
//        } else if ( vf[cxm] == 0.0 && vf[cym] == 0.0 && vf[czm] == 0.0 && vf[czp] == 0.0 ) { //check 4 wall cells corner
//          wallInt[c] = 401;
//        } else if ( vf[cxm] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 402;
//        } else if ( vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[czm] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 403;
//        } else if ( vf[cxp] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 404;
//        } else if ( vf[cxm] == 0.0 && vf[cym] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0 ) {
//          wallInt[c] = 405;
//        } else if ( vf[cxm] == 0.0 && vf[cym] == 0.0 && vf[cyp] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 406;
//        } else if ( vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0 ) {
//          wallInt[c] = 407;
//        } else if ( vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[cyp] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 408;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[czm] == 0.0 ) {
//          wallInt[c] = 409;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 410;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0 ) {
//          wallInt[c] = 411;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 && vf[cyp] == 0.0 && vf[czp] == 0.0 ) {
//          wallInt[c] = 412;
//
//        } else if ( vf[cxm] == 0.0 && vf[cym] == 0.0 & vf[cyp] == 0.0) { //check for 3 wall cell "teeth"
//          wallInt[c] = 301;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 & vf[cym] == 0.0) {
//          wallInt[c] = 302;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 & vf[cyp] == 0.0) {
//          wallInt[c] = 303;
//        } else if ( vf[cxp] == 0.0 && vf[cym] == 0.0 & vf[cyp] == 0.0) {
//          wallInt[c] = 304;
//        } else if ( vf[cxm] == 0.0 && vf[czm] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 305;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 & vf[czm] == 0.0) {
//          wallInt[c] = 306;
//        } else if ( vf[cxm] == 0.0 && vf[cxp] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 307;
//        } else if ( vf[cxp] == 0.0 && vf[czm] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 308;
//        } else if ( vf[cym] == 0.0 && vf[czm] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 309;
//        } else if ( vf[cym] == 0.0 && vf[cyp] == 0.0 & vf[czm] == 0.0) {
//          wallInt[c] = 310;
//        } else if ( vf[cym] == 0.0 && vf[cyp] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 311;
//        } else if ( vf[cyp] == 0.0 && vf[czm] == 0.0 & vf[czp] == 0.0) {
//          wallInt[c] = 312;

        //check for 3 wall cell corners
        } else if ( (vf[cxm] == 0.0 && vf[cym] == 0.0 && vf[czm] == 0.0) || (vf[cxp] == 0.0 && vf[cyp] == 0.0 && vf[czp] == 0.0) ) {
          wallInt[c] = 31;
        } else if ( (vf[cxm] == 0.0 && vf[cym] == 0.0 && vf[czp] == 0.0) || ( vf[cxp] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0) ) {
          wallInt[c] = 32;
        } else if ( (vf[cxm] == 0.0 && vf[cyp] == 0.0 && vf[czm] == 0.0) || ( vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[czp] == 0.0 ) ) {
          wallInt[c] = 33;
        } else if ( (vf[cxm] == 0.0 && vf[cyp] == 0.0 && vf[czp] == 0.0) || ( vf[cxp] == 0.0 && vf[cym] == 0.0 && vf[czm] == 0.0 ) ) {
          wallInt[c] = 34;

        //check for 2 wall cell corners
        } else if ( (vf[cxm] == 0.0 && vf[cym] == 0.0) || ( vf[cxp] == 0.0 && vf[cyp] == 0.0 ) ) {
          wallInt[c] = 21;
        } else if ( (vf[cxm] == 0.0 && vf[cyp] == 0.0) || ( vf[cxp] == 0.0 && vf[cym] == 0.0 ) ) {
          wallInt[c] = 22;
        } else if ( (vf[cxm] == 0.0 && vf[czm] == 0.0) || ( vf[cxp] == 0.0 && vf[czp] == 0.0 ) ) {
          wallInt[c] = 23;
        } else if ( (vf[cxm] == 0.0 && vf[czp] == 0.0) || ( vf[cxp] == 0.0 && vf[czm] == 0.0 ) ) {
          wallInt[c] = 24;
        } else if ( (vf[cym] == 0.0 && vf[czm] == 0.0) || ( vf[cyp] == 0.0 && vf[czp] == 0.0 ) ) {
          wallInt[c] = 25;
        } else if ( (vf[cym] == 0.0 && vf[czp] == 0.0) || ( vf[cyp] == 0.0 && vf[czm] == 0.0 ) ) {
          wallInt[c] = 26;

        //check for flat wall
        } else if ( vf[cxm] == 0.0 ) {
          wallInt[c] = 11;
        } else if ( vf[cxp] == 0.0 ) {
          wallInt[c] = 12;
        } else if ( vf[cym] == 0.0 ) {
          wallInt[c] = 13;
        } else if ( vf[cyp] == 0.0 ) {
          wallInt[c] = 14;
        } else if ( vf[czm] == 0.0 ) {
          wallInt[c] = 15;
        } else if ( vf[czp] == 0.0 ) {
          wallInt[c] = 16;

        } else {
          wallInt[c] = 99;  //all un set cases, add more in future if problems arise
        }

      }
    } //cell loop
  }
}
