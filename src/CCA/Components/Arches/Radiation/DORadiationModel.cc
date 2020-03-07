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

//----- DORadiationModel.cc --------------------------------------------------
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ArchesStatsEnum.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/hypre_defs.h>
#include <sci_defs/kokkos_defs.h>

#ifdef HAVE_HYPRE
#  include <CCA/Components/Arches/Radiation/RadHypreSolver.h>
#endif

#include <CCA/Components/Arches/Radiation/fortran/rordr_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radarray_fort.h>
// #include <CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
// #include <CCA/Components/Arches/Radiation/fortran/radwsgg_fort.h>
// #include <CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
// #include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
// #include <CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <CCA/Components/Arches/Radiation/LegendreChebyshevQuad.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <Core/Parallel/LoopExecution.hpp>

using namespace std;
using namespace Uintah;

#define OUTPUT_ORDINATES

static DebugStream dbg("ARCHES_RADIATION",false);

//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(const ArchesLabel* label,
                                   const MPMArchesLabel* MAlab,
                                   const ProcessorGroup* myworld,
                                   bool sweepMethod ):
                                   d_myworld(myworld),
                                   m_ffield(-1)  //WARNING: Hack -- flow cells set to -1

{
  _sweepMethod = sweepMethod;
  d_linearSolver = 0;
}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  if (!_sweepMethod){
    delete d_linearSolver;
  }

  for (unsigned int i=0; i<_emiss_plus_scat_source_label.size(); i++){
    VarLabel::destroy(_emiss_plus_scat_source_label[i]);
  }
}

//****************************************************************************
// Problem Setup for DORadiationModel
//****************************************************************************

void
DORadiationModel::problemSetup( ProblemSpecP& params )
{

  ProblemSpecP db = params->findBlock("DORadiationModel");

  db->getWithDefault("ReflectOn",m_doReflections,false);  //  reflections are off by default.

  //db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions")
  std::string initialGuessType;
  db->getWithDefault("initialGuess",initialGuessType,"zeros"); //  using the previous solve as initial guess, is off by default

  if(initialGuessType=="zeros"){
    m_initialGuess = ZERO;
  }

//  else if(initialGuessType=="prevDir"){                // This conditional doesn't make sense.  What flag is enabled??
//    m_zeroInitialGuess    = false;
//    _usePreviousIntensity = false;
//  }
  else if(initialGuessType=="prevRadSolve"){
    m_initialGuess = OLD_INTENSITY;
  } else{
    throw ProblemSetupException("Error:DO-radiation initial guess not set!.", __FILE__, __LINE__);
  }

  db->getWithDefault("ScatteringOn",     m_doScattering,       false);
  db->getWithDefault("QuadratureSet",    m_quadratureSet,      "LevelSymmetric");
  db->getWithDefault("addOrthogonalDirs", m_addOrthogonalDirs, false);

  std::string baseNameAbskp;
  std::string modelName;
  std::string baseNameTemperature;

  // Does this system have particles??? Check for particle property models

  _grey_reference_weight=std::vector<double> (1, 1.0);
  m_nQn_part =0;
  m_nbands = 1;

  ProblemSpecP db_propV2 = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModelsV2");
  if  (db_propV2){
    for ( ProblemSpecP db_model = db_propV2->findBlock("model"); db_model != nullptr; db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName);

      if (modelName=="partRadProperties"){
        bool doing_dqmom = ArchesCore::check_for_particle_method(db,ArchesCore::DQMOM_METHOD);
        bool doing_cqmom = ArchesCore::check_for_particle_method(db,ArchesCore::CQMOM_METHOD);

        if ( doing_dqmom ){
          m_nQn_part = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );
        }
        else if ( doing_cqmom ){
          m_nQn_part = ArchesCore::get_num_env( db, ArchesCore::CQMOM_METHOD );
        }
        else {
          throw ProblemSetupException("Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
        }

        db_model->getWithDefault( "part_temp_label",  baseNameTemperature, "heat_pT" );
        db_model->getWithDefault( "radiateAtGasTemp", m_radiateAtGasTemp, true );
        db_model->getAttribute("label",baseNameAbskp);
      }
      else if (modelName=="gasRadProperties"){
        _abskg_name_vector = std::vector<std::string>  (1);
        _abswg_name_vector = std::vector<std::string>  (0);
        db_model->getAttribute("label",_abskg_name_vector[0]);
      }
      else if(modelName=="spectralProperties"){
        _LspectralSolve = true;
        m_nbands        = 4;// hard coded for now (+1 for soot or particles, later on)

        double T_ref=0.0;
        double molarRatio_ref=0.0;
        db_model->getWithDefault( "Temperature_ref",       T_ref, 1750 );
        db_model->getWithDefault( "CO2_H2OMolarRatio_ref", molarRatio_ref, 0.25 );

        T_ref=std::min(std::max(T_ref,500.0),2400.0);  // limit reference to within bounds of fit data
        molarRatio_ref=std::min(std::max(molarRatio_ref,0.01),4.0);

        _abskg_name_vector=std::vector<std::string>  (m_nbands);
        _abswg_name_vector=std::vector<std::string>  (m_nbands);

        for (int i=0; i<m_nbands; i++){
          std::stringstream abskg_name;
          abskg_name << "abskg" <<"_"<< i;
          _abskg_name_vector[i]= abskg_name.str();

          std::stringstream abswg_name;
          abswg_name << "abswg" <<"_"<< i;
          _abswg_name_vector[i]= abswg_name.str();
        }

        std::string soot_name="";
        db_model->get("sootVolumeFrac",soot_name);

        if (soot_name !=""){
          _LspectralSootOn=true;
        }

        _grey_reference_weight=std::vector<double> (m_nbands+1,0.0); // +1 for transparent band

        const double wecel_C_coeff[5][4][5] {

          {{0.7412956,  -0.9412652,   0.8531866,   -.3342806,    0.0431436 },
           {0.1552073,  0.6755648,  -1.1253940, 0.6040543,  -0.1105453},
           {0.2550242,  -0.6065428,  0.8123855,  -0.45322990,  0.0869309},
           {-0.0345199, 0.4112046, -0.5055995, 0.2317509, -0.0375491}},

          {{-0.5244441, 0.2799577 ,   0.0823075,    0.1474987,   -0.0688622},
           {-0.4862117, 1.4092710 ,  -0.5913199,  -0.0553385 , 0.0464663},
           {0.3805403 , 0.3494024 ,  -1.1020090,   0.6784475 , -0.1306996},
           {0.2656726 , -0.5728350,  0.4579559 ,  -0.1656759 , 0.0229520}},

          {{ 0.582286 , -0.7672319,  0.5289430,   -0.4160689,  0.1109773},
           { 0.3668088, -1.3834490,  0.9085441,  -0.1733014 ,  -0.0016129},
           {-0.4249709, 0.1853509 ,  0.4046178,  -0.3432603 ,  0.0741446},
           {-0.1225365, 0.2924490,  -0.2616436,  0.1052608  ,  -0.0160047}},

          {{-.2096994,   0.3204027, -.2468463,   0.1697627,   -0.0420861},
           {-0.1055508,  0.4575210, -0.3334201,  0.0791608,  -0.0035398},
           {0.1429446,  -0.1013694, -0.0811822,  0.0883088,  -0.0202929},
           {0.0300151,  -0.0798076, 0.0764841,  -0.0321935,  0.0050463}},

          {{0.0242031 , -.0391017 ,  0.0310940,  -0.0204066,  0.0049188},
           {0.0105857 , -0.0501976,  0.0384236, -0.0098934 ,  0.0006121},
           {-0.0157408, 0.0130244 ,  0.0062981, -0.0084152 ,  0.0020110},
           {-0.0028205, 0.0079966 , -0.0079084,  0.003387  , -0.0005364}}};

        int nrows=4;
        int n_coeff=5;

        double T_r=std::min(std::max(T_ref/1200.0,0.01),4.0); // 0.01 to 4.0
        std::vector<std::vector<double> > b_vec(nrows,std::vector<double>(n_coeff,0.0)); // minus 1 for transparent band

        double m_k=1.0; // m^k
        for (int kk=0; kk < n_coeff; kk++){
          for (int jj=0; jj < nrows; jj++){
            for (int ii=0; ii < n_coeff; ii++){
              b_vec[jj][ii]+=wecel_C_coeff[kk][jj][ii]*m_k;
            }
          }
          m_k*=molarRatio_ref;
        }

        double T_r_k=1.0; //T_r^k
        m_k=1.0;
        double weight_sum=0.0;

        for (int kk=0; kk < n_coeff;  kk++){
          for (int ii=0; ii< nrows ; ii++){
            _grey_reference_weight[ii]+=b_vec[ii][kk]*T_r_k;
            weight_sum+=b_vec[ii][kk]*T_r_k;
          }
          T_r_k*=T_r;
          m_k*=molarRatio_ref;
        }
       _grey_reference_weight[nrows]=1.0-weight_sum;

      } // end if
    } // end model loop
  } // end spec-check

  for (int qn=0; qn < m_nQn_part; qn++){
    std::stringstream absorp;
    std::stringstream temper;
    absorp <<baseNameAbskp <<"_"<< qn;
    temper <<baseNameTemperature <<"_"<< qn;

    _abskp_name_vector.push_back( absorp.str());
    _temperature_name_vector.push_back( temper.str());
  }

  // solve for transparent band if soot or particles are present
  if(_LspectralSolve && (m_nQn_part > 0 || _LspectralSootOn)){
    std::stringstream abskg_name;
    abskg_name << "abskg" <<"_"<< m_nbands;
    _abskg_name_vector.push_back(abskg_name.str());

    std::stringstream abswg_name;
    abswg_name << "abswg" <<"_"<< m_nbands;
    _abswg_name_vector.push_back(abswg_name.str());
    m_nbands=m_nbands+1;
  }

  if (m_doScattering  && m_nQn_part ==0){
    throw ProblemSetupException("Error: No particle model found in DO-radiation! When scattering is turned on, a particle model is required!", __FILE__, __LINE__);
  }

  if (db) {
    bool ordinates_specified =db->findBlock("ordinates");
    db->getWithDefault("ordinates",m_sn,2);

    if (ordinates_specified == false){
      proc0cout << " Notice: No ordinate number specified.  Defaulting to 2." << endl;
    }
    if ((m_sn)%2 || m_sn <2){
      throw ProblemSetupException("Error:Only positive, even, and non-zero ordinate numbers for discrete-ordinates radiation are permitted.", __FILE__, __LINE__);
    }
  }
  else {
    throw ProblemSetupException("Error: <DORadiation> node not found.", __FILE__, __LINE__);
  }

  computeOrdinatesOPL();

  if ( db->findBlock("print_all_info") ){
    m_print_all_info = true;
  }

  string linear_sol;
  if(db->findBlock("LinearSolver") ==nullptr){
    if (!_sweepMethod){
      throw ProblemSetupException("Error: Linear solver missing for DORadition model! Specify Linear solver parameters or use sweeping method.", __FILE__, __LINE__);
    }
  }
  else{
    db->findBlock("LinearSolver")->getAttribute("type",linear_sol);
    if (!_sweepMethod){
      if (linear_sol == "petsc"){

        d_linearSolver = scinew RadPetscSolver(d_myworld);

      } else if (linear_sol == "hypre"){

        d_linearSolver = scinew RadHypreSolver(d_myworld);

      }
      d_linearSolver->problemSetup(db);
    }
  }

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  for (int iband=0; iband<m_nbands; iband++){
    for( int ord=0; ord<m_totalOrds; ord++){

      ostringstream labelName;
      labelName << "Intensity" << setfill('0') << setw(4)<<  ord << "_"<< setw(2)<< iband ;
      _IntensityLabels.push_back(  VarLabel::create(labelName.str(),  CC_double));

      if(needIntensitiesBool()== false){
        break;  // gets labels for all intensities, otherwise only create 1 label
      }
    }
  }

  _emiss_plus_scat_source_label = std::vector<const VarLabel*> (0);

  if (m_doScattering && _sweepMethod){
    for (int iband=0; iband<m_nbands; iband++){
      for( int ord=0;  ord< m_totalOrds;ord++){
        ostringstream labelName;
        labelName << "scatSrc_absSrc" << setfill('0') << setw(4)<<  ord <<"_"<<iband ;
        _emiss_plus_scat_source_label.push_back(  VarLabel::create(labelName.str(),CC_double));
      }
    }
  }

  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxE"));
  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxW"));
  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxN"));
  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxS"));
  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxT"));
  _radiationFluxLabels.push_back( VarLabel::find("radiationFluxB"));

   // this was added with the hope that it could be extended to one-sided ghost cell requirements
  _gv =std::vector< std::vector < std::vector < Ghost::GhostType > > >   (2,  std::vector < std::vector < Ghost::GhostType > > (2, std::vector < Ghost::GhostType > (2) ));
  _gv[0][0][0] = Ghost::xpypzp;
  _gv[0][0][1] = Ghost::xpypzm;
  _gv[0][1][0] = Ghost::xpymzp;
  _gv[0][1][1] = Ghost::xpymzm;
  _gv[1][0][0] = Ghost::xmypzp;
  _gv[1][0][1] = Ghost::xmypzm;
  _gv[1][1][0] = Ghost::xmymzp;
  _gv[1][1][1] = Ghost::xmymzm;

}

//______________________________________________________________________
//  Insert the orthogonal cosine dirs every nthElement
void
DORadiationModel::insertEveryNth( const std::vector<std::vector<double>>& orthogonalCosineDirs,
                                  const int nthElement,
                                  const int dir,                // x, y, z
                                  std::vector<double>& vec)     // vector to modify
{
  auto it = vec.begin() + nthElement;
  
  for( size_t i = 0; i < orthogonalCosineDirs.size(); i++){

    it = vec.insert( it, orthogonalCosineDirs[i][dir] );
    std::advance( it, ( nthElement +1) );
  }
}

//______________________________________________________________________
//
void
DORadiationModel::computeOrdinatesOPL()
{
  m_totalOrds = m_sn*(m_sn+2);
  m_omu.assign( m_totalOrds,0.0);
  m_oeta.assign(m_totalOrds,0.0);
  m_oxi.assign( m_totalOrds,0.0);
  m_wt.assign(  m_totalOrds,0.0);

  if (m_quadratureSet=="LegendreChebyshev"){

    computeLegendreChebyshevQuadratureSet(m_sn, m_omu,m_oeta,m_oxi,m_wt);

  } else{  // Level-Symmetric

    // base 1 arrays for the fortran subroutine
    OffsetArray1<double> oxi(  1,m_totalOrds + 1);
    OffsetArray1<double> omu(  1,m_totalOrds + 1);
    OffsetArray1<double> oeta( 1,m_totalOrds + 1);
    OffsetArray1<double> wt(   1,m_totalOrds + 1);

    omu.initialize(0.0);
    oeta.initialize(0.0);
    oxi.initialize(0.0);
    wt.initialize(0.0);

    fort_rordr(m_sn, oxi, omu, oeta, wt);

    // convert to stl::vector 0 based
    m_oxi  = oxi.to_stl_vector();
    m_omu  = omu.to_stl_vector();
    m_oeta = oeta.to_stl_vector();
    m_wt   = wt.to_stl_vector();
  }

  //__________________________________
  //  for Orthogonal sweeps
  if( m_addOrthogonalDirs ){

    // unit vector specifying direction of that are orthogonal.  You must have 8 entries.
    const                                             //    omu    oeta      oxi   wt
    std::vector<std::vector<double>> orthogonalCosineDirs{{1.0,    2e-16,   2e-16, 0 }, 
                                                          {-2e-16, 1.0,     2e-16, 0 },
                                                          {2e-16, -2e-16,   1.0,   0 },
                                                          {-1.0,  -2e-16,   2e-16, 0 },
                                                          {2e-16,  1.0,    -2e-16, 0 },  // NULL at 5
                                                          {-2e-16, 1.0,    -2e-16, 0 },  // NULL at 6
                                                          {2e-16, -1.0,    -2e-16, 0 },
                                                          {-2e-16,-2e-16,  -1.0,   0 } };   
    const int nthElement = m_totalOrds/8;   
    
    insertEveryNth( orthogonalCosineDirs, nthElement, 0,  m_omu );
    insertEveryNth( orthogonalCosineDirs, nthElement, 1,  m_oeta );
    insertEveryNth( orthogonalCosineDirs, nthElement, 2,  m_oxi );
    insertEveryNth( orthogonalCosineDirs, nthElement, 3,  m_wt );

    m_totalOrds = m_totalOrds + orthogonalCosineDirs.size();


    // write direction cosines to a file
    if( d_myworld->myRank() == 0 ){

      std::string filename = "dir_cosines.txt";
      ofstream oStream;
      oStream.open(filename);
      oStream<< "# This file contains the Discrete Ordinate ordinate directions and weights\n"
             << "# i     omu         oeta        oxi       wt\n";

      for (int i=0; i< m_totalOrds; i++){
        oStream <<  i<< " " << setw(10) << m_omu[i] << " "<< setw(10) << m_oeta[i] << " "<< setw(10) << m_oxi[i] << " " << setw(10) << m_wt[i] << endl;
      }
      oStream.close();
    }
  }

  //__________________________________
  //
  double sumx=0;
  double sumy=0;
  double sumz=0;

  for (int i=0; i< m_totalOrds/8; i++){
   sumx += m_omu[i] * m_wt[i];
   sumy += m_oeta[i]* m_wt[i];
   sumz += m_oxi[i] * m_wt[i];
  }

  m_xfluxAdjust=M_PI/sumx/4.0;  // sumx, sumy, sumz should equal pi/4 because: Int->0:pi/2 cos(theta) dOmega = pi,
  m_yfluxAdjust=M_PI/sumy/4.0;
  m_zfluxAdjust=M_PI/sumz/4.0;

  m_plusX = vector<bool> (m_totalOrds,false);
  m_plusY = vector<bool> (m_totalOrds,false);
  m_plusZ = vector<bool> (m_totalOrds,false);

  m_xiter = vector<int> (m_totalOrds,-1);
  m_yiter = vector<int> (m_totalOrds,-1);
  m_ziter = vector<int> (m_totalOrds,-1);

  for (int ord = 0; ord <m_totalOrds; ord++){
    if (m_omu[ord] > 0) {
      m_plusX[ord]  = true;
      m_xiter[ord]  = 1;
    }
    if (m_oeta[ord] > 0){
      m_plusY[ord]  = true;
      m_yiter[ord]  = 1;
    }
    if (m_oxi[ord] > 0) {
      m_plusZ[ord]  = true;
      m_ziter[ord]  = 1;
    }
  }

  if(m_doScattering){
    m_cosineTheta      = vector<vector<double>> (m_totalOrds,vector<double>(m_totalOrds,0.0));
    m_solidAngleWeight = vector<double> (m_totalOrds,0.0);

    for (int i=0; i<m_totalOrds ; i++){
      m_solidAngleWeight[i]=  m_wt[i]/(4.0 * M_PI);

      for (int j=0; j<m_totalOrds ; j++){
        m_cosineTheta[i][j]=m_oxi[j]*m_oxi[i] + m_oeta[j]*m_oeta[i] + m_omu[j]*m_omu[i];
      }
    }
    // No adjustment factor appears to be needed for this form of the phase function. PHI=1+f*cos(theta)
    //for (int direction=0; direction<m_totalOrds ; direction++){
        //double  sumpF=0.0;
      //for (int i=0; i<m_totalOrds ; i++){
         //sumpF += (1.0 + 0.333333*m_cosineTheta[direction][i])*m_solidAnbleWeight[i];
      //}
     //proc0cout << sumpF << "\n";
    //}
  }
}

//______________________________________________________________________
//
struct computeAMatrix{
       computeAMatrix( double  _omu, double _oeta, double _oxi,
                       double _areaEW, double _areaNS, double _areaTB, double _vol, int _intFlow,
                       constCCVariable<int>      &_cellType,
                       constCCVariable<double>   &_wallTemp,
                       constCCVariable<double>   &_abskt,
                       CCVariable<double>   &_srcIntensity,
                       CCVariable<double>   &_matrixB,
                       CCVariable<double>   &_west,
                       CCVariable<double>   &_south,
                       CCVariable<double>   &_bottom,
                       CCVariable<double>   &_center,
                       CCVariable<double>   &_scatSource,
                       CCVariable<double>   &_fluxX,
                       CCVariable<double>   &_fluxY,
                       CCVariable<double>   &_fluxZ)  :
                       omu(_omu),
                       oeta(_oeta),
                       oxi(_oxi),
                       areaEW(_areaEW),
                       areaNS(_areaNS),
                       areaTB(_areaTB),
                       vol(_vol),
                       intFlow(_intFlow),
#if defined( KOKKOS_ENABLE_OPENMP )
                       cellType(_cellType.getKokkosView()),
                       wallTemp(_wallTemp.getKokkosView()),
                       abskt(_abskt.getKokkosView()),
                       srcIntensity(_srcIntensity.getKokkosView()),
                       matrixB(_matrixB.getKokkosView()),
                       west(_west.getKokkosView()),
                       south(_south.getKokkosView()),
                       bottom(_bottom.getKokkosView()),
                       center(_center.getKokkosView()),
                       scatSource(_scatSource.getKokkosView()),
                       fluxX(_fluxX.getKokkosView()) ,
                       fluxY(_fluxY.getKokkosView()) ,
                       fluxZ(_fluxZ.getKokkosView())
#else
                       cellType(_cellType),
                       wallTemp(_wallTemp),
                       abskt(_abskt),
                       srcIntensity(_srcIntensity),
                       matrixB(_matrixB),
                       west(_west),
                       south(_south),
                       bottom(_bottom),
                       center(_center),
                       scatSource(_scatSource),
                       fluxX(_fluxX) ,
                       fluxY(_fluxY) ,
                       fluxZ(_fluxZ)
#endif //UINTAH_ENABLE_KOKKOS
                                    { SB=5.67e-8;  // W / m^2 / K^4
                                      dirX = (omu  > 0.0)? -1 : 1;
                                      dirY = (oeta > 0.0)? -1 : 1;
                                      dirZ = (oxi  > 0.0)? -1 : 1;
                                      omu=abs(omu);
                                      oeta=abs(oeta);
                                      oxi=abs(oxi);
                                     }

       void operator()(int i, int j, int k) const {

         if (cellType(i,j,k)==intFlow) {

           matrixB(i,j,k)=(srcIntensity(i,j,k)+ scatSource(i,j,k))*vol;
           center(i,j,k) =  omu*areaEW + oeta*areaNS +  oxi*areaTB +
           abskt(i,j,k) * vol; // out scattering

          int ipm = i + dirX;
          int jpm = j + dirY;
          int kpm = k + dirZ;

           if (cellType(ipm,j,k)==intFlow) {
             west(i,j,k)= omu*areaEW; // signed changed in radhypresolve
           }else{
             matrixB(i,j,k)+= abskt(ipm,j,k)*omu*areaEW*SB/M_PI*pow(wallTemp(ipm,j,k),4.0)+omu*fluxX(i,j,k)/M_PI*(1.0-abskt(ipm,j,k)) ;
           }

           if (cellType(i,jpm,k)==intFlow) {
             south(i,j,k)= oeta*areaNS; // signed changed in radhypresolve
           }else{
             matrixB(i,j,k)+= abskt(i,jpm,k)*oeta*areaNS*SB/M_PI*pow(wallTemp(i,jpm,k),4.0)+oeta*fluxY(i,j,k)/M_PI*(1.0-abskt(i,jpm,k));
           }

           if (cellType(i,j,kpm)==intFlow) {
             bottom(i,j,k) =  oxi*areaTB; // sign changed in radhypresolve
           }else{
             matrixB(i,j,k)+= abskt(i,j,kpm)*oxi*areaTB*SB/M_PI*pow(wallTemp(i,j,kpm),4.0)+oxi*fluxZ(i,j,k)/M_PI*(1.0-abskt(i,j,kpm));
           }
         }else{
           matrixB(i,j,k) = SB/M_PI*pow(wallTemp(i,j,k),4.0);
           center(i,j,k)  = 1.0;
         }
 }

  private:

       double omu;
       double oeta;
       double oxi;
       double areaEW;
       double areaNS;
       double areaTB;
       double vol;
       int    intFlow;

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
       KokkosView3<const int,    Kokkos::HostSpace> cellType;
       KokkosView3<const double, Kokkos::HostSpace> wallTemp;
       KokkosView3<const double, Kokkos::HostSpace> abskt;

       KokkosView3<double, Kokkos::HostSpace> srcIntensity;
       KokkosView3<double, Kokkos::HostSpace> matrixB;
       KokkosView3<double, Kokkos::HostSpace> west;
       KokkosView3<double, Kokkos::HostSpace> south;
       KokkosView3<double, Kokkos::HostSpace> bottom;
       KokkosView3<double, Kokkos::HostSpace> center;
       KokkosView3<double, Kokkos::HostSpace> scatSource;
       KokkosView3<double, Kokkos::HostSpace> fluxX;
       KokkosView3<double, Kokkos::HostSpace> fluxY;
       KokkosView3<double, Kokkos::HostSpace> fluxZ;
#else
       constCCVariable<int>    & cellType;
       constCCVariable<double> & wallTemp;
       constCCVariable<double> & abskt;

       CCVariable<double> & srcIntensity;
       CCVariable<double> & matrixB;
       CCVariable<double> & west;
       CCVariable<double> & south;
       CCVariable<double> & bottom;
       CCVariable<double> & center;
       CCVariable<double> & scatSource;
       CCVariable<double> & fluxX;
       CCVariable<double> & fluxY;
       CCVariable<double> & fluxZ;
#endif

       double SB;
       int    dirX;
       int    dirY;
       int    dirZ;

};


//***************************************************************************
// Sums the intensities to compute the 6 fluxes, and incident radiation
//***************************************************************************
template <typename constCCVar_or_CCVar, typename constDouble_or_double>
struct compute4Flux{
       compute4Flux( double  _omu, double _oeta, double _oxi, double  _wt,
                   constCCVar_or_CCVar &_intensity,  ///< intensity field corresponding to unit direction vector [mu eta xi]
                   CCVariable<double> &_fluxX,  ///< either x+ or x- flux
                   CCVariable<double> &_fluxY,  ///< either y+ or y- flux
                   CCVariable<double> &_fluxZ,  ///< either z+ or z- flux
                   CCVariable<double> &_volQ) :
                   omu(_omu),    ///< absolute value of solid angle weighted x-component
                   oeta(_oeta),  ///< absolute value of solid angle weighted y-component
                   oxi(_oxi),    ///< absolute value of solid angle weighted z-component
                   wt(_wt),
#if defined( KOKKOS_ENABLE_OPENMP )
                   intensity(_intensity.getKokkosView()),
                   fluxX(_fluxX.getKokkosView()) ,
                   fluxY(_fluxY.getKokkosView()) ,
                   fluxZ(_fluxZ.getKokkosView()) ,
                   volQ(_volQ.getKokkosView())
#else
                   intensity(_intensity),
                   fluxX(_fluxX) ,
                   fluxY(_fluxY) ,
                   fluxZ(_fluxZ) ,
                   volQ(_volQ)
#endif //UINTAH_ENABLE_KOKKOS
                     { }

       void operator()(int i , int j, int k ) const {
                   fluxX(i,j,k) += omu*intensity(i,j,k);
                   fluxY(i,j,k) += oeta*intensity(i,j,k);
                   fluxZ(i,j,k) += oxi*intensity(i,j,k);
                   volQ(i,j,k)  += intensity(i,j,k)*wt;

       }

  private:

       double  omu;    ///< x-directional component
       double  oeta;   ///< y-directional component
       double  oxi;    ///< z-directional component
       double  wt;     ///< ordinate weight

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
       KokkosView3<constDouble_or_double, Kokkos::HostSpace> intensity; ///< intensity solution from linear solve

       KokkosView3<double, Kokkos::HostSpace> fluxX;   ///< x-directional flux ( positive or negative direction)
       KokkosView3<double, Kokkos::HostSpace> fluxY;   ///< y-directional flux ( positive or negative direction)
       KokkosView3<double, Kokkos::HostSpace> fluxZ;   ///< z-directional flux ( positive or negative direction)
       KokkosView3<double, Kokkos::HostSpace> volQ;    ///< Incident radiation
#else
       constCCVar_or_CCVar& intensity; ///< intensity solution from linear solve

       CCVariable<double>& fluxX;  ///< x-directional flux ( positive or negative direction)
       CCVariable<double>& fluxY;  ///< y-directional flux ( positive or negative direction)
       CCVariable<double>& fluxZ;  ///< z-directional flux ( positive or negative direction)
       CCVariable<double>& volQ;   ///< Incident radiation
#endif

};

//***************************************************************************
// Solves for intensity in the D.O method
//***************************************************************************
void
DORadiationModel::intensitysolve(const ProcessorGroup* pg,
                                 const Patch* patch,
                                 CellInformation* cellinfo,
                                 ArchesVariables* vars,
                                 ArchesConstVariables* constvars,
                                 CCVariable<double>& divQ,
                                 int wall_type,
                                 int matlIndex,
                                 DataWarehouse* new_dw,
                                 DataWarehouse* old_dw,
                                 bool old_DW_isMissingIntensities)
{
  proc0cout << " Radiation Solve: " << endl;
  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  Timers::Simple timer;
  timer.start();

  d_linearSolver->matrixInit(patch);

  OffsetArray1<double> rgamma(1,29);
  OffsetArray1<double> sd15(1,481);
  OffsetArray1<double> sd(1,2257);
  OffsetArray1<double> sd7(1,49);
  OffsetArray1<double> sd3(1,97);

  rgamma.initialize(0.0);
  sd15.initialize(0.0);
  sd.initialize(0.0);
  sd7.initialize(0.0);
  sd3.initialize(0.0);

  if (m_lambda > 1) {
    fort_radarray(rgamma, sd15, sd, sd7, sd3);
  }

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();
  IntVector domLo = patch->getExtraCellLowIndex();
  IntVector domHi = patch->getExtraCellHighIndex();

  CCVariable<double> su;
  CCVariable<double> aw;
  CCVariable<double> as;
  CCVariable<double> ab;
  CCVariable<double> ap;

  std::vector< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.


  if(m_doReflections){
    for (unsigned int i=0; i< _radiationFluxLabels.size(); i++){
      constCCVariable<double> radiationFlux_temp;
      old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch, m_gn, 0  );

      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].copyData(radiationFlux_temp);
    }
  }
  else{
    for (unsigned int i=0; i< _radiationFluxLabels.size(); i++){  // magic number cooresponds to number of labels tranported, when
      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].initialize(0.0);      // for no reflections, this must be zero
    }
  }

  if( m_initialGuess != OLD_INTENSITY ){
    old_dw->get(constvars->cenint,     _IntensityLabels[0], matlIndex, patch, m_gn, 0  );
    new_dw->getModifiable(vars->cenint,_IntensityLabels[0], matlIndex, patch ); // per the logic in sourceterms/doradiation, old and new dw are the same.
  }

  std::vector< constCCVariable<double> > Intensities((m_doScattering && !old_DW_isMissingIntensities) ? m_totalOrds : 0);

  std::vector< CCVariable<double> > IntensitiesRestart((m_doScattering && old_DW_isMissingIntensities) ? m_totalOrds : 0);

  CCVariable<double> scatIntensitySource;
  constCCVariable<double> scatkt;             //total scattering coefficient
  constCCVariable<double> asymmetryParam;

  scatIntensitySource.allocate(domLo,domHi);
  scatIntensitySource.initialize(0.0);        // needed for non-scattering cases


  Vector Dx = patch->dCell();
  double volume = Dx.x()* Dx.y()* Dx.z();
  double areaEW = Dx.y()*Dx.z();
  double areaNS = Dx.x()*Dx.z();
  double areaTB = Dx.x()*Dx.y();

  if(m_doScattering){
    if(old_DW_isMissingIntensities){
      for( int ord=0;  ord<m_totalOrds ;ord++){
        IntensitiesRestart[ord].allocate(domLo,domHi);
        IntensitiesRestart[ord].initialize(0.0);
      }
    }else{
      for( int ord=0;  ord<m_totalOrds ;ord++){
        old_dw->get(Intensities[ord],_IntensityLabels[ord], matlIndex , patch, m_gn, 0  );
      }
    }
    old_dw->get(asymmetryParam,_asymmetry_label, matlIndex , patch, m_gn, 0);
    old_dw->get(scatkt,        _scatkt_label,    matlIndex , patch, m_gn, 0);
  }

  std::vector< constCCVariable<double> > abskp(m_nQn_part);
  std::vector< constCCVariable<double> > partTemp(m_nQn_part);
  for (int ix=0;  ix< m_nQn_part; ix++){
    old_dw->get(abskp[ix],   _abskp_label_vector[ix],       matlIndex , patch, m_gn, 0  );
    old_dw->get(partTemp[ix],_temperature_label_vector[ix], matlIndex , patch, m_gn, 0  );
  }

  su.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);

  divQ.initialize(0.0);
  vars->qfluxe.initialize(0.0);
  vars->qfluxw.initialize(0.0);
  vars->qfluxn.initialize(0.0);
  vars->qfluxs.initialize(0.0);
  vars->qfluxt.initialize(0.0);
  vars->qfluxb.initialize(0.0);

  //__________________________________
  //begin discrete ordinates
  std::vector< constCCVariable<double> > spectral_weights(0); // spectral not supported for DO linear solve
  std::vector<      CCVariable<double> > Emission_source(1);
  std::vector< constCCVariable<double> > abskgas(1);

  abskgas[0]=constvars->ABSKG;

  Emission_source[0].allocate( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Emission_source[0].initialize(0.0);

  for (int bands =1; bands <=m_lambda; bands++){

    vars->volq.initialize(0.0);

    computeIntensitySource(patch,abskp,partTemp,abskgas,constvars->temperature,Emission_source,spectral_weights);

    for (int ord = 0; ord < m_totalOrds; ord++){
      if( (m_initialGuess == OLD_INTENSITY) && !old_DW_isMissingIntensities){
        old_dw->get(constvars->cenint,     _IntensityLabels[ord], matlIndex, patch, m_gn, 0  );
        new_dw->getModifiable(vars->cenint,_IntensityLabels[ord], matlIndex, patch );
      }
      else if ( m_doScattering){
        new_dw->getModifiable(vars->cenint,_IntensityLabels[ord], matlIndex, patch );
      }
      if(old_DW_isMissingIntensities){
        old_dw->get(constvars->cenint,     _IntensityLabels[0], matlIndex, patch, m_gn, 0  );
      }

      if(m_initialGuess==ZERO){
        vars->cenint.initialize(0.0); // remove once RTs have been checked.
      }

      su.initialize(0.0);
      aw.initialize(0.0);
      as.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);

      bool plusX, plusY, plusZ;
      plusX = (m_omu[ord]  > 0.0)? 1 : 0;
      plusY = (m_oeta[ord] > 0.0)? 1 : 0;
      plusZ = (m_oxi[ord]  > 0.0)? 1 : 0;

      d_linearSolver->gridSetup(plusX, plusY, plusZ);

      if(m_doScattering){
        if(old_DW_isMissingIntensities){
          computeScatteringIntensities(ord, scatkt, IntensitiesRestart,scatIntensitySource, asymmetryParam, patch);
        }else{
          computeScatteringIntensities(ord, scatkt, Intensities,       scatIntensitySource, asymmetryParam, patch);
        }
      }

      // old construction of the A-matrix using fortran
      //fort_rdomsolve( idxLo, idxHi, constvars->cellType, ffield,
                      //cellinfo->sew, cellinfo->sns, cellinfo->stb,
                      //vars->ESRCG, dir, oxi, omu,oeta, wt,
                      //constvars->temperature, constvars->ABSKT,
                      //su, aw, as, ab, ap,
                      //plusX, plusY, plusZ, fraction, bands, //fraction set to 1.0
                      //radiationFlux_old[0] , radiationFlux_old[1],
                      //radiationFlux_old[2] , radiationFlux_old[3],
                      //radiationFlux_old[4] , radiationFlux_old[5],scatIntensitySource); //  this term needed for scattering

     // new (2-2017) construction of A-matrix and b-matrix
      computeAMatrix  doMakeMatrixA( m_omu[ord], m_oeta[ord], m_oxi[ord],
                                     areaEW, areaNS, areaTB, volume, m_ffield,
                                     constvars->cellType,
                                     constvars->temperature,
                                     constvars->ABSKT,
                                     Emission_source[0],
                                     su,
                                     aw,
                                     as,
                                     ab,
                                     ap,
                                     scatIntensitySource,
                                     radiationFlux_old[plusX ? 0 : 1],
                                     radiationFlux_old[plusY ? 2 : 3],
                                     radiationFlux_old[plusZ ? 4 : 5]);


      Uintah::parallel_for( range, doMakeMatrixA );

     // Done constructing A-matrix and matrix, pass to solver object
      d_linearSolver->setMatrix( pg ,patch, vars, constvars, plusX, plusY, plusZ,
                                 su, ab, as, aw, ap, m_print_all_info );

      bool converged =  d_linearSolver->radLinearSolve( ord, m_print_all_info );

      if(m_initialGuess == OLD_INTENSITY){
        vars->cenint.initialize(0.0); // Extra cells of intensity solution are not set when using non-zero initial guess.  Reset field to initialize extra cells
      }

      if (converged) {
        d_linearSolver->copyRadSoln(patch, vars);
      }else {
        throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
      }



      //fort_rdomvolq( idxLo, idxHi, dir, wt, vars->cenint, vars->volq);
      //fort_rdomflux( idxLo, idxHi, dir, oxi, omu, oeta, wt, vars->cenint,
                     //plusX, plusY, plusZ,
                     //vars->qfluxe, vars->qfluxw,
                     //vars->qfluxn, vars->qfluxs,
                     //vars->qfluxt, vars->qfluxb);
                     //
                     //
                     //

      compute4Flux<CCVariable<double>, double> doFlux(m_wt[ord] * abs(m_omu[ord]) * m_xfluxAdjust,
                                                      m_wt[ord] * abs(m_oeta[ord])* m_yfluxAdjust,
                                                      m_wt[ord] * abs(m_oxi[ord]) * m_zfluxAdjust,
                                                      m_wt[ord],  vars->cenint,
                                                      plusX ? vars->qfluxe :  vars->qfluxw,
                                                      plusY ? vars->qfluxn :  vars->qfluxs,
                                                      plusZ ? vars->qfluxt :  vars->qfluxb,
                                                      vars->volq);
      Uintah::parallel_for( range, doFlux );
    }  // ordinate loop

    if(m_doScattering){
      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt, _scatkt_label, matlIndex , patch, m_gn, 0);

      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k) += (constvars->ABSKT(i,j,k) - scatkt(i,j,k)) * vars->volq(i,j,k) - 4.0*M_PI*Emission_source[0](i,j,k);
      });
    }
    else{
      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k) += (constvars->ABSKT(i,j,k)) * vars->volq(i,j,k) - 4.0*M_PI*Emission_source[0](i,j,k);
      });
    }
      //fort_rdomsrcscattering( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ, scatkt,scatIntensitySource);
      //fort_rdomsrc( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ);
  }  // bands loop

  d_linearSolver->destroyMatrix();

  proc0cout << "Total Radiation Solve Time: " << timer().seconds() << " seconds\n";

}


//______________________________________________________________________
// Do the Intensities need to be saved from the previous solve?
// Yes, if we are using the previous intensity as our initial guess in the linear solve.
// Yes, if we modeling scattering physics, by lagging the scattering source term.
//should I compute intensity fields?  sweeps needs to compute for communication purposes
bool
DORadiationModel::needIntensitiesBool(){ return (m_initialGuess == OLD_INTENSITY) || m_doScattering || _sweepMethod; }


//______________________________________________________________________
//
void
DORadiationModel::setLabels(const VarLabel* abskg_label,
                            const VarLabel* abskt_label,
                            const VarLabel* T_label,
                            const VarLabel* cellType_label,
                            std::vector<const VarLabel*> radIntSource,
                            const VarLabel*  fluxE_label,
                            const VarLabel*  fluxW_label,
                            const VarLabel*  fluxN_label,
                            const VarLabel*  fluxS_label,
                            const VarLabel*  fluxT_label,
                            const VarLabel*  fluxB_label,
                            const VarLabel*  volQ_label,
                            const VarLabel*  divQ_label)
{
  _emissSrc_label= radIntSource;
  _abskt_label = abskt_label;
  _T_label     = T_label;
  _cellType_label=cellType_label;
  _fluxE_label = fluxE_label;
  _fluxW_label = fluxW_label;
  _fluxN_label = fluxN_label;
  _fluxS_label = fluxS_label;
  _fluxT_label = fluxT_label;
  _fluxB_label = fluxB_label;
  _volQ_label  = volQ_label;
  _divQ_label  = divQ_label;

  _abskg_label_vector= std::vector<const VarLabel* > (m_nbands);
  _abswg_label_vector= std::vector<const VarLabel* > (_LspectralSolve? m_nbands : 0 );

  if(_LspectralSolve){
    for (int i=0; i<m_nbands; i++){
      _abskg_label_vector[i] = VarLabel::find(_abskg_name_vector[i], "Error: spectral gas absorption coefficient" );
      _abswg_label_vector[i] = VarLabel::find(_abswg_name_vector[i], "Error: spectral gas weighting coefficient" );
    }
  }else{
    _abskg_label_vector[0]=abskg_label;
  }


  for (int qn=0; qn < m_nQn_part; qn++){
    _abskp_label_vector.push_back(VarLabel::find(_abskp_name_vector[qn], "Error: particle absorption coefficient"));

    _temperature_label_vector.push_back(VarLabel::find(_temperature_name_vector[qn], "Error: particle temperature "));
  }


  if(m_doScattering){
    _scatkt_label    = VarLabel::find("scatkt",  "Error: scattering coeff");  // need more descriptive error message
    _asymmetry_label = VarLabel::find("asymmetryParam", "Error:");
  }
  return;
}

//______________________________________________________________________
//
void
DORadiationModel::setLabels()
{
  for (int qn=0; qn < m_nQn_part; qn++){
    _abskp_label_vector.push_back(VarLabel::find(_abskp_name_vector[qn], "Error: particle absorption coefficient" ));

    _temperature_label_vector.push_back(VarLabel::find(_temperature_name_vector[qn], "Error: particle temperature" ));
  }

  if(m_doScattering){
    _scatkt_label   = VarLabel::find("scatkt");
    _asymmetry_label= VarLabel::find("asymmetryParam");
  }
  return;
}

//______________________________________________________________________
//
template<class T>
void
DORadiationModel::computeScatteringIntensities(const int ord,
                                              constCCVariable<double> &scatkt,
                                              std::vector<T>          &Intensities,
                                              CCVariable<double>      &scatIntensitySource,
                                              constCCVariable<double> &asymmetryFactor ,
                                              const Patch* patch)
{
  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  scatIntensitySource.initialize(0.0); //initialize to zero for sum

  int  binSize=8; //optimization parameter since this integral is a bit nasty ~10 appears to be ideal
  int nsets=std::ceil(m_totalOrds/binSize);

  for (int iset=0; iset <nsets; iset++){

    Uintah::parallel_for( range,[&](int i, int j, int k){  // should invert this loop, and remove if-statement
      for (int ii=iset*binSize; ii < std::min((iset+1)*binSize,m_totalOrds) ; ii++) {
        double phaseFunction = (1.0 + asymmetryFactor(i,j,k) * m_cosineTheta[ord][ii]) * m_solidAngleWeight[ii];
        scatIntensitySource(i,j,k)  +=phaseFunction * Intensities[ii](i,j,k);
      }
    });
  }

  // can't use optimization on out-intensities, because only one is available in this function
  Uintah::parallel_for( range,[&](int i, int j, int k){
    scatIntensitySource(i,j,k) *=scatkt(i,j,k);
  });

  return;
}


//-----------------------------------------------------------------//
// This function computes the boundary conditiosn for the intensities
// for sweeping Discrete-Ordinates
//-----------------------------------------------------------------//
void
DORadiationModel::computeIntensitySource( const Patch* patch,
                                          std::vector<constCCVariable<double> > &abskp,
                                          std::vector<constCCVariable<double> > &pTemp,
                                          std::vector<constCCVariable<double> > &abskg,
                                                      constCCVariable<double>   &gTemp,
                                          std::vector<     CCVariable<double> > &emissSrc,
                                          std::vector<constCCVariable<double> > &spectral_weights)
{

  for (unsigned int qn=0; qn < abskp.size(); qn++){
    if( m_radiateAtGasTemp ){

      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double sum = (_sigma/M_PI) * abskp[qn][c] * std::pow(gTemp[c],4.0);

        for ( unsigned int n=0; n< emissSrc.size(); n++){
          emissSrc[n][c] += sum * _grey_reference_weight[n];
        }
      }
    }else{
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double sum = ((_sigma/M_PI) * abskp[qn][c]) * std::pow(pTemp[qn][c],4.0);

        for (unsigned int n=0; n< emissSrc.size(); n++){
          emissSrc[n][c] += sum * _grey_reference_weight[n];
        }
      }
    }  // if radiateAtGas
  }  // abskp loop

  //__________________________________
  //
  if (_LspectralSolve){
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      const double T4=std::pow(gTemp[c],4.0);

      for ( unsigned int n=0; n< emissSrc.size(); n++){
        emissSrc[n][c] += (_sigma/M_PI) * spectral_weights[n][c] * abskg[n][c]*T4;
      }
    }
  }else{
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      const double T4 = std::pow(gTemp[c],4.0);

      for ( unsigned int n=0; n< emissSrc.size(); n++){ // 1 iteration for non-spectral cases
        emissSrc[n][c] += (_sigma/M_PI) * abskg[n][c] * T4;
      }
    }
  }
}


//-----------------------------------------------------------------//
// This function computes the intensities. The fields that are required are
// cellType, radiation temperature, radiation source, and  abskt.
// This function is probably the bottle-neck in the radiation solve (optimized version method).
//-----------------------------------------------------------------//
//New:This method differs from the old method by adding spectral support, AND it
//    assumes that the infrastructure will NOT do a copy of the intensity CCVar
//    for the ghost cells.
//-----------------------------------------------------------------//
void
DORadiationModel::intensitysolveSweepOptimized( const Patch* patch,
                                                const int matlIndex,
                                                DataWarehouse* new_dw,
                                                DataWarehouse* old_dw,
                                                const int dir)
{
  const IntVector idxLo = patch->getFortranCellLowIndex();
  const IntVector idxHi = patch->getFortranCellHighIndex();

  Vector Dx = patch->dCell();
  double areaew = Dx.y()*Dx.z();
  double areans = Dx.x()*Dx.z();
  double areatb = Dx.x()*Dx.y();

  const double vol = Dx.x()* Dx.y()* Dx.z();
  const double abs_oxi  = std::abs(m_oxi[dir])  * areatb;
  const double abs_oeta = std::abs(m_oeta[dir]) * areans;
  const double abs_omu  = std::abs(m_omu[dir])  * areaew;

  const double denom    = std::abs(m_omu[dir])  * areaew +
                          std::abs(m_oeta[dir]) * areans +
                          std::abs(m_oxi[dir])  * areatb; // denomintor for Intensity in current cell

  // -------------------NEEDS TO BE ADDED, REFLCTIONS ON WALLS -----------------//
  //IntVector domLo = patch->getExtraCellLowIndex();
  //IntVector domHi = patch->getExtraCellHighIndex();
  //std::vector< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.

  //if(m_doReflections){
  //for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){
  //constCCVariable<double>  radiationFlux_temp;
  //old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch,Ghost::None, 0  );
  //radiationFlux_old[i].allocate(domLo,domHi);
  //radiationFlux_old[i].copyData(radiationFlux_temp);
  //}
  //}
  // ---------------------------------------------------------------------------//

  constCCVariable<int> cellType;
  old_dw->get( cellType,_cellType_label, matlIndex , patch, m_gn,0 );

  constCCVariable<double> abskt;
  old_dw->get( abskt,_abskt_label, matlIndex , patch, m_gn,0 );


  std::vector<constCCVariable<double> >abskg_array (m_nbands);
  if (_LspectralSolve){
    for (int iband=0; iband<m_nbands; iband++){
      old_dw->get(abskg_array[iband], _abskg_label_vector[iband], matlIndex , patch, m_gn, 0 ); // last abskg element is soot only (or zeros)
    }
  }

  //__________________________________
  //
  for (int iband=0; iband<m_nbands; iband++){

    const int idx = intensityIndx(dir,iband);

    CCVariable<double> intensity;
    new_dw->getModifiable(intensity, _IntensityLabels[idx], matlIndex, patch, _gv[m_plusX[dir] ][m_plusY[dir]  ][m_plusZ[dir]  ],1 );

    constCCVariable<double> emissSrc;
    if(m_doScattering){
      new_dw->get( emissSrc, _emiss_plus_scat_source_label[idx], matlIndex, patch, m_gn,0 );
    }else{
      new_dw->get( emissSrc, _emissSrc_label[iband], matlIndex, patch, m_gn,0 );
    }

    const int kstart = (m_plusZ[dir] ? idxLo.z() : idxHi.z()); // allows for direct logic in triple for loop
    const int jstart = (m_plusY[dir] ? idxLo.y() : idxHi.y());
    const int istart = (m_plusX[dir] ? idxLo.x() : idxHi.x());

    const int kDir = m_plusZ[dir] ? 1 : -1; // reverse logic for negative directions
    const int jDir = m_plusY[dir] ? 1 : -1;
    const int iDir = m_plusX[dir] ? 1 : -1;

    const int kEnd = m_plusZ[dir] ? idxHi.z() : -idxLo.z();  // reverse logic and bound for negative directions
    const int jEnd = m_plusY[dir] ? idxHi.y() : -idxLo.y();
    const int iEnd = m_plusX[dir] ? idxHi.x() : -idxLo.x();

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
    KokkosView3<const int, Kokkos::HostSpace>    kv_cellType  = cellType.getKokkosView();
    KokkosView3<const double, Kokkos::HostSpace> kv_emissSrc  = emissSrc.getKokkosView();
    KokkosView3<const double, Kokkos::HostSpace> kv_abskt     = abskt.getKokkosView();
    KokkosView3<double, Kokkos::HostSpace>       kv_intensity = intensity.getKokkosView();

    const int zdir = m_ziter[dir];
    const int ydir = m_yiter[dir];
    const int xdir = m_xiter[dir];
    ExecutionObject<UintahSpaces::CPU,UintahSpaces::HostSpace> execObj;
    int n_thread_partitions = 4; // 4 appears to optimium on a 4 core machine.......... meaning 1-22 threads per patch.
    if (_LspectralSolve){
      KokkosView3<const double, Kokkos::HostSpace> kv_abskg_array =  abskg_array[iband].getKokkosView();
      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
      Uintah::sweeping_parallel_for(execObj, range, [&](int i, int j, int k) {
        int km=k-zdir;
        int jm=j-ydir; 
        int im=i-xdir;
        kv_intensity(i,j,k) = (kv_emissSrc(i,j,k) + kv_intensity(i,j,km)*abs_oxi +  kv_intensity(i,jm,k)*abs_oeta  +  kv_intensity(im,j,k)*abs_omu)/(denom + (kv_abskg_array(i,j,k)  + kv_abskt(i,j,k))*vol);
        kv_intensity(i,j,k) = (kv_cellType(i,j,k) !=m_ffield) ? kv_emissSrc(i,j,k) : kv_intensity(i,j,k);
      }, m_plusX[dir], m_plusY[dir], m_plusZ[dir], n_thread_partitions);
    }else{
      Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
      Uintah::sweeping_parallel_for(execObj, range, [&](int i, int j, int k) {
        int km=k-zdir;
        int jm=j-ydir; 
        int im=i-xdir;
        kv_intensity(i,j,k) = (kv_emissSrc(i,j,k) + kv_intensity(i,j,km)*abs_oxi  +  kv_intensity(i,jm,k)*abs_oeta  +  kv_intensity(im,j,k)*abs_omu)/(denom + kv_abskt(i,j,k)*vol);
        kv_intensity(i,j,k) = (kv_cellType(i,j,k) !=m_ffield) ? kv_emissSrc(i,j,k) : kv_intensity(i,j,k);
      }, m_plusX[dir], m_plusY[dir], m_plusZ[dir], n_thread_partitions);
    }
#else
    int i;
    int im;
    int j;
    int jm;
    int k ;
    int km;

    //--------------------------------------------------------//
    // definition of abskt -> abskg_soot + abskg + sum(abskp_i) + scatkt
    // definition of abskt(spectral)-> abskg_soot +  sum(abskp_i) + scatkt
    //
    // definition of abskg ->  abskg + abskg_soot
    // definition of abskg_i(spectral)-> abskg_i
    //--------------------------------------------------------//
    if (_LspectralSolve){
      for ( k = kstart ;  kDir*k<=kEnd; k=k+m_ziter[dir]){
        km = k - m_ziter[dir];
        for ( j = jstart;  jDir*j<=jEnd; j=j+m_yiter[dir]){
          jm = j - m_yiter[dir];
          for ( i = istart;  (iDir*i<=iEnd)  ; i=i+ m_xiter[dir]){
            im = i - m_xiter[dir];
            if (cellType(i,j,k) != m_ffield){ // if intrusions
              intensity(i,j,k) = emissSrc(i,j,k) ;
            }
            else{ // else flow cell
              intensity(i,j,k) = (emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + (abskg_array[iband](i,j,k)  + abskt(i,j,k))*vol);
            } // end if intrusion
          } // end i loop
        } // end j loop
      } // end k loop
    }
    else{
      for ( k = kstart ;  kDir*k<=kEnd; k=k+m_ziter[dir]){
        km = k - m_ziter[dir];
        for ( j = jstart;  jDir*j<=jEnd; j=j+m_yiter[dir]){
          jm = j - m_yiter[dir];
          for ( i = istart;  (iDir*i<=iEnd)  ; i=i+m_xiter[dir]){
            im = i - m_xiter[dir];
            if (cellType(i,j,k) != m_ffield){ // if intrusions
              intensity(i,j,k) = emissSrc(i,j,k) ;
            }
            else{ // else flow cell
              intensity(i,j,k) = (emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol);
            } // end if intrusion
          } // end i loop
        } // end j loop
      } // end k loop
    }
#endif // end _OPENMP && KOKKOS_ENABLE_OPENMP
  } // end band loop
  return;
}


//-----------------------------------------------------------------//
// This function computes the source for radiation transport.  It
// requires abskg, abskp, radiation temperature, Intensities from the
// previous solve, particle temperature, asymmetry factor,
// and scatkt (last two only needed when including scattering)
//-----------------------------------------------------------------//
void
DORadiationModel::getDOSource(const Patch* patch,
                              const int matlIndex,
                              DataWarehouse* new_dw,
                              DataWarehouse* old_dw)
{

  _timer.reset(true); // Radiation solve start!

  constCCVariable<double> abskt;
  constCCVariable<double> radTemp;
  constCCVariable<int> cellType;

  old_dw->get(abskt,    _abskt_label,   matlIndex, patch, m_gn, 0);
  new_dw->get(radTemp,  _T_label,       matlIndex, patch, m_gn, 0);
  old_dw->get(cellType, _cellType_label,matlIndex, patch, m_gn, 0);

  std::vector< constCCVariable<double> > abskp(    m_nQn_part + ( (_LspectralSootOn && _LspectralSolve) ? 1:0)); // stor soot in particle array
  std::vector< constCCVariable<double> > partTemp( m_nQn_part + ( (_LspectralSootOn && _LspectralSolve) ? 1:0));
  std::vector<      CCVariable<double> > emissSrc(m_nbands);
  std::vector< constCCVariable<double> > abskg(   m_nbands);

  for (int ix=0;  ix< m_nQn_part; ix++){
    old_dw->get( abskp[ix],    _abskp_label_vector[ix],       matlIndex, patch, m_gn, 0);
    old_dw->get( partTemp[ix], _temperature_label_vector[ix], matlIndex, patch, m_gn, 0);
  }

  if(_LspectralSootOn && _LspectralSolve){
    old_dw->get( abskp[m_nQn_part], VarLabel::find("absksoot"), matlIndex, patch, m_gn, 0);
    new_dw->get( partTemp[m_nQn_part],_T_label,                 matlIndex, patch, m_gn, 0); // soot radiates at gas temp always
  }

  for (int iband=0; iband<m_nbands; iband++){
    old_dw->get(abskg[iband],_abskg_label_vector[iband],           matlIndex, patch, m_gn, 0);
    new_dw->allocateAndPut( emissSrc[iband], _emissSrc_label[iband], matlIndex, patch);  // optimization bug - make this be computed differently for intrusion cells
    emissSrc[iband].initialize(0.0);  // a sum will be performed on this variable, intialize it to zero.
  }

  std::vector< constCCVariable<double > > spectral_weights(_LspectralSolve ? m_nbands : 0 );
  if(_LspectralSolve){
    for (int iband=0; iband<m_nbands; iband++){
       old_dw->get(spectral_weights[iband], _abswg_label_vector[iband], matlIndex , patch, m_gn, 0  );
    }
  }

  computeIntensitySource(patch, abskp, partTemp, abskg, radTemp, emissSrc, spectral_weights);

  Vector Dx = patch->dCell();
  double volume = Dx.x()* Dx.y()* Dx.z();

  //__________________________________
  //
  if(m_doScattering){

    constCCVariable<double> scatkt;   //total scattering coefficient
    constCCVariable<double> asymmetryParam;

    old_dw->get(asymmetryParam,_asymmetry_label, matlIndex , patch, m_gn, 0);
    old_dw->get(scatkt,        _scatkt_label,    matlIndex , patch, m_gn, 0);

    for (int iband=0; iband<m_nbands; iband++){

      std::vector< constCCVariable<double> >IntensitiesOld(m_totalOrds);

      for( int ord=0;  ord<m_totalOrds ;ord++){
        const int idx = intensityIndx(ord,iband);
        old_dw->get(IntensitiesOld[ord],_IntensityLabels[idx], matlIndex , patch, m_gn, 0  );
      }

      // populate scattering source for each band and intensity-direction
      for( int ord=0;  ord<m_totalOrds ;ord++){

        const int idx = intensityIndx(ord,iband);

        CCVariable<double> scatIntensitySource;
        new_dw->allocateAndPut(scatIntensitySource, _emiss_plus_scat_source_label[idx], matlIndex, patch);

        computeScatteringIntensities(ord, scatkt, IntensitiesOld, scatIntensitySource, asymmetryParam, patch);

        Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
        Uintah::parallel_for( range,[&](int i, int j, int k){
          if(cellType(i,j,k) == m_ffield){
            scatIntensitySource(i,j,k) += emissSrc[iband](i,j,k);
            scatIntensitySource(i,j,k) *= volume;
          }else{
            scatIntensitySource(i,j,k) = (_sigma/M_PI)*std::pow(radTemp(i,j,k),4.0)*abskt(i,j,k)*_grey_reference_weight[iband];
          }
       });
      }
    }
  }
  //__________________________________
  //    No Scattering
  else{
    Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    Uintah::parallel_for( range,[&](int i, int j, int k) {
      if(cellType(i,j,k) == m_ffield){
        for (int iband=0; iband<m_nbands; iband++){
          emissSrc[iband](i,j,k)*=volume;
        }
      }else{
        double T4 = std::pow(radTemp(i,j,k),4.0);
        for (int iband=0; iband<m_nbands; iband++){
          emissSrc[iband](i,j,k) = (_sigma/M_PI) * T4 * abskt(i,j,k) * _grey_reference_weight[iband];
        }
      }
    });
  }
  return ;
}

//-----------------------------------------------------------------//
// This function computes 6 radiative fluxes, incident radiation,
//  and divQ. This function requires the required intensity fields
// sn*(2+sn), radiation temperature, abskt, and scatkt (scattering
// only)
//-----------------------------------------------------------------//
void
DORadiationModel::computeFluxDiv(const Patch* patch,
                                 const int matlIndex,
                                 DataWarehouse* new_dw,
                                 DataWarehouse* old_dw){

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  CCVariable<double> fluxE;
  CCVariable<double> fluxW;
  CCVariable<double> fluxN;
  CCVariable<double> fluxS;
  CCVariable<double> fluxT;
  CCVariable<double> fluxB;
  CCVariable<double> volQ;
  CCVariable<double> divQ;

  std::vector<CCVariable<double>> spectral_volQ(m_nbands);
  std::vector<constCCVariable<double>> spectral_abskg(m_nbands);

  if(_LspectralSolve){
    for (int iband=0; iband<m_nbands; iband++){

      old_dw->get(spectral_abskg[iband],_abskg_label_vector[iband], matlIndex, patch, m_gn,0 );
      spectral_volQ[iband].allocate(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
      spectral_volQ[iband].initialize(0.0);
    }
  }

  new_dw->allocateAndPut( divQ, _divQ_label, matlIndex, patch );
  new_dw->allocateAndPut( volQ, _volQ_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxE, _fluxE_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxW, _fluxW_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxN, _fluxN_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxS, _fluxS_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxT, _fluxT_label, matlIndex, patch );
  new_dw->allocateAndPut( fluxB, _fluxB_label, matlIndex, patch );

  divQ.initialize(0.0);
  volQ.initialize(0.0);
  fluxE.initialize(0.0);
  fluxW.initialize(0.0);
  fluxN.initialize(0.0);
  fluxS.initialize(0.0);
  fluxT.initialize(0.0);
  fluxB.initialize(0.0);

  std::vector< constCCVariable<double>> emissSrc (m_nbands);
  for (int iband=0; iband<m_nbands; iband++){
    new_dw->get(emissSrc[iband], _emissSrc_label[iband], matlIndex, patch, m_gn,0 );
  }

  constCCVariable<double> abskt;
  old_dw->get(abskt,_abskt_label, matlIndex , patch, m_gn, 0  );      // should be WHICH DW !!!!!! BUG

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  //__________________________________
  //
  for (int iband=0; iband<m_nbands; iband++){
    for (int ord = 0; ord < m_totalOrds; ord++){

      const int idx = intensityIndx(ord,iband);
      constCCVariable <double > intensity;
      new_dw->get(intensity,_IntensityLabels[idx] , matlIndex, patch, m_gn,0 );   // this should be a requires,  type restriction

      compute4Flux<constCCVariable<double>, const double> doFlux( m_wt[ord] * abs(m_omu[ord]) * m_xfluxAdjust,
                                                                  m_wt[ord] * abs(m_oeta[ord])* m_yfluxAdjust,
                                                                  m_wt[ord] * abs(m_oxi[ord]) * m_zfluxAdjust,
                                                                  m_wt[ord],  intensity,
                                                                  m_plusX[ord]==1 ? fluxE :  fluxW,
                                                                  m_plusY[ord]==1 ? fluxN :  fluxS,
                                                                  m_plusZ[ord]==1 ? fluxT :  fluxB,
                                                                  volQ);

      Uintah::parallel_for( range, doFlux );

      if(_LspectralSolve){
        Uintah::parallel_for( range, [&](int i, int j, int k){
          spectral_volQ[iband](i,j,k) += intensity(i,j,k) * m_wt[ord];
        });
      }
    }
  }

  //__________________________________
  //
  if(_LspectralSolve){
    if(m_doScattering){

      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt, _scatkt_label, matlIndex, patch, m_gn, 0);
      //computeDivQScat<constCCVariable<double> > doDivQ(abskt, emissSrc,volQ, divQ, scatkt);

      Uintah::parallel_for( range, [&](int i, int j, int k){
        for (int iband=0; iband<m_nbands; iband++){
          divQ(i,j,k) += (abskt(i,j,k) - scatkt(i,j,k) + spectral_abskg[iband](i,j,k)) * spectral_volQ[iband](i,j,k) - 4.0*M_PI * emissSrc[iband](i,j,k);
        }
      });
    }
    else{
      Vector Dx = patch->dCell();
      const double volume = Dx.x()* Dx.y()* Dx.z();  // have to do this because of multiplication of volume upstream, its avoided for scattering, since it has its own source terms.

      Uintah::parallel_for( range,  [&](int i, int j, int k){
        for (int iband=0; iband<m_nbands; iband++){
          divQ(i,j,k) += (abskt(i,j,k) + spectral_abskg[iband](i,j,k)) * spectral_volQ[iband](i,j,k) - 4.0*M_PI * emissSrc[iband](i,j,k)/volume;
        }
      });
    }
  }
  else{
    if(m_doScattering){
      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt,_scatkt_label, matlIndex , patch, m_gn, 0);

      Uintah::parallel_for( range, [&](int i, int j, int k){
         divQ(i,j,k)+= (abskt(i,j,k) - scatkt(i,j,k)) * volQ(i,j,k) - 4.0*M_PI*emissSrc[0](i,j,k);
      });
    }
    else{
      Vector Dx = patch->dCell();
      const double volume = Dx.x()* Dx.y()* Dx.z();  // have to do this because of multiplication of volume upstream, its avoided for scattering, since it has its own source terms.

      Uintah::parallel_for( range, [&](int i, int j, int k){
         divQ(i,j,k)+= abskt(i,j,k) * volQ(i,j,k) - 4.0*M_PI*emissSrc[0](i,j,k)/volume;
      });
    }
  }

//#ifdef ADD_PERFORMANCE_STATS
    // Add in the sweep stat.
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) DORadiationTime   ] += _timer().seconds();
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) DORadiationBands  ] += m_nbands;
    m_application->getApplicationStats()[ (ApplicationInterface::ApplicationStatsEnum) DORadiationSweeps ] += m_totalOrds*m_nbands;
    // For each stat recorded increment the count so to get a per patch value.
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) DORadiationTime );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) DORadiationBands );
    m_application->getApplicationStats().incrCount( (ApplicationInterface::ApplicationStatsEnum) DORadiationSweeps );
//#endif

  proc0cout << "//---------------------------------------------------------------------//\n";
  proc0cout << "Total Radiation Solve Time (Approximate): " << _timer().seconds() << " seconds for " << m_totalOrds*m_nbands<< " sweeps (bands=" <<m_nbands << ")\n";
  proc0cout << "//---------------------------------------------------------------------//\n";

  return ;
}


//______________________________________________________________________
//
void
DORadiationModel::setIntensityBC(const Patch* patch,
                                 const int matlIndex,
                                 DataWarehouse* new_dw,
                                 DataWarehouse* old_dw,
                                 const Ghost::GhostType ghostType,
                                 const int ord)
{

  constCCVariable<double> radTemp;
  constCCVariable<int>    cellType;
  new_dw->get(radTemp,  _T_label,        matlIndex , patch, m_gn, 0  );
  old_dw->get(cellType, _cellType_label, matlIndex , patch, m_gn, 0  );

  //__________________________________
  //  Loop over spectral bands
  for (int iband=0; iband<m_nbands; iband++){

    const int idx = intensityIndx(ord,iband);
    CCVariable <double > intensity;
    new_dw->allocateAndPut(intensity, _IntensityLabels[idx] , matlIndex, patch, ghostType, 1);
    intensity.initialize(0.0);

    //__________________________________
    // Loop over computational domain faces
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
      Patch::FaceType face = *iter;

      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

      for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector c = *iter;
        if (cellType[c] != m_ffield ){
          intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0)*_grey_reference_weight[iband]; // No reflections here!  Needs to be developed for reflections!
        }
      }
    }

  }
}

//______________________________________________________________________
//      utilities
//______________________________________________________________________
int
DORadiationModel::intensityIndx(const int ord,
                                const int iband)
{
  return (ord + iband * m_totalOrds );
}
