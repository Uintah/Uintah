/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/Radiation/RadPetscSolver.h>
#include <CCA/Components/Arches/Radiation/RadiationSolver.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ParticleModels/ParticleTools.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/Timers/Timers.hpp>

#include <sci_defs/hypre_defs.h>

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
#include <iomanip>


using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES_RADIATION",false);

//****************************************************************************
// Default constructor for DORadiationModel
//****************************************************************************
DORadiationModel::DORadiationModel(const ArchesLabel* label,
                                   const MPMArchesLabel* MAlab,
                                   const ProcessorGroup* myworld,bool sweepMethod ):
                                   d_lab(label),
                                   d_MAlab(MAlab),
                                   d_myworld(myworld),
                                   ffield(-1)  //WARNING: Hack -- flow cells set to -1

{
  _sweepMethod = sweepMethod;
  d_linearSolver = 0;
  d_perproc_patches = 0;

}

//****************************************************************************
// Destructor
//****************************************************************************
DORadiationModel::~DORadiationModel()
{
  if (!_sweepMethod)
    delete d_linearSolver;

  if(d_perproc_patches && d_perproc_patches->removeReference()){
    delete d_perproc_patches;
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

  db->getWithDefault("ReflectOn",reflectionsTurnedOn,false);  //  reflections are off by default.

  //db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions")
  std::string initialGuessType;
  db->getWithDefault("initialGuess",initialGuessType,"zeros"); //  using the previous solve as initial guess, is off by default
  if(initialGuessType=="zeros"){
    _zeroInitialGuess=true;
    _usePreviousIntensity=false;
  } else if(initialGuessType=="prevDir"){
    _zeroInitialGuess=false;
    _usePreviousIntensity=false;
  } else if(initialGuessType=="prevRadSolve"){
    _zeroInitialGuess=false;
    _usePreviousIntensity=true;
  }  else{
    throw ProblemSetupException("Error:DO-radiation initial guess not set!.", __FILE__, __LINE__);
  }
  //ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModelsV2");

  db->getWithDefault("ScatteringOn",_scatteringOn,false);
  db->getWithDefault("QuadratureSet",d_quadratureSet,"LevelSymmetric");

  std::string baseNameAbskp;
  std::string modelName;
  std::string baseNameTemperature;
  _radiateAtGasTemp=true; // this flag is arbitrary for no particles

  // Does this system have particles??? Check for particle property models

  _grey_reference_weight=std::vector<double> (1, 1.0);
  _nQn_part =0;
  _LspectralSolve=false;
  d_nbands=1;
  bool LsootOn=false;
  ProblemSpecP db_propV2 = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("PropertyModelsV2");
  if  (db_propV2){
    for ( ProblemSpecP db_model = db_propV2->findBlock("model"); db_model != nullptr;
        db_model = db_model->findNextBlock("model")){
      db_model->getAttribute("type", modelName);

      if (modelName=="partRadProperties"){
        bool doing_dqmom = ParticleTools::check_for_particle_method(db,ParticleTools::DQMOM);
        bool doing_cqmom = ParticleTools::check_for_particle_method(db,ParticleTools::CQMOM);

        if ( doing_dqmom ){
          _nQn_part = ParticleTools::get_num_env( db, ParticleTools::DQMOM );
        } else if ( doing_cqmom ){
          _nQn_part = ParticleTools::get_num_env( db, ParticleTools::CQMOM );
        } else {
          throw ProblemSetupException("Error: This method only working for DQMOM/CQMOM.",__FILE__,__LINE__);
        }

        db_model->getWithDefault( "part_temp_label", baseNameTemperature, "heat_pT" );
        db_model->getWithDefault( "radiateAtGasTemp", _radiateAtGasTemp, true );
        db_model->getAttribute("label",baseNameAbskp);

      }else if (modelName=="gasRadProperties"){
        _abskg_name_vector=std::vector<std::string>  (1);
        _abswg_name_vector=std::vector<std::string>  (0);
        db_model->getAttribute("label",_abskg_name_vector[0]);
      }else if(modelName=="spectralProperties"){
        _LspectralSolve=true;
        d_nbands=4;// hard coded for now (+1 for soot or particles, later on)

        double T_ref=0.0;
        double molarRatio_ref=0.0;
        db_model->getWithDefault( "Temperature_ref", T_ref, 1200 );
        db_model->getWithDefault( "CO2_H2OMolarRatio_ref", molarRatio_ref, 0.25 );
        molarRatio_ref=std::min(std::max(molarRatio_ref/1200.0,0.01),4.0);
        T_ref=std::min(std::max(T_ref,500.0),2400.0);  // limit reference to within bounds of fit data

        _abskg_name_vector=std::vector<std::string>  (d_nbands);
        _abswg_name_vector=std::vector<std::string>  (d_nbands);

        for (int i=0; i<d_nbands; i++){
          std::stringstream abskg_name;
          abskg_name << "abskg" <<"_"<< i;
          _abskg_name_vector[i]= abskg_name.str();
          std::stringstream abswg_name;
          abswg_name << "abswg" <<"_"<< i;
          _abswg_name_vector[i]= abswg_name.str();
        }
        std::string soot_name="";
        db_model->get("sootVolumeFrac",soot_name);
        if (soot_name==""){
          LsootOn=false;
        }else{
          LsootOn=true;
        }

        _grey_reference_weight=std::vector<double> (d_nbands+1,0.0); // +1 for transparent band 

        const double wecel_C_coeff[5][4][5] {
     
          {{0.7412956,  -0.9412652,   0.8531866,   -.3342806,    0.0431436 },
           {0.1552073,  0.6755648,  -1.1253940, 0.6040543,  -0.1105453},
           {0.2550242,  -0.605428,  0.8123855,  -0.45322990,  0.0869309},
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
       for (int kk=0; kk < n_coeff ;  kk++){
         for (int jj=0; jj < nrows;  jj++){
           for (int ii=0; ii < n_coeff;  ii++){
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


    for (int qn=0; qn < _nQn_part; qn++){
      std::stringstream absorp;
      std::stringstream temper;
      absorp <<baseNameAbskp <<"_"<< qn;
      temper <<baseNameTemperature <<"_"<< qn;
      _abskp_name_vector.push_back( absorp.str());
      _temperature_name_vector.push_back( temper.str());
    }

    // solve for transparent band if soot or particles are present
    if(_LspectralSolve && (_nQn_part > 0 || LsootOn)){
      std::stringstream abskg_name;
      abskg_name << "abskg" <<"_"<< d_nbands;
      _abskg_name_vector.push_back(abskg_name.str());
      std::stringstream abswg_name;
      abswg_name << "abswg" <<"_"<< d_nbands;
      _abswg_name_vector.push_back(abswg_name.str());
      d_nbands=d_nbands+1;
    }

  if (_scatteringOn  && _nQn_part ==0){
    throw ProblemSetupException("Error: No particle model found in DO-radiation! When scattering is turned on, a particle model is required!", __FILE__, __LINE__);
  }

  if (db) {
    bool ordinates_specified =db->findBlock("ordinates");
    db->getWithDefault("ordinates",d_sn,2);
    if (ordinates_specified == false){
      proc0cout << " Notice: No ordinate number specified.  Defaulting to 2." << endl;
    }
    if ((d_sn)%2 || d_sn <2){
      throw ProblemSetupException("Error:Only positive, even, and non-zero ordinate numbers for discrete-ordinates radiation are permitted.", __FILE__, __LINE__);
    }
  }
  else {
    throw ProblemSetupException("Error: <DORadiation> node not found.", __FILE__, __LINE__);
  }

  //WARNING: Hack -- Hard-coded for now.
  d_lambda      = 1;

  computeOrdinatesOPL();

  d_print_all_info = false;
  if ( db->findBlock("print_all_info") ){
    d_print_all_info = true;
  }

  string linear_sol;
  db->findBlock("LinearSolver")->getAttribute("type",linear_sol);
  if (!_sweepMethod){
    if (linear_sol == "petsc"){

      d_linearSolver = scinew RadPetscSolver(d_myworld);

    } else if (linear_sol == "hypre"){

      d_linearSolver = scinew RadHypreSolver(d_myworld);

    }

    d_linearSolver->problemSetup(db);
  }

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  for (int iband=0; iband<d_nbands; iband++){
    for( int ix=0;  ix<d_totalOrds ;ix++){
      ostringstream my_stringstream_object;
      //my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix << "_"<< iband ;
      my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix << "_"<< setw(2)<< iband ;
      //my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix  ;
      _IntensityLabels.push_back(  VarLabel::create(my_stringstream_object.str(),  CC_double));
      if(needIntensitiesBool()== false){
        break;  // gets labels for all intensities, otherwise only create 1 label
      }
    }
  }

  _emiss_plus_scat_source_label = std::vector<const VarLabel*> (0);
  if (_scatteringOn && _sweepMethod){
    for (int iband=0; iband<d_nbands; iband++){
      for( int ix=0;  ix< d_totalOrds;ix++){
        ostringstream my_stringstream_object;
        my_stringstream_object << "scatSrc_absSrc" << setfill('0') << setw(4)<<  ix <<"_"<<iband ;
        _emiss_plus_scat_source_label.push_back(  VarLabel::create(my_stringstream_object.str(),CC_double));
      }
    }
  }

  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxE"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxW"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxN"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxS"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxT"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxB"));


    // this was added with the hope that it could be extended to one-sided ghost cell requirements
   _gv =std::vector< std::vector < std::vector < Ghost::GhostType > > >   (2,  std::vector < std::vector < Ghost::GhostType > > (2, std::vector < Ghost::GhostType > (2) ));
   _gv[0][0][0] =Ghost::AroundCells ;
   _gv[0][0][1] =Ghost::AroundCells ;
   _gv[0][1][0] =Ghost::AroundCells ;
   _gv[0][1][1] =Ghost::AroundCells ;
   _gv[1][0][0] =Ghost::AroundCells ;
   _gv[1][0][1] =Ghost::AroundCells ;
   _gv[1][1][0] =Ghost::AroundCells ;
   _gv[1][1][1] =Ghost::AroundCells ;

}
//______________________________________________________________________
//
void
DORadiationModel::computeOrdinatesOPL(){



  d_totalOrds = d_sn*(d_sn+2);
  omu.resize( 1,d_totalOrds + 1);
  oeta.resize(1,d_totalOrds + 1);
  oxi.resize( 1,d_totalOrds + 1);
  wt.resize(  1,d_totalOrds + 1);


  omu.initialize(0.0);
  oeta.initialize(0.0);
  oxi.initialize(0.0);
  wt.initialize(0.0);
  if (d_quadratureSet=="LegendreChebyshev"){
    std::vector<double> xx(d_totalOrds,0.0);
    std::vector<double> yy(d_totalOrds,0.0);
    std::vector<double> zz(d_totalOrds,0.0);
    std::vector<double> ww(d_totalOrds,0.0);
    computeLegendreChebyshevQuadratureSet(d_sn, xx,yy,zz,ww);

    for (int i=0; i< d_totalOrds; i++){
      omu[i+1]= xx[i];
      oeta[i+1]=yy[i];
      oxi[i+1]=zz[i];
      wt[i+1]=ww[i];
    }
  } else{  // Level-Symmetric
    fort_rordr(d_sn, oxi, omu, oeta, wt);
  }

   double sumx=0;  double sumy=0;  double sumz=0;
  for (int i=0; i< d_totalOrds/8; i++){
   sumx+=omu[i+1]*wt[i+1];
   sumy+=oeta[i+1]*wt[i+1];
   sumz+=oxi[i+1]*wt[i+1];
  }

  d_xfluxAdjust=M_PI/sumx/4.0;  // sumx, sumy, sumz should equal pi/4 because: Int->0:pi/2 cos(theta) dOmega = pi,
  d_yfluxAdjust=M_PI/sumy/4.0;
  d_zfluxAdjust=M_PI/sumz/4.0;

  _plusX = vector< bool > (d_totalOrds,false);
  _plusY = vector< bool > (d_totalOrds,false);
  _plusZ = vector< bool > (d_totalOrds,false);
  xiter = vector< int > (d_totalOrds,-1);
  yiter = vector< int > (d_totalOrds,-1);
  ziter = vector< int > (d_totalOrds,-1);

  for (int direcn = 1; direcn <=d_totalOrds; direcn++){
    if (omu[direcn] > 0) {
      _plusX[direcn-1]= true;
      xiter[direcn-1]= 1;
    }
    if (oeta[direcn] > 0){
      _plusY[direcn-1]= true;
      yiter[direcn-1] = 1;
    }
    if (oxi[direcn] > 0) {
      _plusZ[direcn-1]= true;
      ziter[direcn-1]= 1;
    }
  }

  _sigma=5.67e-8;  //  w / m^2 k^4

  if(_scatteringOn){
    cosineTheta    = vector<vector< double > > (d_totalOrds,vector<double>(d_totalOrds,0.0));
    solidAngleWeight = vector< double >  (d_totalOrds,0.0);
    for (int i=0; i<d_totalOrds ; i++){
        solidAngleWeight[i]=  wt[i+1]/(4.0 * M_PI);
      for (int j=0; j<d_totalOrds ; j++){
        cosineTheta[i][j]=oxi[j+1]*oxi[i+1]+oeta[j+1]*oeta[i+1]+omu[j+1]*omu[i+1];
      }
    }
    // No adjustment factor appears to be needed for this form of the phase function. PHI=1+f*cos(theta)
    //for (int direction=0; direction<d_totalOrds ; direction++){
        //double  sumpF=0.0;
      //for (int i=0; i<d_totalOrds ; i++){
         //sumpF += (1.0 + 0.333333*cosineTheta[direction][i])*solidAngleWeight[i];
      //}
     //proc0cout << sumpF << "\n";
    //}
  }
}

//***************************************************************************
// Sets the radiation boundary conditions for the D.O method
//***************************************************************************
void
DORadiationModel::boundarycondition(const ProcessorGroup*,
                                    const Patch* patch,
                                    CellInformation* cellinfo,
                                    ArchesVariables* vars,
                                    ArchesConstVariables* constvars)
{

  //This should be done in the property calculator
//  //__________________________________
//  // loop over computational domain faces
//  vector<Patch::FaceType> bf;
//  patch->getBoundaryFaces(bf);
//
//  for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
//    Patch::FaceType face = *iter;
//
//    Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;
//
//    for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
//      IntVector c = *iter;
//      if (constvars->cellType[c] != ffield ){
//        vars->ABSKG[c]       = d_wall_abskg;
//      }
//    }
//  }

}


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
#ifdef UINTAH_ENABLE_KOKKOS
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

          int ipm = i+dirX;
          int jpm = j+dirY;
          int kpm = k+dirZ;

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




#ifdef UINTAH_ENABLE_KOKKOS
       KokkosView3<const int> cellType;
       KokkosView3<const double> wallTemp;
       KokkosView3<const double> abskt;

       KokkosView3<double> srcIntensity;
       KokkosView3<double> matrixB;
       KokkosView3<double> west;
       KokkosView3<double> south;
       KokkosView3<double> bottom;
       KokkosView3<double> center;
       KokkosView3<double> scatSource;
       KokkosView3<double> fluxX;
       KokkosView3<double> fluxY;
       KokkosView3<double> fluxZ;
#else
       constCCVariable<int>      &cellType;
       constCCVariable<double>   &wallTemp;
       constCCVariable<double>   &abskt;

       CCVariable<double>   &srcIntensity;
       CCVariable<double>   &matrixB;
       CCVariable<double>   &west;
       CCVariable<double>   &south;
       CCVariable<double>   &bottom;
       CCVariable<double>   &center;
       CCVariable<double>   &scatSource;
       CCVariable<double>   &fluxX;
       CCVariable<double>   &fluxY;
       CCVariable<double>   &fluxZ;
#endif //UINTAH_ENABLE_KOKKOS

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
#ifdef UINTAH_ENABLE_KOKKOS
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

#ifdef UINTAH_ENABLE_KOKKOS
       KokkosView3<constDouble_or_double> intensity; ///< intensity solution from linear solve
       KokkosView3<double> fluxX;   ///< x-directional flux ( positive or negative direction)
       KokkosView3<double> fluxY;   ///< y-directional flux ( positive or negative direction)
       KokkosView3<double> fluxZ;   ///< z-directional flux ( positive or negative direction)
       KokkosView3<double> volQ;    ///< Incident radiation
#else
       constCCVar_or_CCVar& intensity; ///< intensity solution from linear solve
       CCVariable<double>& fluxX;  ///< x-directional flux ( positive or negative direction)
       CCVariable<double>& fluxY;  ///< y-directional flux ( positive or negative direction)
       CCVariable<double>& fluxZ;  ///< z-directional flux ( positive or negative direction)
       CCVariable<double>& volQ;   ///< Incident radiation
#endif //UINTAH_ENABLE_KOKKOS
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
                                 int wall_type, int matlIndex,
                                 DataWarehouse* new_dw,
                                 DataWarehouse* old_dw,
                                 bool old_DW_isMissingIntensities)
{

  proc0cout << " Radiation Solve: " << endl;

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  Timers::Simple timer;
  timer.start();

  d_linearSolver->matrixInit(patch);

  rgamma.resize(1,29);
  sd15.resize(1,481);
  sd.resize(1,2257);
  sd7.resize(1,49);
  sd3.resize(1,97);

  rgamma.initialize(0.0);
  sd15.initialize(0.0);
  sd.initialize(0.0);
  sd7.initialize(0.0);
  sd3.initialize(0.0);

  if (d_lambda > 1) {
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


  if(reflectionsTurnedOn){
    for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){
      constCCVariable<double>  radiationFlux_temp;
      old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch,Ghost::None, 0  );
      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].copyData(radiationFlux_temp);
    }
  }
  else{
    for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){  // magic number cooresponds to number of labels tranported, when
      radiationFlux_old[i].allocate(domLo,domHi);
      radiationFlux_old[i].initialize(0.0);      // for no reflections, this must be zero
    }
  }


  if(_usePreviousIntensity==false){
    old_dw->get(constvars->cenint,_IntensityLabels[0], matlIndex , patch,Ghost::None, 0  );
    new_dw->getModifiable(vars->cenint,_IntensityLabels[0] , matlIndex, patch ); // per the logic in sourceterms/doradiation, old and new dw are the same.
  }


  std::vector< constCCVariable<double> > Intensities((_scatteringOn && !old_DW_isMissingIntensities) ? d_totalOrds : 0);

  std::vector< CCVariable<double> > IntensitiesRestart((_scatteringOn && old_DW_isMissingIntensities) ? d_totalOrds : 0);

  CCVariable<double> scatIntensitySource;
  constCCVariable<double> scatkt;   //total scattering coefficient
  constCCVariable<double> asymmetryParam;   //total scattering coefficient

  scatIntensitySource.allocate(domLo,domHi);
  scatIntensitySource.initialize(0.0); // needed for non-scattering cases


   Vector Dx = patch->dCell();
   double volume = Dx.x()* Dx.y()* Dx.z();
   double areaEW = Dx.y()*Dx.z();
   double areaNS = Dx.x()*Dx.z();
   double areaTB = Dx.x()*Dx.y();


  if(_scatteringOn){
    if(old_DW_isMissingIntensities){
      for( int ix=0;  ix<d_totalOrds ;ix++){
        IntensitiesRestart[ix].allocate(domLo,domHi);
        IntensitiesRestart[ix].initialize(0.0);
      }
    }else{
      for( int ix=0;  ix<d_totalOrds ;ix++){
        old_dw->get(Intensities[ix],_IntensityLabels[ix], matlIndex , patch,Ghost::None, 0  );
      }
    }
    old_dw->get(asymmetryParam,_asymmetryLabel, matlIndex , patch,Ghost::None, 0);
    old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
  }

  std::vector< constCCVariable<double> > abskp(_nQn_part);
  std::vector< constCCVariable<double> > partTemp(_nQn_part);
  for (int ix=0;  ix< _nQn_part; ix++){
      old_dw->get(abskp[ix],_abskp_label_vector[ix], matlIndex , patch,Ghost::None, 0  );
      old_dw->get(partTemp[ix],_temperature_label_vector[ix], matlIndex , patch,Ghost::None, 0  );
  }

  su.allocate(domLo,domHi);
  aw.allocate(domLo,domHi);
  as.allocate(domLo,domHi);
  ab.allocate(domLo,domHi);
  ap.allocate(domLo,domHi);


  srcbm.resize(domLo.x(),domHi.x());
  srcbm.initialize(0.0);
  srcpone.resize(domLo.x(),domHi.x());
  srcpone.initialize(0.0);
  qfluxbbm.resize(domLo.x(),domHi.x());
  qfluxbbm.initialize(0.0);

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
  std::vector< CCVariable<double> > Emission_source(1);
  std::vector< constCCVariable<double> > abskgas(1);
  abskgas[0]=constvars->ABSKG;
  Emission_source[0].allocate( patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() );
  Emission_source[0].initialize(0.0);
  for (int bands =1; bands <=d_lambda; bands++){


    vars->volq.initialize(0.0);
    computeIntensitySource(patch,abskp,partTemp,abskgas,constvars->temperature,Emission_source,spectral_weights);

    for (int direcn = 1; direcn <=d_totalOrds; direcn++){
      if(_usePreviousIntensity  && !old_DW_isMissingIntensities){
        old_dw->get(constvars->cenint,_IntensityLabels[direcn-1], matlIndex , patch,Ghost::None, 0  );
        new_dw->getModifiable(vars->cenint,_IntensityLabels[direcn-1] , matlIndex, patch );
      }
      else if ( _scatteringOn){
        new_dw->getModifiable(vars->cenint,_IntensityLabels[direcn-1] , matlIndex, patch );
      }
      if(old_DW_isMissingIntensities){
        old_dw->get(constvars->cenint,_IntensityLabels[0], matlIndex , patch,Ghost::None, 0  );
      }

      if(_zeroInitialGuess)
        vars->cenint.initialize(0.0); // remove once RTs have been checked.


      su.initialize(0.0);
      aw.initialize(0.0);
      as.initialize(0.0);
      ab.initialize(0.0);
      ap.initialize(0.0);


      bool plusX, plusY, plusZ;
      plusX = (omu[direcn]  > 0.0)? 1 : 0;
      plusY = (oeta[direcn] > 0.0)? 1 : 0;
      plusZ = (oxi[direcn]  > 0.0)? 1 : 0;

      d_linearSolver->gridSetup(plusX, plusY, plusZ);

      if(_scatteringOn){
      if(old_DW_isMissingIntensities)
        computeScatteringIntensities(direcn,scatkt, IntensitiesRestart,scatIntensitySource,asymmetryParam, patch);
      else
        computeScatteringIntensities(direcn,scatkt, Intensities,scatIntensitySource,asymmetryParam, patch);
      }

      // old construction of the A-matrix using fortran
      //fort_rdomsolve( idxLo, idxHi, constvars->cellType, ffield,
                      //cellinfo->sew, cellinfo->sns, cellinfo->stb,
                      //vars->ESRCG, direcn, oxi, omu,oeta, wt,
                      //constvars->temperature, constvars->ABSKT,
                      //su, aw, as, ab, ap,
                      //plusX, plusY, plusZ, fraction, bands, //fraction set to 1.0
                      //radiationFlux_old[0] , radiationFlux_old[1],
                      //radiationFlux_old[2] , radiationFlux_old[3],
                      //radiationFlux_old[4] , radiationFlux_old[5],scatIntensitySource); //  this term needed for scattering

     // new (2-2017) construction of A-matrix and b-matrix
      computeAMatrix  doMakeMatrixA( omu[direcn], oeta[direcn], oxi[direcn],
                                     areaEW, areaNS, areaTB, volume, ffield,
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
                                 su, ab, as, aw, ap, d_print_all_info );

      bool converged =  d_linearSolver->radLinearSolve( direcn, d_print_all_info );

      if(_usePreviousIntensity){
        vars->cenint.initialize(0.0); // Extra cells of intensity solution are not set when using non-zero initial guess.  Reset field to initialize extra cells
      }

      if (converged) {
        d_linearSolver->copyRadSoln(patch, vars);
      }else {
        throw InternalError("Radiation solver not converged", __FILE__, __LINE__);
      }



      //fort_rdomvolq( idxLo, idxHi, direcn, wt, vars->cenint, vars->volq);
      //fort_rdomflux( idxLo, idxHi, direcn, oxi, omu, oeta, wt, vars->cenint,
                     //plusX, plusY, plusZ,
                     //vars->qfluxe, vars->qfluxw,
                     //vars->qfluxn, vars->qfluxs,
                     //vars->qfluxt, vars->qfluxb);
                     //
                     //
                     //



      compute4Flux<CCVariable<double>, double> doFlux(wt[direcn]*abs(omu[direcn])*d_xfluxAdjust,wt[direcn]*abs(oeta[direcn])*d_yfluxAdjust,wt[direcn]*abs(oxi[direcn])*d_zfluxAdjust,
                                                                    wt[direcn],  vars->cenint,
                                                                    plusX ? vars->qfluxe :  vars->qfluxw,
                                                                    plusY ? vars->qfluxn :  vars->qfluxs,
                                                                    plusZ ? vars->qfluxt :  vars->qfluxb,
                                                                    vars->volq);

      Uintah::parallel_for( range, doFlux );

    }  // ordinate loop

    if(_scatteringOn){
      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k)+= (constvars->ABSKT(i,j,k)-scatkt(i,j,k))*vars->volq(i,j,k) - 4.0*M_PI*Emission_source[0](i,j,k);
      });
    }else{
      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k)+= (constvars->ABSKT(i,j,k))*vars->volq(i,j,k) - 4.0*M_PI*Emission_source[0](i,j,k);
      });
    }
      //fort_rdomsrcscattering( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ, scatkt,scatIntensitySource);
      //fort_rdomsrc( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ);

  }  // bands loop

  d_linearSolver->destroyMatrix();

  proc0cout << "Total Radiation Solve Time: " << timer().seconds() << " seconds\n";

}
// returns the total number of directions, sn*(sn+2)
int
DORadiationModel::getIntOrdinates(){
return d_totalOrds;
}

// Do the walls reflect? (should only be off if emissivity of walls = 1.0)
bool
DORadiationModel::reflectionsBool(){
return reflectionsTurnedOn;
}

// Do the Intensities need to be saved from the previous solve?
// Yes, if we are using the previous intensity as our initial guess in the linear solve.
// Yes, if we modeling scattering physics, by lagging the scattering source term.
bool
DORadiationModel::needIntensitiesBool(){ //should I compute intensity fields?  sweeps needs to compute for communication purposes
return _usePreviousIntensity || _scatteringOn || _sweepMethod;
}

// Model scattering physics of particles?
bool
DORadiationModel::ScatteringOnBool(){
return _scatteringOn;
}

void
DORadiationModel::setLabels(const VarLabel* abskg_label,
                            const VarLabel* abskt_label,
                            const VarLabel* T_label,
                            const VarLabel* cellTypeLabel,
                            std::vector<const VarLabel*> radIntSource,
                            const VarLabel*  fluxE,
                            const VarLabel*  fluxW,
                            const VarLabel*  fluxN,
                            const VarLabel*  fluxS,
                            const VarLabel*  fluxT,
                            const VarLabel*  fluxB,
                            const VarLabel*  volQ,
                            const VarLabel*  divQ){

    _abskg_label_vector= std::vector<const VarLabel* > (d_nbands);
    _abswg_label_vector= std::vector<const VarLabel* > (_LspectralSolve? d_nbands : 0 );
    _radIntSource=radIntSource;
    //_radIntSource=std::vector<const VarLabel*>(d_nbands);
    //for (int iband=0; iband<d_nbands; iband++){
      //_radIntSource[iband]=radIntSource[iband];
    //}
    _abskt_label=abskt_label;
    _T_label=T_label;
    _cellTypeLabel=cellTypeLabel;
    _fluxE=fluxE;
    _fluxW=fluxW;
    _fluxN=fluxN;
    _fluxS=fluxS;
    _fluxT=fluxT;
    _fluxB=fluxB;
    _volQ=volQ;
    _divQ=divQ;

  if(_LspectralSolve){
    for (int i=0; i<d_nbands; i++){
      _abskg_label_vector[i]=VarLabel::find(_abskg_name_vector[i]);
      _abswg_label_vector[i]=VarLabel::find(_abswg_name_vector[i]);
      if (_abskg_label_vector[i]==nullptr){
        throw ProblemSetupException("Error: spectral gas absorption coefficient label not found."+_abskg_name_vector[i], __FILE__, __LINE__);
      }
      if (_abswg_label_vector[i]==nullptr){
        throw ProblemSetupException("Error: spectral gas weighting coefficient label not found."+_abswg_name_vector[i], __FILE__, __LINE__);
      }
    }
  }else{
    _abskg_label_vector[0]=abskg_label;
  }


  for (int qn=0; qn < _nQn_part; qn++){
    _abskp_label_vector.push_back(VarLabel::find(_abskp_name_vector[qn]));
    if (_abskp_label_vector[qn]==0){
      throw ProblemSetupException("Error: particle absorption coefficient node not found."+_abskp_name_vector[qn], __FILE__, __LINE__);
    }

    _temperature_label_vector.push_back(VarLabel::find(_temperature_name_vector[qn]));

    if (_temperature_label_vector[qn]==0){
      throw ProblemSetupException("Error: particle temperature node not foundr! "+_temperature_name_vector[qn], __FILE__, __LINE__);
    }
  }


  if(_scatteringOn){
    _scatktLabel= VarLabel::find("scatkt");
    _asymmetryLabel=VarLabel::find("asymmetryParam");
  }
  return;
}


void
DORadiationModel::setLabels(){

  for (int qn=0; qn < _nQn_part; qn++){
    _abskp_label_vector.push_back(VarLabel::find(_abskp_name_vector[qn]));
    if (_abskp_label_vector[qn]==0){
      throw ProblemSetupException("Error: particle absorption coefficient node not found."+_abskp_name_vector[qn], __FILE__, __LINE__);
    }

    _temperature_label_vector.push_back(VarLabel::find(_temperature_name_vector[qn]));

    if (_temperature_label_vector[qn]==0){
      throw ProblemSetupException("Error: particle temperature node not foundr! "+_temperature_name_vector[qn], __FILE__, __LINE__);
    }
  }


  if(_scatteringOn){
    _scatktLabel= VarLabel::find("scatkt");
    _asymmetryLabel=VarLabel::find("asymmetryParam");
  }
  return;
}

template<class TYPE>
void
DORadiationModel::computeScatteringIntensities(int direction, constCCVariable<double> &scatkt, std::vector < TYPE > &Intensities, CCVariable<double> &scatIntensitySource,constCCVariable<double> &asymmetryFactor , const Patch* patch){


  direction -=1;   // change from fortran vector to c++ vector
  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  scatIntensitySource.initialize(0.0); //initialize to zero for sum

  int  binSize=8; //optimization parameter since this integral is a bit nasty ~10 appears to be ideal
  int nsets=std::ceil(d_totalOrds/binSize);
  for (int iset=0; iset <nsets; iset++){
    Uintah::parallel_for( range,[&](int i, int j, int k){  // should invert this loop, and remove if-statement
         for (int ii=iset*binSize; ii < std::min((iset+1)*binSize,d_totalOrds) ; ii++) {
           double phaseFunction = (1.0 + asymmetryFactor(i,j,k)*cosineTheta[direction][ii])*solidAngleWeight[ii];
           scatIntensitySource(i,j,k)  +=phaseFunction*Intensities[ii](i,j,k);
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
DORadiationModel::computeIntensitySource( const Patch* patch, std::vector <constCCVariable<double> >&abskp,
    std::vector <constCCVariable<double> > &pTemp,
    std::vector< constCCVariable<double> > &abskg,
                  constCCVariable<double>  &gTemp,
    std::vector<     CCVariable<double> > &b_sourceArray,
    std::vector<  constCCVariable<double> > &spectral_weights){


    //Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
  for (int qn=0; qn < _nQn_part; qn++){
    if( _radiateAtGasTemp ){
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        double sum=(_sigma/M_PI)*abskp[qn][*iter]*std::pow(gTemp[*iter],4.0);
    for ( int iStatic=0; iStatic< b_sourceArray.size(); iStatic++){
        b_sourceArray[iStatic][*iter]+=sum*_grey_reference_weight[iStatic];
    }


      //Uintah::parallel_for( range,[&](int i, int j, int k){
              //double T2 =gTemp(i,j,k)*gTemp(i,j,k);
              //b_sourceArray(i,j,k)+=(_sigma/M_PI)*abskp[qn](i,j,k)*T2*T2;
      //});
      }
    }else{
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        double sum=((_sigma/M_PI)*abskp[qn][*iter])*std::pow(pTemp[qn][*iter],4.0);
          for (int iStatic=0; iStatic< b_sourceArray.size(); iStatic++){
            b_sourceArray[iStatic][*iter]+=sum*_grey_reference_weight[iStatic];
          }
      //Uintah::parallel_for( range,[&](int i, int j, int k){
              //double T2 =pTemp[qn](i,j,k)*pTemp[qn](i,j,k);
              //b_sourceArray(i,j,k)+=((_sigma/M_PI)*abskp[qn](i,j,k))*T2*T2;
//});
    }
 }
  
}

    if (_LspectralSolve){
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
         const double T4=std::pow(gTemp[*iter],4.0);
        for ( int iStatic=0; iStatic< b_sourceArray.size(); iStatic++){
          b_sourceArray[iStatic][*iter]+=(_sigma/M_PI)*spectral_weights[iStatic][*iter]*abskg[iStatic][*iter]*T4;
        }
      }
    }else{
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        for ( int iStatic=0; iStatic< b_sourceArray.size(); iStatic++){ // 1 iteration for non-spectral cases
          b_sourceArray[iStatic][*iter]+=(_sigma/M_PI)*abskg[iStatic][*iter]*std::pow(gTemp[*iter],4.0);
        }
      }
    }
      //Uintah::parallel_for( range,[&](int i, int j, int k){
      //double T2 =gTemp(i,j,k)*gTemp(i,j,k);
      //b_sourceArray(i,j,k)+=(_sigma/M_PI)*abskg(i,j,k)*T2*T2;
  //});
//}

  return;
}

//-----------------------------------------------------------------//
// This function computes the intensities. The fields that are required are
// cellType, radiation temperature, radiation source, and  abskt.
// This function is probably the bottle-neck in the radiation solve (sweeping method).
//-----------------------------------------------------------------//
void
DORadiationModel::intensitysolveSweepOptimizedOLD( const Patch* patch,
                                       const int matlIndex,
                                       DataWarehouse* new_dw,
                                       DataWarehouse* old_dw,
                                       const int cdirecn){

  const int direcn = cdirecn+1;
  const IntVector idxLo = patch->getFortranCellLowIndex();
  const IntVector idxHi = patch->getFortranCellHighIndex();

  // -------------------NEEDS TO BE ADDED, REFLCTIONS ON WALLS -----------------//
  //IntVector domLo = patch->getExtraCellLowIndex();
  //IntVector domHi = patch->getExtraCellHighIndex();
  //std::vector< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.

  //if(reflectionsTurnedOn){
  //for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){
  //constCCVariable<double>  radiationFlux_temp;
  //old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch,Ghost::None, 0  );
  //radiationFlux_old[i].allocate(domLo,domHi);
  //radiationFlux_old[i].copyData(radiationFlux_temp);
  //}
  //}
  // ---------------------------------------------------------------------------//


  CCVariable <double > intensity;
  new_dw->getModifiable(intensity,_IntensityLabels[cdirecn] , matlIndex, patch);   // change to computes when making it its own task

  constCCVariable <double > ghost_intensity;
  new_dw->get( ghost_intensity, _IntensityLabels[cdirecn], matlIndex, patch, _gv[(int) _plusX[cdirecn]][(int) _plusY[cdirecn]][(int) _plusZ[cdirecn]],1 );


  constCCVariable <double > emissSrc;
  if(_scatteringOn){
    new_dw->get( emissSrc, _emiss_plus_scat_source_label[cdirecn], matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
  }else{
    for (int iband=0; iband<d_nbands; iband++){
      new_dw->get( emissSrc, _radIntSource[iband], matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
    }
  }

  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<double> abskt;
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );


  Vector Dx = patch->dCell();
  double  areaew = Dx.y()*Dx.z();
  double  areans = Dx.x()*Dx.z();
  double  areatb = Dx.x()*Dx.y();

  const double vol = Dx.x()* Dx.y()* Dx.z();  // const to increase speed?
  const double  abs_oxi= std::abs(oxi[direcn])*areatb;
  const double  abs_oeta=std::abs(oeta[direcn])*areans;
  const double  abs_omu= std::abs(omu[direcn])*areaew;
  const double  denom = abs(omu[direcn])*areaew+abs(oeta[direcn])*areans+abs(oxi[direcn])*areatb; // denomintor for Intensity in current cell



  ///--------------------------------------------------//
  ///------------perform sweep on one patch -----------//
  ///--------------------------------------------------//
  //--------------------------------------------------------//
  // Step 1:
  //  Set seed cell (three ghost cells and no normal cells)
  //--------------------------------------------------------//
  int i= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  int im=i-xiter[cdirecn];
  int j= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  int jm=j-yiter[cdirecn];
  int k = _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  int km=k-ziter[cdirecn];
  if (cellType(i,j,k) !=ffield){ // if intrusions
    intensity(i,j,k) = emissSrc(i,j,k) ;
  } else{ // else flow cell
    intensity(i,j,k) = (emissSrc(i,j,k) +ghost_intensity(i,j,km)*abs_oxi  +  ghost_intensity(i,jm,k)*abs_oeta  +  ghost_intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol);
  } // end if
  //--------------------------------------------------------//
  // Step 2:
  //  Set seed rows (two ghost cells and one normal cells)
  //--------------------------------------------------------//
  ////--------------------set zy----------------------//
  j = _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  jm=j-yiter[cdirecn];
  k = _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  km=k-ziter[cdirecn];
  for ( i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
    im=i-xiter[cdirecn];
    if (cellType(i,j,k) !=ffield){ // if intrusions
      intensity(i,j,k) = emissSrc(i,j,k) ;
    } else{ // else flow cell
      intensity(i,j,k) = ( emissSrc(i,j,k) +ghost_intensity(i,j,km)*abs_oxi  +  ghost_intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
    } // end if
  }
  ////--------------------set xz----------------------//
  i= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  im=i-xiter[cdirecn];
  k= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  km=k-ziter[cdirecn];
  for (j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
    jm=j-yiter[cdirecn];
    if (cellType(i,j,k) !=ffield){ // if intrusions
      intensity(i,j,k) = emissSrc(i,j,k) ;
    } else{ // else flow cell
      intensity(i,j,k) = ( emissSrc(i,j,k) + ghost_intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  ghost_intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
    } // end if
  }
  ////--------------------set yx----------------------//
  i= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  im=i-xiter[cdirecn];
  j= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  jm=j-yiter[cdirecn];
  for ( k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
    km=k-ziter[cdirecn];
    if (cellType(i,j,k) !=ffield){ // if intrusions
      intensity(i,j,k) = emissSrc(i,j,k) ;
    } else{ // else flow cell
      intensity(i,j,k) = ( emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  ghost_intensity(i,jm,k)*abs_oeta  +  ghost_intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
    } // end if
  }
  //
  //
  //
  //-------------------- ghost cells stored in different structure requires this----------------------//
  //--------------------------------------------------------//
  // Step 3:
  //  Set seed faces (one ghost cells and two normal cells)
  //--------------------------------------------------------//
  //-------------------- set z ----------------------//
  k= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  km=k-ziter[cdirecn];
  for ( j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
    jm=j-yiter[cdirecn];
    for ( i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
      im=i-xiter[cdirecn];
      if (cellType(i,j,k) !=ffield){ // if intrusions
        intensity(i,j,k) = emissSrc(i,j,k) ;
      } else{ // else flow cell
        intensity(i,j,k) = ( emissSrc(i,j,k) + ghost_intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
      } // end if
    }
  }
  ////--------------------set y----------------------//
  j= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  jm=j-yiter[cdirecn];
  for ( k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
    km=k-ziter[cdirecn];
    for ( i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
      im=i-xiter[cdirecn];
      if (cellType(i,j,k) !=ffield){ // if intrusions
        intensity(i,j,k) = emissSrc(i,j,k) ;
      } else{ // else flow cell
        intensity(i,j,k) = ( emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  ghost_intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
      } // end if
    }
  }
  ////--------------------set x----------------------//
  i= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  im=i-xiter[cdirecn];
  for ( k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
    km=k-ziter[cdirecn];
    for (j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
      jm=j-yiter[cdirecn];
      if (cellType(i,j,k) !=ffield){ // if intrusions
        intensity(i,j,k) = emissSrc(i,j,k) ;
      } else{ // else flow cell
        intensity(i,j,k) = ( emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  ghost_intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol);
        //intensity[c] = 0.0 ;
      } // end if
    }
  }
  ///  --------------------------------------------------//
  //--------------------------------------------------------//
  // Step 4:
  //  Set interior cells (no ghost cells and three normal cells)
  //--------------------------------------------------------//

  for ( k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
    km=k-ziter[cdirecn];
    for ( j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
      jm=j-yiter[cdirecn];
      for ( i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
        im=i-xiter[cdirecn];
        if (cellType(i,j,k) !=ffield){ // if intrusions
          intensity(i,j,k) = emissSrc(i,j,k) ;
        } else{ // else flow cell
          intensity(i,j,k) = (emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol );
        } // end if
      } // end i loop
    } // end j loop
  } // end k loop
  // --------------------------------------------------//

  return;
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
                                       const int cdirecn){

  const int direcn = cdirecn+1;
  const IntVector idxLo = patch->getFortranCellLowIndex();
  const IntVector idxHi = patch->getFortranCellHighIndex();

  Vector Dx = patch->dCell();
  double  areaew = Dx.y()*Dx.z();
  double  areans = Dx.x()*Dx.z();
  double  areatb = Dx.x()*Dx.y();

  const double vol = Dx.x()* Dx.y()* Dx.z();  // const to increase speed?
  const double  abs_oxi= std::abs(oxi[direcn])*areatb;
  const double  abs_oeta=std::abs(oeta[direcn])*areans;
  const double  abs_omu= std::abs(omu[direcn])*areaew;
  const double  denom = abs(omu[direcn])*areaew+abs(oeta[direcn])*areans+abs(oxi[direcn])*areatb; // denomintor for Intensity in current cell

  // -------------------NEEDS TO BE ADDED, REFLCTIONS ON WALLS -----------------//
  //IntVector domLo = patch->getExtraCellLowIndex();
  //IntVector domHi = patch->getExtraCellHighIndex();
  //std::vector< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.

  //if(reflectionsTurnedOn){
  //for (unsigned int i=0; i<  _radiationFluxLabels.size(); i++){
  //constCCVariable<double>  radiationFlux_temp;
  //old_dw->get(radiationFlux_temp,_radiationFluxLabels[i], matlIndex , patch,Ghost::None, 0  );
  //radiationFlux_old[i].allocate(domLo,domHi);
  //radiationFlux_old[i].copyData(radiationFlux_temp);
  //}
  //}
  // ---------------------------------------------------------------------------//
    
  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<double> abskt;
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );


  constCCVariable<double> abskg;
  std::vector<constCCVariable<double> >abskg_array (d_nbands);
  if (_LspectralSolve){
    old_dw->get(abskg,_abskg_label_vector[d_nbands-1], matlIndex , patch,Ghost::None, 0  ); // last abskg element is soot only (or zeros)
    for (int iband=0; iband<d_nbands; iband++){
      old_dw->get(abskg_array[iband],_abskg_label_vector[iband], matlIndex , patch,Ghost::None, 0  ); // last abskg element is soot only (or zeros)
    }
  }

  for (int iband=0; iband<d_nbands; iband++){

    CCVariable <double > intensity;
    new_dw->getModifiable(intensity,_IntensityLabels[cdirecn+iband*d_totalOrds] , matlIndex, patch,Ghost::AroundCells, 1);   

    constCCVariable <double > emissSrc;
    if(_scatteringOn){
      new_dw->get( emissSrc, _emiss_plus_scat_source_label[cdirecn+iband*d_totalOrds], matlIndex, patch, Ghost::None,0 );  
    }else{
      new_dw->get( emissSrc, _radIntSource[iband], matlIndex, patch, Ghost::None,0 );  
    }



    int i ;
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
    // definition of abskg_i(spectral)-> abskg_soot_i + abskg_soot_i 
    //--------------------------------------------------------//
    if (_LspectralSolve){
      for ( k = (_plusZ[cdirecn] ? idxLo.z() : idxHi.z());  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
        km=k-ziter[cdirecn];
        for ( j = (_plusY[cdirecn] ? idxLo.y() : idxHi.y());  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
          jm=j-yiter[cdirecn];
          for ( i = (_plusX[cdirecn] ? idxLo.x() : idxHi.x());  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
            im=i-xiter[cdirecn];
            if (cellType(i,j,k) !=ffield){ // if intrusions
              intensity(i,j,k) = emissSrc(i,j,k) ;
            } else{ // else flow cell
              intensity(i,j,k) = (emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + (abskg_array[iband](i,j,k) - abskg(i,j,k) + abskt(i,j,k))*vol);
            } // end if intrusion
          } // end i loop
        } // end j loop
      } // end k loop
    }else{
      for ( k = (_plusZ[cdirecn] ? idxLo.z() : idxHi.z());  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){
        km=k-ziter[cdirecn];
        for ( j = (_plusY[cdirecn] ? idxLo.y() : idxHi.y());  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
          jm=j-yiter[cdirecn];
          for ( i = (_plusX[cdirecn] ? idxLo.x() : idxHi.x());  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
            im=i-xiter[cdirecn];
            if (cellType(i,j,k) !=ffield){ // if intrusions
              intensity(i,j,k) = emissSrc(i,j,k) ;
            } else{ // else flow cell
              intensity(i,j,k) = (emissSrc(i,j,k) + intensity(i,j,km)*abs_oxi  +  intensity(i,jm,k)*abs_oeta  +  intensity(im,j,k)*abs_omu)/(denom + abskt(i,j,k)*vol);
            } // end if intrusion
          } // end i loop
        } // end j loop
      } // end k loop
    }
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
                              int matlIndex,
                              DataWarehouse* new_dw,
                              DataWarehouse* old_dw){

  _timer.reset(true); // Radiation solve start!

  constCCVariable<double> abskt;
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<double> radTemp ;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );

  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel,matlIndex,patch,Ghost::None,0);

  std::vector< constCCVariable<double> > abskp(_nQn_part);
  std::vector< constCCVariable<double> > partTemp(_nQn_part);
  std::vector< CCVariable <double > > emissSrc(d_nbands);
  std::vector< constCCVariable<double> > abskg(d_nbands);

  for (int ix=0;  ix< _nQn_part; ix++){
    old_dw->get(abskp[ix],_abskp_label_vector[ix], matlIndex , patch,Ghost::None, 0  );
    old_dw->get(partTemp[ix],_temperature_label_vector[ix], matlIndex , patch,Ghost::None, 0  );
  }

    for (int iband=0; iband<d_nbands; iband++){
      old_dw->get(abskg[iband],_abskg_label_vector[iband], matlIndex , patch,Ghost::None, 0  );
      new_dw->allocateAndPut( emissSrc[iband], _radIntSource[iband], matlIndex, patch);  // optimization bug - make this be computed differently for intrusion cells
      emissSrc[iband].initialize(0.0);  // a sum will be performed on this variable, intialize it to zero.
    }

    std::vector< constCCVariable<double > > spectral_weights(_LspectralSolve ? d_nbands : 0 );
    if(_LspectralSolve){
      for (int iband=0; iband<d_nbands; iband++){
         old_dw->get(spectral_weights[iband],_abswg_label_vector[iband], matlIndex , patch,Ghost::None, 0  );
      }
    }
    computeIntensitySource(patch,abskp,partTemp,abskg,radTemp,emissSrc,spectral_weights);


   Vector Dx = patch->dCell();
   double volume = Dx.x()* Dx.y()* Dx.z();
  if(_scatteringOn){

    constCCVariable<double> scatkt;   //total scattering coefficient
    constCCVariable<double> asymmetryParam;

    old_dw->get(asymmetryParam,_asymmetryLabel, matlIndex , patch,Ghost::None, 0);
    old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
   
    for (int iband=0; iband<d_nbands; iband++){

      std::vector< constCCVariable<double> >IntensitiesOld(d_totalOrds);
      // reconstruct oldIntensity staticArray for each band
      for( int ix=0;  ix<d_totalOrds ;ix++){
        old_dw->get(IntensitiesOld[ix],_IntensityLabels[ix+(iband)*d_totalOrds], matlIndex , patch,Ghost::None, 0  );
      }
      // populate scattering source for each band and intensity-direction
      for( int ix=0;  ix<d_totalOrds ;ix++){
        CCVariable<double> scatIntensitySource;
        new_dw->allocateAndPut(scatIntensitySource,_emiss_plus_scat_source_label[ix+(iband)*d_totalOrds], matlIndex, patch);

        computeScatteringIntensities(ix+1,scatkt, IntensitiesOld,scatIntensitySource,asymmetryParam, patch); // function expects fortran indices; spectral element handled by changing IntensitiesOld
        Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());

        Uintah::parallel_for( range,[&](int i, int j, int k){
               if(cellType(i,j,k)==ffield){
                 scatIntensitySource(i,j,k)+=emissSrc[iband](i,j,k);
                 scatIntensitySource(i,j,k)*=volume;
               }else{
                 scatIntensitySource(i,j,k)=(_sigma/M_PI)*std::pow(radTemp(i,j,k),4.0)*abskt(i,j,k)*_grey_reference_weight[iband];
               }
            });
      }
    }
  }else{
    Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    Uintah::parallel_for( range,[&](int i, int j, int k) {
              if(cellType(i,j,k)==ffield){
                for (int iband=0; iband<d_nbands; iband++){
                  emissSrc[iband](i,j,k)*=volume;
                }
              }else{
                double T4=std::pow(radTemp(i,j,k),4.0);
                for (int iband=0; iband<d_nbands; iband++){
                  emissSrc[iband](i,j,k)=(_sigma/M_PI)*T4*abskt(i,j,k)*_grey_reference_weight[iband];
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
                                 int matlIndex,
                                 DataWarehouse* new_dw,
                                 DataWarehouse* old_dw){

  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  CCVariable <double > fluxE;
  CCVariable <double > fluxW;
  CCVariable <double > fluxN;
  CCVariable <double > fluxS;
  CCVariable <double > fluxT;
  CCVariable <double > fluxB;
  CCVariable <double > volQ;
  CCVariable <double > divQ;

  std::vector<CCVariable <double > > spectral_volQ(d_nbands);
  std::vector<constCCVariable <double > > spectral_abskg(d_nbands);

  if(_LspectralSolve){
    for (int iband=0; iband<d_nbands; iband++){
      old_dw->get(spectral_abskg[iband],_abskg_label_vector[iband], matlIndex, patch, Ghost::None,0 );
      spectral_volQ[iband].allocate(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
      spectral_volQ[iband].initialize(0.0);
    }
  }

  new_dw->allocateAndPut( divQ, _divQ, matlIndex, patch );
  new_dw->allocateAndPut( volQ, _volQ, matlIndex, patch );
  new_dw->allocateAndPut( fluxE, _fluxE, matlIndex, patch );
  new_dw->allocateAndPut( fluxW, _fluxW, matlIndex, patch );
  new_dw->allocateAndPut( fluxN, _fluxN, matlIndex, patch );
  new_dw->allocateAndPut( fluxS, _fluxS, matlIndex, patch );
  new_dw->allocateAndPut( fluxT, _fluxT, matlIndex, patch );
  new_dw->allocateAndPut( fluxB, _fluxB, matlIndex, patch );

  divQ.initialize(0.0);
  volQ.initialize(0.0);
  fluxE.initialize(0.0);
  fluxW.initialize(0.0);
  fluxN.initialize(0.0);
  fluxS.initialize(0.0);
  fluxT.initialize(0.0);
  fluxB.initialize(0.0);

  std::vector< constCCVariable <double > > emissSrc (d_nbands);
  constCCVariable<double> abskt ;

  for (int iband=0; iband<d_nbands; iband++){
    new_dw->get(emissSrc[iband], _radIntSource[iband], matlIndex, patch, Ghost::None,0 );
  }

  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );      // should be WHICH DW !!!!!! BUG

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();


  for (int iband=0; iband<d_nbands; iband++){
    for (int direcn = 1; direcn <=d_totalOrds; direcn++){

      constCCVariable <double > intensity;
      new_dw->get(intensity,_IntensityLabels[direcn-1+iband*d_totalOrds] , matlIndex, patch, Ghost::None,0 );   // this should be a requires,  type restriction

      compute4Flux<constCCVariable<double>, const double> doFlux(wt[direcn]*abs(omu[direcn])*d_xfluxAdjust,wt[direcn]*abs(oeta[direcn])*d_yfluxAdjust,wt[direcn]*abs(oxi[direcn])*d_zfluxAdjust,
                                                    wt[direcn],  intensity,
                                                    _plusX[direcn-1]==1 ? fluxE :  fluxW,
                                                    _plusY[direcn-1]==1 ? fluxN :  fluxS,
                                                    _plusZ[direcn-1]==1 ? fluxT :  fluxB,
                                                    volQ);

      Uintah::parallel_for( range, doFlux );

      if(_LspectralSolve){ 
        Uintah::parallel_for( range,   [&](int i, int j, int k){
            spectral_volQ[iband](i,j,k) += intensity(i,j,k)*wt[direcn];
            });
      }
    }
  }


  if(_LspectralSolve){ 
    if(_scatteringOn){
      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
      //computeDivQScat<constCCVariable<double> > doDivQ(abskt, emissSrc,volQ, divQ, scatkt);
      Uintah::parallel_for( range,   [&](int i, int j, int k){
      for (int iband=0; iband<d_nbands; iband++){
         divQ(i,j,k) += (abskt(i,j,k)-scatkt(i,j,k)+spectral_abskg[iband](i,j,k))*spectral_volQ[iband](i,j,k) - 4.0*M_PI*emissSrc[iband](i,j,k);
      }
      });
    }else{
      Uintah::parallel_for( range,   [&](int i, int j, int k){
      for (int iband=0; iband<d_nbands; iband++){
         //divQ(i,j,k) += (abskt(i,j,k)+spectral_abskg[iband](i,j,k))*spectral_volQ[iband](i,j,k) - 4.0*M_PI*emissSrc[iband](i,j,k);
         divQ(i,j,k) += (abskt(i,j,k)+spectral_abskg[iband](i,j,k))*spectral_volQ[iband](i,j,k) - 4.0*M_PI*emissSrc[iband](i,j,k);
      }
      });
    }
  }else{
    if(_scatteringOn){
      constCCVariable<double> scatkt;   //total scattering coefficient
      old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k)+= (abskt(i,j,k)-scatkt(i,j,k))*volQ(i,j,k) - 4.0*M_PI*emissSrc[0](i,j,k);
      });
    }else{
      Uintah::parallel_for( range,   [&](int i, int j, int k){
         divQ(i,j,k)+= (abskt(i,j,k))*volQ(i,j,k) - 4.0*M_PI*emissSrc[0](i,j,k);
      });
    }
  }



  proc0cout << "//---------------------------------------------------------------------//\n";
  proc0cout << "Total Radiation Solve Time (Approximate): " << _timer().seconds() << " seconds for " << d_totalOrds*d_nbands<< " sweeps (bands=" <<d_nbands << ")\n";
  proc0cout << "//---------------------------------------------------------------------//\n";
  return ;

}



void
DORadiationModel::setIntensityBC(const Patch* patch,
                                 int matlIndex,
                                 CCVariable<double>& intensity,
                                 constCCVariable<double>& radTemp,
                                 constCCVariable<int>& cellType,
                                 int iSpectralBand){


    //-------------- Compute Intensity on boundarys------------//
    // loop over computational domain faces
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( vector<Patch::FaceType>::const_iterator iter = bf.begin(); iter != bf.end(); ++iter ){
      Patch::FaceType face = *iter;

      Patch::FaceIteratorType PEC = Patch::ExtraPlusEdgeCells;

      for (CellIterator iter =  patch->getFaceIterator(face, PEC); !iter.done(); iter++) {
        IntVector c = *iter;
        if (cellType[c] != ffield ){
          intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0)*_grey_reference_weight[iSpectralBand]; // No reflections here!  Needs to be developed for reflections!
        }
      }
    }

    // ----------faster way to set BCs----------//
      //IntVector c(0,0,0); // current cell
  //IntVector idxLo = patch->getFortranCellLowIndex();
  //IntVector idxHi = patch->getFortranCellHighIndex();

      /////  -------1 - patch boundary conditions set----------//
      ////--------------------set z----------------------//
      //for (int k = 0;  k<2 ; k++){
      //c[2]= k ? idxHi.z()+1 : idxLo.z()-1;
       ////if (cellType[IntVector(idxLo.x()+1,idxLo.y()+1, c.z()] ==
      //for (int j = idxLo.y() ;  j<=idxHi.y(); j++){
      //c[1]=j;
      //for (int i = idxLo.x() ;  i<=idxHi.x(); i++){
      //c[0]=i;
      //intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0) ;
      //}
      //}
      //}
      //////--------------------set y----------------------//
      //for (int k = idxLo.z();  k<=idxHi.z() ; k++){
      //c[2]=k;
      //for (int j = 0;  j<2 ; j++){
      //c[1]= j ? idxHi.y()+1 : idxLo.y()-1;
      //for (int i = idxLo.x();  i<=idxHi.x() ; i++){
      //c[0]=i;
      //intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0) ;
      //}
      //}
      //}
      //////--------------------set x----------------------//
      //for (int k = idxLo.z();  k<=idxHi.z() ; k++){
      //c[2]=k;
      //for (int j = idxLo.y();  j<=idxHi.y() ; j++){
      //c[1]=j;
      //for (int i = 0;  i<2 ; i++){
      //c[0]= i ? idxHi.x()+1 : idxLo.x()-1;
      //intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0) ;
      //}
      //}
      //}
      ///  --------------------------------------------------//

  return;
}

void
DORadiationModel::setIntensityBC2Orig(const Patch* patch,
                                 int matlIndex,
                                       DataWarehouse* new_dw,
                                       DataWarehouse* old_dw, int ix){




  constCCVariable<double> radTemp;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );

  for (int iband=0; iband<d_nbands; iband++){
  CCVariable <double > intensity;
  new_dw->getModifiable(intensity,_IntensityLabels[ix+iband*d_totalOrds] , matlIndex, patch);   // change to computes when making it its own task
    setIntensityBC(patch, matlIndex,intensity, radTemp,cellType,iband);
  }

  return;
}

void
DORadiationModel::setExtraSweepingLabels(int nphase){

  IntVector patchIntVector (0,0,0);
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
  _nphase=nphase;
  for( int ixx=0;  ixx<_nphase;ixx++){
    _patchIntensityLabels.push_back(std::vector< const VarLabel*> (0) );
    for( int ix=0;  ix<d_totalOrds ;ix++){
      ostringstream my_stringstream_object;
      my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix << ixx ;
      _patchIntensityLabels[ixx].push_back(VarLabel::create(my_stringstream_object.str(), CC_double));
    }
  }
}

