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
#include <CCA/Components/Arches/Radiation/fortran/radcoef_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radwsgg_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/radcal_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsolve_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrc_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomsrcscattering_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomflux_fort.h>
#include <CCA/Components/Arches/Radiation/fortran/rdomvolq_fort.h>
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

  db->getWithDefault("ScatteringOn",_scatteringOn,false);
  db->getWithDefault("QuadratureSet",d_quadratureSet,"LevelSymmetric");

  std::string baseNameAbskp;
  std::string modelName;
  std::string baseNameTemperature;
  _radiateAtGasTemp=true; // this flag is arbitrary for no particles

  // Does this system have particles??? Check for particle property models
   
  _nQn_part =0; 
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
        //  db_model->findBlock("calculator")->findBlock("abskg")->getAttribute("label",_abskg_label_name);
        break;
      }
    }
  }

    for (int qn=0; qn < _nQn_part; qn++){
      std::stringstream absorp;
      std::stringstream temper;
      absorp <<baseNameAbskp <<"_"<< qn;
      temper <<baseNameTemperature <<"_"<< qn;
      _abskp_name_vector.push_back( absorp.str());
      _temperature_name_vector.push_back( temper.str());
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

  fraction.resize(1,100);
  fraction.initialize(0.0);
  fraction[1]=1.0;  // This a hack to fix DORad with the new property model interface - Derek 6-14

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
  for( int ix=0;  ix<d_totalOrds ;ix++){
    ostringstream my_stringstream_object;
    my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix ;
    _IntensityLabels.push_back(  VarLabel::create(my_stringstream_object.str(),  CC_double));
    if(needIntensitiesBool()== false){
     break;  // gets labels for all intensities, otherwise only create 1 label
    }
  }

  if (_scatteringOn && _sweepMethod){
    for( int ix=0;  ix< d_totalOrds;ix++){
      ostringstream my_stringstream_object;
      my_stringstream_object << "scatSrc_absSrc" << setfill('0') << setw(4)<<  ix  ;
      _emiss_plus_scat_source_label.push_back(  VarLabel::create(my_stringstream_object.str(),CC_double));
    }
  }

  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxE"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxW"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxN"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxS"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxT"));
  _radiationFluxLabels.push_back(  VarLabel::find("radiationFluxB"));

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
template <typename constCCVar_or_CCVar>
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
       KokkosView3<double> intensity; ///< intensity solution from linear solve
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
// Compute the heat flux divergence with scattering on.  (This is necessary because abskt includes scattering coefficients)
//***************************************************************************
template <typename constCCVar_or_CCVar>
struct computeDivQScat{
       computeDivQScat(constCCVariable<double> &_abskt,
                       constCCVar_or_CCVar &_intensitySource,
                       CCVariable<double> &_volQ,
                       CCVariable<double> &_divQ,
                       constCCVariable<double> &_scatkt) :
#ifdef UINTAH_ENABLE_KOKKOS
                       abskt(_abskt.getKokkosView()),
                       intensitySource(_intensitySource.getKokkosView()),
                       volQ(_volQ.getKokkosView()),
                       divQ(_divQ.getKokkosView()),
                       scatkt(_scatkt.getKokkosView())
#else
                       abskt(_abskt),
                       intensitySource(_intensitySource),
                       volQ(_volQ),
                       divQ(_divQ),
                       scatkt(_scatkt)
#endif //UINTAH_ENABLE_KOKKOS
                       { }

       void operator()(int i , int j, int k ) const {
         divQ(i,j,k)+= (abskt(i,j,k)-scatkt(i,j,k))*volQ(i,j,k) - 4.0*M_PI*intensitySource(i,j,k);
       }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
       KokkosView3<const double> abskt;
       KokkosView3<double>  intensitySource;
       KokkosView3<double>  volQ;
       KokkosView3<double>  divQ;
       KokkosView3<const double> scatkt;
#else
       constCCVariable<double> &abskt;
       constCCVar_or_CCVar  &intensitySource;
       CCVariable<double>  &volQ;
       CCVariable<double>  &divQ;
       constCCVariable<double> &scatkt;
#endif //UINTAH_ENABLE_KOKKOS
};

//***************************************************************************
// Compute the heat flux divergence with scattering off.
//***************************************************************************
template <typename constCCVar_or_CCVar>
struct computeDivQ{
       computeDivQ(    constCCVariable<double> &_abskt,
                       constCCVar_or_CCVar     &_intensitySource,
                       CCVariable<double> &_volQ,
                       CCVariable<double> &_divQ) :
#ifdef UINTAH_ENABLE_KOKKOS
                       abskt(_abskt.getKokkosView()),
                       intensitySource(_intensitySource.getKokkosView()),
                       volQ(_volQ.getKokkosView()),
                       divQ(_divQ.getKokkosView())
#else
                       abskt(_abskt),
                       intensitySource(_intensitySource),
                       volQ(_volQ),
                       divQ(_divQ)
#endif //UINTAH_ENABLE_KOKKOS
                       { }

       void operator()(int i , int j, int k ) const {
         divQ(i,j,k)+= abskt(i,j,k)*volQ(i,j,k) - 4.0*M_PI*intensitySource(i,j,k);
       }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
       KokkosView3<const double> abskt;
       KokkosView3<double>  intensitySource;
       KokkosView3<double>  volQ;
       KokkosView3<double>  divQ;
#else
       constCCVariable<double> &abskt;
       constCCVar_or_CCVar &intensitySource;
       CCVariable<double>  &volQ;
       CCVariable<double>  &divQ;
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

  StaticArray< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.


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


  StaticArray< constCCVariable<double> > Intensities((_scatteringOn && !old_DW_isMissingIntensities) ? d_totalOrds : 0);

  StaticArray< CCVariable<double> > IntensitiesRestart((_scatteringOn && old_DW_isMissingIntensities) ? d_totalOrds : 0);

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

  StaticArray< constCCVariable<double> > abskp(_nQn_part);
  StaticArray< constCCVariable<double> > partTemp(_nQn_part);
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
  for (int bands =1; bands <=d_lambda; bands++){

    vars->volq.initialize(0.0);
    vars->ESRCG.initialize(0.0);
    computeIntensitySource(patch,abskp,partTemp,constvars->ABSKG,constvars->temperature,vars->ESRCG);

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
                      //plusX, plusY, plusZ, fraction, bands,
                      //radiationFlux_old[0] , radiationFlux_old[1],
                      //radiationFlux_old[2] , radiationFlux_old[3],
                      //radiationFlux_old[4] , radiationFlux_old[5],scatIntensitySource); //  this term needed for scattering

     // new (2-2017) construction of A-matrix and b-matrix
      computeAMatrix  doMakeMatrixA( omu[direcn], oeta[direcn], oxi[direcn],
                                     areaEW, areaNS, areaTB, volume, ffield,
                                     constvars->cellType,
                                     constvars->temperature,
                                     constvars->ABSKT,
                                     vars->ESRCG,
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



      compute4Flux<CCVariable<double> > doFlux(wt[direcn]*abs(omu[direcn])*d_xfluxAdjust,wt[direcn]*abs(oeta[direcn])*d_yfluxAdjust,wt[direcn]*abs(oxi[direcn])*d_zfluxAdjust,
                                                                    wt[direcn],  vars->cenint,
                                                                    plusX ? vars->qfluxe :  vars->qfluxw,
                                                                    plusY ? vars->qfluxn :  vars->qfluxs,
                                                                    plusZ ? vars->qfluxt :  vars->qfluxb,
                                                                    vars->volq);

      Uintah::parallel_for( range, doFlux );

    }  // ordinate loop

    if(_scatteringOn){
      computeDivQScat<CCVariable<double> > doDivQ(constvars->ABSKT, vars->ESRCG,vars->volq, divQ, scatkt);
      Uintah::parallel_for( range, doDivQ );
      //fort_rdomsrcscattering( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ, scatkt,scatIntensitySource);
    }else{
      computeDivQ<CCVariable<double> > doDivQ(constvars->ABSKT, vars->ESRCG,vars->volq, divQ);
      Uintah::parallel_for( range, doDivQ );
      //fort_rdomsrc( idxLo, idxHi, constvars->ABSKT, vars->ESRCG,vars->volq, divQ);
    }

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
                            const VarLabel* radIntSource,
                            const VarLabel*  fluxE,
                            const VarLabel*  fluxW,
                            const VarLabel*  fluxN,
                            const VarLabel*  fluxS,
                            const VarLabel*  fluxT,
                            const VarLabel*  fluxB,
                            const VarLabel*  volQ,
                            const VarLabel*  divQ){

    _radIntSource=radIntSource;
    _abskg_label=abskg_label;
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
DORadiationModel::computeScatteringIntensities(int direction, constCCVariable<double> &scatkt, StaticArray < TYPE > &Intensities, CCVariable<double> &scatIntensitySource,constCCVariable<double> &asymmetryFactor , const Patch* patch){


  direction -=1;   // change from fortran vector to c++ vector
  Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  scatIntensitySource.initialize(0.0); //reinitialize to zero for sum

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
DORadiationModel::computeIntensitySource( const Patch* patch, StaticArray <constCCVariable<double> >&abskp,
    StaticArray <constCCVariable<double> > &pTemp,
                  constCCVariable<double>  &abskg,
                  constCCVariable<double>  &gTemp,
                  CCVariable<double> &b_sourceArray){


    //Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
  for (int qn=0; qn < _nQn_part; qn++){
    if( _radiateAtGasTemp ){
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        b_sourceArray[*iter]+=(_sigma/M_PI)*abskp[qn][*iter]*std::pow(gTemp[*iter],4.0);
      //Uintah::parallel_for( range,[&](int i, int j, int k){ 
              //double T2 =gTemp(i,j,k)*gTemp(i,j,k);
              //b_sourceArray(i,j,k)+=(_sigma/M_PI)*abskp[qn](i,j,k)*T2*T2;
      //});
}
    }else{
      for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
        b_sourceArray[*iter]+=((_sigma/M_PI)*abskp[qn][*iter])*std::pow(pTemp[qn][*iter],4.0);
      //Uintah::parallel_for( range,[&](int i, int j, int k){ 
              //double T2 =pTemp[qn](i,j,k)*pTemp[qn](i,j,k);
              //b_sourceArray(i,j,k)+=((_sigma/M_PI)*abskp[qn](i,j,k))*T2*T2;
//});
    }
 }
  }

  for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      b_sourceArray[*iter]+=(_sigma/M_PI)*abskg[*iter]*std::pow(gTemp[*iter],4.0);
      //Uintah::parallel_for( range,[&](int i, int j, int k){ 
      //double T2 =gTemp(i,j,k)*gTemp(i,j,k);
      //b_sourceArray(i,j,k)+=(_sigma/M_PI)*abskg(i,j,k)*T2*T2;
  //});
}

  return;
}

//-----------------------------------------------------------------//
// This function computes the intensities. The fields that are required are
// cellType, radiation temperature, radiation source, and  abskt.
// This function is probably the bottle-neck in the radiation solve (sweeping method).
//-----------------------------------------------------------------//
void
DORadiationModel::intensitysolveSweep( const Patch* patch,
                                       int matlIndex,
                                       DataWarehouse* new_dw, 
                                       DataWarehouse* old_dw,
                                       int cdirecn, int phase){ 

  int direcn = cdirecn+1;
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  CCVariable <double > intensity;
  new_dw->allocateAndPut(intensity,_patchIntensityLabels[phase][cdirecn] , matlIndex, patch);   // change to computes when making it its own task
  //intensity.initialize(0.0);
  //new_dw->getModifiable(intensity,_patchIntensityLabels[phase][cdirecn] , matlIndex, patch);   // change to computes when making it its own task
  constCCVariable <double > ghost_intensity;
  new_dw->get( ghost_intensity, _patchIntensityLabels[phase-1][cdirecn], matlIndex, patch, _gv[(int) _plusX[cdirecn]][(int) _plusY[cdirecn]][(int) _plusZ[cdirecn]],1 );  


  constCCVariable <double > emissSrc;
  if(_scatteringOn){
    new_dw->get( emissSrc, _emiss_plus_scat_source_label[cdirecn], matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells

  }else{
    new_dw->get( emissSrc, _radIntSource, matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
  }

  constCCVariable<double> radTemp;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<double> abskt ; // should be WHICH DW !!!!!! BUG - only if rad solve is computed on RK step 2
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );


  setIntensityBC(patch, matlIndex,intensity, radTemp,cellType);






  Vector Dx = patch->dCell(); 
  const double vol = Dx.x()* Dx.y()* Dx.z();  // const for optimization
  const double  areaew = Dx.y()*Dx.z();
  const double  areans = Dx.x()*Dx.z();
  const double  areatb = Dx.x()*Dx.y(); // move into constructor or problem setup?
  const double  denom = abs(omu[direcn])*areaew+abs(oeta[direcn])*areans+abs(oxi[direcn])*areatb; // denomintor for Intensity in current cell



  IntVector c(0,0,0); // current cell

  ///  --------------------------------------------------//
  // ------------perform sweep on one patch -----------//
  //
  IntVector czm(0,0,0); // current cell, minus 1 in z
  IntVector cym(0,0,0); // current cell, minus 1 in y
  IntVector cxm(0,0,0); // current cell, minus 1 in x
  // ------------perform sweep on one patch -----------//
  //--------------------------------------------------------//
  // Step 1:
  //  Set seed cell (three ghost cells and no normal cells)
  //--------------------------------------------------------//
  c[0]= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  czm[0]=c[0];
  cym[0]=c[0];
  cxm[0]=c[0]-xiter[cdirecn];
  c[1]= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  czm[1]=c[1];
  cym[1]=c[1]-yiter[cdirecn];
  cxm[1]=c[1];
  c[2]= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  czm[2]=c[2]-ziter[cdirecn];
  cym[2]=c[2];
  cxm[2]=c[2];
  if (cellType[c] !=ffield){ // if intrusions
    intensity[c] = emissSrc[c] ;
  } else{ // else flow cell
    intensity[c] = ( emissSrc[c] +ghost_intensity[czm]*abs(oxi[direcn])*areatb  +  ghost_intensity[cym]*abs(oeta[direcn])*areans  +  ghost_intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
  } // end if
  //--------------------------------------------------------//
  // Step 2:
  //  Set seed rows (two ghost cells and one normal cells)
  //--------------------------------------------------------//
  ////--------------------set z----------------------//
  c[1]= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  czm[1]=c[1];
  cym[1]=c[1]-yiter[cdirecn];
  cxm[1]=c[1];
  c[2]= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  czm[2]=c[2]-ziter[cdirecn];
  cym[2]=c[2];
  cxm[2]=c[2];
  for (int i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
    c[0]=i;
    czm[0]=i;
    cym[0]=i;
    cxm[0]=i-xiter[cdirecn];
    if (cellType[c] !=ffield){ // if intrusions
      intensity[c] = emissSrc[c] ;
    } else{ // else flow cell
      intensity[c] = ( emissSrc[c] +ghost_intensity[czm]*abs(oxi[direcn])*areatb  +  ghost_intensity[cym]*abs(oeta[direcn])*areans  +  intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
    } // end if
  }
  ////--------------------set y----------------------//
  c[0]= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  czm[0]=c[0];
  cym[0]=c[0];
  cxm[0]=c[0]-xiter[cdirecn];
  c[2]= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  czm[2]=c[2]-ziter[cdirecn];
  cym[2]=c[2];
  cxm[2]=c[2];
  for (int j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
    c[1]=j;
    czm[1]=j;
    cym[1]=j-yiter[cdirecn];
    cxm[1]=j;
    if (cellType[c] !=ffield){ // if intrusions
      intensity[c] = emissSrc[c] ;
    } else{ // else flow cell
      intensity[c] = ( emissSrc[c] + ghost_intensity[czm]*abs(oxi[direcn])*areatb  +  intensity[cym]*abs(oeta[direcn])*areans  +  ghost_intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
    } // end if
  }
  ////--------------------set x----------------------//
  c[0]= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  czm[0]=c[0];
  cym[0]=c[0];
  cxm[0]=c[0]-xiter[cdirecn];
  c[1]= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  czm[1]=c[1];
  cym[1]=c[1]-yiter[cdirecn];
  cxm[1]=c[1];
  for (int k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){  
    c[2]=k;
    czm[2]=k-ziter[cdirecn];
    cym[2]=k;
    cxm[2]=k;
    if (cellType[c] !=ffield){ // if intrusions
      intensity[c] = emissSrc[c] ;
    } else{ // else flow cell
      intensity[c] = ( emissSrc[c] + intensity[czm]*abs(oxi[direcn])*areatb  +  ghost_intensity[cym]*abs(oeta[direcn])*areans  +  ghost_intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
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
  c[2]= _plusZ[cdirecn] ? idxLo.z() : idxHi.z();
  czm[2]=c[2]-ziter[cdirecn];
  cym[2]=c[2];
  cxm[2]=c[2];
  for (int j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
    c[1]=j;
    czm[1]=j;
    cym[1]=j-yiter[cdirecn];
    cxm[1]=j;
    for (int i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
      c[0]=i;
      czm[0]=i;
      cym[0]=i;
      cxm[0]=i-xiter[cdirecn];
      if (cellType[c] !=ffield){ // if intrusions
        intensity[c] = emissSrc[c] ;
      } else{ // else flow cell
        intensity[c] = ( emissSrc[c] + ghost_intensity[czm]*abs(oxi[direcn])*areatb  +  intensity[cym]*abs(oeta[direcn])*areans  +  intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
      } // end if
    }
  }
  ////--------------------set y----------------------//
  c[1]= _plusY[cdirecn] ? idxLo.y() : idxHi.y();
  czm[1]=c[1];
  cym[1]=c[1]-yiter[cdirecn];
  cxm[1]=c[1];
  for (int k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){  
    c[2]=k;
    czm[2]=k-ziter[cdirecn];
    cym[2]=k;
    cxm[2]=k;
    for (int i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
      c[0]=i;
      czm[0]=i;
      cym[0]=i;
      cxm[0]=i-xiter[cdirecn];
      if (cellType[c] !=ffield){ // if intrusions
        intensity[c] = emissSrc[c] ;
      } else{ // else flow cell
        intensity[c] = ( emissSrc[c] + intensity[czm]*abs(oxi[direcn])*areatb  +  ghost_intensity[cym]*abs(oeta[direcn])*areans  +  intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
      } // end if
    }
  }
  ////--------------------set x----------------------//
  c[0]= _plusX[cdirecn] ? idxLo.x() : idxHi.x();
  czm[0]=c[0];
  cym[0]=c[0];
  cxm[0]=c[0]-xiter[cdirecn];
  for (int k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){  
    c[2]=k;
    czm[2]=k-ziter[cdirecn];
    cym[2]=k;
    cxm[2]=k;
    for (int j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
      c[1]=j;
      czm[1]=j;
      cym[1]=j-yiter[cdirecn];
      cxm[1]=j;
      if (cellType[c] !=ffield){ // if intrusions
        intensity[c] = emissSrc[c] ;
      } else{ // else flow cell
        intensity[c] = ( emissSrc[c] + intensity[czm]*abs(oxi[direcn])*areatb  +  intensity[cym]*abs(oeta[direcn])*areans  +  ghost_intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
      } // end if
    }
  }
  ///  --------------------------------------------------//
  //--------------------------------------------------------//
  // Step 4:
  //  Set interior cells (no ghost cells and three normal cells)
  //--------------------------------------------------------//

  for (int k = (_plusZ[cdirecn] ? idxLo.z()+ziter[cdirecn] : idxHi.z()+ziter[cdirecn]);  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())) ; k=k+ziter[cdirecn]){  
    c[2]=k;
    czm[2]=k-ziter[cdirecn];
    cym[2]=k;
    cxm[2]=k;
    for (int j = (_plusY[cdirecn] ? idxLo.y()+yiter[cdirecn] : idxHi.y()+yiter[cdirecn]);  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
      c[1]=j;
      czm[1]=j;
      cym[1]=j-yiter[cdirecn];
      cxm[1]=j;
      for (int i = (_plusX[cdirecn] ? idxLo.x()+xiter[cdirecn] : idxHi.x()+xiter[cdirecn]);  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
        c[0]=i;
        czm[0]=i;
        cym[0]=i;
        cxm[0]=i-xiter[cdirecn];
        if (cellType[c] !=ffield){ // if intrusions
          intensity[c] = emissSrc[c] ;
        } else{ // else flow cell
          //intensity[c] = ((emissSrc[c] + scatSource[c])*vol + intensity[czm]*abs(oxi[direcn])*areatb  +  
          intensity[c] = (emissSrc[c] + intensity[czm]*abs(oxi[direcn])*areatb  +  
              intensity[cym]*abs(oeta[direcn])*areans  +  intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
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
// This function is probably the bottle-neck in the radiation solve (sweeping method).
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

  // -------------------NEEDS TO BE ADDED, REFLCTIONS ON WALLS -----------------//
  //IntVector domLo = patch->getExtraCellLowIndex();
  //IntVector domHi = patch->getExtraCellHighIndex();
  //StaticArray< CCVariable<double> > radiationFlux_old(_radiationFluxLabels.size()); // must always 6, even when reflections are off.

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
    new_dw->get( emissSrc, _radIntSource, matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
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


void
DORadiationModel::intensitysolveSweepOrigProto( const Patch* patch,
                                                int matlIndex,
                                                DataWarehouse* new_dw, 
                                                DataWarehouse* old_dw,
                                                int cdirecn){ 


  int direcn = cdirecn+1; // fortran direcn is 1 plus c++ iterator (cdirecn)
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  idxLo=IntVector(idxLo[0]-1,idxLo[1]-1,idxLo[2]-1);
  idxHi=IntVector(idxHi[0]+1,idxHi[1]+1,idxHi[2]+1);
  cout << idxLo << "  " << idxHi << " \n";

  constCCVariable<double> radTemp;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );

  CCVariable <double > intensity;
  new_dw->allocateAndPut(intensity,_IntensityLabels[cdirecn] , matlIndex, patch);   // change to computes when making it its own task
  intensity.initialize(0.0);
  constCCVariable <double > emissSrc;
  if(_scatteringOn){
    new_dw->get( emissSrc, _emiss_plus_scat_source_label[cdirecn], matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
  }else{
    new_dw->get( emissSrc, _radIntSource, matlIndex, patch, Ghost::None,0 );  // optimization bug - make this be computed differently for intrusion cells
  }

  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<double> abskt;  
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );


  Vector Dx = patch->dCell(); 
  const double vol = Dx.x()* Dx.y()* Dx.z();//   const for speed optimizations from compiler
  const double  areaew = Dx.y()*Dx.z();
  const double  areans = Dx.x()*Dx.z();
  const double  areatb = Dx.x()*Dx.y();//  move into constructor or problem setup?
  const double  denom = abs(omu[direcn])*areaew+abs(oeta[direcn])*areans+abs(oxi[direcn])*areatb; // denomintor for Intensity in current cell

  IntVector czm(0,0,0); // current cell, minus 1 in z
  IntVector cym(0,0,0);//  current cell, minus 1 in y
  IntVector cxm(0,0,0);//  current cell, minus 1 in x
  IntVector c(0,0,0); // current cell

  for (int k = (_plusZ[cdirecn] ? idxLo.z() : idxHi.z());  (_plusZ[cdirecn] ? (k<=idxHi.z()) : (k>=idxLo.z())); k=k+ziter[cdirecn]){  
    c[2]=k;
    czm[2]=k-ziter[cdirecn];
    cym[2]=k;
    cxm[2]=k;
    for (int j = (_plusY[cdirecn] ? idxLo.y() : idxHi.y());  (_plusY[cdirecn] ? (j<=idxHi.y()) : (j>=idxLo.y())) ; j=j+yiter[cdirecn]){
      c[1]=j;
      czm[1]=j;
      cym[1]=j-yiter[cdirecn];
      cxm[1]=j;
      for (int i = (_plusX[cdirecn] ? idxLo.x() : idxHi.x());  (_plusX[cdirecn] ? (i<=idxHi.x()) : (i>=idxLo.x())) ; i=i+xiter[cdirecn]){
        c[0]=i;
        czm[0]=i;
        cym[0]=i;
        cxm[0]=i-xiter[cdirecn];
        if (cellType[c] !=ffield){  //if intrusions
          intensity[c] = (_sigma/M_PI)*std::pow(radTemp[c],4.0) ;
        } else{  //else flow cell
          intensity[c] = (emissSrc[c] + intensity[czm]*abs(oxi[direcn])*areatb  +  intensity[cym]*abs(oeta[direcn])*areans  +  intensity[cxm]*abs(omu[direcn])*areaew)/(denom + abskt[c]*vol );
        }  //end if
      } // end i loop
    } // end j loop   
  } // end k loop
  // --------------------------------------------------//

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
  CCVariable <double > emissSrc;
  new_dw->allocateAndPut( emissSrc, _radIntSource, matlIndex, patch );  // optimization bug - make this be computed differently for intrusion cells
  emissSrc.initialize(0.0);  // a sum will be performed on this variable, intialize it to zero.

      constCCVariable<double> abskt;
      old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );
      constCCVariable<double> abskg;
      old_dw->get(abskg,_abskg_label, matlIndex , patch,Ghost::None, 0  );
      constCCVariable<double> radTemp ; 
      new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );

  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel,matlIndex,patch,Ghost::None,0);

  StaticArray< constCCVariable<double> > abskp(_nQn_part);
  StaticArray< constCCVariable<double> > partTemp(_nQn_part);
  for (int ix=0;  ix< _nQn_part; ix++){
    old_dw->get(abskp[ix],_abskp_label_vector[ix], matlIndex , patch,Ghost::None, 0  ); 
    old_dw->get(partTemp[ix],_temperature_label_vector[ix], matlIndex , patch,Ghost::None, 0  );
  }


  computeIntensitySource(patch,abskp,partTemp,abskg,radTemp,emissSrc);


   Vector Dx = patch->dCell();
   double volume = Dx.x()* Dx.y()* Dx.z();
  if(_scatteringOn){
    constCCVariable<double> scatkt;   //total scattering coefficient
    constCCVariable<double> asymmetryParam;  
    StaticArray< constCCVariable<double> > IntensitiesOld( d_totalOrds);

    old_dw->get(asymmetryParam,_asymmetryLabel, matlIndex , patch,Ghost::None, 0);
    old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);

    for( int ix=0;  ix<d_totalOrds ;ix++){
      old_dw->get(IntensitiesOld[ix],_IntensityLabels[ix], matlIndex , patch,Ghost::None, 0  );
    }
    for( int ix=0;  ix<d_totalOrds ;ix++){
      CCVariable<double> scatIntensitySource;  
      new_dw->allocateAndPut(scatIntensitySource,_emiss_plus_scat_source_label[ix], matlIndex, patch);

      computeScatteringIntensities(ix+1,scatkt, IntensitiesOld,scatIntensitySource,asymmetryParam, patch); // function expects fortran indices
    Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    Uintah::parallel_for( range,[&](int i, int j, int k){ 
        if(cellType(i,j,k)==ffield){
          scatIntensitySource(i,j,k)+=emissSrc(i,j,k);
          scatIntensitySource(i,j,k)*=volume;
        }else{
          scatIntensitySource(i,j,k)=(_sigma/M_PI)*std::pow(radTemp(i,j,k),4.0)*abskt(i,j,k);
        }
        });
              
    }
  }else{
    Uintah::BlockRange range(patch->getExtraCellLowIndex(),patch->getExtraCellHighIndex());
    Uintah::parallel_for( range,[&](int i, int j, int k) { 
        if(cellType(i,j,k)==ffield){
          emissSrc(i,j,k)*=volume;
        }else{
          emissSrc(i,j,k)=(_sigma/M_PI)*std::pow(radTemp(i,j,k),4.0)*abskt(i,j,k);
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

  constCCVariable <double > emissSrc;
  constCCVariable<double> abskt ; 
  new_dw->get(emissSrc, _radIntSource, matlIndex, patch, Ghost::None,0 ); 
  old_dw->get(abskt,_abskt_label, matlIndex , patch,Ghost::None, 0  );      // should be WHICH DW !!!!!! BUG

  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();


  for (int direcn = 1; direcn <=d_totalOrds; direcn++){

    constCCVariable <double > intensity;
    new_dw->get(intensity,_IntensityLabels[direcn-1] , matlIndex, patch, Ghost::None,0 );   // this should be a requires,  type restriction

    compute4Flux<constCCVariable<double> > doFlux(wt[direcn]*abs(omu[direcn])*d_xfluxAdjust,wt[direcn]*abs(oeta[direcn])*d_yfluxAdjust,wt[direcn]*abs(oxi[direcn])*d_zfluxAdjust,
                                                  wt[direcn],  intensity,
                                                  _plusX[direcn-1]==1 ? fluxE :  fluxW,
                                                  _plusY[direcn-1]==1 ? fluxN :  fluxS,
                                                  _plusZ[direcn-1]==1 ? fluxT :  fluxB,
                                                  volQ);
    Uintah::parallel_for( range, doFlux );
  }

  if(_scatteringOn){
    constCCVariable<double> scatkt;   //total scattering coefficient
    old_dw->get(scatkt,_scatktLabel, matlIndex , patch,Ghost::None, 0);
    computeDivQScat<constCCVariable<double> > doDivQ(abskt, emissSrc,volQ, divQ, scatkt);
    Uintah::parallel_for( range, doDivQ );
  }else{
    computeDivQ<constCCVariable<double> > doDivQ(abskt, emissSrc,volQ, divQ);
    Uintah::parallel_for( range, doDivQ );
  }



  proc0cout << "//---------------------------------------------------------------------//\n";
  proc0cout << "Total Radiation Solve Time (Approximate): " << _timer().seconds() << " seconds for " << d_totalOrds<< " sweeps  \n";
  proc0cout << "//---------------------------------------------------------------------//\n";
  return ;

}



void
DORadiationModel::setIntensityBC(const Patch* patch,
                                 int matlIndex,  
                                 CCVariable<double>& intensity,
                                 constCCVariable<double>& radTemp,
                                 constCCVariable<int>& cellType){


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
          intensity[c] =  _sigma/M_PI*pow(radTemp[c],4.0) ; // No reflections here!  Needs to be developed for reflections!
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



  CCVariable <double > intensity;
  new_dw->getModifiable(intensity,_IntensityLabels[ix] , matlIndex, patch);   // change to computes when making it its own task

  constCCVariable<double> radTemp;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  ); 
  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );

 setIntensityBC(patch, matlIndex,intensity, radTemp,cellType);
  return;
}



void
DORadiationModel::setIntensityBC2(const Patch* patch,
                                 int matlIndex,  
                                       DataWarehouse* new_dw, 
                                       DataWarehouse* old_dw, int ix, int phase){



  CCVariable <double > intensity;
  new_dw->getModifiable(intensity,_patchIntensityLabels[phase][ix] , matlIndex, patch);   // change to computes when making it its own task

  constCCVariable<double> radTemp;
  new_dw->get(radTemp,_T_label, matlIndex , patch,Ghost::None, 0  );
  constCCVariable<int> cellType;
  old_dw->get(cellType,_cellTypeLabel, matlIndex , patch,Ghost::None, 0  );

 setIntensityBC(patch, matlIndex,intensity, radTemp,cellType);
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

