#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <iomanip>

using namespace std;
using namespace Uintah;

DORadiation::DORadiation( std::string src_name, ArchesLabel* labels, MPMArchesLabel* MAlab,
                          vector<std::string> req_label_names, const ProcessorGroup* my_world,
                          std::string type )
: SourceTermBase( src_name, labels->d_sharedState, req_label_names, type ),
  _labels( labels ),
  _MAlab(MAlab),
  _my_world(my_world){

  // NOTE: This boundary condition here is bogus.  Passing it for
  // now until the boundary condition reference can be stripped out of
  // the radiation model.

  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();

  _src_label = VarLabel::create( src_name, CC_double );

  // Add any other local variables here.
  _radiationFluxELabel = VarLabel::create("radiationFluxE",  CC_double);
  _extra_local_labels.push_back(_radiationFluxELabel);

  _radiationFluxWLabel = VarLabel::create("radiationFluxW",  CC_double);
  _extra_local_labels.push_back(_radiationFluxWLabel);

  _radiationFluxNLabel = VarLabel::create("radiationFluxN",  CC_double);
  _extra_local_labels.push_back(_radiationFluxNLabel);

  _radiationFluxSLabel = VarLabel::create("radiationFluxS",  CC_double);
  _extra_local_labels.push_back(_radiationFluxSLabel);

  _radiationFluxTLabel = VarLabel::create("radiationFluxT",  CC_double);
  _extra_local_labels.push_back(_radiationFluxTLabel);

  _radiationFluxBLabel = VarLabel::create("radiationFluxB",  CC_double);
  _extra_local_labels.push_back(_radiationFluxBLabel);

  _radiationVolqLabel = VarLabel::create("radiationVolq",  CC_double);
  _extra_local_labels.push_back(_radiationVolqLabel);

  //Declare the source type:
  _source_grid_type = CC_SRC; // or FX_SRC, or FY_SRC, or FZ_SRC, or CCVECTOR_SRC

  _DO_model = 0;



}

DORadiation::~DORadiation()
{

  // source label is destroyed in the base class

  for (vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
       iter != _extra_local_labels.end(); iter++) {

    VarLabel::destroy( *iter );

  }

  if (_sweepMethod){
      
    for (int iband=0; iband<d_nbands; iband++){
      VarLabel::destroy(_radIntSource[iband]);
    }
  }
  delete _DO_model;

}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
DORadiation::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->getWithDefault( "calc_frequency",   _radiation_calc_freq, 3 );
  db->getWithDefault( "checkForMissingIntensities", _checkForMissingIntensities  , false );
  db->getWithDefault( "calc_on_all_RKsteps", _all_rk, false );
  if (db->findBlock("abskt")!= nullptr ){
    db->findBlock("abskt")->getAttribute("label",_abskt_label_name);
  }else{
    throw InvalidValue("Error: Couldn't find label for total absorption coefficient in src DORadiation.    \n", __FILE__, __LINE__);
  }
  _T_label_name = "radiation_temperature";


    std::string solverType;
    db->findBlock("DORadiationModel")->getAttribute("type", solverType );
   if(solverType=="sweepSpatiallyParallel"){
     _sweepMethod=enum_sweepSpatiallyParallel;
   }else{
     _sweepMethod=enum_linearSolve;
   }

  _DO_model = scinew DORadiationModel( _labels, _MAlab, _my_world, _sweepMethod);
  _DO_model->problemSetup( db );

  d_nbands=_DO_model->spectralBands();

  if (!((d_nbands>1 && _sweepMethod==enum_sweepSpatiallyParallel)  || d_nbands==1)){
    throw ProblemSetupException("DORadiation: Spectral Radiation only supported when using sweepsSpatiallyParallel.",__FILE__, __LINE__);
  }



  std::vector<std::string> abskg_vec =  _DO_model->gasAbsorptionNames();
  std::vector<std::string> abswg_vec =  _DO_model->gasWeightsNames();
      
  if(abskg_vec.size()<1){
    _abskg_label_name=_abskt_label_name; // No abskg model found, use custom label
  }else{
    _abskg_label_name=abskg_vec[0];
    for (unsigned int i=0; i<abskg_vec.size(); i++){
      if ( _abskt_label_name == abskg_vec[i]){
        throw ProblemSetupException("DORadiation: abskg and abskt cannot use the same label.  abskt = abskg + emissivity.",__FILE__, __LINE__);
      }
    }
  }

  proc0cout << " --- DO Radiation Model Summary: --- " << endl;
  proc0cout << "   -> calculation frequency:      " << _radiation_calc_freq << endl;
  proc0cout << "   -> temperature label:          " << _T_label_name << endl;
  for (unsigned int i=0; i<abskg_vec.size(); i++){
  proc0cout << "   -> abskg label(s):             " << abskg_vec[i] << endl;}
  for (unsigned int i=0; i<abswg_vec.size(); i++){
  proc0cout << "   -> absorption weights label(s):" << abswg_vec[i] << endl;}
  proc0cout << "   -> abskt label:                " << _abskt_label_name << endl;
  proc0cout << " --- end DO Radiation Summary ------ " << endl;




  for (int iband=0; iband<d_nbands; iband++){
    for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
      ostringstream my_stringstream_object;
      //my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix ;
      my_stringstream_object << "Intensity" << setfill('0') << setw(4)<<  ix << "_"<< setw(2)<< iband;
      _IntensityLabels.push_back(  VarLabel::find(my_stringstream_object.str()));
      _extra_local_labels.push_back(_IntensityLabels[ix+iband*_DO_model->getIntOrdinates()]);
      if(_DO_model->needIntensitiesBool()==false){
        break;  // create labels for all intensities, otherwise only create 1 label
      }
    }
  }

  if (_sweepMethod){
    _radIntSource= std::vector<const VarLabel*> (d_nbands);
    for (int iband=0; iband<d_nbands; iband++){
      ostringstream my_stringstream_object;
      my_stringstream_object << "radIntSource" << "_"<<setfill('0')<< setw(2)<< iband ;
      _radIntSource[iband] = VarLabel::create(my_stringstream_object.str(), CCVariable<double>::getTypeDescription());
    }

    _patchIntVector =  IntVector(0,0,0);
    _xPatch_boundary=std::vector<double>(0);//3 1-D vectors since domains aren't square ( can 2D vectors be non-square?)
    _yPatch_boundary=std::vector<double>(0);
    _zPatch_boundary=std::vector<double>(0);
    ProblemSpecP db_level = db->getRootNode()->findBlock("Grid")->findBlock("Level");
    std::vector<Vector> boxDimensionsHi(0);
    std::vector<Vector> boxDimensionsLo(0);
    std::vector<Box> boxDimensions(0);
    double pTol=1e-10;// absolute_tolerance_for reduncant patch boundaries, avoided relative because of zeros
    _multiBox=false;
    for ( ProblemSpecP db_box = db_level->findBlock("Box"); db_box != nullptr; db_box = db_box->findNextBlock("Box")){
      if (db_box->findNextBlock("Box") != nullptr){
        _multiBox=true;
      }

      IntVector tempPatchIntVector(0,0,0);
      db_box->require( "patches", tempPatchIntVector );
      Vector boxLo(0,0,0);
      Vector boxHi(0,0,0);
      db_box->require( "lower", boxLo );
      db_box->require( "upper", boxHi );
      if(_multiBox){
        boxDimensions.push_back(Box((Point) boxLo,(Point) boxHi));
        boxDimensionsHi.push_back(boxLo);
        boxDimensionsHi.push_back(boxHi);
      }

      for (int j=0; j<3; j++){
        for (int i=0; i<tempPatchIntVector[j]+1; i++){
          double patch_node=(boxHi[j]-boxLo[j])*((double) i / (double) tempPatchIntVector[j])+boxLo[j];
          if (j==0){
            bool abscissa_exists=false;
            for (unsigned int k=0; k <_xPatch_boundary.size(); k++){
              if (patch_node < (_xPatch_boundary[k]+pTol) && patch_node > (_xPatch_boundary[k]-pTol)){
                abscissa_exists=true;
                break;
              }
            }
            if (abscissa_exists==false){
              _xPatch_boundary.push_back(patch_node);
            }
          } else if (j==1){
            bool abscissa_exists=false;
            for (unsigned int k=0; k <_yPatch_boundary.size(); k++){
              if (patch_node < (_yPatch_boundary[k]+pTol) && patch_node > (_yPatch_boundary[k]-pTol)){
                abscissa_exists=true;
                break;
              }
            }
            if (abscissa_exists==false){
              _yPatch_boundary.push_back(patch_node);
            }
          } else if (j==2){
            bool abscissa_exists=false;
            for (unsigned int k=0; k <_zPatch_boundary.size(); k++){
              if (patch_node < (_zPatch_boundary[k]+pTol) && patch_node > (_zPatch_boundary[k]-pTol)){
                abscissa_exists=true;
                break;
              }
            }
            if (abscissa_exists==false){
              _zPatch_boundary.push_back(patch_node);
            }
            //xPatchBoundary.push_back( );
          } // end logic checking for redundant abscissa
        } // end iterating over elements in a single direction
      } // end iterating over x y z, patch dimensions

    } // end iterating over boxes

    std::sort(_xPatch_boundary.begin(),_xPatch_boundary.end());
    std::sort(_yPatch_boundary.begin(),_yPatch_boundary.end());
    std::sort(_zPatch_boundary.begin(),_zPatch_boundary.end());

    _patchIntVector[0]=_xPatch_boundary.size()-1;
    _patchIntVector[1]=_yPatch_boundary.size()-1;
    _patchIntVector[2]=_zPatch_boundary.size()-1;

  _doesPatchExist =  std::vector< std::vector < std::vector < bool > > >(_patchIntVector[0] ,std::vector < std::vector < bool > > (_patchIntVector[1],std::vector < bool > (_patchIntVector[2] , _multiBox ? false : true )));

    _nphase=_patchIntVector[0]+_patchIntVector[1]+_patchIntVector[2]-2;
    _nDir=_DO_model->getIntOrdinates();
    _nstage=_nphase+_nDir/8-1;
    _directional_phase_adjustment= std::vector <std::vector< std::vector<int> > >(2,std::vector< std::vector< int > > (2,std::vector< int > (2,0)));

   // for optimization of multi-box problems (not needed)
    if(_multiBox){
      bool firstTimeThrough=true;// Need to construct full patch domain using Booleans, because infrastructure crashes if you look for patches that aren't there.
      for (int idir=0; idir<2; idir++){
        for (int jdir=0; jdir<2; jdir++){
          for (int kdir=0; kdir<2; kdir++){
            // basic concept  i + j + k  must alwasy equal iphase, and be within the bounds of the super-imposed box
            bool foundIntersection=false;
            for (int iphase=0; iphase <_nphase; iphase++ ){
              if (foundIntersection==false){
                _directional_phase_adjustment[idir][jdir][kdir]=iphase;
              }
              for (int j=std::max(0,std::min(iphase-_patchIntVector[0]-_patchIntVector[2],_patchIntVector[1]-1));  j< std::min(iphase+1,_patchIntVector[1]);  j++ ){
                for (int k=std::max(0,std::min(iphase-j-_patchIntVector[0],_patchIntVector[2]-1));  k< std::min(iphase-j+1,_patchIntVector[2]);  k++ ){
                  if (iphase-j-k < _patchIntVector[0] ){


                    // adjust for non- x+ y+ z+ directions
                    int iAdj =iphase-j-k;
                    int jAdj =j;
                    int kAdj =k;
                    if(idir==1){
                      iAdj=_patchIntVector[0]-iAdj;
                    }
                    if(jdir==1){
                      jAdj=_patchIntVector[1]-jAdj;
                    }
                    if(kdir==1){
                      kAdj=_patchIntVector[2]-kAdj;
                    }


                    for (unsigned int ibox=0; ibox < boxDimensions.size(); ibox++){
                      if( boxDimensions[ibox].contains(Point((_xPatch_boundary[iAdj]+_xPatch_boundary[iAdj + (idir==0 ? 1 : -1)])/2.0,(_yPatch_boundary[jAdj]+_yPatch_boundary[jAdj + (jdir==0 ? 1 : -1)])/2.0,(_zPatch_boundary[kAdj] + _zPatch_boundary[kAdj + (kdir==0 ? 1 : -1)])/2.0)) ){

                        _doesPatchExist[iAdj+(idir==0 ? 0 : -1)][jAdj+(jdir==0 ? 0 : -1)][kAdj+(kdir==0 ? 0 : -1)]=true ;

                        foundIntersection=true;
                        if (firstTimeThrough==false){ // no need to populate boolean matrix
                          break;
                        }
                      } //check for intersection
                    } // loop over all uintah boxes
                  } // check on i (is it in domain?)
                } // k
                if(foundIntersection==true && firstTimeThrough==false){ break;}
              } // j
              if(foundIntersection==true && firstTimeThrough==false){ break;}
            } // iphase
           firstTimeThrough=false;
          } // z+ z- dir
        } // y+ y- dir
      } // x+ x- dir
    } // end multi-box check

    if (_DO_model->ScatteringOnBool()){
      for (int iband=0; iband<d_nbands; iband++){
        for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
          ostringstream my_stringstream_object;
          my_stringstream_object << "scatSrc_absSrc" << setfill('0') << setw(4)<<  ix <<"_"<<iband ;
          _emiss_plus_scat_source_label.push_back(  VarLabel::find(my_stringstream_object.str()));
        }
      }
    }

   _RelevantPatchesXpYpZp=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYpZm=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYmZp=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYmZm=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYpZp=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYpZm=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYmZp=std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYmZm=std::vector<const PatchSubset*> (0);

  }

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
DORadiation::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{


  _T_label = VarLabel::find(_T_label_name);
  if ( _T_label == 0){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the radiation temperature label.", __FILE__, __LINE__);
  }

  _abskg_label = VarLabel::find(_abskg_label_name);
  if ( _abskg_label == 0){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the abskg label.", __FILE__, __LINE__);
  }
  _abskt_label = VarLabel::find(_abskt_label_name);
  if ( _abskt_label == 0){
    throw InvalidValue("Error: For DO Radiation source term -- Could not find the abskt label:"+_abskt_label_name, __FILE__, __LINE__);
  }

  if (_DO_model->ScatteringOnBool()){
    _scatktLabel =  VarLabel::find("scatkt");
    _asymmetryLabel= VarLabel::find("asymmetryParam");
    if ( _scatktLabel == 0 ){
      throw ProblemSetupException("Error: scatkt label not found! This label should be created in the Radiation property calculator!",__FILE__, __LINE__);
    }
    if (_asymmetryLabel == 0 ){
      throw ProblemSetupException("Error: asymmetry label not found! This label should be created in the Radiation property calculator!",__FILE__, __LINE__);
    }
  }

  if (_DO_model->get_nQn_part() >0 && _abskt_label_name == _abskg_label_name){
    throw ProblemSetupException("DORadiation: You must use a gas phase aborptoin coefficient if particles are included, as well as a seperate label for abskt.",__FILE__, __LINE__);
  }
  if (_sweepMethod>0){
    sched_computeSourceSweep( level, sched, timeSubStep );
  }else{
  std::string taskname = "DORadiation::computeSource";
  Task* tsk = scinew Task(taskname, this, &DORadiation::computeSource, timeSubStep);

  _perproc_patches = level->eachPatch();

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;


  if (timeSubStep == 0) {

    tsk->computes(_src_label);
    tsk->requires( Task::NewDW, _T_label, gac, 1 );
    tsk->requires( Task::OldDW, _abskt_label, gac, 1 );
    tsk->requires( Task::OldDW, _abskg_label, gn, 0 );


    tsk->requires( Task::OldDW, _src_label, gn, 0 ); // should be removed, for temporal scheduling

    _DO_model->setLabels();


    for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
      tsk->requires( Task::OldDW,_DO_model->getAbskpLabels()[i], gn, 0 );
      tsk->requires( Task::OldDW,_DO_model->getPartTempLabels()[i], gn, 0 );
    }

    if (_DO_model->ScatteringOnBool()){
      tsk->requires( Task::OldDW, _scatktLabel, gn, 0 );
      tsk->requires( Task::OldDW,_asymmetryLabel, gn, 0 );
    }

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
        iter != _extra_local_labels.end(); iter++){

      tsk->computes( *iter );
      //if (*(*iter)=="radiationVolq"){
        //continue;
      //}
      tsk->requires( Task::OldDW, *iter, gn, 0 );

    }

  } else {

    tsk->modifies(_src_label);
    tsk->requires( Task::NewDW, _T_label, gac, 1 );
    tsk->requires( Task::OldDW, _abskt_label, gac, 1 );
    tsk->requires( Task::OldDW, _abskg_label, gn, 0 );

    for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
      tsk->requires( Task::NewDW,_DO_model->getAbskpLabels()[i], gn, 0 );
      tsk->requires( Task::NewDW,_DO_model->getPartTempLabels()[i], gn, 0 );
    }

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
         iter != _extra_local_labels.end(); iter++){

      tsk->modifies( *iter );

    }
  }

  tsk->requires(Task::OldDW, _labels->d_cellTypeLabel, gac, 1 );
  tsk->requires(Task::NewDW, _labels->d_cellInfoLabel, gn);

  int Rad_TG=1;
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials(),Rad_TG);


////---------------------carry forward task-------------------------------//

  if (timeSubStep == 0) {
    std::string taskNoCom = "DORadiation::TransferRadFieldsFromOldDW";
    Task* tsk_noRad = scinew Task(taskNoCom, this, &DORadiation::TransferRadFieldsFromOldDW);

    tsk_noRad->requires(Task::OldDW, _src_label,gn,0);
    tsk_noRad->computes( _src_label);

    // fluxes and intensities and radvolq
    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
        iter != _extra_local_labels.end(); iter++){
      tsk_noRad->requires( Task::OldDW, *iter, gn, 0 );
      tsk_noRad->computes( *iter );
    }

    int no_Rad_TG=0;
    sched->addTask(tsk_noRad, level->eachPatch(), _shared_state->allArchesMaterials(),no_Rad_TG);
  }

////----------------------------------------------------------------------//

}
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
DORadiation::computeSource( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   int timeSubStep )
{
  _DO_model->d_linearSolver->matrixCreate( _perproc_patches, patches );

    //int timestep = _labels->d_sharedState->getCurrentTopLevelTimeStep(); 
    bool do_radiation = false;
    if ( _all_rk ) {
      do_radiation = true;
    } else if ( timeSubStep == 0 && !_all_rk ) {
      do_radiation = true;
    }

    bool old_DW_isMissingIntensities=0;
    if(_checkForMissingIntensities){  // should only be true for first time step of a restart
        if(_DO_model->needIntensitiesBool()==false ){
          for(unsigned int ix=0; ix< _IntensityLabels.size() ;ix++){
            for (int p=0; p < patches->size(); p++){
              const Patch* patch = patches->get(p);
              int archIndex = 0;
              int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();  // have to do it this way because overloaded exists(arg) doesn't seem to work!!
              if (!old_dw->exists(_IntensityLabels[ix],matlIndex,patch)){
                proc0cout << "WARNING:  Intensity" << *_IntensityLabels[ix] << "from previous solve are missing!   Using zeros. \n";
                CCVariable< double> temp;
                new_dw->allocateAndPut(temp  , _IntensityLabels[ix]  , matlIndex , patch );
                temp.initialize(0.0);
                old_DW_isMissingIntensities=true;
              }
              else{
                new_dw->transferFrom(old_dw,_IntensityLabels[ix],  patches, matls);
                break;
              }
            }
          }
        }
        else{
          for(unsigned int ix=0; ix< _IntensityLabels.size() ;ix++){
            for (int p=0; p < patches->size(); p++){
              const Patch* patch = patches->get(p);
              int archIndex = 0;
              int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();  // have to do it this way because overloaded exists(arg) doesn't seem to work!!
              if (!old_dw->exists(_IntensityLabels[ix],matlIndex,patch)){
                proc0cout << "WARNING:  Intensity" << *_IntensityLabels[ix] << "from previous solve are missing!   Using zeros. \n";
                old_DW_isMissingIntensities=true;
                break;
              }
            }
          }
        }
    }else{
      if(_DO_model->needIntensitiesBool()==false ){  // we do this for a feature that is never used  (solve using previous direction)
        if ( timeSubStep == 0 ) {
          for(unsigned int ix=0;  ix< _IntensityLabels.size();ix++){
            new_dw->transferFrom(old_dw,_IntensityLabels[ix],  patches, matls);
          }
        }
      }
    }


  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, _labels->d_cellInfoLabel, matlIndex, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    CCVariable<double> divQ;


    ArchesVariables radiation_vars;
    ArchesConstVariables const_radiation_vars;

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;

    if ( timeSubStep == 0 ) {

      new_dw->get( const_radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );

      new_dw->allocateAndPut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch );
      new_dw->allocateAndPut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch );
      new_dw->allocateAndPut( divQ, _src_label, matlIndex, patch );

      // copy old solution into new
      old_dw->copyOut( divQ, _src_label, matlIndex, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxe , _radiationFluxELabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxw , _radiationFluxWLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxn , _radiationFluxNLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxs , _radiationFluxSLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxt , _radiationFluxTLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.qfluxb , _radiationFluxBLabel                , matlIndex , patch , gn , 0 );
      old_dw->copyOut( radiation_vars.volq   , _radiationVolqLabel                 , matlIndex , patch , gn , 0 );
      old_dw->get( const_radiation_vars.ABSKG  , _abskg_label , matlIndex , patch , gn , 0 );
      old_dw->get( const_radiation_vars.ABSKT  , _abskt_label , matlIndex , patch , gac , 1 );

    } else {

      new_dw->get( const_radiation_vars.temperature, _T_label, matlIndex , patch , gac , 1 );

      new_dw->getModifiable( radiation_vars.qfluxe , _radiationFluxELabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxw , _radiationFluxWLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxn , _radiationFluxNLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxs , _radiationFluxSLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxt , _radiationFluxTLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.qfluxb , _radiationFluxBLabel , matlIndex , patch );
      new_dw->getModifiable( radiation_vars.volq   , _radiationVolqLabel  , matlIndex , patch );
      new_dw->getModifiable( divQ, _src_label, matlIndex, patch );

      old_dw->get( const_radiation_vars.ABSKG  , _abskg_label, matlIndex , patch, gn, 0 ); // wrong DW
      old_dw->get( const_radiation_vars.ABSKT  , _abskt_label , matlIndex , patch , gac , 1 );
    }

    old_dw->get( const_radiation_vars.cellType , _labels->d_cellTypeLabel, matlIndex, patch, gac, 1 );

    if ( do_radiation ){

      if ( timeSubStep == 0 ) {

        if(_DO_model->needIntensitiesBool()){
          for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
            CCVariable<double> cenint;
            new_dw->allocateAndPut(cenint,_IntensityLabels[ix] , matlIndex, patch );
          }
        }

        //Note: The final divQ is initialized (to zero) and set after the solve in the intensity solve itself.
        _DO_model->intensitysolve( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars, divQ, BoundaryCondition::WALL, matlIndex, new_dw, old_dw, old_DW_isMissingIntensities );

      }
    }
  } // end patch loop
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
DORadiation::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DORadiation::initialize";
  Task* tsk = scinew Task(taskname, this, &DORadiation::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
       iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void
DORadiation::initialize( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex();

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch );

    src.initialize(0.0);

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin();
         iter != _extra_local_labels.end(); iter++){

      CCVariable<double> temp_var;
      new_dw->allocateAndPut(temp_var, *iter, matlIndex, patch );
      temp_var.initialize(0.0);
    }
  }
}


void
DORadiation::init_all_intensities( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{

  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    _DO_model->getDOSource(patch, matlIndex, new_dw, old_dw);
  }
}


void
DORadiation::doSweepAdvanced( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw ,
                        const int ixx_orig       , int intensity_iter        )
{

  int archIndex = 0;
  int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
  for (int p=0; p < patches->size(); p++){
  int ixx=ixx_orig;
    const Patch* patch = patches->get(p);

      int xPatch=0;
      int yPatch=0;
      int zPatch=0;
      if(_multiBox){
        Vector patchLo= patch->getBox().lower().toVector();
        Vector PatchHi= patch->getBox().upper().toVector();
        Vector patchMid=(patchLo+PatchHi)/2.0;

        xPatch=getSweepPatchIndex(patchMid[0],_xPatch_boundary);
        yPatch=getSweepPatchIndex(patchMid[1],_yPatch_boundary);
        zPatch=getSweepPatchIndex(patchMid[2],_zPatch_boundary);



      }else{
        int patchNumber= patch->getID() ;  // patches are assigned z y x 0 -> npatches-1
        xPatch =    floor(patchNumber/(_patchIntVector[1]*_patchIntVector[2]));
        yPatch =    floor((patchNumber-xPatch*_patchIntVector[1]*_patchIntVector[2])/_patchIntVector[2]); // possible type (int) problems here?
        zPatch =    patchNumber%(_patchIntVector[2]);

      }


    int xPatchAdjusted =xPatch;  // account for varying direction of intensities
    int yPatchAdjusted =yPatch;
    int zPatchAdjusted =zPatch;
    if(_DO_model->xDir(intensity_iter)==0){
      xPatchAdjusted=_patchIntVector[0]-xPatch-1;
    }
    if(_DO_model->yDir(intensity_iter)==0){
      yPatchAdjusted=_patchIntVector[1]-yPatch-1;
    }
    if(_DO_model->zDir(intensity_iter)==0){
      zPatchAdjusted=_patchIntVector[2]-zPatch-1;
    }

  ixx+=  _directional_phase_adjustment[1-_DO_model->xDir(intensity_iter)][1-_DO_model->yDir(intensity_iter)][1-_DO_model->zDir(intensity_iter)];// L-shaped domain adjustment



    if ((intensity_iter%(_nDir/8))== (ixx-(xPatchAdjusted+yPatchAdjusted+zPatchAdjusted))){
        _DO_model->intensitysolveSweepOptimized(patch,matlIndex, new_dw,old_dw, intensity_iter );
    }
  }
}

void
DORadiation::computeFluxDivQ( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    _DO_model->computeFluxDiv( patch, matlIndex, new_dw, old_dw);
  }
}

void
DORadiation::sched_computeSourceSweep( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
 if (timeSubStep==0){


  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;
    std::vector< std::vector < std::vector < Ghost::GhostType > > >  gv (2,  std::vector < std::vector < Ghost::GhostType > > (2, std::vector < Ghost::GhostType > (2) ));

   gv[0][0][0] =gac ;
   gv[0][0][1] =gac ;
   gv[0][1][0] =gac ;
   gv[0][1][1] =gac ;
   gv[1][0][0] =gac ;
   gv[1][0][1] =gac ;
   gv[1][1][0] =gac ;
   gv[1][1][1] =gac ;

      _DO_model->setLabels(_abskg_label, _abskt_label, _T_label,_labels->d_cellTypeLabel ,_radIntSource,
                           _radiationFluxELabel,_radiationFluxWLabel,_radiationFluxNLabel,
                           _radiationFluxSLabel,_radiationFluxTLabel,_radiationFluxBLabel,
                           _radiationVolqLabel, _src_label);






////-----------for timesteps w/o radiation----------//
  int Radiation_TG=1;
  int  no_Rad_TG=0;
  std::string taskNoCom = "DORadiation::TransferRadFieldsFromOldDW";
  Task* tsk_noRadiation = scinew Task(taskNoCom, this, &DORadiation::TransferRadFieldsFromOldDW);

    tsk_noRadiation->requires(Task::OldDW,_radiationFluxELabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationFluxWLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationFluxNLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationFluxSLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationFluxTLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationFluxBLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW,_radiationVolqLabel,gn,0);
    tsk_noRadiation->requires(Task::OldDW, _src_label,gn,0);


    tsk_noRadiation->computes(_radiationFluxELabel);
    tsk_noRadiation->computes(_radiationFluxWLabel);
    tsk_noRadiation->computes(_radiationFluxNLabel);
    tsk_noRadiation->computes(_radiationFluxSLabel);
    tsk_noRadiation->computes(_radiationFluxTLabel);
    tsk_noRadiation->computes(_radiationFluxBLabel);
    tsk_noRadiation->computes(_radiationVolqLabel);
    tsk_noRadiation->computes( _src_label);

    if (_DO_model->ScatteringOnBool()){
      for (int iband=0; iband<d_nbands; iband++){
        for (int j=0; j< _nDir; j++){
          tsk_noRadiation->requires( Task::OldDW,_IntensityLabels[j+iband*_DO_model->getIntOrdinates()], gn, 0 );
          tsk_noRadiation->computes( _IntensityLabels[j+iband*_DO_model->getIntOrdinates()]);
        }
      }
    }

    sched->addTask(tsk_noRadiation, level->eachPatch(), _shared_state->allArchesMaterials(),no_Rad_TG);


  std::vector<const VarLabel* > spectral_gas_absorption = _DO_model->getAbskgLabels();
  std::vector<const VarLabel* > spectral_gas_weight = _DO_model->getAbswgLabels();

//-----------------------------------------------------------------------------------------------------//
// advanced sweeping algorithm uses spatial parallisms to improve efficiency
//  Total number of tasks = N_sweepingPhases+N_ordinates-1
  if(_sweepMethod==enum_sweepSpatiallyParallel){  // try to avoid redundant sweeps

    //---------create patch subsets for each spatial task -----------------------//
    const Uintah::PatchSet* const allPatches = sched->getLoadBalancer()->getPerProcessorPatchSet(level);
    std::vector<const Patch*> localPatches = allPatches->getSubset(Uintah::Parallel::getMPIRank())->getVector();

    for (int idir=0; idir<2; idir++){
      for (int jdir=0; jdir<2; jdir++){
        for (int kdir=0; kdir<2; kdir++){
          // basic concept  i + j + k  must alwasy equal iphase, and be within the bounds of the super-imposed box
          for (int iphase=_directional_phase_adjustment[idir][jdir][kdir]; iphase <_nphase-_directional_phase_adjustment[1-idir][1-jdir][1-kdir]; iphase++ ){
            std::vector<const  Patch* >  RelevantPatches(0);
            for (int j=std::max(0,std::min(iphase-_patchIntVector[0]-_patchIntVector[2],_patchIntVector[1]-1));  j< std::min(iphase+1,_patchIntVector[1]);  j++ ){
              for (int k=std::max(0,std::min(iphase-j-_patchIntVector[0],_patchIntVector[2]-1));  k< std::min(iphase-j+1,_patchIntVector[2]);  k++ ){
                if (iphase-j-k < _patchIntVector[0] ){


                  // adjust for non- x+ y+ z+ directions
                  int iAdj =iphase-j-k;
                  int jAdj =j;
                  int kAdj =k;
                  if(idir==1){
                    iAdj=_patchIntVector[0]-iAdj;
                  }
                  if(jdir==1){
                    jAdj=_patchIntVector[1]-jAdj;
                  }
                  if(kdir==1){
                    kAdj=_patchIntVector[2]-kAdj;
                  }


                  //proc0cout <<_doesPatchExist[iAdj+(idir==0 ? 0 : -1)][jAdj+(jdir==0 ? 0 : -1)][kAdj+(kdir==0 ? 0 : -1)] << " " <<  IntVector( iAdj+(idir==0 ? 0 : -1),jAdj+(jdir==0 ? 0 : -1),kAdj+(kdir==0 ? 0 : -1))  << " \n";
                  if( _doesPatchExist[iAdj+(idir==0 ? 0 : -1)][jAdj+(jdir==0 ? 0 : -1)][kAdj+(kdir==0 ? 0 : -1)]==false) { // needed for multi-box problems
                    continue ;
                  }
                  Point patchCenter((_xPatch_boundary[iAdj]+_xPatch_boundary[iAdj + (idir==0 ? 1 : -1)])/2.0,(_yPatch_boundary[jAdj]+_yPatch_boundary[jAdj + (jdir==0 ? 1 : -1)])/2.0,(_zPatch_boundary[kAdj] + _zPatch_boundary[kAdj + (kdir==0 ? 1 : -1)])/2.0);
                  RelevantPatches.push_back(  level.get_rep()->getPatchFromPoint( patchCenter, false ));

                } // check on i (is it in domain?)
              } // k
            } // j
            if(RelevantPatches.size()==0){
              continue; // No patches in this patch set, this could be reformulated, to create emtpy patchsets, because it makes downstream logic less friendly
            }

            PatchSubset* sweepingPatches= scinew PatchSubset(RelevantPatches);
            sweepingPatches->sort();
            if (idir==0 && jdir==0 && kdir==0)
              _RelevantPatchesXpYpZp.push_back(sweepingPatches);
            if (idir==0 && jdir==0 && kdir==1)
              _RelevantPatchesXpYpZm.push_back(sweepingPatches);
            if (idir==0 && jdir==1 && kdir==0)
              _RelevantPatchesXpYmZp.push_back(sweepingPatches);
            if (idir==0 && jdir==1 && kdir==1)
              _RelevantPatchesXpYmZm.push_back(sweepingPatches);
            if (idir==1 && jdir==0 && kdir==0)
              _RelevantPatchesXmYpZp.push_back(sweepingPatches);
            if (idir==1 && jdir==0 && kdir==1)
              _RelevantPatchesXmYpZm.push_back(sweepingPatches);
            if (idir==1 && jdir==1 && kdir==0)
              _RelevantPatchesXmYmZp.push_back(sweepingPatches);
            if (idir==1 && jdir==1 && kdir==1)
              _RelevantPatchesXmYmZm.push_back(sweepingPatches);
          } // iphase
        } // z+ z- dir
      } // y+ y- dir
    } // x+ x- dir
    //----------------------------------------------------------------------//


    //--------------------------------------------------------------------//
    //      Scedule initialization task.  Initializes all intensities
    //--------------------------------------------------------------------//
    std::string taskname1 = "DORadiation::init_all_intensities";
    Task* tsk1 = scinew Task(taskname1, this, &DORadiation::init_all_intensities);
    tsk1->requires( Task::OldDW,_abskg_label, gn, 0 );
    tsk1->requires( Task::OldDW,_abskt_label, gn, 0 );
    tsk1->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
    tsk1->requires( Task::NewDW,_T_label, gn, 0 );
    for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
      tsk1->requires( Task::OldDW,_DO_model->getAbskpLabels()[i], gn, 0 );
      tsk1->requires( Task::OldDW,_DO_model->getPartTempLabels()[i], gn, 0 );
    }

    if (_DO_model->ScatteringOnBool()){
      for (int j=0; j< _nDir; j++){
        for (int iband=0; iband<d_nbands; iband++){
          tsk1->requires( Task::OldDW,_IntensityLabels[j+iband*_DO_model->getIntOrdinates()], gn, 0 );
          tsk1->computes( _emiss_plus_scat_source_label[j+iband*_DO_model->getIntOrdinates()]);
        }
        tsk1->requires( Task::OldDW, _scatktLabel, gn, 0 );
        tsk1->requires( Task::OldDW,_asymmetryLabel, gn, 0 );
      }
    }
    for (int iband=0; iband<d_nbands; iband++){
      tsk1->computes( _radIntSource[iband]); // 
    }

    for (int iband=0; iband<d_nbands; iband++){
      tsk1->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
      if(d_nbands>1){
        tsk1->requires( Task::OldDW,spectral_gas_weight[iband], gn, 0 );
      }
    }

    for (int iband=0; iband<d_nbands; iband++){
      tsk1->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
    }

    //--------------------------------------------------------------------//
    //    Scedule set BCs task.  Sets the intensity fields in the walls.  //
    //--------------------------------------------------------------------//
    sched->addTask(tsk1, level->eachPatch(), _shared_state->allArchesMaterials(),Radiation_TG);

    for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
      std::stringstream tasknamec;
      tasknamec << "DORadiation::sweeping_initialize_" <<ix;
      Task* tskc = scinew Task(tasknamec.str(), this, &DORadiation::setIntensityBC,ix);
      tskc->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
      tskc->requires( Task::NewDW,_T_label, gn, 0 );
      for (int iband=0; iband<d_nbands; iband++){
        tskc->computes( _IntensityLabels[ix+iband*_DO_model->getIntOrdinates()]);
      }

      sched->addTask(tskc, level->eachPatch(), _shared_state->allArchesMaterials(),Radiation_TG);
    }


    //-- create material subset from arches materials (needed to satifiy interface for spatial scheduling) --//
    const MaterialSet* matlset = _labels->d_sharedState->allArchesMaterials();
    const MaterialSubset* matlDS=nullptr;
    for ( auto i_matSubSet = (matlset->getVector()).begin();
        i_matSubSet != (matlset->getVector()).end(); i_matSubSet++ ){
      matlDS =  *i_matSubSet;
      break;  
    }

    //--------------------------------------------------------------------//
    // These Tasks computes the intensities, on a per patch basis. The pseudo
    // spatial scheduling acheieved could use features from the infrastructure
    // to be improved. The spatial tasks were developed by looking at single direction,
    // then re-used for the other directions by re-mapping the processor IDs.
    // These tasks are the bottle-neck in the radiation solve (sweeping method).
    //--------------------------------------------------------------------//
    int nOctants=8;
    for( int istage=0;  istage< _nstage;istage++){ // loop over stages
      for (int idir=0; idir< nOctants; idir++){  // loop over octants
        int first_intensity=idir*_nDir/nOctants;
        int pAdjm = _directional_phase_adjustment[1-_DO_model->xDir(first_intensity)][1-_DO_model->yDir(first_intensity)][1-_DO_model->zDir(first_intensity)];// L-shaped domain adjustment, assumes that ordiantes are stored in octants (8 bins), with similiar directional properties in each bin
        int pAdjp = _directional_phase_adjustment[_DO_model->xDir(first_intensity)][_DO_model->yDir(first_intensity)][_DO_model->zDir(first_intensity)];// L-shaped domain adjustment, assumes that ordiantes are stored in octants (8 bins), with similiar directional properties in each bin
        for( int int_x=std::max(0,istage-_nphase+pAdjm+1);  int_x<std::min(_nDir/nOctants,istage+1); int_x++){ // loop over per-octant-intensities (intensity_within_octant_x)
          if(istage-int_x>_nphase-pAdjp-pAdjm-1){ // Terminte sweep early for multi-box problems
            continue;
          }
          // combine stages into single task?
          int intensity_iter=int_x+idir*_nDir/nOctants;
          std::stringstream taskname2;
          //                    base name
          taskname2 << "DORadiation::doSweepAdvanced_" <<istage<< "_"<<intensity_iter;
          Task* tsk2 = scinew Task(taskname2.str(), this, &DORadiation::doSweepAdvanced, istage,intensity_iter);
          tsk2->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
          tsk2->requires( Task::OldDW,_abskt_label, gn, 0 );

          if (_DO_model->ScatteringOnBool()){
            for (int iband=0; iband<d_nbands; iband++){
              tsk2->requires( Task::NewDW, _emiss_plus_scat_source_label[intensity_iter+iband*_DO_model->getIntOrdinates()],gn,0);
            }
          }else{
            for (int iband=0; iband<d_nbands; iband++){
              tsk2->requires( Task::NewDW, _radIntSource[iband], gn, 0 );
            }
          }
          if (d_nbands > 1){
            for (int iband=0; iband<d_nbands; iband++){
              tsk2->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
            }
          }
          for (int iband=0; iband<d_nbands; iband++){
            tsk2->modifies( _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()]);
            // --- Turn on and off communication depending on phase and intensity using equation:  iStage = iPhase + intensity_within_octant_x, 8 different patch subsets, due to 8 octants ---//
            if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==1)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXpYpZp[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==0)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXpYpZm[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==1)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXpYmZp[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==0)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXpYmZm[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==1)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXmYpZp[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==0)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXmYpZm[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==1)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXmYmZp[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
            if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==0)
              tsk2->requires( Task::NewDW, _IntensityLabels[intensity_iter+iband*_DO_model->getIntOrdinates()] ,_RelevantPatchesXmYmZm[istage-int_x], Uintah::Task::PatchDomainSpec::ThisLevel, matlDS, Uintah::Task::MaterialDomainSpec::NormalDomain, gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
          }

          sched->addTask(tsk2, level->eachPatch(), _shared_state->allArchesMaterials(),Radiation_TG);
        }
      }
    }

    //-----------------------------------------------------------------//
    // This task computes 6 radiative fluxes, incident radiation,
    //  and divQ. This function requries the required intensity fields
    // sn*(2+sn), radiation temperature, abskt, and scatkt (scattering
    // only)
    //-----------------------------------------------------------------//
    std::string taskname3 = "DORadiation::computeFluxDivQ";
    Task* tsk3 = scinew Task(taskname3, this, &DORadiation::computeFluxDivQ);
    for (int iband=0; iband<d_nbands; iband++){
      for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
        tsk3->requires( Task::NewDW, _IntensityLabels[ix+iband*_DO_model->getIntOrdinates()], gn, 0 );
      }
    }


    for (int iband=0; iband<d_nbands; iband++){
      tsk3->requires( Task::NewDW,_radIntSource[iband], gn, 0 );
    }
    if(d_nbands>1){
      for (int iband=0; iband<d_nbands; iband++){
        tsk3->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
      }
    }


    tsk3->requires( Task::NewDW,_T_label, gn, 0 );
    tsk3->requires( Task::OldDW,_abskt_label, gn, 0 );

    tsk3->computes(_radiationFluxELabel);
    tsk3->computes(_radiationFluxWLabel);
    tsk3->computes(_radiationFluxNLabel);
    tsk3->computes(_radiationFluxSLabel);
    tsk3->computes(_radiationFluxTLabel);
    tsk3->computes(_radiationFluxBLabel);
    tsk3->computes(_radiationVolqLabel);
    tsk3->computes( _src_label);

    sched->addTask(tsk3, level->eachPatch(), _shared_state->allArchesMaterials(),Radiation_TG);
  }
 }
}

void
DORadiation::setIntensityBC( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         int ix )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _labels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    for (int iband=0; iband<d_nbands; iband++){
      CCVariable <double > intensity;
      new_dw->allocateAndPut(intensity,_IntensityLabels[ix+iband*_DO_model->getIntOrdinates()] , matlIndex, patch,Ghost::AroundCells,1); // not supported long term (avoids copy)
      //new_dw->allocateAndPut(intensity,_IntensityLabels[ix] , matlIndex, patch); // portable and slower, Required to use sweepOptimizedOLD for full portability
      intensity.initialize(0.0);
    }
  _DO_model->setIntensityBC2Orig( patch, matlIndex, new_dw, old_dw, ix);

  }
}


// Table search, nothing fancy linear search
int
DORadiation::getSweepPatchIndex( double patchMid, std::vector<double>& indep_var  ){
  int j = -1;
  for (unsigned int i=0; i < indep_var.size(); i++ ) {
    if ( patchMid > indep_var[i] ) {
      j++;
    }else{
      break;
    }
  }
  return j;
}




void
DORadiation::TransferRadFieldsFromOldDW( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw)
{



  if (_DO_model->ScatteringOnBool() || (!_sweepMethod) ){
    for (int iband=0; iband<d_nbands; iband++){
      for( int ix=0;  ix< _DO_model->getIntOrdinates();ix++){
        new_dw->transferFrom(old_dw,_IntensityLabels[ix+iband*_DO_model->getIntOrdinates()],  patches, matls);
        if (_DO_model->needIntensitiesBool()==false){ // this is always true for scattering
          break; // need 1 intensity, for a feature that is never used  =(
        }
      }
    }
  }

     new_dw->transferFrom(old_dw,_radiationFluxELabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationFluxWLabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationFluxNLabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationFluxSLabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationFluxTLabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationFluxBLabel,patches,matls);
     new_dw->transferFrom(old_dw,_radiationVolqLabel,patches,matls);
     new_dw->transferFrom(old_dw, _src_label,patches,matls);

}
