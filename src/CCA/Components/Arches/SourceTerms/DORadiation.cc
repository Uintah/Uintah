#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/Radiation/DORadiationModel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesStatsEnum.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <iomanip>

using namespace std;
using namespace Uintah;

DORadiation::DORadiation( std::string src_name,
                          ArchesLabel* labels,
                          MPMArchesLabel* MAlab,
                          vector<std::string> req_label_names,
                          const ProcessorGroup* my_world,
                          std::string type )
: SourceTermBase( src_name, labels->d_materialManager, req_label_names, type ),
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

//______________________________________________________________________
//
DORadiation::~DORadiation()
{
  // source label is destroyed in the base class

  for (auto iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++) {
    VarLabel::destroy( *iter );

  }

  if (_sweepMethod){
    for (int iband=0; iband<d_nbands; iband++){
      VarLabel::destroy(_radIntSource[iband]);
    }
  }

  if(_dynamicSolveFrequency){
    VarLabel::destroy(_dynamicSolveCountPatchLabel);
    VarLabel::destroy(_lastRadSolvePatchLabel);
    VarLabel::destroy(VarLabel::find("min_time") );
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

  db->getWithDefault( "calc_frequency", _radiation_calc_freq, 3 );

  // Check to see if the dynamic frequency radiation solve should be used.
  if(1 < _radiation_calc_freq && db->findBlock("use_dynamic_frequency") != nullptr) {

    db->getWithDefault( "use_dynamic_frequency", _nsteps_calc_freq, 25 );
    _dynamicSolveFrequency = true;

    _dynamicSolveCountPatchLabel =
      VarLabel::create( dynamicSolveCountPatch_name,
                        SoleVariable< double >::getTypeDescription() );
    _lastRadSolvePatchLabel =
      VarLabel::create( "last_radiation_solve_timestep_index",
                        SoleVariable< int >::getTypeDescription() );
  }

  db->getWithDefault( "checkForMissingIntensities", _checkForMissingIntensities  , false );
  db->getWithDefault( "calc_on_all_RKsteps", _all_rk, false );

  if (db->findBlock("abskt")!= nullptr ){
    db->findBlock("abskt")->getAttribute("label",_abskt_label_name);
  }
  else{
    throw InvalidValue("Error: Couldn't find label for total absorption coefficient in src DORadiation.    \n", __FILE__, __LINE__);
  }

  _T_label_name = "radiation_temperature";


  std::string solverType;
  db->findBlock("DORadiationModel")->getAttribute("type", solverType );

  if(solverType=="sweepSpatiallyParallel"){
    _sweepMethod = enum_sweepSpatiallyParallel;
  }else{
    _sweepMethod = enum_linearSolve;
  }

  _DO_model = scinew DORadiationModel( _labels, _MAlab, _my_world, _sweepMethod);
  _DO_model->problemSetup( db );

  d_nbands=_DO_model->spectralBands();

  if (!((d_nbands>1 && _sweepMethod == enum_sweepSpatiallyParallel)  || d_nbands==1)){
    throw ProblemSetupException("DORadiation: Spectral Radiation only supported when using sweepsSpatiallyParallel.",__FILE__, __LINE__);
  }

  std::vector<std::string> abskg_vec = _DO_model->gasAbsorptionNames();
  std::vector<std::string> abswg_vec = _DO_model->gasWeightsNames();

  if(abskg_vec.size()<1){
    _abskg_label_name = _abskt_label_name; // No abskg model found, use custom label
  }
  else{
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
    for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
      ostringstream labelName;
      //labelName << "Intensity" << setfill('0') << setw(4)<<  ord ;
      labelName << "Intensity" << setfill('0') << setw(4)<<  ord << "_"<< setw(2)<< iband;
      _IntensityLabels.push_back(  VarLabel::find(labelName.str()));

      const int indx = ord + iband * _DO_model->getIntOrdinates();
      _extra_local_labels.push_back(_IntensityLabels[indx]);

      if(_DO_model->needIntensitiesBool()==false){
        break;  // create labels for all intensities, otherwise only create 1 label
      }
    }
  }

  if (_sweepMethod){
    // check to see if a sweep compatible scheduler is being used
    ProblemSpecP db_sched = db->getRootNode()->findBlock("Scheduler");
    if (db_sched != nullptr){
      std::string scheduler_name;
      db_sched->getAttribute ("type",scheduler_name);
      if (scheduler_name=="DynamicMPI"){
        throw ProblemSetupException("Error: DORadiation sweeps is not compatible with the DynamicMPI scheduler. ",__FILE__, __LINE__);
      }
    }

    _radIntSource= std::vector<const VarLabel*> (d_nbands);
    for (int iband=0; iband<d_nbands; iband++){
      ostringstream labelName;
      labelName << "radIntSource" << "_"<<setfill('0')<< setw(2)<< iband ;
      _radIntSource[iband] = VarLabel::create(labelName.str(), CCVariable<double>::getTypeDescription());
    }

    _patchIntVector =  IntVector(0,0,0);

    _xyzPatch_boundary=std::vector<std::vector<double> > (3,std::vector<double> (0) );// 3 vectors for x, y, and z.

    ProblemSpecP db_level = db->getRootNode()->findBlock("Grid")->findBlock("Level");

    std::vector<Box> boxDimensions(0);
    double pTol=1e-10;// (meters)  absolute_tolerance_for redundant patch boundaries, avoided relative because of zeros
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
      }

      for (int idir=0; idir<3; idir++){
        for (int i=0; i<tempPatchIntVector[idir]+1; i++){
          double patch_node=(boxHi[idir]-boxLo[idir])*((double) i / (double) tempPatchIntVector[idir])+boxLo[idir];

          bool abscissa_exists=false; // check if patch boundary is present in current matrix
          for (unsigned int k=0; k <_xyzPatch_boundary[idir].size(); k++){
            if (patch_node < (_xyzPatch_boundary[idir][k]+pTol) && patch_node > (_xyzPatch_boundary[idir][k]-pTol)){
              abscissa_exists=true;
              break;
            }
          }

          if (abscissa_exists==false){
            _xyzPatch_boundary[idir].push_back(patch_node);
          }
        } // end iterating over elements in a single direction
      } // end iterating over x y z, patch dimensions
    } // end iterating over boxes

    std::sort( _xyzPatch_boundary[0].begin(), _xyzPatch_boundary[0].end());
    std::sort( _xyzPatch_boundary[1].begin(), _xyzPatch_boundary[1].end());
    std::sort( _xyzPatch_boundary[2].begin(), _xyzPatch_boundary[2].end());

    if(_multiBox){  // check for staggered patch layouts and throw error if found.  We do this by checking to see if all patch boundarys line up with all patch-box boundaries.
      for ( ProblemSpecP db_box = db_level->findBlock("Box"); db_box != nullptr; db_box = db_box->findNextBlock("Box")){

        IntVector tempPatchIntVector(0,0,0);
        db_box->require( "patches", tempPatchIntVector );

        Vector boxLo(0,0,0);
        Vector boxHi(0,0,0);
        db_box->require( "lower", boxLo );
        db_box->require( "upper", boxHi );

        IntVector checkBoxLo, checkBoxHi;

        for (int idir=0; idir < 3; idir++){
          for (unsigned int k=0; k <_xyzPatch_boundary[idir].size(); k++){
            if (boxLo[idir] < (_xyzPatch_boundary[idir][k]+pTol) && boxLo[idir] > (_xyzPatch_boundary[idir][k]-pTol)){
              checkBoxLo[idir]=k;
            }
            if (boxHi[idir] < (_xyzPatch_boundary[idir][k]+pTol) && boxHi[idir] > (_xyzPatch_boundary[idir][k]-pTol)){
              checkBoxHi[idir]=k;
              break;
            }
          }
        }

        if( tempPatchIntVector[0] !=  (checkBoxHi[0]-checkBoxLo[0]) || tempPatchIntVector[1] !=  (checkBoxHi[1]-checkBoxLo[1]) ||  tempPatchIntVector[2] !=  (checkBoxHi[2]-checkBoxLo[2]) ) {
          throw InvalidValue("Error: The selected patch layout doesn't appear to be compatible with DORadiation Sweeps.", __FILE__, __LINE__);
        }
      }
    }

    _patchIntVector[0]=_xyzPatch_boundary[0].size()-1;
    _patchIntVector[1]=_xyzPatch_boundary[1].size()-1;
    _patchIntVector[2]=_xyzPatch_boundary[2].size()-1;

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
                      if( boxDimensions[ibox].contains( Point( (_xyzPatch_boundary[0][iAdj] + _xyzPatch_boundary[0][iAdj + (idir==0 ? 1 : -1)])/2.0,
                                                               (_xyzPatch_boundary[1][jAdj] + _xyzPatch_boundary[1][jAdj + (jdir==0 ? 1 : -1)])/2.0,
                                                               (_xyzPatch_boundary[2][kAdj] + _xyzPatch_boundary[2][kAdj + (kdir==0 ? 1 : -1)])/2.0)) ){

                        _doesPatchExist[iAdj+(idir==0 ? 0 : -1)][jAdj+(jdir==0 ? 0 : -1)][kAdj+(kdir==0 ? 0 : -1)]=true;

                        foundIntersection=true;         // Intersection found with a single box, ergo, the patch must exist
                        if (firstTimeThrough==false){   // no need to populate boolean matrix
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
        for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
          ostringstream labelName;
          labelName << "scatSrc_absSrc" << setfill('0') << setw(4)<<  ord <<"_"<<iband ;
          _emiss_plus_scat_source_label.push_back(  VarLabel::find(labelName.str()));
        }
      }
    }

    // Vector of PatchSubset pointers
   _RelevantPatchesXpYpZp = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYpZm = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYmZp = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXpYmZm = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYpZp = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYpZm = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYmZp = std::vector<const PatchSubset*> (0);
   _RelevantPatchesXmYmZm = std::vector<const PatchSubset*> (0);

    // Vector of PatchSet pointers
   _RelevantPatchesXpYpZp2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXpYpZm2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXpYmZp2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXpYmZm2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXmYpZp2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXmYpZm2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXmYmZp2 = std::vector<const PatchSet*> (0);
   _RelevantPatchesXmYmZm2 = std::vector<const PatchSet*> (0);
  }

  //Check to see if any labels are requested for saving:
  std::vector<std::string> intensity_names;
  for ( auto i = _IntensityLabels.begin(); i != _IntensityLabels.end(); i++ ){
    std::string labelName = (*i)->getName();
    intensity_names.push_back(labelName);
  }

  std::vector<bool> save_requested = ArchesCore::save_in_archiver(intensity_names, db);
  const int my_len = save_requested.size();
  for ( int i = 0; i < my_len; i++){
    if ( save_requested[i] ){
      m_user_intensity_save = true;
      m_user_intensity_save_labels.push_back(_IntensityLabels[i]);
    }
  }

}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
DORadiation::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  _T_label     = VarLabel::find( _T_label_name,     "Error: radiation temperature");
  _abskg_label = VarLabel::find( _abskg_label_name, "Error: gas absorption coefficient");
  _abskt_label = VarLabel::find( _abskt_label_name, "Error: total absorption coefficient");

  if (_DO_model->ScatteringOnBool()){
    _scatktLabel   = VarLabel::find("scatkt", "Error scattering coefficien");                 // Need a better error message
    _asymmetryLabel= VarLabel::find("asymmetryParam", "Error");
  }

  if (_DO_model->get_nQn_part() >0 && _abskt_label_name == _abskg_label_name){
    throw ProblemSetupException("DORadiation: You must use a gas phase aborptoin coefficient if particles are included, as well as a seperate label for abskt.",__FILE__, __LINE__);
  }

  int Rad_TG=1; // solve radiation in this taskgraph
  int no_Rad_TG=0; // don't solve radiation in this taskgraph
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn = Ghost::None;


  if (_sweepMethod>0){
    sched_computeSourceSweep( level, sched, timeSubStep );
  }
  else{
    //______________________________________________________________________
    //
    std::string taskname = "DORadiation::computeSource";
    Task* tsk = scinew Task(taskname, this, &DORadiation::computeSource, timeSubStep);

    _perproc_patches = level->eachPatch();

    if (timeSubStep == 0) {

      tsk->computes(_src_label);
      tsk->requires( Task::NewDW, _T_label,     gac, 1 );
      tsk->requires( Task::OldDW, _abskt_label, gac, 1 );
      tsk->requires( Task::OldDW, _abskg_label, gn, 0 );
      tsk->requires( Task::OldDW, _src_label,   gn, 0 ); // should be removed, for temporal scheduling

      _DO_model->setLabels();

      for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
        tsk->requires( Task::OldDW,_DO_model->getAbskpLabels()[i],    gn, 0 );
        tsk->requires( Task::OldDW,_DO_model->getPartTempLabels()[i], gn, 0 );
      }

      if (_DO_model->ScatteringOnBool()){
        tsk->requires( Task::OldDW, _scatktLabel,   gn, 0 );
        tsk->requires( Task::OldDW,_asymmetryLabel, gn, 0 );
      }

      for ( auto  iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        tsk->computes( *iter );
        //if (*(*iter)=="radiationVolq"){
        //continue;
        //}
        tsk->requires( Task::OldDW, *iter, gn, 0 );
      }
    } else {

      tsk->modifies(_src_label);
      tsk->requires( Task::NewDW, _T_label,     gac, 1 );
      tsk->requires( Task::OldDW, _abskt_label, gac, 1 );
      tsk->requires( Task::OldDW, _abskg_label, gn, 0 );

      for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
        tsk->requires( Task::NewDW,_DO_model->getAbskpLabels()[i],    gn, 0 );
        tsk->requires( Task::NewDW,_DO_model->getPartTempLabels()[i], gn, 0 );
      }

      for ( auto iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        tsk->modifies( *iter );
      }
    }

    tsk->requires( Task::OldDW, _labels->d_cellTypeLabel, gac, 1 );
    tsk->requires( Task::NewDW, _labels->d_cellInfoLabel, gn);

    sched->addTask(tsk, level->eachPatch(), m_matls ,Rad_TG);

    //---------------------carry forward task-------------------------------
    //
    if (timeSubStep == 0) {

      std::string taskNoCom = "DORadiation::TransferRadFieldsFromOldDW";
      Task* tsk_noRad = scinew Task(taskNoCom, this, &DORadiation::TransferRadFieldsFromOldDW);

      tsk_noRad->requires( Task::OldDW, _src_label,gn,0);
      tsk_noRad->computes( _src_label);

      // fluxes and intensities and radvolq
      for ( auto iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
        tsk_noRad->requires( Task::OldDW, *iter, gn, 0 );
        tsk_noRad->computes( *iter );
      }

      if (_dynamicSolveFrequency ) {
        tsk_noRad->requires( Task::OldDW, _dynamicSolveCountPatchLabel, gn, 0 );
        tsk_noRad->requires( Task::OldDW, _lastRadSolvePatchLabel,      gn, 0 );
        tsk_noRad->requires( Task::OldDW, _simulationTimeLabel, gn, 0 );
        tsk_noRad->requires( Task::OldDW, _labels->d_delTLabel,         gn, 0 );

        tsk_noRad->computes( _dynamicSolveCountPatchLabel );
        tsk_noRad->computes( _lastRadSolvePatchLabel );
      }

      sched->addTask(tsk_noRad, level->eachPatch(), m_matls ,no_Rad_TG);
    }
  }

  //---------------------profile dynamic radiation task-------------------
  //
  if (timeSubStep == 0 && _dynamicSolveFrequency) {
    VarLabel::create("min_time",  min_vartype::getTypeDescription());
    std::string taskname4 = "DORadiation::profileDynamicRadiation";
    Task* tsk4 = scinew Task(taskname4, this, &DORadiation::profileDynamicRadiation);

    tsk4->requires( Task::NewDW, VarLabel::find("radiationVolq"), gn, 0 );
    tsk4->requires( Task::NewDW, VarLabel::find("divQ"), gn, 0 );
    tsk4->requires( Task::NewDW, _T_label, gn, 0 );

    tsk4->computes(VarLabel::find("min_time"));
    sched->addTask(tsk4, level->eachPatch(), m_matls, Rad_TG);


    //______________________________________________________________________
    //
    std::string taskname5 = "DORadiation::checkReductionVars";
    Task* tsk5 = scinew Task(taskname5, this, &DORadiation::checkReductionVars);
    tsk5->requires( Task::NewDW, VarLabel::find("min_time"), gn,0 );
    tsk5->requires( Task::OldDW, _simulationTimeLabel,       gn,0 );
    tsk5->requires( Task::OldDW, _labels->d_timeStepLabel,    gn,0 );
    tsk5->requires( Task::OldDW, _labels->d_delTLabel,       gn,0 );

    tsk5->computes( _dynamicSolveCountPatchLabel );
    tsk5->computes( _lastRadSolvePatchLabel );

    timeStep_vartype timeStep(0);

    sched->addTask(tsk5, level->eachPatch(), m_matls, Rad_TG); // in both taskGraphs
  }
}

//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
DORadiation::computeSource( const ProcessorGroup  * pc,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matls,
                                  DataWarehouse   * old_dw,
                                  DataWarehouse   * new_dw,
                                  int timeSubStep )
{
  _DO_model->d_linearSolver->matrixCreate( _perproc_patches, patches );

  bool do_radiation = false;
  if ( _all_rk ) {
    do_radiation = true;
  } else if ( timeSubStep == 0 && !_all_rk ) {
    do_radiation = true;
  }

  bool old_DW_isMissingIntensities=0;


  if( _checkForMissingIntensities){  // should only be true for first time step of a restart
      if(_DO_model->needIntensitiesBool()==false ){

        for(unsigned int ix=0; ix< _IntensityLabels.size() ;ix++){
          for (int p=0; p < patches->size(); p++){

            const Patch* patch = patches->get(p);

            if ( !old_dw->exists(_IntensityLabels[ix], m_matIdx, patch)){
              proc0cout << "WARNING:  Intensity" << *_IntensityLabels[ix] << "from previous solve are missing!   Using zeros. \n";
              CCVariable< double> temp;
              new_dw->allocateAndPut(temp, _IntensityLabels[ix], m_matIdx, patch );
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

            if (!old_dw->exists(_IntensityLabels[ix], m_matIdx, patch)){
              proc0cout << "WARNING:  Intensity" << *_IntensityLabels[ix] << "from previous solve are missing!   Using zeros. \n";
              old_DW_isMissingIntensities=true;
              break;
            }
          }
        }
      }
  }else{
    if( _DO_model->needIntensitiesBool()==false ){  // we do this for a feature that is never used  (solve using previous direction)
      if ( timeSubStep == 0 ) {
        for(unsigned int ix=0;  ix< _IntensityLabels.size();ix++){
          new_dw->transferFrom(old_dw,_IntensityLabels[ix],  patches, matls);
        }
      }
    }
  }

  //__________________________________
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);

    PerPatch<CellInformationP> cellInfoP;
    new_dw->get(cellInfoP, _labels->d_cellInfoLabel, m_matIdx, patch);
    CellInformation* cellinfo = cellInfoP.get().get_rep();

    CCVariable<double> divQ;
    ArchesVariables radiation_vars;
    ArchesConstVariables const_radiation_vars;

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn = Ghost::None;

    if ( timeSubStep == 0 ) {

      new_dw->get( const_radiation_vars.temperature, _T_label, m_matIdx, patch, gac, 1 );

      new_dw->allocateAndPut( radiation_vars.qfluxe, _radiationFluxELabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.qfluxw, _radiationFluxWLabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.qfluxn, _radiationFluxNLabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.qfluxs, _radiationFluxSLabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.qfluxt, _radiationFluxTLabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.qfluxb, _radiationFluxBLabel, m_matIdx, patch );
      new_dw->allocateAndPut( radiation_vars.volq  , _radiationVolqLabel, m_matIdx, patch );
      new_dw->allocateAndPut( divQ, _src_label, m_matIdx, patch );

      // copy old solution into new
      old_dw->copyOut( divQ, _src_label, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxe, _radiationFluxELabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxw, _radiationFluxWLabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxn, _radiationFluxNLabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxs, _radiationFluxSLabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxt, _radiationFluxTLabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.qfluxb, _radiationFluxBLabel, m_matIdx, patch, gn, 0 );
      old_dw->copyOut( radiation_vars.volq  , _radiationVolqLabel, m_matIdx, patch, gn, 0 );
      old_dw->get( const_radiation_vars.ABSKG , _abskg_label, m_matIdx, patch, gn, 0 );
      old_dw->get( const_radiation_vars.ABSKT , _abskt_label, m_matIdx, patch, gac, 1 );

    } else {

      new_dw->get( const_radiation_vars.temperature, _T_label, m_matIdx, patch, gac, 1 );

      new_dw->getModifiable( radiation_vars.qfluxe, _radiationFluxELabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.qfluxw, _radiationFluxWLabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.qfluxn, _radiationFluxNLabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.qfluxs, _radiationFluxSLabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.qfluxt, _radiationFluxTLabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.qfluxb, _radiationFluxBLabel, m_matIdx, patch );
      new_dw->getModifiable( radiation_vars.volq  , _radiationVolqLabel , m_matIdx, patch );
      new_dw->getModifiable( divQ, _src_label, m_matIdx, patch );

      old_dw->get( const_radiation_vars.ABSKG , _abskg_label, m_matIdx, patch, gn, 0 ); // wrong DW
      old_dw->get( const_radiation_vars.ABSKT , _abskt_label, m_matIdx, patch, gac, 1 );
    }

    old_dw->get( const_radiation_vars.cellType, _labels->d_cellTypeLabel, m_matIdx, patch, gac, 1 );

    if ( do_radiation ){

      if ( timeSubStep == 0 ) {

        if(_DO_model->needIntensitiesBool()){
          for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
            CCVariable<double> cenint;
            new_dw->allocateAndPut(cenint,_IntensityLabels[ord] , m_matIdx, patch );
          }
        }

        //Note: The final divQ is initialized (to zero) and set after the solve in the intensity solve itself.
        _DO_model->intensitysolve( pc, patch, cellinfo, &radiation_vars, &const_radiation_vars, divQ, BoundaryCondition::WALL, m_matIdx, new_dw, old_dw, old_DW_isMissingIntensities );

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

  //__________________________________
  // Define the arches material sets and index.  This is the first place after problemSetup
  // and finalizeMaterials() where you can do it;
  m_matls = _materialManager->allMaterials( "Arches" );

  const int arches  = 0;                       // HARDWIRED arches material index!!!!
  m_matIdx  = _materialManager->getMaterial(arches)->getDWIndex();


  // A pointer to the application so to get a handle to the
  // performanance stats.  This step is a hack so to get the
  // application passed down to other classes like the model.
  m_arches = sched->getApplication();

  if( _DO_model ){
    _DO_model->setApplicationInterface( m_arches );
  }

  //__________________________________
  //
  Task* tsk = scinew Task("DORadiation::initialize", this, &DORadiation::initialize);

  tsk->computes(_src_label);

  for (auto iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter);
  }

  if(_dynamicSolveFrequency) {
    tsk->computes(_dynamicSolveCountPatchLabel);
    tsk->computes( _lastRadSolvePatchLabel );
  }

  sched->addTask(tsk, level->eachPatch(), m_matls);
}

//---------------------------------------------------------------------------
// Method: initialization
//---------------------------------------------------------------------------
void
DORadiation::initialize( const ProcessorGroup * pc,
                         const PatchSubset    * patches,
                         const MaterialSubset * matls,
                         DataWarehouse        * old_dw,
                         DataWarehouse        * new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, m_matIdx, patch );
    src.initialize(0.0);

    for ( auto iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){

      CCVariable<double> temp_var;
      new_dw->allocateAndPut(temp_var, *iter, m_matIdx, patch );
      temp_var.initialize(0.0);
    }
  }

  if(_dynamicSolveFrequency) {
    // Add the per patch dynamicSolveCount so there is something to
    // transfer initially.
    double firstRadSolveAtTime=.1; // unless otherwise dictated by the solve frequency
    SoleVariable< double > ppVar =firstRadSolveAtTime;
    new_dw->put( ppVar, _dynamicSolveCountPatchLabel);

    SoleVariable< int > ppLastRadTimeStep = 0;
    new_dw->put( ppLastRadTimeStep, _lastRadSolvePatchLabel);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule restart initialization
//---------------------------------------------------------------------------
void
DORadiation::sched_restartInitialize( const LevelP& level, SchedulerP& sched )
{

  //__________________________________
  // Define the arches material sets and index.  This is the first place after problemSetup
  // and finalizeMaterials(), where you can do it;
  m_matls = _materialManager->allMaterials( "Arches" );
  const int arches  = 0;                       // HARDWIRED arches material index!!!!
  m_matIdx  = _materialManager->getMaterial(arches)->getDWIndex();

  // A pointer to the application so to get a handle to the
  // performanance stats.  This step is a hack so to get the
  // application passed down to other classes like the model.
  m_arches = sched->getApplication();

  if( _DO_model ){      // When is this ever false?  --Todd
    _DO_model->setApplicationInterface( m_arches );
  }

  //__________________________________
  //  Are all needed varLabels in the checkpoint?
  // If not then initialize them to 0.0
  DataWarehouse* new_dw = sched->getLastDW();

  // Find the first patch, on the arches level, that this mpi rank owns.
  const Uintah::PatchSet* const ps = sched->getLoadBalancer()->getPerProcessorPatchSet( level ); 
  const int rank                   = Uintah::Parallel::getMPIRank();
  const PatchSubset* myPatches     = ps->getSubset( rank );
  
  if ( myPatches->size() > 0 ){
    const Patch* firstPatch  = myPatches->get(0);

    for ( auto  iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      const VarLabel* varLabel = *iter;

      if( !new_dw->exists( varLabel, arches, firstPatch ) ){
        _missingCkPt_labels.push_back( varLabel );
      }
    }
  }

  if (_dynamicSolveFrequency || _missingCkPt_labels.size() > 0) {

    std::string taskNoCom = "DORadiation::restartInitialize";
    Task* tsk = scinew Task(taskNoCom, this, &DORadiation::restartInitialize);
    //tsk->requires( Task::NewDW, _dynamicSolveCountPatchLabel, Ghost::None, 0 );  // These appear to cause problems.  Perhaps it is best to ignore requires since the simulation doesn't care about satifying requires() on restart. NEW is the only DW present on restart.
    //tsk->requires( Task::NewDW, _lastRadSolvePatchLabel, Ghost::None, 0 );
    //tsk->requires( Task::NewDW, _simulationTimeLabel,Ghost::None,0);

    for ( auto  iter = _missingCkPt_labels.begin(); iter != _missingCkPt_labels.end(); iter++){
      tsk->computes( *iter );
    }

    sched->addTask(tsk, level->eachPatch(), m_matls);
  }
}
//______________________________________________________________________
//
inline bool needRadSolveNextTimeStep( const int radSolveCounter,const int &calc_freq, const double nextCFDTime, const double& targetTime ){
  return nextCFDTime  >= targetTime || radSolveCounter >=calc_freq;
}


//---------------------------------------------------------------------------
// Method: restart initialization
//---------------------------------------------------------------------------
void
DORadiation::restartInitialize( const ProcessorGroup  * pc,
                                const PatchSubset     * patches,
                                const MaterialSubset  * matls,
                                      DataWarehouse   * old_dw,
                                      DataWarehouse   * new_dw )
{
  //__________________________________
  //  Initialize any variable missing from the
  // checkpoints to 0.  Enabling addOrthogonalDirs on
  // a restart


  if( _missingCkPt_labels.size() > 0 ){
    static bool doCout=( pc->myRank() == 0 );  // this won't work if this level is owned by another rank.

    DOUT( doCout, "__________________________________\n"
              << "  DORadiation::restartInitialize \n"
              << "    These variables were not found in the checkpoints\n"
              << "    and will be initialized to 0\n");

    for ( auto  iter = _missingCkPt_labels.begin(); iter != _missingCkPt_labels.end(); iter++){
      const VarLabel* QLabel = *iter;
      DOUT( doCout, "    Label:  " << QLabel-> getName() );

      for (int p=0; p < patches->size(); p++){
        const Patch* patch = patches->get(p);
        CCVariable<double> Q;
        new_dw->allocateAndPut( Q, QLabel, m_matIdx, patch);
        Q.initialize( 0.0 );
      }
    }
    doCout=false;
  }


  //__________________________________
  //  Is this ever used?
  if (_dynamicSolveFrequency){
     //// ONLY NEW DW USED, it appears that at restart only the newDW is available.
    simTime_vartype simTime(0);
    new_dw->get(simTime, _simulationTimeLabel );

    timeStep_vartype timeStep(0);
    new_dw->get(timeStep, _labels->d_timeStepLabel ); // For this to be totally correct, should have corresponding requires.

    SoleVariable< double > ppTargetTimeStep;
    SoleVariable< int > lastRadSolveIndex;

    new_dw->get( ppTargetTimeStep,  _dynamicSolveCountPatchLabel );
    new_dw->get( lastRadSolveIndex, _lastRadSolvePatchLabel );

    m_arches->setTaskGraphIndex(needRadSolveNextTimeStep(timeStep - lastRadSolveIndex +1,_radiation_calc_freq,simTime,ppTargetTimeStep));
  }
}

//---------------------------------------------------------------------------
// Method: init_all_intensities
//---------------------------------------------------------------------------
void
DORadiation::init_all_intensities( const ProcessorGroup * pc,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                         DataWarehouse  * old_dw,
                                         DataWarehouse  * new_dw )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    _DO_model->getDOSource(patch, m_matIdx, new_dw, old_dw);
  }
}

//______________________________________________________________________
//
void
DORadiation::doSweepAdvanced( const ProcessorGroup * pc,
                              const PatchSubset    * patches,
                              const MaterialSubset * matls,
                                    DataWarehouse  * old_dw,
                                    DataWarehouse  * new_dw,
                                    int intensity_iter )
{
  // This version relies on FULL spatial scheduling to reduce work, to see
  // logic needed for partial spatial scheduling see revision 57848 or earlier
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    _DO_model->intensitysolveSweepOptimized(patch, m_matIdx, new_dw, old_dw, intensity_iter );
  }
}
//______________________________________________________________________
//
void
DORadiation::computeFluxDivQ( const ProcessorGroup  * pc,
                              const PatchSubset     * patches,
                              const MaterialSubset  * matls,
                                    DataWarehouse   * old_dw,
                                    DataWarehouse   * new_dw )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);
    _DO_model->computeFluxDiv( patch, m_matIdx, new_dw, old_dw);
  }
}
//______________________________________________________________________
//
void
DORadiation::sched_computeSourceSweep( const LevelP & level,
                                       SchedulerP   & sched,
                                       int timeSubStep )
{
if (timeSubStep==0){

  Ghost::GhostType  gn = Ghost::None;

  _gv = std::vector< std::vector < std::vector < Ghost::GhostType > > >   (2,  std::vector < std::vector < Ghost::GhostType > > (2, std::vector < Ghost::GhostType > (2) ));
  _gv[0][0][0] =Ghost::xpypzp;
  _gv[0][0][1] =Ghost::xpypzm;
  _gv[0][1][0] =Ghost::xpymzp;
  _gv[0][1][1] =Ghost::xpymzm;
  _gv[1][0][0] =Ghost::xmypzp;
  _gv[1][0][1] =Ghost::xmypzm;
  _gv[1][1][0] =Ghost::xmymzp;
  _gv[1][1][1] =Ghost::xmymzm;

  _DO_model->setLabels(_abskg_label, _abskt_label, _T_label,_labels->d_cellTypeLabel ,_radIntSource,
                       _radiationFluxELabel, _radiationFluxWLabel,
                       _radiationFluxNLabel, _radiationFluxSLabel,
                       _radiationFluxTLabel, _radiationFluxBLabel,
                       _radiationVolqLabel, _src_label);

////-----------for timesteps w/o radiation----------//
  int Radiation_TG = 1;
  int  no_Rad_TG   = 0;

  std::string taskNoCom = "DORadiation::TransferRadFieldsFromOldDW";
  Task* tsk_noRadiation = scinew Task(taskNoCom, this, &DORadiation::TransferRadFieldsFromOldDW);

  tsk_noRadiation->requires( Task::OldDW, _radiationFluxELabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationFluxWLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationFluxNLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationFluxSLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationFluxTLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationFluxBLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW, _radiationVolqLabel, gn, 0);
  tsk_noRadiation->requires( Task::OldDW,  _src_label, gn, 0);

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
        const int idx = intensityIndx(j, iband);
        tsk_noRadiation->requires( Task::OldDW,_IntensityLabels[idx], gn, 0 );
        tsk_noRadiation->computes( _IntensityLabels[idx]);
      }
    }
  } else if ( m_user_intensity_save ){
    for ( auto i = m_user_intensity_save_labels.begin(); i != m_user_intensity_save_labels.end(); i++ ){
      tsk_noRadiation->requires( Task::OldDW, *i, gn, 0 );
      tsk_noRadiation->computes( *i );
    }
  }

  if (_dynamicSolveFrequency ) {
    tsk_noRadiation->requires( Task::OldDW, _dynamicSolveCountPatchLabel, gn, 0 );
    tsk_noRadiation->requires( Task::OldDW, _lastRadSolvePatchLabel,      gn, 0 );

    tsk_noRadiation->computes( _dynamicSolveCountPatchLabel );
    tsk_noRadiation->computes( _lastRadSolvePatchLabel );
  }

  sched->addTask(tsk_noRadiation, level->eachPatch(), m_matls, no_Rad_TG);

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


                  if( _doesPatchExist[iAdj+(idir==0 ? 0 : -1)][jAdj+(jdir==0 ? 0 : -1)][kAdj+(kdir==0 ? 0 : -1)]==false) { // needed for multi-box problems
                    continue ;
                  }
                  Point patchCenter((_xyzPatch_boundary[0][iAdj] + _xyzPatch_boundary[0][iAdj + (idir==0 ? 1 : -1)])/2.0,
                                    (_xyzPatch_boundary[1][jAdj] + _xyzPatch_boundary[1][jAdj + (jdir==0 ? 1 : -1)])/2.0,
                                    (_xyzPatch_boundary[2][kAdj] + _xyzPatch_boundary[2][kAdj + (kdir==0 ? 1 : -1)])/2.0);
                  RelevantPatches.push_back(  level.get_rep()->getPatchFromPoint( patchCenter, false ));

                } // check on i (is it in domain?)
              } // k
            } // j

            if(RelevantPatches.size()==0){
              continue; // No patches in this patch set, this could be reformulated, to create emtpy patchsets, because it makes downstream logic less friendly
            }

            PatchSubset* sweepingPatches= scinew PatchSubset(RelevantPatches);

            PatchSet* sweepingPatches2= scinew PatchSet();

              //sweepingPatches2->addReference();
            for (unsigned int ix2=0; ix2 < RelevantPatches.size(); ix2++){
              sweepingPatches2->add(RelevantPatches[ix2]);
            }

            sweepingPatches->sort();
            if (idir==0 && jdir==0 && kdir==0){
              _RelevantPatchesXpYpZp.push_back(sweepingPatches);
              _RelevantPatchesXpYpZp2.push_back(sweepingPatches2);
            }
            if (idir==0 && jdir==0 && kdir==1){
              _RelevantPatchesXpYpZm.push_back(sweepingPatches);
              _RelevantPatchesXpYpZm2.push_back(sweepingPatches2);
            }
            if (idir==0 && jdir==1 && kdir==0){
              _RelevantPatchesXpYmZp.push_back(sweepingPatches);
              _RelevantPatchesXpYmZp2.push_back(sweepingPatches2);
            }
            if (idir==0 && jdir==1 && kdir==1){
              _RelevantPatchesXpYmZm.push_back(sweepingPatches);
              _RelevantPatchesXpYmZm2.push_back(sweepingPatches2);
            }
            if (idir==1 && jdir==0 && kdir==0){
              _RelevantPatchesXmYpZp.push_back(sweepingPatches);
              _RelevantPatchesXmYpZp2.push_back(sweepingPatches2);
            }
            if (idir==1 && jdir==0 && kdir==1){
              _RelevantPatchesXmYpZm.push_back(sweepingPatches);
              _RelevantPatchesXmYpZm2.push_back(sweepingPatches2);
            }
            if (idir==1 && jdir==1 && kdir==0){
              _RelevantPatchesXmYmZp.push_back(sweepingPatches);
              _RelevantPatchesXmYmZp2.push_back(sweepingPatches2);
            }
            if (idir==1 && jdir==1 && kdir==1){
              _RelevantPatchesXmYmZm.push_back(sweepingPatches);
              _RelevantPatchesXmYmZm2.push_back(sweepingPatches2);
            }
          } // iphase
        } // z+ z- dir
      } // y+ y- dir
    } // x+ x- dir
    //----------------------------------------------------------------------//


    //--------------------------------------------------------------------//
    //      Schedule initialization task.  Initializes all intensities
    //--------------------------------------------------------------------//
    std::string taskname1 = "DORadiation::init_all_intensities";
    Task* tsk1 = scinew Task(taskname1, this, &DORadiation::init_all_intensities);

    tsk1->requires( Task::OldDW,_abskg_label, gn, 0 );
    tsk1->requires( Task::OldDW,_abskt_label, gn, 0 );
    tsk1->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
    tsk1->requires( Task::NewDW,_T_label, gn, 0 );

    for (int i=0 ; i< _DO_model->get_nQn_part(); i++){
      tsk1->requires( Task::OldDW,_DO_model->getAbskpLabels()[i],    gn, 0 );
      tsk1->requires( Task::OldDW,_DO_model->getPartTempLabels()[i], gn, 0 );
    }

    if (_DO_model->ScatteringOnBool()){
      for (int j=0; j< _nDir; j++){
        for (int iband=0; iband<d_nbands; iband++){
          const int idx = intensityIndx(j, iband);

          tsk1->requires( Task::OldDW,_IntensityLabels[idx], gn, 0 );
          tsk1->computes( _emiss_plus_scat_source_label[idx]);
        }
        tsk1->requires( Task::OldDW, _scatktLabel,   gn, 0 );
        tsk1->requires( Task::OldDW,_asymmetryLabel, gn, 0 );
      }
    }

    for (int iband=0; iband<d_nbands; iband++){
      tsk1->computes( _radIntSource[iband]);
    }

    for (int iband=0; iband<d_nbands; iband++){
      tsk1->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
      if(d_nbands>1){
        tsk1->requires( Task::OldDW,spectral_gas_weight[iband], gn, 0 );
      }
    }

    if(_DO_model->spectralSootOn()){
      tsk1->requires( Task::OldDW,VarLabel::find("absksoot"), gn, 0 );
    }

    for (int iband=0; iband<d_nbands; iband++){
      tsk1->requires( Task::OldDW,spectral_gas_absorption[iband], gn, 0 );
    }

    sched->addTask(tsk1, level->eachPatch(), m_matls, Radiation_TG);

    //--------------------------------------------------------------------//
    //    Schedule set BCs task.  Sets the intensity fields in the walls.  //
    //--------------------------------------------------------------------//

    for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
      std::stringstream tasknamec;
      tasknamec << "DORadiation::sweeping_initialize_" <<ord;
      Task* tskc = scinew Task(tasknamec.str(), this, &DORadiation::setIntensityBC,ord);

      tskc->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
      tskc->requires( Task::NewDW,_T_label, gn, 0 );

      for (int iband=0; iband<d_nbands; iband++){
        const int idx = intensityIndx( ord, iband);
        tskc->computes( _IntensityLabels[idx]);
      }
      sched->addTask(tskc, level->eachPatch(), m_matls, Radiation_TG);
    }

    //-- create material subset from arches materials (needed to satifiy interface for spatial scheduling) --//
    const MaterialSubset* matlDS=nullptr;
    for ( auto i_matSubSet = (m_matls->getVector()).begin(); i_matSubSet != (m_matls->getVector()).end(); i_matSubSet++ ){
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

        const int first_intensity=idir*_nDir/nOctants;

        int pAdjm = _directional_phase_adjustment[1-_DO_model->xDir(first_intensity)][1-_DO_model->yDir(first_intensity)][1-_DO_model->zDir(first_intensity)]; // L-shaped domain adjustment, assumes that ordiantes are stored in octants (8 bins), with similiar directional properties in each bin
        int pAdjp = _directional_phase_adjustment[  _DO_model->xDir(first_intensity)][  _DO_model->yDir(first_intensity)][  _DO_model->zDir(first_intensity)]; // L-shaped domain adjustment, assumes that ordiantes are stored in octants (8 bins), with similiar directional properties in each bin

        for( int int_x=std::max(0,istage-_nphase+pAdjm+1);  int_x<std::min(_nDir/nOctants,istage+1); int_x++){ // loop over per-octant-intensities (intensity_within_octant_x)
          if(istage-int_x>_nphase-pAdjp-pAdjm-1){ // Terminte sweep early for multi-box problems
            continue;
          }

          // combine stages into single task?
          int intensity_iter = int_x + idir*_nDir/nOctants;

          std::stringstream taskname2;
          taskname2 << "DORadiation::doSweepAdvanced_" <<istage<< "_"<<intensity_iter;

          Task* tsk2 = scinew Task(taskname2.str(), this, &DORadiation::doSweepAdvanced,intensity_iter);

          tsk2->requires( Task::OldDW,_labels->d_cellTypeLabel, gn, 0 );
          tsk2->requires( Task::OldDW,_abskt_label,             gn, 0 );

          if (_DO_model->ScatteringOnBool()){
            for (int iband=0; iband<d_nbands; iband++){
              const int idx = intensityIndx(intensity_iter, iband);

              tsk2->requires( Task::NewDW, _emiss_plus_scat_source_label[idx],gn,0);
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

          // requires->modifies chaining to facilitate inter-patch communication.
          // We may be able to use wrapped requires() due to using the reduced patchset when calling addTask (True spatial scheduling)


          const Task::PatchDomainSpec thisLevel = Task::ThisLevel;   // for readability
          const Task::MaterialDomainSpec ND     = Task::NormalDomain;

          for (int iband=0; iband<d_nbands; iband++){

            const int idx = intensityIndx( intensity_iter, iband);

            tsk2->modifies( _IntensityLabels[idx]);

            // --- Turn on and off communication depending on phase and intensity using equation:  iStage = iPhase + intensity_within_octant_x, 8 different patch subsets, due to 8 octants ---//
            if ( _DO_model->xDir(first_intensity) ==1 &&
                 _DO_model->yDir(first_intensity) ==1 &&
                 _DO_model->zDir(first_intensity) ==1){

              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXpYpZp[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){ // Adding bands, made this logic painful. To fit the original abstraction, the bands loop should be merged with int_x loop.
                sched->addTask( tsk2,_RelevantPatchesXpYpZp2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==0){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXpYpZm[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXpYpZm2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==1){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXpYmZp[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXpYmZp2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==1 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==0){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXpYmZm[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXpYmZm2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==1){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXmYpZp[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXmYpZp2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==1 && _DO_model->zDir(first_intensity)==0){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXmYpZm[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXmYpZm2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==1){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXmYmZp[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXmYmZp2[istage-int_x] , m_matls, Radiation_TG);
              }
            }else if (_DO_model->xDir(first_intensity) ==0 && _DO_model->yDir(first_intensity)==0 && _DO_model->zDir(first_intensity)==0){
              tsk2->requires( Task::NewDW, _IntensityLabels[idx] ,_RelevantPatchesXmYmZm[istage-int_x], thisLevel, matlDS, ND, _gv[_DO_model->xDir(intensity_iter)][_DO_model->yDir(intensity_iter)][_DO_model->zDir(intensity_iter)], 1, false);
              if((iband+1)==d_nbands){
                sched->addTask( tsk2,_RelevantPatchesXmYmZm2[istage-int_x] , m_matls, Radiation_TG);
              }
            }

          //sched->addTask(tsk2, level->eachPatch(), matls,Radiation_TG); // partial spatial scheduling
          } // iband
        } // int_x (octant intensities)
      } // octants
    } //istage

    //-----------------------------------------------------------------//
    // This task computes 6 radiative fluxes, incident radiation,
    //  and divQ. This function requries the required intensity fields
    // sn*(2+sn), radiation temperature, abskt, and scatkt (scattering
    // only)
    //-----------------------------------------------------------------//
    std::string taskname3 = "DORadiation::computeFluxDivQ";
    Task* tsk3 = scinew Task(taskname3, this, &DORadiation::computeFluxDivQ);

    for (int iband=0; iband<d_nbands; iband++){
      for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
        const int idx = intensityIndx(ord, iband);
        tsk3->requires( Task::NewDW, _IntensityLabels[idx], gn, 0 );
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

    sched->addTask(tsk3, level->eachPatch(), _materialManager->allMaterials( "Arches" ),Radiation_TG);

  }  //  if enum_sweepSpatiallyParallel
 }  // if timesubstep ==0
}

//______________________________________________________________________
//
void
DORadiation::profileDynamicRadiation( const ProcessorGroup * pc,
                                      const PatchSubset    * patches,
                                      const MaterialSubset * matls,
                                            DataWarehouse  * old_dw,
                                            DataWarehouse  * new_dw)
{
  double dt_min=1 ; // min
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    constCCVariable <double > volQ;
    constCCVariable <double > divQ;
    constCCVariable <double > gasTemp;

    new_dw->get(gasTemp, _T_label            , m_matIdx, patch, Ghost::None, 0);
    new_dw->get(volQ,    _radiationVolqLabel , m_matIdx, patch, Ghost::None, 0);
    new_dw->get(divQ,    _src_label          , m_matIdx, patch, Ghost::None, 0);

    double maxdelT=0.;
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    const double Cp_vol=400.; // j/m^3/K            air at 1000K                       HARDWIRED!!!!

    Uintah::parallel_for( range, [&](int i, int j, int k) {
      double   T_eql =    sqrt(sqrt(volQ(i,j,k) / 4. /  5.67e-8));

      maxdelT= max(fabs(T_eql - gasTemp(i,j,k)), maxdelT);
      double timescale = fabs((T_eql - gasTemp(i,j,k) *Cp_vol) / divQ(i,j,k));
      dt_min = std::min( timescale / _nsteps_calc_freq,  dt_min ); // min for zero divQ

      } );

    simTime_vartype simTime(0);
    old_dw->get(simTime, _simulationTimeLabel);

    new_dw->put(min_vartype(simTime+dt_min), VarLabel::find("min_time"));
  }
}


//---------------------------------------------------------------------------
// Method: setIntensityBC
//---------------------------------------------------------------------------
void
DORadiation::setIntensityBC( const ProcessorGroup * pc,
                             const PatchSubset    * patches,
                             const MaterialSubset * matls,
                                   DataWarehouse  * old_dw,
                                   DataWarehouse  * new_dw,
                                   int ord )
{
  for (int p=0; p < patches->size(); p++){
    const Patch* patch = patches->get(p);

    Ghost::GhostType ghostType = _gv[_DO_model->xDir(ord)][_DO_model->yDir(ord)][_DO_model->zDir(ord)];

    _DO_model->setIntensityBC( patch, m_matIdx, new_dw, old_dw, ghostType, ord );

  }
}

//---------------------------------------------------------------------------
// Method: TransferRadFieldsFromOldDW
//---------------------------------------------------------------------------
void
DORadiation::TransferRadFieldsFromOldDW( const ProcessorGroup * pc,
                                         const PatchSubset    * patches,
                                         const MaterialSubset * matls,
                                               DataWarehouse  * old_dw,
                                               DataWarehouse  * new_dw)
{
  if (_DO_model->ScatteringOnBool() || (!_sweepMethod) ){
    for (int iband=0; iband<d_nbands; iband++){
      for( int ord=0;  ord< _DO_model->getIntOrdinates();ord++){
        const int indx = intensityIndx( ord, iband);


        new_dw->transferFrom(old_dw,_IntensityLabels[indx],  patches, matls);
        if (_DO_model->needIntensitiesBool()==false){ // this is always true for scattering
          break; // need 1 intensity, for a feature that is never used  =(
        }
      }
    }
  } else if ( m_user_intensity_save ){

    for ( auto i = m_user_intensity_save_labels.begin(); i != m_user_intensity_save_labels.end(); i++ ){

      for (int p=0; p < patches->size(); p++){
        const Patch* patch = patches->get(p);
        if ( old_dw->exists(*i, 0, patch) ){
          new_dw->transferFrom( old_dw, *i, patches, matls );
        } else {
          //If the user adds a new label to the archiver this will create and initialize to zero.
          CCVariable<double> temp;
          new_dw->allocateAndPut( temp, *i, m_matIdx, patch);
          temp.initialize(0.0);
        }
      }
    }
  }

  new_dw->transferFrom(old_dw, _radiationFluxELabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationFluxWLabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationFluxNLabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationFluxSLabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationFluxTLabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationFluxBLabel, patches, matls);
  new_dw->transferFrom(old_dw, _radiationVolqLabel, patches, matls);
  new_dw->transferFrom(old_dw, _src_label, patches, matls);

  //Reduce the dynamic radiation solve time step counter for each
  //patch and then store the minimum value.
  if(_dynamicSolveFrequency) {
    simTime_vartype simTime(0);
    old_dw->get(simTime, _simulationTimeLabel );

    timeStep_vartype timeStep(0);
    old_dw->get(timeStep, _labels->d_timeStepLabel ); // For this to be totally correct, should have corresponding requires.

    delt_vartype delT;
    old_dw->get(delT,_labels->d_delTLabel);

    SoleVariable< double > ppTargetTimeStep;
    SoleVariable< int > lastRadSolveIndex;

    old_dw->get( ppTargetTimeStep, _dynamicSolveCountPatchLabel );
    new_dw->put( ppTargetTimeStep, _dynamicSolveCountPatchLabel );

    old_dw->get( lastRadSolveIndex, _lastRadSolvePatchLabel );
    new_dw->put( lastRadSolveIndex, _lastRadSolvePatchLabel );

    m_arches->setTaskGraphIndex(needRadSolveNextTimeStep(timeStep - lastRadSolveIndex +1,_radiation_calc_freq,delT+simTime,ppTargetTimeStep));
  }
}

//---------------------------------------------------------------------------
// Method: checkReductionVars
//---------------------------------------------------------------------------
void
DORadiation::checkReductionVars( const ProcessorGroup * pg,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * matls,
                                       DataWarehouse  * old_dw,
                                       DataWarehouse  * new_dw )
{
  min_vartype target_rad_solve_time;
  new_dw->get(target_rad_solve_time,VarLabel::find("min_time"));

  simTime_vartype simTime(0);
  old_dw->get(simTime, _simulationTimeLabel );

  timeStep_vartype timeStep(0);
  old_dw->get(timeStep, _labels->d_timeStepLabel ); // For this to be totally correct, should have corresponding requires.

  delt_vartype delT;
  old_dw->get(delT,_labels->d_delTLabel);

  SoleVariable< double > ppVar=(double) target_rad_solve_time;
  SoleVariable< int > ppCurrentTimeStep = (int) timeStep;

  new_dw->put( ppVar, _dynamicSolveCountPatchLabel );
  new_dw->put( ppCurrentTimeStep, _lastRadSolvePatchLabel );

  m_arches->setTaskGraphIndex(needRadSolveNextTimeStep(timeStep - ppCurrentTimeStep +1,_radiation_calc_freq,delT+simTime,target_rad_solve_time));
}
//______________________________________________________________________
//      utilities
//______________________________________________________________________
int
DORadiation::intensityIndx(const int ord,
                           const int iband)
{
  return (ord + iband * _DO_model->getIntOrdinates() );
}
