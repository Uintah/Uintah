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

#include <CCA/Components/OnTheFlyAnalysis/controlVolFluxes.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DOUT.hpp>

#include <dirent.h>
#include <iostream>
#include <cstdio>

using namespace Uintah;
using namespace std;
//______________________________________________________________________
// ToDo:
//       Verify the face iterators.
//       Test in multiple patches and mpi ranks
//       Generalize so doubles and Vectors based fluxes can be computed
//       generalize so user inputs face centered velocities
//       each CV should keep track of output face fluxes
//______________________________________________________________________

Dout cout_OTF_CVF("controlVolFluxes",     "OnTheFlyAnalysis", "controlVolFluxes task exec", false);
Dout dbg_OTF_CVF("controlVolFluxes_dbg",  "OnTheFlyAnalysis", "controlVolFluxes debug info", false);

controlVolFluxes::controlVolFluxes( const ProcessorGroup  * myworld,
                                    const MaterialManagerP materialManager,
                                    const ProblemSpecP    & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  m_zeroMatl     = 0;
  m_zeroMatlSet  = 0;
  m_zeroPatch    = 0;
  m_matIdx       = -9;
  m_matl         = nullptr;
  m_matlSet      = nullptr;

  m_lb = scinew labels();

  m_lb->lastCompTime      = VarLabel::create( "lastCompTime_CVF", max_vartype::getTypeDescription() );
  m_lb->fileVarsStruct    = VarLabel::create( "FileInfo_CVF", PerPatch<FileInfoP>::getTypeDescription() );
}

//______________________________________________________________________
//
controlVolFluxes::~controlVolFluxes()
{
  DOUT(cout_OTF_CVF, " Doing: destorying fluxes " );
  if( m_zeroMatlSet  && m_zeroMatlSet->removeReference() ) {
    delete m_zeroMatlSet;
  }
  if( m_zeroMatl && m_zeroMatl->removeReference() ) {
    delete m_zeroMatl;
  }
  if( m_zeroPatch && m_zeroPatch->removeReference() ) {
    delete m_zeroPatch;
  }
  if( m_matl && m_matl->removeReference() ) {
    delete m_matl;
  }
  if( m_matlSet && m_matlSet->removeReference() ) {
    delete m_matlSet;
  }

  VarLabel::destroy( m_lb->lastCompTime );
  VarLabel::destroy( m_lb->fileVarsStruct );

  // labels for each CV
  for( size_t i=0; i< m_controlVols.size(); i++ ){

    VarLabel::destroy( m_lb->totalQ_CV[i] );
    VarLabel::destroy( m_lb->net_Q_faceFluxes[i] );

   for( int f=cvFace::startFace; f <= cvFace::endFace; f++){
      cvFace face = static_cast<cvFace>(f);
      VarLabel::destroy( m_lb->Q_faceFluxes[i][face] );
    }
  }

  delete m_lb;

  // This must be last
  for( size_t i=0; i< m_controlVols.size(); i++ ){
    delete m_controlVols[i];
  }


}

//______________________________________________________________________
//
void controlVolFluxes::problemSetup(const ProblemSpecP& ,
                                    const ProblemSpecP& ,
                                    GridP& grid,
                                    std::vector<std::vector<const VarLabel* > > &PState,
                                    std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  DOUT(cout_OTF_CVF, "Doing problemSetup \t\t\t\tfluxes" );

  if(grid->numLevels() > 1 ) {
    proc0cout << "______________________________________________________________________\n"
              << " DataAnalysis:controlVolFluxes\n"
              << " ERROR:  Currently, this analysis module only works on a single level\n"
              << "______________________________________________________________________\n";
    throw ProblemSetupException("", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  m_module_spec->require( "samplingFrequency", m_analysisFreq );
  m_module_spec->require( "timeStart",         d_startTime );
  m_module_spec->require( "timeStop",          d_stopTime );

  m_zeroMatlSet = scinew MaterialSet();
  m_zeroMatlSet->add(0);
  m_zeroMatlSet->addReference();

  m_zeroMatl = m_zeroMatlSet->getUnion();

  // one patch
  const Patch* p = grid->getPatchByID(0,0);
  m_zeroPatch = scinew PatchSet();
  m_zeroPatch->add(p);
  m_zeroPatch->addReference();

  //__________________________________
  // find the material .  Default is matl 0.
  //  <material>   atmosphere </material>

  ProblemSpecP matl_ps = m_module_spec->findBlock("material");

  if( matl_ps == nullptr ){
    throw ProblemSetupException("ERROR: Couldn't find <material> xml tag", __FILE__, __LINE__);
  }

  Material* matl= m_materialManager->parseAndLookupMaterial( m_module_spec, "material" );
  m_matIdx = matl->getDWIndex();

  vector<int> m;
  m.push_back(0);            // matl index for FileInfo label
  m.push_back( m_matIdx );

  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  m_matlSet = scinew MaterialSet();
  m_matlSet->addAll(m);
  m_matlSet->addReference();

  m_matl = m_matlSet->getUnion();

  // HARDWIRED FOR ICE/MPMICE
  m_lb->vel_CC    = VarLabel::find( "vel_CC", "ERROR  controlVolFluxes::problemSetup"  );
  m_lb->rho_CC    = VarLabel::find( "rho_CC", "ERROR  controlVolFluxes::problemSetup"  );

  m_lb->uvel_FC   = VarLabel::find( "uvel_FCME", "ERROR  controlVolFluxes::problemSetup"  );
  m_lb->vvel_FC   = VarLabel::find( "vvel_FCME", "ERROR  controlVolFluxes::problemSetup"  );
  m_lb->wvel_FC   = VarLabel::find( "wvel_FCME", "ERROR  controlVolFluxes::problemSetup"  );


  //__________________________________
  // Add the control volumes to a vector
  ProblemSpecP cv_ps = m_module_spec->findBlock("controlVolumes");
  if( cv_ps == nullptr ) {
    throw ProblemSetupException("ERROR: Couldn't find <controlVolumes> xml node", __FILE__, __LINE__);
  }

  for( ProblemSpecP box_ps = cv_ps->findBlock( "box" ); box_ps != nullptr; box_ps=box_ps->findNextBlock( "box" ) ) {
    controlVolume* cv  = scinew controlVolume(box_ps, grid);
    m_controlVols.push_back( cv );
  }
}

//______________________________________________________________________
//
void controlVolFluxes::scheduleInitialize( SchedulerP   & sched,
                                           const LevelP & level )
{
  printSchedule(level,cout_OTF_CVF,"controlVolFluxes::scheduleInitialize");

  Task* t = scinew Task("controlVolFluxes::initialize",
                  this, &controlVolFluxes::initialize);

  t->computes( m_lb->lastCompTime );
  t->computes( m_lb->fileVarsStruct, m_zeroMatl );

  // only run task once per proc
  t->setType( Task::OncePerProc );
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);

  sched->addTask(t, perProcPatches, m_zeroMatlSet);
}

//______________________________________________________________________
//
void controlVolFluxes::initialize( const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset *,
                                         DataWarehouse  *,
                                         DataWarehouse  * new_dw )
{
  // initialize each of the control volumes
  const Level* level = getLevel(patches);
  for( size_t i=0; i< m_controlVols.size(); i++ ){
    controlVolume* cv = m_controlVols[i];
    cv->initialize(level);
    //cv->print();
  }
  //__________________________________
  //  Create varLabels needed for each CV
  // ONLY create these labels once per proc, otherwise nPatches labels are created
  m_lb->totalQ_CV        = createLabels("totalQ_CV",    sum_vartype::getTypeDescription() );
  m_lb->net_Q_faceFluxes = createLabels("net_Q_fluxes", sumvec_vartype::getTypeDescription() );

  for( size_t i=0; i< m_controlVols.size(); i++ ){
    controlVolume* cv = m_controlVols[i];

    std::string cvName = cv->getName();

    FaceLabelsMap Q_faceLabels;
    FaceNamesMap  Q_faceNames;

    for( auto f : cv->allFaces){
      std::string name = cvName + "_Q_FaceFlux_" + cv->getFaceName(f);
      Q_faceNames[f]  = name;
      Q_faceLabels[f] = VarLabel::create( name, sum_vartype::getTypeDescription() );
    }

    m_lb->Q_faceNames.push_back( Q_faceNames );
    m_lb->Q_faceFluxes.push_back( Q_faceLabels );
  }

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_OTF_CVF,"Doing initialize");

    // last computational time
    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put( max_vartype(tminus), m_lb->lastCompTime );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo, m_lb->fileVarsStruct, 0, patch);

    //__________________________________
    //  does the uda exist?
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = m_output->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:controlVolFluxes  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}

//______________________________________________________________________
//
void controlVolFluxes::scheduleRestartInitialize(SchedulerP   & sched,
                                                 const LevelP & level)
{
  scheduleInitialize( sched, level);
}


//______________________________________________________________________
//
void controlVolFluxes::scheduleDoAnalysis(SchedulerP   & sched,
                                          const LevelP & level)
{

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_CVF", false, false, false, true, true);

  //__________________________________
  //  compute the total Q and net fluxes
  printSchedule( level,cout_OTF_CVF,"controlVolFluxes::scheduleDoAnalysis" );

  Task* t0 = scinew Task( "controlVolFluxes::integrate_Q_overCV",
                     this,&controlVolFluxes::integrate_Q_overCV );

  Ghost::GhostType gn  = Ghost::None;
  Ghost::GhostType gac = Ghost::AroundCells;

  sched_TimeVars( t0, level, m_lb->lastCompTime, false );

//  t0->requires( Task::NewDW, m_lb->vel_CC,    m_matl, gn );
  t0->requires( Task::NewDW, m_lb->rho_CC,    m_matl, gac, 1 );

  t0->requires( Task::NewDW, m_lb->uvel_FC,   m_matl, gn );
  t0->requires( Task::NewDW, m_lb->vvel_FC,   m_matl, gn );
  t0->requires( Task::NewDW, m_lb->wvel_FC,   m_matl, gn );

  for( size_t i=0; i< m_controlVols.size(); i++ ){
    controlVolume* cv = m_controlVols[i];

    t0->computes( m_lb->totalQ_CV[i] );
    t0->computes( m_lb->net_Q_faceFluxes[i] );

    for( auto f : cv->allFaces){
      t0->computes( m_lb->Q_faceFluxes[i][f] );
    }
  }

  sched->addTask( t0, level->eachPatch(), m_matlSet );

  //__________________________________
  //  Task that outputs the contributions
  Task* t1 = scinew Task("controlVolFluxes::doAnalysis",
                    this,&controlVolFluxes::doAnalysis );

  sched_TimeVars( t1, level, m_lb->lastCompTime, true );
  t1->requires( Task::OldDW, m_lb->fileVarsStruct, m_zeroMatl, gn, 0 );

  for( size_t i=0; i< m_controlVols.size(); i++ ){
    controlVolume* cv = m_controlVols[i];

    t1->requires( Task::NewDW, m_lb->totalQ_CV[i] );
    t1->requires( Task::NewDW, m_lb->net_Q_faceFluxes[i] );

    for( auto f : cv->allFaces){
      t1->requires( Task::NewDW, m_lb->Q_faceFluxes[i][f] );
    }
  }

  t1->computes( m_lb->fileVarsStruct, m_zeroMatl );
  sched->addTask( t1, m_zeroPatch, m_zeroMatlSet);        // you only need to schedule patch 0 since all you're doing is writing out data
}

//______________________________________________________________________
//  Compute the total Q of the control volume and the fluxes passing
//  through the control surfaces
//______________________________________________________________________
//
void controlVolFluxes::integrate_Q_overCV(const ProcessorGroup * pg,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * matl_sub ,
                                                DataWarehouse  * old_dw,
                                                DataWarehouse  * new_dw)
{

  const Level* level = getLevel(patches);

  // Ignore the task if a recompute time step has been requested upstream
  bool rts      = new_dw->recomputeTimeStep();
  bool itItTime = isItTime( old_dw, level, m_lb->lastCompTime );

  if( itItTime == false  || rts ) {
    return;
  }

  //__________________________________
  //  Loop over patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_OTF_CVF,"Doing controlVolFluxes::integrate_Q_overCV");
    DOUT(dbg_OTF_CVF, "     Patch: " << patch->getCellLowIndex() << " " << patch->getCellHighIndex() << "\n");

    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    Ghost::GhostType gn  = Ghost::None;
    Ghost::GhostType gac = Ghost::AroundCells;
    new_dw->get(rho_CC,  m_lb->rho_CC,   m_matIdx, patch, gac,1);
  //  new_dw->get(vel_CC,  m_lb->vel_CC,   m_matIdx, patch, gn,0);
    new_dw->get(uvel_FC, m_lb->uvel_FC,  m_matIdx, patch, gn,0);
    new_dw->get(vvel_FC, m_lb->vvel_FC,  m_matIdx, patch, gn,0);
    new_dw->get(wvel_FC, m_lb->wvel_FC,  m_matIdx, patch, gn,0);

    //__________________________________
    //  loop over control volumes
    for( size_t cv=0; cv< m_controlVols.size(); cv++ ){

      controlVolume* contVol = m_controlVols[cv];

      if ( !contVol->controlVolume::hasBoundaryFaces( patch ) ) {
        continue;
      }

      //__________________________________
      //  Sum the total Q over the patch
      double  cellVol = patch->cellVolume();
      double totalQ_CV = 0;

      for (CellIterator iter=contVol->getCellIterator( patch );!iter.done();iter++){
        IntVector c = *iter;
        totalQ_CV += rho_CC[c] * cellVol;
      }

      //__________________________________
      // Sum the fluxes passing through control volume surface

      vector<controlVolume::FaceType> bf;
      contVol->getBoundaryFaces(bf, patch);

      faceQuantities* faceQ = scinew faceQuantities;

      initializeVars( faceQ );

      for( vector<cvFace>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        cvFace face = *itr;

        string faceName = contVol->getFaceName( face );
        Vector norm     = contVol->getFaceNormal( face );
        IntVector axis  = contVol->getFaceAxes( face );

        DOUT( dbg_OTF_CVF, std::setprecision(15) << std::left
                 << setw(7)    << contVol->getName()
                 << setw(7)   << "Face:"     << setw(7) << faceName
                 << setw(10)  << "faceType:" << setw(5) << face
                 << setw(7)   << "norm:"     << setw(5) << norm << std::left
                 << setw(15)  << "faceAxes:" << setw(7) <<  axis );


        //__________________________________
        //           X faces
        if (face == controlVolume::xminus || face == controlVolume::xplus) {
          integrate_Q_overFace ( face, contVol, patch, faceQ, uvel_FC, rho_CC, vel_CC);
        }

        //__________________________________
        //        Y faces
        if (face == controlVolume::yminus || face == controlVolume::yplus) {
          integrate_Q_overFace( face, contVol, patch, faceQ, vvel_FC, rho_CC, vel_CC);
        }
        //__________________________________
        //        Z faces
        if (face == controlVolume::zminus || face == controlVolume::zplus) {
          integrate_Q_overFace( face, contVol, patch, faceQ, wvel_FC, rho_CC, vel_CC);
        }

        new_dw->put( sum_vartype( faceQ->Q_faceFluxes[face] ), m_lb->Q_faceFluxes[cv][face] );
      }  // boundary faces

      //__________________________________
      //  Now compute the net fluxes from all of the face quantites
      Vector net_Q_flux = L_minus_R( faceQ->Q_faceFluxes );

      //__________________________________
      // put in the dw
      DOUT( dbg_OTF_CVF,  std::setprecision(15) << contVol->getName() << " Total CV : " << totalQ_CV << " Net face fluxes: " << net_Q_flux );

      new_dw->put( sum_vartype( totalQ_CV ),     m_lb->totalQ_CV[cv] );
      new_dw->put( sumvec_vartype( net_Q_flux ), m_lb->net_Q_faceFluxes[cv] );

      delete faceQ;
    }  // controlVol loop
  }  // patch loop
}

//______________________________________________________________________
//
void controlVolFluxes::doAnalysis(const ProcessorGroup * pg,
                                  const PatchSubset    * patches,
                                  const MaterialSubset * matls ,
                                        DataWarehouse  * old_dw,
                                        DataWarehouse  * new_dw)
{

  // Ignore the task if a recompute time step has been requested upstream
  bool rts = new_dw->recomputeTimeStep();

  const Level* level = getLevel(patches);
  timeVars tv;

  getTimeVars( old_dw, level, m_lb->lastCompTime, tv );
  putTimeVars( new_dw, m_lb->lastCompTime, tv );

  if( rts || tv.isItTime == false) {
    return;
  }

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_OTF_CVF,"Doing doAnalysis");

    //__________________________________
    // open the struct that contains the file pointer map.  We use FileInfoP types
    // and store them in the DW to avoid doing system calls (SLOW).
    // Note: after regridding this may not exist for this patch in the old_dw
    PerPatch<FileInfoP> fileInfo;

    if( old_dw->exists( m_lb->fileVarsStruct, 0, patch ) ){
      old_dw->get(fileInfo, m_lb->fileVarsStruct, 0, patch);
    }
    else{
      FileInfo* myFileInfo = scinew FileInfo();
      fileInfo.get() = myFileInfo;
    }

    std::map<string, FILE *> myFiles;

    if( fileInfo.get().get_rep() ){
      myFiles = fileInfo.get().get_rep()->files;
    }

    string udaDir = m_output->getOutputLocation();

    //__________________________________
    //  loop over control volumes
    for( size_t i=0; i< m_controlVols.size(); i++ ){
      controlVolume* cv = m_controlVols[i];


      string filename = udaDir + "/" + cv->getName() + ".dat";
      FILE *fp=nullptr;


      if( myFiles.count(filename) == 0 ){
        createFile(filename, fp, cv);
        myFiles[filename] = fp;
      }
      else {
        fp = myFiles[filename];
      }

      if (!fp){
        throw InternalError("\nERROR:dataAnalysisModule:fluxes:  failed opening file"+filename,__FILE__, __LINE__);
      }
      //__________________________________
      //  Write out the total and net fluxes
      sum_vartype    totalQ_CV;
      sumvec_vartype Q_flux;

      new_dw->get( totalQ_CV, m_lb->totalQ_CV[i] );
      new_dw->get( Q_flux,    m_lb->net_Q_faceFluxes[i] );

      // so fprintf can deal with it
      const double Q          = totalQ_CV;
      const Vector net_Q_flux = Q_flux;
      const int w = m_col_width;
      const int p = m_precision;

      fprintf(fp, "%-*.*E %-*.*E %-*.*E %-*.*E %-*.*E ",
              w, p, tv.now,
              w, p, Q,
              w, p, net_Q_flux.x(),
              w, p, net_Q_flux.y(),
              w, p, net_Q_flux.z() );

      //__________________________________
      //   write out each face flux
      for( auto f : cv->allFaces){
        sum_vartype Q_faceFlux;
        new_dw->get( Q_faceFlux, m_lb->Q_faceFluxes[i][f] );

        const double Q_ff = Q_faceFlux;
        fprintf(fp, "%-*.*E ",w, p, Q_ff);
      }
      fprintf(fp, "\n");

  //      fflush(fp);   If you want to write the data right now, no buffering.
    }
    //__________________________________
    // put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;
    new_dw->put(fileInfo, m_lb->fileVarsStruct, 0, patch);
  }
}

//______________________________________________________________________
//  Integrate fluxes over the control volume face
//______________________________________________________________________
//
template < class SFC_D >
void controlVolFluxes::integrate_Q_overFace( controlVolume::FaceType face,
                                             const controlVolume * cv,
                                             const Patch         * patch,
                                             faceQuantities      * faceQ,
                                             SFC_D               &  vel_FC,
                                             constCCVariable<double>& rho_CC,
                                             constCCVariable<Vector>& vel_CC)
{
  double faceArea = cv->getCellArea(face, patch);
  const int pDir  = cv->getFaceAxes( face )[0];

  double Q_flux( 0. );

  //__________________________________
  //  get the iterator on this face
  CellIterator iter = cv->getFaceIterator(face, controlVolume::SFC_Cells, patch);

 DOUT( dbg_OTF_CVF, std::right << setw(10) <<  " faceIter: " << setw(10) << iter );

  for(; !iter.done(); iter++) {
    IntVector c = *iter;
    double velNorm = vel_FC[c];  // normal component of Velocity ( V^\vec \dot n^\vec)

    // find upwind cell
    IntVector uw = c;
    if (velNorm > 0 ){
      uw[pDir] = c[pDir] - 1;
    }

    // One way to define m dot through face
    double mdot  =  velNorm * rho_CC[uw] * faceArea;

    // Another way
    // Vector mdotV  = faceArea * vel_CC[uw] * rho_CC[uw];

    Q_flux  += mdot;
  }
  faceQ->Q_faceFluxes[face] = Q_flux;

  DOUT( dbg_OTF_CVF, "     Face: " << cv->getFaceName(face) << "\t dir: " << pDir << " Q_Flux = " <<  Q_flux );
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
//______________________________________________________________________
void controlVolFluxes::createFile(string& filename,
                                  FILE*& fp,
                                  const controlVolume * cv)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen( filename.c_str(), "a" );
    return;
  }

  fp = fopen(filename.c_str(), "w");

  const int w = m_col_width;

  std::string mesg = cv->getExtents_string();
  fprintf(fp, "#%s \n", mesg.c_str() );

  fprintf(fp, "%-*s %-*s %-*s %-*s %-*s %-*s\n", w, "# ",
                                                 w, "ControlVol",
                                                 w, "Net face fluxes",
                                                 w, " ",
                                                 w, " ",
                                                 w, "Face fluxes");

  fprintf(fp,"%-*s %-*s %-*s %-*s %-*s", w, "#Time [s]",
                                         w, "Total",
                                         w, "X faces",
                                         w, "Y faces",
                                         w, "Z faces" );

  fprintf(fp," %-*s %-*s %-*s %-*s %-*s %-*s\n", w, "x-",
                                                w, "x+",
                                                w, "y-",
                                                w, "y+",
                                                w, "z-",
                                                w, "z+" );

  proc0cout << Parallel::getMPIRank() << " controlVolFluxes:Created file " << filename << endl;
}

//______________________________________________________________________
//    Initialize the face quantities
//______________________________________________________________________
void controlVolFluxes::initializeVars( faceQuantities* faceQ)
{
  const double  zero(0.0);

  for ( int f = 0; f < 6; f++ ){
    faceQ->Q_faceFluxes[f] = zero;
  }
}


//______________________________________________________________________
//
Vector controlVolFluxes::L_minus_R( std::map <int, double >& faceFlux)
{
  double X_flux  = faceFlux[0] - faceFlux[1];
  double Y_flux  = faceFlux[2] - faceFlux[3];
  double Z_flux  = faceFlux[4] - faceFlux[5];

  Vector net_flux(X_flux, Y_flux, Z_flux);
  return net_flux;
}

//______________________________________________________________________
//
Vector controlVolFluxes::L_minus_R( std::map <int, Vector >& faceFlux)
{
  double X_flux  = ( faceFlux[0].x() - faceFlux[1].x() ) +
                   ( faceFlux[2].x() - faceFlux[3].x() ) +
                   ( faceFlux[4].x() - faceFlux[5].x() );

  double Y_flux  = ( faceFlux[0].y() - faceFlux[1].y() ) +
                   ( faceFlux[2].y() - faceFlux[3].y() ) +
                   ( faceFlux[4].y() - faceFlux[5].y() );

  double Z_flux  = ( faceFlux[0].z() - faceFlux[1].z() ) +
                   ( faceFlux[2].z() - faceFlux[3].z() ) +
                   ( faceFlux[4].z() - faceFlux[5].z() );

  Vector net_flux(X_flux, Y_flux, Z_flux);
  return net_flux;
}

//______________________________________________________________________
//  For each CV create a vector of labels   desc_i
std::vector<VarLabel*>
controlVolFluxes::createLabels(std::string desc,
                               const Uintah::TypeDescription* td)
{
  std::vector<VarLabel*> labels;

  for( size_t i=0; i< m_controlVols.size(); i++ ){
    controlVolume* cv = m_controlVols[i];
    std::string cvName = cv->getName();

    string name = cvName + "_" + desc;
    VarLabel* l = VarLabel::create( name, td );

    labels.push_back(l);
  }
  return labels;
}
