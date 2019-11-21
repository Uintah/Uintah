/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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
#include <Core/Util/DebugStream.h>

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
//______________________________________________________________________

static DebugStream cout_doing("controlVolFluxes",   false);
static DebugStream cout_dbg("controlVolFluxes_dbg", false);

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

  labels = scinew MA_Labels();

  labels->lastCompTime      = VarLabel::create( "lastCompTime", max_vartype::getTypeDescription() );
  labels->fileVarsStruct    = VarLabel::create( "FileInfo_MA",  PerPatch<FileInfoP>::getTypeDescription() );
  labels->totalQ_CV         = VarLabel::create( "totalQ_CV",    sumvec_vartype::getTypeDescription() );
  labels->net_Q_faceFluxes  = VarLabel::create( "net_Q_fluxes", sumvec_vartype::getTypeDescription() );
}

//______________________________________________________________________
//
controlVolFluxes::~controlVolFluxes()
{
  cout_doing << " Doing: destorying fluxes " << endl;
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

  VarLabel::destroy( labels->lastCompTime );
  VarLabel::destroy( labels->fileVarsStruct );
  VarLabel::destroy( labels->totalQ_CV );
  VarLabel::destroy( labels->net_Q_faceFluxes );

  for( size_t i=0; i< d_controlVols.size(); i++ ){
    delete d_controlVols[i];
  }

  delete labels;
}

//______________________________________________________________________
//
void controlVolFluxes::problemSetup(const ProblemSpecP& ,
                                    const ProblemSpecP& ,
                                    GridP& grid,
                                    std::vector<std::vector<const VarLabel* > > &PState,
                                    std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  cout_doing << "Doing problemSetup \t\t\t\tfluxes" << endl;

  //__________________________________
  //  Read in timing information
  m_module_spec->require( "samplingFrequency", m_analysisFreq );
  m_module_spec->require( "timeStart",         d_startTime );
  m_module_spec->require( "timeStop",          d_stopTime );

  m_zeroMatl = scinew MaterialSubset();
  m_zeroMatl->add(0);
  m_zeroMatl->addReference();

  m_zeroMatlSet = scinew MaterialSet();
  m_zeroMatlSet->add(0);
  m_zeroMatlSet->addReference();

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
  labels->vel_CC    = assignLabel( "vel_CC" );
  labels->rho_CC    = assignLabel( "rho_CC" );

  labels->uvel_FC   = assignLabel( "uvel_FCME" );
  labels->vvel_FC   = assignLabel( "vvel_FCME" );
  labels->wvel_FC   = assignLabel( "wvel_FCME" );


  //__________________________________
  // Loop over each face and find the extents
  ProblemSpecP cv_ps = m_module_spec->findBlock("controlVolumes");
  if( ! cv_ps) {
    throw ProblemSetupException("ERROR: Couldn't find <controlVolumes> xml node", __FILE__, __LINE__);
  }

  for( ProblemSpecP box_ps = cv_ps->findBlock( "box" ); box_ps != nullptr; box_ps=box_ps->findNextBlock( "box" ) ) {
    controlVolume* cv  = scinew controlVolume(box_ps, grid);
    d_controlVols.push_back( cv );
  }
}

//______________________________________________________________________
//
void controlVolFluxes::scheduleInitialize( SchedulerP   & sched,
                                           const LevelP & level )
{
  printSchedule(level,cout_doing,"controlVolFluxes::scheduleInitialize");

  Task* t = scinew Task("controlVolFluxes::initialize",
                  this, &controlVolFluxes::initialize);

  t->computes( labels->lastCompTime );
  t->computes( labels->fileVarsStruct, m_zeroMatl );
  sched->addTask(t, m_zeroPatch, m_zeroMatlSet);
}

//______________________________________________________________________
//
void controlVolFluxes::initialize( const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset *,
                                         DataWarehouse  *,
                                         DataWarehouse  * new_dw )
{
  // initialize the control volumes
  const Level* level = getLevel(patches);
  for( size_t i=0; i< d_controlVols.size(); i++ ){
    controlVolume* cv = d_controlVols[i];
    cv->initialize(level);
    cv->print();
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialize");

    double tminus = d_startTime - 1.0/m_analysisFreq;
    new_dw->put( max_vartype(tminus), labels->lastCompTime );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo, labels->fileVarsStruct, 0, patch);

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
  sched->overrideVariableBehavior("FileInfo_MA", false, false, false, true, true);

  //__________________________________
  //  compute the total Q and net fluxes
  printSchedule( level,cout_doing,"controlVolFluxes::scheduleDoAnalysis" );

  Task* t0 = scinew Task( "controlVolFluxes::integrate_Q_overCV",
                     this,&controlVolFluxes::integrate_Q_overCV );

  Ghost::GhostType  gn  = Ghost::None;

  sched_TimeVars( t0, level, labels->lastCompTime, false );

  t0->requires( Task::NewDW, labels->vel_CC,    m_matl, gn );
  t0->requires( Task::NewDW, labels->rho_CC,    m_matl, gn );

  t0->requires( Task::NewDW, labels->uvel_FC,   m_matl, gn );
  t0->requires( Task::NewDW, labels->vvel_FC,   m_matl, gn );
  t0->requires( Task::NewDW, labels->wvel_FC,   m_matl, gn );

  t0->computes( labels->totalQ_CV );
  t0->computes( labels->net_Q_faceFluxes );

  sched->addTask( t0, level->eachPatch(), m_matlSet );

  //__________________________________
  //  Task that outputs the contributions
  Task* t1 = scinew Task("controlVolFluxes::doAnalysis",
                    this,&controlVolFluxes::doAnalysis );

  sched_TimeVars( t1, level, labels->lastCompTime, true );
  t1->requires( Task::OldDW, labels->fileVarsStruct, m_zeroMatl, gn, 0 );

  t1->requires( Task::NewDW, labels->totalQ_CV );
  t1->requires( Task::NewDW, labels->net_Q_faceFluxes );

  t1->computes( labels->fileVarsStruct, m_zeroMatl );
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
  bool itItTime = isItTime( old_dw, level, labels->lastCompTime );

  if( itItTime == false  || rts ) {
    return;
  }


  //__________________________________
  //  Loop over patches
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,cout_doing,"Doing controlVolFluxes::integrate_Q_overCV");
    
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    Ghost::GhostType gn = Ghost::None;
    new_dw->get(rho_CC,  labels->rho_CC,   m_matIdx, patch, gn,0);
    new_dw->get(vel_CC,  labels->vel_CC,   m_matIdx, patch, gn,0);
    new_dw->get(uvel_FC, labels->uvel_FC,  m_matIdx, patch, gn,0);
    new_dw->get(vvel_FC, labels->vvel_FC,  m_matIdx, patch, gn,0);
    new_dw->get(wvel_FC, labels->wvel_FC,  m_matIdx, patch, gn,0);

    double totalQ_CV = 0;

    //__________________________________
    //  loop over control volumes
    for( size_t i=0; i< d_controlVols.size(); i++ ){

      controlVolume* cv = d_controlVols[i];

      if ( !cv->controlVolume::hasBoundaryFaces( patch ) ) {
        continue;
      }

      //__________________________________
      //  Sum the total Q over the patch
      double  cellVol = patch->cellVolume();
      for (CellIterator iter=cv->getCellIterator( patch );!iter.done();iter++){
        IntVector c = *iter;
        totalQ_CV += rho_CC[c] * cellVol;
      }


      cout_dbg.precision(15);
      //__________________________________
      // Sum the fluxes passing through control volume surface

      vector<controlVolume::FaceType> bf;
      cv->getBoundaryFaces(bf, patch);

      for( vector<controlVolume::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        controlVolume::FaceType face = *itr;

        string faceName = cv->getFaceName( face );
        Vector norm     = cv->getFaceNormal( face );
        IntVector axis  = cv->getFaceAxes( face );


        cout_dbg << std::left << setw(7) << cv->getName()
                 << setw(7)   << "Face:"     << setw(7) << faceName
                 << setw(10)  << "faceType:" << setw(5) << face
                 << setw(7)   << "norm:"     << setw(5) << norm << std::left
                 << setw(15)  << "faceAxes:" << setw(7) <<  axis;
                 
        faceQuantities* faceQ = scinew faceQuantities;

        initializeVars( faceQ );         

        //__________________________________
        //           X faces
        if (face == controlVolume::xminus || face == controlVolume::xplus) {
          integrate_Q_overFace ( face, cv, patch, faceQ, uvel_FC, rho_CC, vel_CC);
        }

        //__________________________________
        //        Y faces
        if (face == controlVolume::yminus || face == controlVolume::yplus) {
          integrate_Q_overFace( face, cv, patch, faceQ, vvel_FC, rho_CC, vel_CC);
        }
        //__________________________________
        //        Z faces
        if (face == controlVolume::zminus || face == controlVolume::zplus) {
          integrate_Q_overFace( face, cv, patch, faceQ, wvel_FC, rho_CC, vel_CC);
        }
      }  // boundary faces
    }  // patch has faces

#if 0
    //__________________________________
    //  Now compute the net fluxes from all of the face quantites
    Vector net_Q_flux = L_minus_R( faceQ->Q_faceFluxes );

    //__________________________________
    // put in the dw
    //cout.precision(15);
    //cout <<  " Total CV momentum: " << totalQ_CV << " Net  convectiveFlux: " << net_convect_flux << endl;

    new_dw->put( sumvec_vartype( totalQ_CV ),     labels->totalQ_CV );
    new_dw->put( sumvec_vartype( net_Q_flux ),    labels->net_Q_faceFluxes );
#endif
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

  getTimeVars( old_dw, level, labels->lastCompTime, tv );
  putTimeVars( new_dw, labels->lastCompTime, tv );

  if( rts || tv.isItTime == false) {
    return;
  }

  //__________________________________
  //
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing doAnalysis");

    //__________________________________
    // open the struct that contains the file pointer map.  We use FileInfoP types
    // and store them in the DW to avoid doing system calls (SLOW).
    // Note: after regridding this may not exist for this patch in the old_dw
    PerPatch<FileInfoP> fileInfo;

    if( old_dw->exists( labels->fileVarsStruct, 0, patch ) ){
      old_dw->get(fileInfo, labels->fileVarsStruct, 0, patch);
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
    string filename = udaDir + "/" + "fluxes.dat";
    FILE *fp=nullptr;


    if( myFiles.count(filename) == 0 ){
      createFile(filename, fp);
      myFiles[filename] = fp;
    }
    else {
      fp = myFiles[filename];
    }

    if (!fp){
      throw InternalError("\nERROR:dataAnalysisModule:fluxes:  failed opening file"+filename,__FILE__, __LINE__);
    }
    //__________________________________
    //
    sumvec_vartype totalQ_CV;
    sumvec_vartype Q_flux;
    new_dw->get( totalQ_CV, labels->totalQ_CV );
    new_dw->get( Q_flux,    labels->net_Q_faceFluxes );

    // so fprintf can deal with it
    Vector Q          = totalQ_CV;
    Vector net_Q_flux = Q_flux;

    fprintf(fp, "%16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E\n",
                tv.now,
                (double)Q.x(),
                (double)Q.y(),
                (double)Q.z(),
                (double)net_Q_flux.x(),
                (double)net_Q_flux.y(),
                (double)net_Q_flux.z()
            );

//      fflush(fp);   If you want to write the data right now, no buffering.

    //__________________________________
    // put the file pointers into the DataWarehouse
    // these could have been altered. You must
    // reuse the Handle fileInfo and just replace the contents
    fileInfo.get().get_rep()->files = myFiles;
    new_dw->put(fileInfo, labels->fileVarsStruct, 0, patch);
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
  Vector norm     = cv->getFaceNormal( face );
  const int pDir  = cv->getFaceAxes( face )[0];
  
  double unitNormal = norm[pDir];  
  double Q_flux( 0. );

  //__________________________________
  //  get the iterator on this face
  controlVolume::FaceIteratorType MEC = controlVolume::MinusEdgeCells;
  CellIterator iter = cv->getFaceIterator(face, MEC, patch);

  cout_dbg << std::right << setw(10) <<  " faceIter: " << setw(10) << iter << endl;

  for(; !iter.done(); iter++) {
    IntVector c = *iter;
    double vel = vel_FC[c];

    // find upwind cell
    IntVector uw = c;
    if (vel > 0 ){
      uw[pDir] = c[pDir] - 1;
    }

    // One way to define m dot through face
    double mdot  =  vel * rho_CC[uw] * faceArea * unitNormal;

    // Another way
    // Vector mdotV  = faceArea * vel_CC[uw] * rho_CC[uw];

    Q_flux  += mdot;
  }
  faceQ->Q_faceFluxes[face] = Q_flux;

//  cout << "face: " << faceName << "\t dir: " << dir << " convect_Flux = " <<  convect_flux << endl;
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
//______________________________________________________________________
//
void controlVolFluxes::createFile(string& filename,
                                  FILE*& fp)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen( filename.c_str(), "a" );
    return;
  }

  fp = fopen(filename.c_str(), "w");
  fprintf(fp, "#                                                 total momentum in the control volume                                          Net convective momentum flux\n");
  fprintf(fp, "#Time                    CV_mom.x                 CV_mom.y                  CV_mom.z                  momFlux.x               momFlux.y                momFlux.z\n");


  proc0cout << Parallel::getMPIRank() << " fluxes:Created file " << filename << endl;
}


//______________________________________________________________________
//   This is a rip off of what's done in the boundary condition code
//______________________________________________________________________
//
void controlVolFluxes::faceInfo( const std::string fc,
                                 Vector& norm,
                                 int& p_dir,
                                 int& index)
{
  if (fc == "x-" || fc == "xminus"){
    norm = Vector(-1, 0, 0);
    p_dir = 0;
    index = 0;
    return;
  }
  if (fc == "x+" || fc == "xplus" ){
    norm = Vector(1, 0, 0);
    p_dir = 0;
    index = 1;
    return;
  }
  if (fc == "y-" || fc == "yminus" ){
    norm = Vector(0, -1, 0);
    p_dir = 1;
    index = 2;
    return;
  }
  if (fc == "y+" || fc == "yplus" ){
    norm = Vector(0, 1, 0);
    p_dir = 1;
    index = 3;
    return;
  }
  if (fc == "z-" || fc == "zminus" ){
    norm = Vector(0, 0, -1);
    p_dir = 2;
    index = 4;
    return;
  }
  if (fc == "z+" || fc == "zplus" ){
    norm = Vector(0, 0, 1);
    p_dir = 2;
    index = 5;
    return;
  }

  ostringstream warn;
  warn <<" ERROR:fluxes face name (" << fc << ") unknown. ";

  throw InternalError( warn.str(), __FILE__, __LINE__ );
}
//______________________________________________________________________
//  bulletProofing on the user inputs
//______________________________________________________________________
//
void controlVolFluxes::bulletProofing(GridP& grid,
                                      const string & side,
                                      const Point  & start,
                                      const Point  & end)
{
   //__________________________________
   // plane must be parallel to the coordinate system
   bool X = ( start.x() == end.x() );
   bool Y = ( start.y() == end.y() );  // 1 out of 3 of these must be true
   bool Z = ( start.z() == end.z() );

   bool validPlane = false;

   if( !X && !Y && Z ){
     validPlane = true;
   }
   if( !X && Y && !Z ){
     validPlane = true;
   }
   if( X && !Y && !Z ){
     validPlane = true;
   }

   if( validPlane == false ){
     ostringstream warn;
     warn << "\n ERROR:fluxes: the plane on face ("<< side
          << ") that you've specified " << start << " " << end
          << " is not parallel to the coordinate system. \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }

   //__________________________________
   //  plane must be on the edge of the domain
   validPlane = true;
   BBox compDomain;
   grid->getInteriorSpatialRange(compDomain);
   Point min = compDomain.min();
   Point max = compDomain.max();

   Point me = min;
   if (side == "x+" || side == "y+" || side == "z+" ){
     me = max;
   }

   if(side == "x+" || side == "x-" ){
     if(start.x() != me.x() ){
       validPlane = false;
     }
   }
   if(side == "y+" || side == "y-" ){
     if(start.y() != me.y() ){
       validPlane = false;
     }
   }
   if(side == "z+" || side == "z-" ){
     if(start.z() != me.z() ){
       validPlane = false;
     }
   }
   if( validPlane == false ){
     ostringstream warn;
     warn << "\n ERROR:controlVolFluxes: the plane on face ("<< side
          << ") that you've specified " << start << " to " << end
          << " is not at the edge of the computational domain. \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }

   //__________________________________
   //the plane can't exceed computational domain
   if( start.x() < min.x() || start.y() < min.y() ||start.z() < min.z() ||
       end.x() > max.x()   || end.y() > max.y()   || end.z() > max.z() ){
     ostringstream warn;
     warn << "\n ERROR:controlVolFluxes: a portion of plane that you've specified " << start
          << " " << end << " lies outside of the computational domain. \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }

   if( start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
     ostringstream warn;
     warn << "\n ERROR:controlVolFluxes: the plane that you've specified " << start
          << " " << end << " the starting point is > than the ending point \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }
}

//______________________________________________________________________
//  Find the VarLabel
//______________________________________________________________________
//
VarLabel* controlVolFluxes::assignLabel( const std::string& varName )
{
  VarLabel* myLabel  = VarLabel::find( varName );

  if( myLabel == nullptr ){
    ostringstream warn;
    warn << "ERROR fluxes One of the VarLabels for the analysis does not exist or could not be found\n"
         << varName << "  address: " << myLabel << "\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }

  return myLabel;
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

