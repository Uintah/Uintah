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

#include <CCA/Components/OnTheFlyAnalysis/momentumAnalysis.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
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
//     T O D O
//
//  Create a vector of control volumes.  Remove the assumption that
//  the entire domain is the CV.
//  This assumes that the control volume is cubic and aligned with the grid.
//  The face centered velocities are used to compute the fluxes through
//  the control surface.
//  This assumes that the variables all come from the new_dw
//  This assumes the entire computational domain is being used as the control volume!!!         <<<<<<<<<<<<,



static DebugStream cout_doing("momentumAnalysis",   false);
static DebugStream cout_dbg("momentumAnalysis_dbg", false);
//______________________________________________________________________
//______________________________________________________________________
//
momentumAnalysis::momentumAnalysis(ProblemSpecP& module_spec,
                                   SimulationStateP& sharedState,
                                   Output* dataArchiver)

  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_zeroMatl     = 0;
  d_zeroMatlSet  = 0;
  d_zeroPatch    = 0;
  d_matlIndx     = -9;
  d_pressIndx    = 0;              // For ICE/MPMICE it's always 0.
  
  d_pressMatl = scinew MaterialSubset();
  d_pressMatl->add(0);
  d_pressMatl->addReference();
  

  labels = scinew MA_Labels();

  labels->lastCompTime       = VarLabel::create( "lastCompTime",      max_vartype::getTypeDescription() );
  labels->fileVarsStruct     = VarLabel::create( "FileInfo_MA",       PerPatch<FileInfoP>::getTypeDescription() );
  labels->totalCVMomentum    = VarLabel::create( "totalCVMomentum",   sumvec_vartype::getTypeDescription() );
  labels->convectMom_fluxes  = VarLabel::create( "convectMom_fluxes", sumvec_vartype::getTypeDescription() );
  labels->viscousMom_fluxes  = VarLabel::create( "viscousMom_fluxes", sumvec_vartype::getTypeDescription() );
  labels->pressForces        = VarLabel::create( "pressForces",       sumvec_vartype::getTypeDescription() );
  labels->delT               = d_sharedState->get_delt_label();
}

//__________________________________
momentumAnalysis::~momentumAnalysis()
{
  cout_doing << " Doing: destorying momentumAnalysis " << endl;
  if( d_zeroMatlSet  && d_zeroMatlSet->removeReference() ) {
    delete d_zeroMatlSet;
  }
  if( d_zeroMatl && d_zeroMatl->removeReference() ) {
    delete d_zeroMatl;
  }
  if( d_zeroPatch && d_zeroPatch->removeReference() ) {
    delete d_zeroPatch;
  }
  if( d_matl_set && d_matl_set->removeReference() ) {
    delete d_matl_set;
  }
  if( d_pressMatl && d_pressMatl->removeReference() ) {
    delete d_pressMatl;
  }

  VarLabel::destroy( labels->lastCompTime );
  VarLabel::destroy( labels->fileVarsStruct );
  VarLabel::destroy( labels->totalCVMomentum );
  VarLabel::destroy( labels->convectMom_fluxes );
  VarLabel::destroy( labels->viscousMom_fluxes );
  VarLabel::destroy( labels->pressForces );

  delete labels;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
//______________________________________________________________________
//
void momentumAnalysis::problemSetup(const ProblemSpecP&,
                                 const ProblemSpecP& restart_prob_spec,
                                 GridP& grid,
                                 SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tmomentumAnalysis" << endl;

  if(!d_dataArchiver){
    throw InternalError("momentumAnalysis:couldn't get output port", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in timing information
  d_prob_spec->require( "samplingFrequency", d_analysisFreq );
  d_prob_spec->require( "timeStart",         d_StartTime );
  d_prob_spec->require( "timeStop",          d_StopTime );

  d_zeroMatl = scinew MaterialSubset();
  d_zeroMatl->add(0);
  d_zeroMatl->addReference();

  d_zeroMatlSet = scinew MaterialSet();
  d_zeroMatlSet->add(0);
  d_zeroMatlSet->addReference();

  // one patch
  const Patch* p = grid->getPatchByID(0,0);
  d_zeroPatch = scinew PatchSet();
  d_zeroPatch->add(p);
  d_zeroPatch->addReference();

  //__________________________________
  // find the material .  Default is matl 0.
  // The user can use either
  //  <material>   atmosphere </material>
  //  <materialIndex> 1 </materialIndex>
  Material* matl;
  if( d_prob_spec->findBlock("material") ){
    matl = d_sharedState->parseAndLookupMaterial( d_prob_spec, "material" );
  } else if ( d_prob_spec->findBlock("materialIndex") ){
    int indx;
    d_prob_spec->get("materialIndex", indx);
    matl = d_sharedState->getMaterial(indx);
  } else {
    matl = d_sharedState->getMaterial(0);
  }

  d_matlIndx = matl->getDWIndex();

  vector<int> m;
  m.push_back(0);            // matl index for FileInfo label
  m.push_back( d_matlIndx );
  
  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it;
  it = unique(m.begin(), m.end());
  m.erase(it, m.end());
  
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // HARDWIRED FOR ICE/MPMICE
  labels->vel_CC    = assignLabel( "vel_CC" );
  labels->rho_CC    = assignLabel( "rho_CC" );

  labels->uvel_FC   = assignLabel( "uvel_FCME" );
  labels->vvel_FC   = assignLabel( "vvel_FCME" );
  labels->wvel_FC   = assignLabel( "wvel_FCME" );

  labels->pressX_FC = assignLabel( "pressX_FC" );
  labels->pressY_FC = assignLabel( "pressY_FC" );
  labels->pressZ_FC = assignLabel( "pressZ_FC" );

  labels->tau_X_FC  = assignLabel( "tau_X_FC" );
  labels->tau_Y_FC  = assignLabel( "tau_Y_FC" );
  labels->tau_Z_FC  = assignLabel( "tau_Z_FC" );


  //__________________________________
  // Loop over each face and find the extents
  ProblemSpecP ma_ps = d_prob_spec->findBlock("controlVolume");
  if(! ma_ps) {
    throw ProblemSetupException("ERROR Radiometer: Couldn't find <controlVolume> xml node", __FILE__, __LINE__);
  }

  for( ProblemSpecP face_ps = ma_ps->findBlock( "Face" ); face_ps != nullptr; face_ps=face_ps->findNextBlock( "Face" ) ) {

    map<string,string> faceMap;
    face_ps->getAttributes(faceMap);

    string side = faceMap["side"];
    int p_dir;
    int index;
    Vector norm;
    Point start(-9,-9,-9);
    Point end(-9,-9,-9);
    FaceType type=none;

    faceInfo(side, norm, p_dir, index);

    if (faceMap["extents"] == "partialFace"){

      face_ps->get( "startPt", start );
      face_ps->get( "endPt",   end );
      type = partialFace;

      bulletProofing(grid, side, start, end);
    }else{
      type = entireFace;
    }

    // put the input variables into the global struct
    cv_face* cvFace    = scinew cv_face;
    cvFace->p_dir      = p_dir;
    cvFace->normalDir  = norm;
    cvFace->face       = type;
    cvFace->startPt    = start;
    cvFace->endPt      = end;
    d_cv_faces[index]  = cvFace;
  }
}
//______________________________________________________________________
//
//______________________________________________________________________
//
void momentumAnalysis::scheduleInitialize( SchedulerP& sched,
                                        const LevelP& level )
{
  printSchedule(level,cout_doing,"momentumAnalysis::scheduleInitialize");

  Task* t = scinew Task("momentumAnalysis::initialize",
                  this, &momentumAnalysis::initialize);

  t->computes( labels->lastCompTime );
  t->computes( labels->fileVarsStruct, d_zeroMatl );
  sched->addTask(t, d_zeroPatch, d_zeroMatlSet);
}
//______________________________________________________________________
//
//______________________________________________________________________
//
void momentumAnalysis::initialize( const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw )
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialize");

    double tminus = -1.0/d_analysisFreq;
    new_dw->put( max_vartype(tminus), labels->lastCompTime );

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;

    new_dw->put(fileInfo,    labels->fileVarsStruct, 0, patch);

    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == nullptr){
        ostringstream warn;
        warn << "ERROR:momentumAnalysis  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }
  }
}

void momentumAnalysis::restartInitialize()
{
}

//______________________________________________________________________
//
//______________________________________________________________________
//
void momentumAnalysis::scheduleDoAnalysis(SchedulerP& sched,
                                        const LevelP& level)
{

  // Tell the scheduler to not copy this variable to a new AMR grid and
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo_MA", false, false, false, true, true);

  //__________________________________
  //  compute the total momentum and fluxes
  printSchedule( level,cout_doing,"momentumAnalysis::scheduleDoAnalysis" );

  Task* t0 = scinew Task( "momentumAnalysis::integrateMomentumField",
                     this,&momentumAnalysis::integrateMomentumField );

  Ghost::GhostType  gn  = Ghost::None;

  MaterialSubset* matl_SS = scinew MaterialSubset();
  matl_SS->add( d_matlIndx );
  matl_SS->addReference();

  t0->requires( Task::OldDW, labels->lastCompTime );
  t0->requires( Task::OldDW, labels->delT, level.get_rep() );

  t0->requires( Task::NewDW, labels->vel_CC,    matl_SS, gn );
  t0->requires( Task::NewDW, labels->rho_CC,    matl_SS, gn );

  t0->requires( Task::NewDW, labels->uvel_FC,   matl_SS, gn );
  t0->requires( Task::NewDW, labels->vvel_FC,   matl_SS, gn );
  t0->requires( Task::NewDW, labels->wvel_FC,   matl_SS, gn );

  t0->requires( Task::NewDW, labels->pressX_FC, d_pressMatl, gn );
  t0->requires( Task::NewDW, labels->pressY_FC, d_pressMatl, gn );
  t0->requires( Task::NewDW, labels->pressZ_FC, d_pressMatl, gn );

  t0->requires( Task::NewDW, labels->tau_X_FC,  matl_SS, gn );
  t0->requires( Task::NewDW, labels->tau_Y_FC,  matl_SS, gn );
  t0->requires( Task::NewDW, labels->tau_Z_FC,  matl_SS, gn );

  t0->computes( labels->totalCVMomentum );
  t0->computes( labels->convectMom_fluxes );
  t0->computes( labels->viscousMom_fluxes );
  t0->computes( labels->pressForces );  

  sched->addTask( t0, level->eachPatch(), d_matl_set );

  //__________________________________
  //  Task that outputs the contributions
  Task* t1 = scinew Task("momentumAnalysis::doAnalysis",
                    this,&momentumAnalysis::doAnalysis );

  t1->requires( Task::OldDW, labels->lastCompTime );
  t1->requires( Task::OldDW, labels->fileVarsStruct, d_zeroMatl, gn, 0 );

  t1->requires( Task::NewDW, labels->totalCVMomentum );
  t1->requires( Task::NewDW, labels->convectMom_fluxes );
  t1->requires( Task::NewDW, labels->viscousMom_fluxes );
  t1->requires( Task::NewDW, labels->pressForces );

  t1->computes( labels->lastCompTime );
  t1->computes( labels->fileVarsStruct, d_zeroMatl );
  sched->addTask( t1, d_zeroPatch, d_zeroMatlSet);        // you only need to schedule patch 0 since all you're doing is writing out data
}

//______________________________________________________________________
//  Compute the total momentum of the control volume, the fluxes passing
//  through the control surfaces and the pressure forces
//______________________________________________________________________
//
void momentumAnalysis::integrateMomentumField(const ProcessorGroup* pg,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matl_sub ,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{

  const Level* level = getLevel(patches);
  max_vartype analysisTime;
  delt_vartype delT;

  old_dw->get( analysisTime, labels->lastCompTime );
  old_dw->get( delT,         labels->delT ,level);

  double lastCompTime = analysisTime;
  double nextCompTime = lastCompTime + 1.0/d_analysisFreq;
  double now = d_sharedState->getElapsedSimTime();

  bool tsr = new_dw->timestepRestarted();  // ignore if a timestep restart has been requested.

  if( now < nextCompTime  || tsr ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing momentumAnalysis::integrateMomentumField");

    Vector totalCVMomentum = Vector(0.,0.,0.);

    faceQuantities* faceQ = scinew faceQuantities;

    initializeVars( faceQ );

    Vector dx = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;

    Ghost::GhostType gn  = Ghost::None;
    new_dw->get(rho_CC,    labels->rho_CC,     d_matlIndx, patch, gn,0);
    new_dw->get(vel_CC,    labels->vel_CC,     d_matlIndx, patch, gn,0);

    //__________________________________
    //  Sum the total momentum over the patch
    // This assumes the entire computational domain is being used as the control volume!!!         <<<<<<<<<<<<,
    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      totalCVMomentum += rho_CC[c] * vol * vel_CC[c];
    }

    //__________________________________
    //
    if ( patch->hasBoundaryFaces() ) {

      constSFCXVariable<double> uvel_FC, pressX_FC;
      constSFCYVariable<double> vvel_FC, pressY_FC;
      constSFCZVariable<double> wvel_FC, pressZ_FC;

      constSFCXVariable<Vector> tau_X_FC;
      constSFCYVariable<Vector> tau_Y_FC;
      constSFCZVariable<Vector> tau_Z_FC;

      new_dw->get(uvel_FC,   labels->uvel_FC,    d_matlIndx, patch, gn,0);
      new_dw->get(vvel_FC,   labels->vvel_FC,    d_matlIndx, patch, gn,0);
      new_dw->get(wvel_FC,   labels->wvel_FC,    d_matlIndx, patch, gn,0);

      new_dw->get(pressX_FC, labels->pressX_FC,  d_pressIndx, patch, gn,0);
      new_dw->get(pressY_FC, labels->pressY_FC,  d_pressIndx, patch, gn,0);
      new_dw->get(pressZ_FC, labels->pressZ_FC,  d_pressIndx, patch, gn,0);

      new_dw->get(tau_X_FC,  labels->tau_X_FC,   d_matlIndx, patch, gn,0);
      new_dw->get(tau_Y_FC,  labels->tau_Y_FC,   d_matlIndx, patch, gn,0);
      new_dw->get(tau_Z_FC,  labels->tau_Z_FC,   d_matlIndx, patch, gn,0);


      cout_dbg.precision(15);
      //__________________________________
      // Sum the fluxes passing through control volume surface
      // and the pressure forces
      vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);

      for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

        Patch::FaceType face = *itr;
        string faceName = patch->getFaceName(face );
        cv_face* cvFace = d_cv_faces[face];

        cout_dbg << "\ncvFace: " <<  faceName << " faceType " << cvFace->face
                 << " startPt: " << cvFace->startPt << " endPt: " << cvFace->endPt << endl;
        cout_dbg << "          norm: " << cvFace->normalDir << " p_dir: " << cvFace->p_dir << endl;

        // define the iterator on this face  The default is the entire face
        Patch::FaceIteratorType SFC = Patch::SFCVars;
        CellIterator iterLimits=patch->getFaceIterator(face, SFC);

        if( cvFace->face == partialFace ){

          IntVector lo  = level->getCellIndex( cvFace->startPt );
          IntVector hi  = level->getCellIndex( cvFace->endPt );
          IntVector pLo = patch->getCellLowIndex();
          IntVector pHi = patch->getCellHighIndex();

          IntVector low  = Max(lo, pLo);    // find the intersection
          IntVector high = Min(hi, pHi);

          iterLimits = CellIterator(low,high);
        }


        //__________________________________
        //           X faces
        if (face == Patch::xminus || face == Patch::xplus) {
          double area = dx.y() * dx.z();
          integrateOverFace ( faceName, area, iterLimits, faceQ, uvel_FC, pressX_FC, tau_X_FC, rho_CC, vel_CC);
        }

        //__________________________________
        //        Y faces
        if (face == Patch::yminus || face == Patch::yplus) {
          double area = dx.x() * dx.z();
          integrateOverFace( faceName, area, iterLimits,  faceQ, vvel_FC, pressY_FC, tau_Y_FC, rho_CC, vel_CC);
        }
        //__________________________________
        //        Z faces
        if (face == Patch::zminus || face == Patch::zplus) {
          double area = dx.x() * dx.y();
          integrateOverFace( faceName, area, iterLimits,  faceQ, wvel_FC, pressZ_FC, tau_Z_FC, rho_CC, vel_CC);

        }

      }  // boundary faces

    }  // patch has faces


    //__________________________________
    //  Now compute the net fluxes from the face quantites
    Vector net_convect_flux = L_minus_R( faceQ->convect_faceFlux );

    Vector net_viscous_flux = L_minus_R( faceQ->viscous_faceFlux );

    // net force on control volume due to pressure forces
    map<int, double> pressForce = faceQ->pressForce_face;  // for readability
    double pressForceX = pressForce[0] - pressForce[1];
    double pressForceY = pressForce[2] - pressForce[3];
    double pressForceZ = pressForce[4] - pressForce[5];

    Vector net_press_forces( pressForceX, pressForceY, pressForceZ );

    //__________________________________
    // put in the dw
    //cout.precision(15);
    //cout <<  " Total CV momentum: " << totalCVMomentum << " Net  convectiveFlux: " << net_convect_flux << " viscousFlux: " << net_viscous_flux << " pressForce " << net_press_forces << endl;

    new_dw->put( sumvec_vartype( totalCVMomentum ),     labels->totalCVMomentum );
    new_dw->put( sumvec_vartype( net_convect_flux ),    labels->convectMom_fluxes );
    new_dw->put( sumvec_vartype( net_viscous_flux ),    labels->viscousMom_fluxes );
    new_dw->put( sumvec_vartype( net_press_forces ),    labels->pressForces );
  }  // patch loop
}

//______________________________________________________________________
//
//______________________________________________________________________
//
void momentumAnalysis::doAnalysis(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls ,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{

  // ignore task if a timestep restart has been requested upstream
  bool tsr = new_dw->timestepRestarted();
  if( tsr ){
    return;
  }

  max_vartype lastTime;
  old_dw->get( lastTime, labels->lastCompTime );

  double now      = d_sharedState->getElapsedSimTime();
  double nextTime = lastTime + ( 1.0 / d_analysisFreq );

  double time_dw  = lastTime;
  if( now >= nextTime ){

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
      }else{
        FileInfo* myFileInfo = scinew FileInfo();
        fileInfo.get() = myFileInfo;
      }

      std::map<string, FILE *> myFiles;

      if( fileInfo.get().get_rep() ){
        myFiles = fileInfo.get().get_rep()->files;
      }

      string udaDir = d_dataArchiver->getOutputLocation();
      string filename = udaDir + "/" + "momentumAnalysis.dat";
      FILE *fp=nullptr;


      if( myFiles.count(filename) == 0 ){
        createFile(filename, fp);
        myFiles[filename] = fp;

      } else {
        fp = myFiles[filename];
      }

      if (!fp){
        throw InternalError("\nERROR:dataAnalysisModule:momentumAnalysis:  failed opening file"+filename,__FILE__, __LINE__);
      }
      //__________________________________
      //
      sumvec_vartype totalCVMomentum, convectFlux, viscousFlux, pressForce;
      new_dw->get( totalCVMomentum, labels->totalCVMomentum );
      new_dw->get( convectFlux,      labels->convectMom_fluxes );
      new_dw->get( viscousFlux,      labels->viscousMom_fluxes );
      new_dw->get( pressForce,       labels->pressForces );

      // so fprintf can deal with it
      Vector momentum = totalCVMomentum;
      Vector conFlux = convectFlux;
      Vector visFlux = viscousFlux;
      Vector pForce  = pressForce;

      fprintf(fp, "%16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E,   %16.15E\n", 
                  now,
                  (double)momentum.x(),
                  (double)momentum.y(),
                  (double)momentum.z(),
                  (double)conFlux.x(),
                  (double)conFlux.y(),
                  (double)conFlux.z(),
                  (double)visFlux.x(),
                  (double)visFlux.y(),
                  (double)visFlux.z(),
                  (double)pForce.x(),
                  (double)pForce.y(),
                  (double)pForce.z()
              );

//      fflush(fp);   If you want to write the data right now, no buffering.
      time_dw = now;

      //__________________________________
      // put the file pointers into the DataWarehouse
      // these could have been altered. You must
      // reuse the Handle fileInfo and just replace the contents
      fileInfo.get().get_rep()->files = myFiles;

      new_dw->put(fileInfo, labels->fileVarsStruct, 0, patch);
    }
  }
  new_dw->put(max_vartype( time_dw ), labels->lastCompTime);
}

//______________________________________________________________________
//  Integrate fluxes/forces over the control volume face
//______________________________________________________________________
//
template < class SFC_D, class SFC_V >
void momentumAnalysis::integrateOverFace( const std::string faceName,
                                          const double faceArea,
                                          CellIterator iter,
                                          faceQuantities* faceQ,
                                          SFC_D&  vel_FC,
                                          SFC_D&  press_FC,
                                          SFC_V&  tau_FC,
                                          constCCVariable<double>& rho_CC,
                                          constCCVariable<Vector>& vel_CC)
{
  int dir;    // x, y or z
  int f;      // face index
  Vector norm;
  faceInfo(faceName, norm, dir, f);

  Vector convect_flux( 0. );
  Vector viscous_flux( 0. );
  double pressForce( 0. );

  //__________________________________
  //  Loop over a face
  for(; !iter.done(); iter++) {
    IntVector c = *iter;
    double vel = vel_FC[c];

    // find upwind cell
    IntVector uw = c;
    if (vel > 0 ){
      uw[dir] = c[dir] - 1;
    }

    // One way to define m dot through face
    double mdot  =  vel * faceArea * rho_CC[uw];

    // Another way
    // Vector mdotV  = faceArea * vel_CC[uw] * rho_CC[uw];
    // double mdot = mdotV[dir];

    convect_flux  += mdot * vel_CC[uw];
    viscous_flux  += ( faceArea * tau_FC[c] );
    pressForce    += ( faceArea * press_FC[c] );

  }
  faceQ->convect_faceFlux[f] = convect_flux;
  faceQ->viscous_faceFlux[f] = viscous_flux;
  faceQ->pressForce_face[f]  = pressForce;

//  cout << "face: " << faceName << "\t dir: " << dir << " convect_Flux = " <<  convect_flux << " ViscousFlux " << viscous_flux << " pressForce " << pressForce << endl;
}

//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
//______________________________________________________________________
//
void momentumAnalysis::createFile(string& filename,  FILE*& fp)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen( filename.c_str(), "a" );
    return;
  }

  fp = fopen(filename.c_str(), "w");
  fprintf(fp, "#                                                 total momentum in the control volume                                          Net convective momentum flux                                               net viscous flux                                                             pressure force on control vol.\n");                                                                      
  fprintf(fp, "#Time                    CV_mom.x                 CV_mom.y                  CV_mom.z                  momFlux.x               momFlux.y                momFlux.z                 visFlux.x                 visFlux.y                visFlux.z                 pressForce.x              pressForce.y             pressForce.z\n");

  
  proc0cout << Parallel::getMPIRank() << " momentumAnalysis:Created file " << filename << endl;
}


//______________________________________________________________________
//   This is a rip off of what's done in the boundary condition code
//______________________________________________________________________
//
void momentumAnalysis::faceInfo( const std::string fc,
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
  warn <<" ERROR:MomentumAnalysis face name (" << fc << ") unknown. ";

  throw InternalError( warn.str(), __FILE__, __LINE__ );
}
//______________________________________________________________________
//  bulletProofing on the user inputs
//______________________________________________________________________
//
void momentumAnalysis::bulletProofing(GridP& grid,
                                    const string& side,
                                    const Point& start,
                                    const Point& end)
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
     warn << "\n ERROR:momentumAnalysis: the plane on face ("<< side
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
     warn << "\n ERROR:1stLawThermo: the plane on face ("<< side
          << ") that you've specified " << start << " to " << end
          << " is not at the edge of the computational domain. \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }

   //__________________________________
   //the plane can't exceed computational domain
   if( start.x() < min.x() || start.y() < min.y() ||start.z() < min.z() ||
       end.x() > max.x()   || end.y() > max.y()   || end.z() > max.z() ){
     ostringstream warn;
     warn << "\n ERROR:1stLawThermo: a portion of plane that you've specified " << start
          << " " << end << " lies outside of the computational domain. \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }

   if( start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
     ostringstream warn;
     warn << "\n ERROR:1stLawThermo: the plane that you've specified " << start
          << " " << end << " the starting point is > than the ending point \n" << endl;
     throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
   }
}

//______________________________________________________________________
//  Find the VarLabel
//______________________________________________________________________
//
VarLabel* momentumAnalysis::assignLabel( const std::string& varName )
{
  VarLabel* myLabel  = VarLabel::find( varName );

  if( myLabel == nullptr ){
    ostringstream warn;
    warn << "ERROR momentumAnalysis One of the VarLabels for the analysis does not exist or could not be found\n"
         << varName << "  address: " << myLabel << "\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }

  return myLabel;
}


//______________________________________________________________________
//    Initialize the face quantities
//______________________________________________________________________
//
void momentumAnalysis::initializeVars( faceQuantities* faceQ)
{
  for ( int f = 0; f < 6; f++ ){
    faceQ->convect_faceFlux[f] = Vector( 0. ) ;
    faceQ->viscous_faceFlux[f] = Vector( 0. ) ;
    faceQ->pressForce_face[f]  = 0.;
  }
}

//______________________________________________________________________
//
//______________________________________________________________________
//
Vector momentumAnalysis::L_minus_R( std::map <int, Vector >& faceFlux)
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

