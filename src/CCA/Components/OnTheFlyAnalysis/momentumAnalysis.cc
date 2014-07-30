/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
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
//  Need to add viscous and pressure forces on the faces
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

  labels = scinew MA_Labels();

  labels->lastCompTime    = VarLabel::create( "lastCompTime",     max_vartype::getTypeDescription() );
  labels->fileVarsStruct  = VarLabel::create( "FileInfo_MA",      PerPatch<FileInfoP>::getTypeDescription() );
  labels->totalCVMomentum = VarLabel::create( "totalCVMomentum",  sumvec_vartype::getTypeDescription() );
  labels->CS_fluxes       = VarLabel::create( "CS_Fluxes",        sumvec_vartype::getTypeDescription() );
  labels->delT            = d_sharedState->get_delt_label();
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

  VarLabel::destroy( labels->lastCompTime );
  VarLabel::destroy( labels->fileVarsStruct );
  VarLabel::destroy( labels->totalCVMomentum );
  VarLabel::destroy( labels->CS_fluxes );
  VarLabel::destroy( labels->delT );
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
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  //__________________________________
  //  read in the VarLabel names
  string uvelFC  = "NULL";
  string vvelFC  = "NULL";
  string wvelFC  = "NULL";
  string velCC  = "NULL";
  string rhoCC  = "NULL";

  if ( d_prob_spec->findBlock( "uvel_FC" ) ){ 
    d_prob_spec->findBlock( "uvel_FC" )->getAttribute( "label", uvelFC ); 
  } 
  if ( d_prob_spec->findBlock( "vvel_FC" ) ){ 
    d_prob_spec->findBlock( "vvel_FC" )->getAttribute( "label", vvelFC ); 
  }
  if ( d_prob_spec->findBlock( "wvel_FC" ) ){ 
    d_prob_spec->findBlock( "wvel_FC" )->getAttribute( "label", wvelFC ); 
  }
  if ( d_prob_spec->findBlock( "vel_CC" ) ){ 
    d_prob_spec->findBlock( "vel_CC" )->getAttribute( "label", velCC ); 
  }
  if ( d_prob_spec->findBlock( "rho_CC" ) ){ 
    d_prob_spec->findBlock( "rho_CC" )->getAttribute( "label", rhoCC ); 
  }  

  //__________________________________
  //  bulletproofing
  labels->uvel_FC  = VarLabel::find( uvelFC );
  labels->vvel_FC  = VarLabel::find( vvelFC );
  labels->wvel_FC  = VarLabel::find( wvelFC );
  labels->vel_CC  = VarLabel::find( velCC );
  labels->rho_CC  = VarLabel::find( rhoCC );
  
  if( labels->uvel_FC == NULL || labels->uvel_FC == NULL || labels->uvel_FC == NULL ||
      labels->vel_CC == NULL  || labels->rho_CC == NULL ){
    ostringstream warn;
    warn << "ERROR momentumAnalysis One of the VarLabels need to do the analysis does not exist\n"
         << "    uvelFC address: " << labels->uvel_FC << "\n"
         << "    vvelFC:         " << labels->vvel_FC << "\n"
         << "    wvelFC:         " << labels->wvel_FC << "\n"
         << "    vel_CC:         " << labels->vel_CC << "\n"
         << "    rho_CC:         " << labels->rho_CC << "\n";
    throw InternalError(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  // Loop over each face and find the extents
  ProblemSpecP ma_ps = d_prob_spec->findBlock("controlVolume");
  if(! ma_ps) {
    throw ProblemSetupException("ERROR Radiometer: Couldn't find <controlVolume> xml node", __FILE__, __LINE__);
  }
  

  for (ProblemSpecP face_ps = ma_ps->findBlock("Face");
      face_ps != 0; face_ps=face_ps->findNextBlock("Face")) {

    map<string,string> faceMap;
    face_ps->getAttributes(faceMap);

    string side = faceMap["side"];
    int p_dir;
    Vector norm;
    Patch::FaceType f;
    Point start(-9,-9,-9);
    Point end(-9,-9,-9);
    FaceType type=none;

    faceInfo(side, f, norm, p_dir);

    if (faceMap["extents"] == "partialFace"){

      face_ps->get( "startPt", start );
      face_ps->get( "endPt",   end );
      type = partialFace;

      bulletProofing(grid, side, start, end);
    }else{
      type = entireFace;
    }
    
    // put the input variables into the global struct
    cv_face* cvFace   = scinew cv_face;
    cvFace->p_dir     = p_dir;
    cvFace->normalDir = norm;
    cvFace->face      = type;
    cvFace->startPt   = start;
    cvFace->endPt     = end;
    d_cv_faces[f]     = cvFace;
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
      if ( check == NULL){
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

  t0->requires( Task::NewDW, labels->vel_CC,   matl_SS, gn );   
  t0->requires( Task::NewDW, labels->rho_CC,   matl_SS, gn );   
  t0->requires( Task::NewDW, labels->uvel_FC,  matl_SS, gn );   
  t0->requires( Task::NewDW, labels->vvel_FC,  matl_SS, gn );
  t0->requires( Task::NewDW, labels->wvel_FC,  matl_SS, gn );

  t0->computes( labels->totalCVMomentum );
  t0->computes( labels->CS_fluxes );

  sched->addTask( t0, level->eachPatch(), d_matl_set );

  //__________________________________
  //  Task that outputs the contributions
  Task* t1 = scinew Task("momentumAnalysis::doAnalysis",
                    this,&momentumAnalysis::doAnalysis );

  t1->requires( Task::OldDW, labels->lastCompTime );
  t1->requires( Task::OldDW, labels->fileVarsStruct, d_zeroMatl, gn, 0 );

  t1->requires( Task::NewDW, labels->totalCVMomentum );
  t1->requires( Task::NewDW, labels->CS_fluxes );

  t1->computes( labels->lastCompTime );
  t1->computes( labels->fileVarsStruct, d_zeroMatl );
  sched->addTask( t1, d_zeroPatch, d_zeroMatlSet);        // you only need to schedule this  patch 0 since all you're doing is writing out data
}

//______________________________________________________________________
//  Compute the total momentum of the control volume and the fluxes passing
//  through the control surfaces.
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
  double now = d_dataArchiver->getCurrentTime();

  if( now < nextCompTime  ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing momentumAnalysis::integrateMomentumField");
    
    constCCVariable<double> rho_CC;
    constCCVariable<Vector> vel_CC;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    Vector dx = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();

    Ghost::GhostType gn  = Ghost::None;
    new_dw->get(rho_CC,  labels->rho_CC,     d_matlIndx, patch, gn,0);  
    new_dw->get(vel_CC,  labels->vel_CC,     d_matlIndx, patch, gn,0);  
    new_dw->get(uvel_FC, labels->uvel_FC,    d_matlIndx, patch, gn,0);
    new_dw->get(vvel_FC, labels->vvel_FC,    d_matlIndx, patch, gn,0);
    new_dw->get(wvel_FC, labels->wvel_FC,    d_matlIndx, patch, gn,0);

    Vector totalCVMomentum = Vector(0.,0.,0.);
    Vector total_flux      = Vector(0.,0.,0.);

    //__________________________________
    //  Sum the total momentum over the patch
    // This assumes the entire computational domain is being used as the control volume!!!         <<<<<<<<<<<<,
    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      totalCVMomentum = rho_CC[c] * vol * vel_CC[c];
    }

    cout_dbg.precision(15);
    //__________________________________
    // Sum the fluxes passing through control volume surface
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){

      Patch::FaceType face = *itr;
      string faceName = patch->getFaceName(face );
      cv_face* cvFace = d_cv_faces[face];

      cout_dbg << "\ncvFace: " <<  faceName << " faceType " << cvFace->face
               << " startPt: " << cvFace->startPt << " endPt: " << cvFace->endPt << endl;
      cout_dbg << "          norm: " << cvFace->normalDir << " p_dir: " << cvFace->p_dir << endl;

      // define the iterator on this face  The defauls is the entire face
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

      IntVector axes = patch->getFaceAxes(face);
      int P_dir = axes[0];  // principal direction
      double plus_minus_one = (double) patch->faceDirection(face)[P_dir];

      cout_dbg << "    face Direction " << patch->faceDirection(face) << endl;

      //__________________________________
      //           X faces
      if (face == Patch::xminus || face == Patch::xplus) {
        double area = dx.y() * dx.z();
        Vector sumMom(0.);
        cout_dbg << "    iterLimits: " << iterLimits << endl;

        for(CellIterator iter = iterLimits; !iter.done();iter++) {
          IntVector c = *iter;
          double vel = uvel_FC[c];

          // find upwind cell
          IntVector uw = c;
          if (vel > 0 ){
            uw.x( uw.x() -1 );
          }

          double mdot   = plus_minus_one * vel * area * rho_CC[uw];
          sumMom += mdot * vel_CC[uw];
          //cout << "face: " << faceName << " c: " << c << " offset: " << offset << " vel = " << vel << " mdot = " << mdot << endl;
        }
        total_flux += sumMom;
        cout_dbg << "    face: " << faceName << " mdot = " << sumMom << endl;
      }

      //__________________________________
      //        Y faces
      if (face == Patch::yminus || face == Patch::yplus) {
        double area = dx.x() * dx.z();
        Vector sumMom(0.);
        cout_dbg << "    iterLimits: " << iterLimits << endl;


        for(CellIterator iter = iterLimits; !iter.done();iter++) {
          IntVector c = *iter;
          double vel = vvel_FC[c];

          // find upwind cell
          IntVector uw = c;
          if (vel > 0 ){
            uw.y( uw.y() -1 );
          }

          double mdot   = plus_minus_one * vel * area * rho_CC[uw];
          sumMom += mdot * vel_CC[uw];

          //cout << "face: " << faceName << " c: " << c << " offset: " << offset << " vel = " << vel << " mdot = " << mdot << endl;
        }
        total_flux += sumMom;
        cout_dbg << "    face: " << faceName << " mdot = "<< sumMom << endl;;
      }

      //__________________________________
      //        Z faces
      if (face == Patch::zminus || face == Patch::zplus) {
        double area = dx.x() * dx.y();
        Vector sumMom(0.);
        cout_dbg << "    iterLimits: " << iterLimits << endl;

        for(CellIterator iter = iterLimits; !iter.done();iter++) {
          IntVector c = *iter;
          double vel = wvel_FC[c];
          
          // find upwind cell
          IntVector uw = c;
          if (vel > 0 ){
            uw.z( uw.z() -1 );
          }

          // compute the average values
          IntVector cc = c - IntVector(0,0,1);

          double mdot   = plus_minus_one * vel * area * rho_CC[uw];
          sumMom += mdot * vel_CC[uw];
        }
        total_flux += sumMom;
        cout_dbg << "    face: " << faceName << " mdot = "<< sumMom << endl;;
      }
    }  // boundary faces

    cout_dbg << "Patch: " << patch->getID() << " totalFlux: " << total_flux <<endl;

    new_dw->put( sumvec_vartype(totalCVMomentum), labels->totalCVMomentum );
    new_dw->put( sumvec_vartype(total_flux),      labels->CS_fluxes );
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
  max_vartype lastTime;
  old_dw->get( lastTime, labels->lastCompTime );

  double now = d_dataArchiver->getCurrentTime();
  double nextTime = lastTime + 1.0/d_analysisFreq;

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
      FILE *fp=NULL;


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
      sumvec_vartype totalCVMomentum, total_flux;
      new_dw->get( totalCVMomentum, labels->totalCVMomentum );
      new_dw->get( total_flux,      labels->CS_fluxes );

      Vector momentum = totalCVMomentum;
      Vector flux = total_flux;

      fprintf(fp, "%16.15E      %16.15E       %16.15E       %16.15E      %16.15E       %16.15E       %16.15E\n", now,
                  (double)momentum.x(),
                  (double)momentum.y(),
                  (double)momentum.z(),
                  (double)flux.x(),
                  (double)flux.y(),
                  (double)flux.z() );

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
  fprintf( fp,"# Definitions:\n");
  fprintf( fp,"#    totalCVMomentum:  the total momentum in the control volume at that instant in time\n" );
  fprintf( fp,"#    netFlux:          the net flux of momentum through the control surfaces\n" );
  fprintf( fp,"#Time                      totalCVMomentum.x()         totalCVMomentum.y()         totalCVMomentum.z()         netFlux.x()                  netFlux.y()                  netFlux.z()\n");
  cout << Parallel::getMPIRank() << " momentumAnalysis:Created file " << filename << endl;
}


//______________________________________________________________________
//   This is a rip off of what's done int the boundary condition code
//______________________________________________________________________
//
void momentumAnalysis::faceInfo(const std::string fc,
                              Patch::FaceType& face_side,
                              Vector& norm,
                              int& p_dir)
{
  if (fc ==  "x-"){
    norm = Vector(-1, 0, 0);
    p_dir = 0;
    face_side = Patch::xminus;
  }
  if (fc == "x+"){
    norm = Vector(1, 0, 0);
    p_dir = 0;
    face_side = Patch::xplus;
  }
  if (fc == "y-"){
    norm = Vector(0, -1, 0);
    p_dir = 1;
    face_side = Patch::yminus;
  }
  if (fc == "y+"){
    norm = Vector(0, 1, 0);
    p_dir = 1;
    face_side = Patch::yplus;
  }
  if (fc == "z-"){
    norm = Vector(0, 0, -1);
    p_dir = 2;
    face_side = Patch::zminus;
  }
  if (fc == "z+"){
    norm = Vector(0, 0, 1);
    p_dir = 2;
    face_side = Patch::zplus;
  }
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
