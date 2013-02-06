/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/1stLawThermo.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/PerPatch.h>
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
//  To turn on the output
//  setenv SCI_DEBUG "FirstLawThermo_DBG_COUT:+" 
static DebugStream cout_doing("FirstLawThermo",   false);
static DebugStream cout_dbg("FirstLawThermo_dbg", false);
//______________________________________________________________________              
FirstLawThermo::FirstLawThermo(ProblemSpecP& module_spec,
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
  d_conversion   = 1.0/1000.;     // in SI units this is J/KJ
  
  FL_lb = scinew FL_Labels();
  I_lb  = scinew ICELabel();
  M_lb  = scinew MPMLabel();
  
  FL_lb->lastCompTimeLabel    = VarLabel::create( "lastCompTime",     max_vartype::getTypeDescription() );
  FL_lb->fileVarsStructLabel  = VarLabel::create( "FileInfo",         PerPatch<FileInfoP>::getTypeDescription() );
  FL_lb->ICE_totalIntEngLabel = VarLabel::create( "ICE_totalIntEng",  sum_vartype::getTypeDescription() );
  FL_lb->MPM_totalIntEngLabel = VarLabel::create( "MPM_totalIntEng",  sum_vartype::getTypeDescription() );
  FL_lb->totalFluxesLabel     = VarLabel::create( "totalFluxes",      sum_vartype::getTypeDescription() );
}

//__________________________________
FirstLawThermo::~FirstLawThermo()
{
  cout_doing << " Doing: destorying FirstLawThermo " << endl;
  if( d_zeroMatlSet  && d_zeroMatlSet->removeReference() ) {
    delete d_zeroMatlSet;
  }
  if( d_zeroMatl && d_zeroMatl->removeReference() ) {
    delete d_zeroMatl;
  }
  if( d_zeroPatch && d_zeroPatch->removeReference() ) {
    delete d_zeroPatch;
  }
  
  VarLabel::destroy( FL_lb->lastCompTimeLabel );
  VarLabel::destroy( FL_lb->fileVarsStructLabel );
  VarLabel::destroy( FL_lb->ICE_totalIntEngLabel );
  VarLabel::destroy( FL_lb->MPM_totalIntEngLabel );
  VarLabel::destroy( FL_lb->totalFluxesLabel );  
  
  delete FL_lb;
  delete M_lb;
  delete I_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void FirstLawThermo::problemSetup(const ProblemSpecP&,
                                  GridP& grid,
                                  SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tFirstLawThermo" << endl;
  
  if(!d_dataArchiver){
    throw InternalError("FirstLawThermo:couldn't get output port", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  Read in timing information
  d_prob_spec->require( "samplingFrequency", d_analysisFreq );
  d_prob_spec->require( "timeStart",         d_StartTime );            
  d_prob_spec->require( "timeStop",          d_StopTime );
  d_prob_spec->get(     "engy_convt_factor", d_conversion );   // energy conversion factor in SI it KJ->J   
  
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
  // Loop over each face and find the extents
  ProblemSpecP cv_ps = d_prob_spec->findBlock("controlVolume");

  for (ProblemSpecP face_ps = cv_ps->findBlock("Face");
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
    
    d_cv_faces[f] = cvFace;  
  }
  
  //__________________________________
  //  Loop over all the MPM Matls and pull out the specific heat
  //  This Assumes that MPM matls are listed from 0 to N
  //  It also assumes that cp is constant
  int matl = 0;
  ProblemSpecP root_ps = d_prob_spec->getRootNode();
  ProblemSpecP mat_ps =  root_ps->findBlockWithOutAttribute("MaterialProperties");
  
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  if(mpm_mat_ps){
    for (ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != 0; 
         ps = ps->findNextBlock("material") ) {
      double cp;   
      ps->require("specific_heat",cp);
      d_mpm_specificHeat[matl] = cp;
      matl +=1;
    } 
  }
}

//______________________________________________________________________
void FirstLawThermo::scheduleInitialize(SchedulerP& sched,
                                        const LevelP& level)
{
  printSchedule(level,cout_doing,"FirstLawThermo::scheduleInitialize");
  
  Task* t = scinew Task("FirstLawThermo::initialize",
                  this, &FirstLawThermo::initialize);
  
  t->computes(FL_lb->lastCompTimeLabel);
  t->computes(FL_lb->fileVarsStructLabel, d_zeroMatl); 
  sched->addTask(t, d_zeroPatch, d_zeroMatlSet);
}
//______________________________________________________________________
void FirstLawThermo::initialize(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw)
{  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing initialize");
    
    double tminus = -1.0/d_analysisFreq;    
    new_dw->put(max_vartype(tminus), FL_lb->lastCompTimeLabel);

    //__________________________________
    //  initialize fileInfo struct
    PerPatch<FileInfoP> fileInfo;
    FileInfo* myFileInfo = scinew FileInfo();
    fileInfo.get() = myFileInfo;
    
    new_dw->put(fileInfo,    FL_lb->fileVarsStructLabel, 0, patch);
    
    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:FirstLawThermo  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);      
    } 
  }  
}

void FirstLawThermo::restartInitialize()
{
}

//______________________________________________________________________
void FirstLawThermo::scheduleDoAnalysis(SchedulerP& sched,
                                        const LevelP& level)
{

  // Tell the scheduler to not copy this variable to a new AMR grid and 
  // do not checkpoint it.
  sched->overrideVariableBehavior("FileInfo", false, false, false, true, true);
  
  //__________________________________  
  //  compute the ICE contributions
  printSchedule( level,cout_doing,"FirstLawThermo::scheduleDoAnalysis" );
  
  Task* t0 = scinew Task( "FirstLawThermo::compute_ICE_Contributions", 
                     this,&FirstLawThermo::compute_ICE_Contributions );

  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSet* all_matls = d_sharedState->allMaterials();
  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  const MaterialSubset* ice_ss = ice_matls->getUnion();
  
  t0->requires( Task::OldDW, FL_lb->lastCompTimeLabel );
  t0->requires( Task::OldDW, I_lb->delTLabel, level.get_rep() ); 
  
  t0->requires( Task::NewDW, I_lb->rho_CCLabel,        ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->temp_CCLabel,       ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->specific_heatLabel, ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->gammaLabel,         ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->uvel_FCMELabel,     ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->vvel_FCMELabel,     ice_ss, gn );
  t0->requires( Task::NewDW, I_lb->wvel_FCMELabel,     ice_ss, gn );
  
  t0->computes( FL_lb->ICE_totalIntEngLabel );
  t0->computes( FL_lb->totalFluxesLabel );
  
  sched->addTask( t0, level->eachPatch(), all_matls );
  
  //__________________________________
  //  compute the MPM contributions
  Task* t1 = scinew Task( "FirstLawThermo::compute_MPM_Contributions", 
                     this,&FirstLawThermo::compute_MPM_Contributions );

  const MaterialSet* mpm_matls = d_sharedState->allMPMMaterials();
  const MaterialSubset* mpm_ss = mpm_matls->getUnion();
  
  t1->requires( Task::OldDW, FL_lb->lastCompTimeLabel );
  t1->requires( Task::NewDW, M_lb->pMassLabel_preReloc,        mpm_ss, gn );
  t1->requires( Task::NewDW, M_lb->pTemperatureLabel_preReloc, mpm_ss, gn );
  t1->computes( FL_lb->MPM_totalIntEngLabel );
  
  sched->addTask( t1, level->eachPatch(), all_matls ); 


  //__________________________________
  //  output the contributions
  Task* t2 = scinew Task("FirstLawThermo::doAnalysis", 
                    this,&FirstLawThermo::doAnalysis );
                    
  t2->requires( Task::OldDW, FL_lb->lastCompTimeLabel );
  t2->requires( Task::OldDW, FL_lb->fileVarsStructLabel, d_zeroMatl, gn, 0 );
  
  t2->requires( Task::NewDW, FL_lb->ICE_totalIntEngLabel );
  t2->requires( Task::NewDW, FL_lb->MPM_totalIntEngLabel );
  t2->requires( Task::NewDW, FL_lb->totalFluxesLabel );
  
  t2->computes( FL_lb->lastCompTimeLabel );
  t2->computes( FL_lb->fileVarsStructLabel, d_zeroMatl );
  sched->addTask( t2, d_zeroPatch, d_zeroMatlSet);        // you only need to schedule this  patch 0 since all you're doing is writing out data

}

//______________________________________________________________________
//        ICE Contributions to the energy
void FirstLawThermo::compute_ICE_Contributions(const ProcessorGroup* pg,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matl_sub ,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  max_vartype analysisTime;
  delt_vartype delT;
  
  old_dw->get(analysisTime, FL_lb->lastCompTimeLabel);
  old_dw->get(delT, d_sharedState->get_delt_label(),level);
  
  double lastCompTime = analysisTime;
  double nextCompTime = lastCompTime + 1.0/d_analysisFreq;  
  double now = d_dataArchiver->getCurrentTime();

  if( now < nextCompTime  ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing FirstLawThermo::compute_ICE_Contributions");

    CCVariable<double> int_eng;
    constCCVariable<double> temp_CC;
    constCCVariable<double> rho_CC;
    constCCVariable<double> cv;
    constCCVariable<double> gamma;

    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;

    new_dw->allocateTemporary(int_eng,patch);

    Vector dx = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();

    int numICEmatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType gn  = Ghost::None;

    double ICE_totalIntEng = 0.0;
    double total_flux      = 0.0;

    for (int m = 0; m < numICEmatls; m++ ) {

      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      new_dw->get(rho_CC,  I_lb->rho_CCLabel,       indx, patch, gn,0);
      new_dw->get(temp_CC, I_lb->temp_CCLabel,      indx, patch, gn,0);
      new_dw->get(cv,      I_lb->specific_heatLabel,indx, patch, gn,0);
      new_dw->get(uvel_FC, I_lb->uvel_FCMELabel,    indx, patch, gn,0);
      new_dw->get(vvel_FC, I_lb->vvel_FCMELabel,    indx, patch, gn,0);
      new_dw->get(wvel_FC, I_lb->wvel_FCMELabel,    indx, patch, gn,0);
      new_dw->get(gamma,   I_lb->gammaLabel,        indx, patch, gn,0);

      double mat_int_eng = 0.0;

      //__________________________________
      //  Sum contributions over patch        
      for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        int_eng[c] = rho_CC[c] * vol * cv[c] * temp_CC[c];
        mat_int_eng += int_eng[c];
      }
      
      ICE_totalIntEng += mat_int_eng;

      //__________________________________
      // Sum the fluxes passing through the boundaries      
      vector<Patch::FaceType> bf;
      patch->getBoundaryFaces(bf);
      double mat_fluxes = 0.0;

      for( vector<Patch::FaceType>::const_iterator itr = bf.begin(); itr != bf.end(); ++itr ){
        Patch::FaceType face = *itr;
        
        cv_face* cvFace = d_cv_faces[face];

        cout_dbg << " cvFace: " << patch->getFaceName(face ) << " faceType " << cvFace->face 
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

        if (face == Patch::xminus || face == Patch::xplus) {    // X faces
          double area = dx.y() * dx.z();
          for(CellIterator iter = iterLimits; !iter.done();iter++) {
            IntVector c = *iter;
            double vel    = uvel_FC[c];
            double mdot   = plus_minus_one * vel * rho_CC[c] * area;
            double KE     = 0.5 * vel * vel;
            double enthpy = temp_CC[c] * gamma[c] * cv[c];
            
	     mat_fluxes +=  mdot * (enthpy * KE * d_conversion);
          }
        }

        if (face == Patch::yminus || face == Patch::yplus) {    // Y faces
          double area = dx.x() * dx.z();
          for(CellIterator iter = iterLimits; !iter.done();iter++) {
            IntVector c = *iter;
            double vel    = vvel_FC[c];
            double mdot   = plus_minus_one * vel * rho_CC[c] * area;
            double KE     = 0.5 * vel * vel;
            double enthpy = temp_CC[c] * gamma[c] * cv[c];
            
	     mat_fluxes +=  mdot * (enthpy * KE * d_conversion);
          }
        }

        if (face == Patch::zminus || face == Patch::zplus) {    // Z faces
          double area = dx.x() * dx.y();
          for(CellIterator iter = iterLimits; !iter.done();iter++) {
            IntVector c = *iter;
            double vel    = wvel_FC[c];
            double mdot   = plus_minus_one * vel * rho_CC[c] * area;
            double KE     = 0.5 * vel * vel;
            double enthpy = temp_CC[c] * gamma[c] * cv[c];
            
	     mat_fluxes +=  mdot * (enthpy * KE * d_conversion);
          }
        }
      }  // boundary faces
      
      total_flux += mat_fluxes;
    }  // ICE Matls loop
    
    new_dw->put( sum_vartype(ICE_totalIntEng), FL_lb->ICE_totalIntEngLabel );
    new_dw->put( sum_vartype(total_flux),      FL_lb->totalFluxesLabel );
  }  // patch loop
}


//______________________________________________________________________
//        MPM Contributions to the energy
void FirstLawThermo::compute_MPM_Contributions(const ProcessorGroup* pg,
                                               const PatchSubset* patches,
                                               const MaterialSubset* matl_sub ,
                                               DataWarehouse* old_dw,
                                               DataWarehouse* new_dw)
{
  max_vartype analysisTime;  
  old_dw->get(analysisTime, FL_lb->lastCompTimeLabel);

  double lastCompTime = analysisTime;
  double nextCompTime = lastCompTime + 1.0/d_analysisFreq;  
  double now = d_dataArchiver->getCurrentTime();

  if( now < nextCompTime  ){
    return;
  }

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing,"Doing FirstLawThermo::compute_MPM_Contributions");

    //__________________________________
    //  compute the thermal energy of the solids
    int numMPMMatls=d_sharedState->getNumMPMMatls();
    double MPM_totalIntEng = 0.0;
    
    for(int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      constParticleVariable<double> pmassNew, pTempNew;
      
      ParticleSubset* pset = old_dw->getParticleSubset( dwi, patch );
      new_dw->get( pTempNew, M_lb->pTemperatureLabel_preReloc, pset );
      new_dw->get( pmassNew, M_lb->pMassLabel_preReloc,        pset );
      
      double Cp = d_mpm_specificHeat[dwi];
    
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
        particleIndex idx = *iter;    
    
        MPM_totalIntEng += pTempNew[idx] * pmassNew[idx] * Cp;
      }
    }  // matl loop
    
    new_dw->put( sum_vartype(MPM_totalIntEng), FL_lb->MPM_totalIntEngLabel );
  }
}




//______________________________________________________________________
// 
void FirstLawThermo::doAnalysis(const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset* matls ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{

  
  max_vartype lastTime;
  old_dw->get( lastTime, FL_lb->lastCompTimeLabel );

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

      if( old_dw->exists( FL_lb->fileVarsStructLabel, 0, patch ) ){
        old_dw->get(fileInfo, FL_lb->fileVarsStructLabel, 0, patch);
      }else{  
        FileInfo* myFileInfo = scinew FileInfo();
        fileInfo.get() = myFileInfo;
      }
      
      std::map<string, FILE *> myFiles;

      if( fileInfo.get().get_rep() ){
        myFiles = fileInfo.get().get_rep()->files;
      } 
      
      string udaDir = d_dataArchiver->getOutputLocation();
      string filename = udaDir + "/" + "1stLawThermo.dat";
      FILE *fp;


      if( myFiles.count(filename) == 0 ){
        createFile(filename, fp);
        myFiles[filename] = fp;

      } else {
        fp = myFiles[filename];
      }

      if (!fp){
        throw InternalError("\nERROR:dataAnalysisModule:1stLawThermo:  failed opening file"+filename,__FILE__, __LINE__);
      } 
      //__________________________________
      //
      sum_vartype ICE_totalIntEng, MPM_totalIntEng, total_flux;
      new_dw->get( ICE_totalIntEng, FL_lb->ICE_totalIntEngLabel );
      new_dw->get( MPM_totalIntEng, FL_lb->MPM_totalIntEngLabel );
      new_dw->get( total_flux,      FL_lb->totalFluxesLabel );
      
      double totalIntEng = (double)ICE_totalIntEng + (double)MPM_totalIntEng;
      
      fprintf(fp, "%16.15E      %16.15E      %16.15E       %16.15E       %16.15E\n", now, 
                  (double)ICE_totalIntEng, 
                  (double)MPM_totalIntEng, 
                  totalIntEng,
                  (double)total_flux );
      
      time_dw = now;
      
      //__________________________________
      // put the file pointers into the DataWarehouse
      // these could have been altered. You must
      // reuse the Handle fileInfo and just replace the contents   
      fileInfo.get().get_rep()->files = myFiles;

      new_dw->put(fileInfo, FL_lb->fileVarsStructLabel, 0, patch);
    }
  }
  new_dw->put(max_vartype( time_dw ), FL_lb->lastCompTimeLabel);
}


//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void FirstLawThermo::createFile(string& filename,  FILE*& fp)
{
  // if the file already exists then exit.  The file could exist but not be owned by this processor
  ifstream doExists( filename.c_str() );
  if(doExists){
    fp = fopen(filename.c_str(), "a");
    return;
  }
  
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"# This assumes:\n");
  fprintf(fp,"#    - mpm matls have constant specific heat\n");
  fprintf(fp,"#    - mpm matls are listed in order 0, 1, 2, 3\n");
  fprintf(fp,"#    - Energy conversion factor, in SI units KJ ->J %E\n",d_conversion);
  fprintf(fp,"#Time                      ICE_totalIntEng            MPM_totalIntEng             totalIntEng                 total_ICE_Flux\n");
  cout << Parallel::getMPIRank() << " FirstLawThermo:Created file " << filename << endl;
}


//______________________________________________________________________
//   This is a rip off of what's done int the boundary condition code
void FirstLawThermo::faceInfo(const std::string fc,
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
void FirstLawThermo::bulletProofing(GridP& grid,
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
     warn << "\n ERROR:1stLawThermo: the plane on face ("<< side
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
