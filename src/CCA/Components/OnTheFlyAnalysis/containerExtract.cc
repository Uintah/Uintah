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

/*!
  \class containerExtract containerExtract.h
  \brief Analyze data from any geom_object's interior or surface.
  \author Steve Brown \n
          Todd Harman
          (c) 2007-2008 University of Utah
 */
 
/**
  * TODO: 
  *  - do we need to delete VarLabels in the destructor?
  *  - abstract extractCell class out to other AnalysisModules?
  *  - v&v against faceextract.cc
  *  - file format?  Todd's tabular format is nice; this file tries to be
  *    backwards-compatable with faceextract.cc & matlab scripts
  *  - validate against ups_spec.xml
  **/

#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/OnTheFlyAnalysis/containerExtract.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/StaticArray.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <dirent.h>
#endif
#include <iostream>
#include <fstream>

#include <cstdio>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "CONTAINEREXTRACT_DBG_COUT:+" 
static DebugStream cout_doing("CONTAINEREXTRACT_DOING_COUT", false);
static DebugStream cout_dbg("CONTAINEREXTRACT_DBG_COUT", false);



//______________________________________________________________________              
containerExtract::containerExtract(ProblemSpecP& module_spec,
                         SimulationStateP& sharedState,
                         Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState = sharedState;
  d_prob_spec = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  ps_lb = scinew containerExtractLabel();
}

//__________________________________
containerExtract::~containerExtract()
{
  cout_doing << " Doing: destorying containerExtract " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  VarLabel::destroy(ps_lb->lastWriteTimeLabel);
  delete ps_lb;
  

  //delete the container cell list
  for (unsigned int i = 0; i < d_containers.size(); i++) {
    container* cnt = d_containers[i];
    vector<extractVarLabel*>::iterator eity;
    for (eity = cnt->vls.begin(); eity != cnt->vls.end(); eity++) {
      delete *eity;
    }

    vector<extractCell*>::iterator ity;
    for( ity = cnt->extractCells.begin(); ity != cnt->extractCells.end(); ity++) {
      delete *ity;
    }
    delete cnt;
  }

  
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void containerExtract::problemSetup(const ProblemSpecP& prob_spec,
                               const ProblemSpecP& restart_prob_spec,
                               GridP& grid,
                               SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tcontainerExtract" << endl;

  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  
  if(!d_dataArchiver){
    throw InternalError("containerExtract:couldn't get output port", __FILE__, __LINE__);
  }
  
  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();                               
  
  ps_lb->lastWriteTimeLabel =  VarLabel::create("lastWriteTime", 
                                            max_vartype::getTypeDescription());

  //__________________________________
  //  Read in timing information
  d_prob_spec->require("samplingFrequency", d_writeFreq);
  d_prob_spec->require("timeStart",         d_StartTime);            
  d_prob_spec->require("timeStop",          d_StopTime);

  
  //__________________________________
  //  Read in containers 
  map<string,string> attribute;
  ProblemSpecP objects_ps = d_prob_spec->findBlock("objects"); 
  if (!objects_ps){
    throw ProblemSetupException("\n ERROR:containerExtract: Couldn't find <objects> tag \n", __FILE__, __LINE__);    
  }        
    
  /* foreach <geom_object> */
  for (ProblemSpecP object_spec = objects_ps->findBlock("geom_object"); object_spec != 0; 
                    object_spec = object_spec->findNextBlock("geom_object")) {
                    
    // put input variables into the global struct
    container* c = scinew container;
    
    object_spec->getAttributes(attribute);
    string name = attribute["name"];

    ProblemSpecP var_spec;
    /* foreach <variable> */
    for (var_spec = object_spec->findBlock("extract"); var_spec != 0;
         var_spec = var_spec->findNextBlock("extract") ) {
      
      var_spec->getAttributes(attribute);                  
      string mode = attribute["mode"];
      enum EXTRACT_MODE fxmode;
      string var = attribute["var"];
      VarLabel* label = VarLabel::find(var);
      if(label == NULL && (mode == "interior" || mode == "surface") ){
        throw ProblemSetupException("containerExtract: var label not found: "
            + var + " mode: " + mode , __FILE__, __LINE__);
      }

      /* For normal interior and surface modes: check and verify the analyze label */
      /* (for incident net veolicty modes: this is done automatically) */
      if (label != NULL && (mode == "interior" || mode == "surface") ) {

        const Uintah::TypeDescription* td = label->typeDescription();
        const Uintah::TypeDescription* subtype = td->getSubType();

        //__________________________________
        // Bulletproofing
        bool throwException = false;  

        // only CC, SFCX, SFCY, SFCZ variables
        if(td->getType() != TypeDescription::CCVariable &&
            td->getType() != TypeDescription::SFCXVariable &&
            td->getType() != TypeDescription::SFCYVariable &&
            td->getType() != TypeDescription::SFCZVariable ){
          throwException = true;
        }
        // CC Variables, only Doubles and Vectors 
        if(td->getType() != TypeDescription::CCVariable &&
            subtype->getType() != TypeDescription::double_type &&
            subtype->getType() != TypeDescription::int_type &&
            subtype->getType() != TypeDescription::Vector  ){
          throwException = true;
        }
        // Face Centered Vars, only Doubles
        if( (td->getType() == TypeDescription::SFCXVariable ||
              td->getType() == TypeDescription::SFCYVariable ||
              td->getType() == TypeDescription::SFCZVariable) &&
            subtype->getType() != TypeDescription::double_type) {
          throwException = true;
        } 
        if(throwException){       
          ostringstream warn;
          warn << "ERROR:AnalysisModule:containerExtract: ("<<label->getName() << " " 
            << td->getName() << " ) has not been implemented" << endl;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }
        d_varLabels.push_back(label);
      }

      // Set the object's mode and schedule the VarLabels for analysis
      if (mode == "interior") {
      fxmode = INTERIOR;
      //user specifies VarLabel above
      } else if (mode == "surface") {
        fxmode = SURFACE;
        //user specifies VarLabel above
        /* You may only specify surface mode with SFC{X Y Z} vars. */
        const Uintah::TypeDescription* td = label->typeDescription();
        if(   td->getType() != TypeDescription::SFCXVariable &&
            td->getType() != TypeDescription::SFCYVariable &&
            td->getType() != TypeDescription::SFCZVariable) {
          ostringstream warn;
          warn << "ERROR:AnalysisModule:containerExtract: only SFC X Y Z variables are allowed in 'surface' mode." 
            << " (You spefified "<<label->getName() << ", which is a " 
          << td->getName() << " )." << endl;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }

      } else if (mode == "incident") {
        fxmode = INCIDENT;
        d_varLabels.push_back(VarLabel::find("radiationFluxWIN"));
        d_varLabels.push_back(VarLabel::find("radiationFluxEIN"));
        d_varLabels.push_back(VarLabel::find("radiationFluxSIN"));
        d_varLabels.push_back(VarLabel::find("radiationFluxNIN"));
        d_varLabels.push_back(VarLabel::find("radiationFluxTIN"));
        d_varLabels.push_back(VarLabel::find("radiationFluxBIN"));
      } else if (mode == "net") {
        fxmode = NET;
        d_varLabels.push_back(VarLabel::find("htfluxRadX"));
        d_varLabels.push_back(VarLabel::find("htfluxRadY"));
        d_varLabels.push_back(VarLabel::find("htfluxRadZ"));
      } else if (mode == "velocity") {
        fxmode = VELOCITY;
        d_varLabels.push_back(VarLabel::find("newCCUVelocity"));
        d_varLabels.push_back(VarLabel::find("newCCVVelocity"));
        d_varLabels.push_back(VarLabel::find("newCCWVelocity"));
      } else {
        throw ProblemSetupException("\n ERROR:containerExtract: extraction mode not supported: "+ mode + "\n",
            __FILE__, __LINE__);
      }
      
      extractVarLabel* evl = scinew extractVarLabel;
      evl->vl = label;
      evl->mode = fxmode;

      c->vls.push_back(evl);
    } /* foreach <variable> */

    //Construct a vector of geom_objs
    vector<GeometryPieceP> geomObjs;
    GeometryPieceFactory::create(object_spec, geomObjs); 
    if (geomObjs.size() < 1) {
      throw ParameterNotFound("ContainerExtract: You didn't specify a proper geom_object.", __FILE__, __LINE__);
    }

    
    //__________________________________
    // bullet proofing
 
    // -every object must have a name
    if(name == ""){
      throw ProblemSetupException("\n ERROR:containerExtract: You must name each object <geom_object name=\"something\">\n", 
          __FILE__, __LINE__);
    }

    //object can't exceed computational domain
    IntVector min, max;
    grid->getLevel(0)->findCellIndexRange(min,max);

    //remove ghost cells
    min = min + IntVector(1,1,1);
    max = max - IntVector(1,1,1);

    Box domainBox = grid->getLevel(0)->getBox(min,max);

    for (unsigned int i = 0; i < geomObjs.size(); i++) {
      Box bbox = geomObjs[i]->getBoundingBox();

      Point dl = domainBox.lower();
      Point du = domainBox.upper();

      Point bl = bbox.lower();
      Point bu = bbox.upper();

      if (bl.x() < dl.x() || bl.y() < dl.y() || bl.z() < dl.z()
       || bu.x() > du.x() || bu.y() > du.y() || bu.z() > du.z() ) {
        ostringstream warn;
        warn << "\n ERROR:containerExtract: the object that you've specified "  
          << " begins or ends outside of the computational domain. \n" << endl;
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }

      if(bbox.intersect(domainBox).degenerate() ) {
        ostringstream warn;
        warn << "\n ERROR:containerExtract: the object that you've specified is degenerate" << endl; 
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }

    }
    
    // Start time < stop time
    if(d_StartTime > d_StopTime){
      throw ProblemSetupException("\n ERROR:containerExtract: startTime > stopTime. \n", __FILE__, __LINE__);
    }
   
 
    /* global container list from above; already has vls populated */
    c->name    = name;
    c->geomObjs= geomObjs;
    
    d_containers.push_back(c);

  } //foreach container
}

//______________________________________________________________________
void containerExtract::scheduleInitialize(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "containerExtract::scheduleInitialize " << endl;
  Task* t = scinew Task("containerExtract::initialize", 
                  this, &containerExtract::initialize);
  
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void containerExtract::initialize(const ProcessorGroup*, 
    const PatchSubset* patches,
    const MaterialSubset*,
    DataWarehouse*,
    DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tcontainerExtract" << endl;
  
  /************************
   * Set up the grid. 
   *************************/
  //face can't exceed computational domain
  IntVector min, max;
  const Patch* patch = patches->get(0);
  patch->getLevel()->findCellIndexRange(min,max);
  Box patchBox = patch->getExtraBox();

  //remove ghost cells
  min = min + IntVector(1,1,1);
  max = max - IntVector(1,1,1);
  cout_dbg << "Min "<<min<<" Max "<<max<<endl;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    double tminus = -1.0/d_writeFreq;
    new_dw->put(max_vartype(tminus), ps_lb->lastWriteTimeLabel);

    if(patch->getGridIndex() == 0){   // only need to do this once
      string udaDir = d_dataArchiver->getOutputLocation();

      //  Bulletproofing
      DIR *check = opendir(udaDir.c_str());
      if ( check == NULL){
        ostringstream warn;
        warn << "ERROR:containerExtract  The main uda directory does not exist. ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      closedir(check);
    }

    /************************
     * Translate the geometry onto the grid.  Save the result in containerPoints. 
     * XXX Arches and MPM disagree on this translation: 
     * Mpm uses the volume fraction of solid:
     * if (volume fraction of solid > 0.5) then cellType  = wall
     * where as the intrusion mechanism uses geometric location:
     * if (cell center is inside geometry) then cellType = wall
     *************************/
    for (unsigned int i = 0; i < d_containers.size(); i++) {
      container* cnt = d_containers[i];

      vector<GeometryPieceP>::iterator ity; 
      for(ity = cnt->geomObjs.begin(); ity!= cnt->geomObjs.end(); ity++) {
        GeometryPieceP piece = *ity;
        Box geomBox = piece->getBoundingBox();
        
        //XXX if on edge of domain it will go to -1 when we subtract later
        Box b = geomBox.intersect(patchBox);
        cout_dbg <<"Iterating from " <<b.lower() <<" to " <<b.upper()<<endl;

        // if this is a special case, skip it. 
        if (b.degenerate())
          continue; 

        for (CellIterator ci = patch->getCellCenterIterator(b); !ci.done(); ci++) {
          IntVector c = *ci;

          Point here = patch->getCellPosition(c);

          if (piece->inside(here)) {
            cnt->containerPoints.push_back(c);
          } //inside geom_object
        } //foreach cell on patch
      } //foreach geom_object
    } //foreach d_containers
  } //foreach patch 

  
  /************************
   * Create a list of extractCells depending on the desired extraction mode. 
   *************************/

  for(unsigned int i = 0; i < d_containers.size(); i++) {
    container* cnt = d_containers[i];
    for (vector<IntVector>::iterator ci = cnt->containerPoints.begin(); ci != cnt->containerPoints.end(); ci++) {
      IntVector c = *ci;

      // loop over each variable associated with this geom_object
      for (unsigned int evl = 0; evl < cnt->vls.size(); evl++) {
        enum EXTRACT_MODE exmode = cnt->vls[evl]->mode; 
        
        bool istop,isbot,isnorth,issouth,iseast,iswest;
        istop=isbot=isnorth=issouth=iseast=iswest = true;
        /* Misleading terms here!
           'iseast == true' means that we have located the WEST boundary of the object.
           We will then analyze data from the EAST face of the neighbor that
           is one cell to the west.  
         */
        for (vector<IntVector>::iterator cellIty = cnt->containerPoints.begin(); 
            cellIty != cnt->containerPoints.end(); cellIty++) {
          IntVector ici = *cellIty;
          if (ici == c - IntVector(0,0,1)) istop   = false;
          if (ici == c + IntVector(0,0,1)) isbot   = false;
          if (ici == c - IntVector(0,1,0)) isnorth = false;
          if (ici == c + IntVector(0,1,0)) issouth = false;
          if (ici == c - IntVector(1,0,0)) iseast  = false;
          if (ici == c + IntVector(1,0,0)) iswest  = false;
        }

        switch (exmode) {
          case INTERIOR:
            if (!(c.x() <= max.x() && c.y() <= max.y() && c.z() <= max.z()
                  && c.x() >= min.x() && c.y() >= min.y() && c.z() >= min.z())) break;
            cnt->extractCells.push_back(scinew extractCell(exmode, NONE, c, cnt->vls[evl]->vl)); 
            break;

          case SURFACE:
            if (iswest  && c.x() < max.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), cnt->vls[evl]->vl));
            if (iseast  && c.x() > min.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c - IntVector(1,0,0), cnt->vls[evl]->vl));
            if (issouth && c.y() < max.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), cnt->vls[evl]->vl));
            if (isnorth && c.y() > min.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c - IntVector(0,1,0), cnt->vls[evl]->vl));
            if (isbot   && c.z() < max.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), cnt->vls[evl]->vl));
            if (istop   && c.z() > min.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c - IntVector(0,0,1), cnt->vls[evl]->vl));
            break;

          case NET:
            if (iswest  && c.x() < max.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), VarLabel::find("htfluxRadX")));
            if (iseast  && c.x() > min.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c /*cf faceextract*/, VarLabel::find("htfluxRadX")));
            if (issouth && c.y() < max.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), VarLabel::find("htfluxRadY")));
            if (isnorth && c.y() > min.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c /*cf faceextract*/, VarLabel::find("htfluxRadY")));
            if (isbot   && c.z() < max.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), VarLabel::find("htfluxRadZ")));
            if (istop   && c.z() > min.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c /*cf faceextract*/, VarLabel::find("htfluxRadZ")));
            break;

          case INCIDENT:
            if (iswest  && c.x() < max.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), VarLabel::find("radiationFluxWIN")));
            if (iseast  && c.x() > min.x())
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c - IntVector(1,0,0), VarLabel::find("radiationFluxEIN")));
            if (issouth && c.y() < max.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), VarLabel::find("radiationFluxSIN")));
            if (isnorth && c.y() > min.y())
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c - IntVector(0,1,0), VarLabel::find("radiationFluxNIN")));
            if (isbot   && c.z() < max.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), VarLabel::find("radiationFluxBIN")));
            if (istop   && c.z() > min.z())
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c - IntVector(0,0,1), VarLabel::find("radiationFluxTIN")));
            break;

          case VELOCITY:
            if (iswest  && c.x() < max.x()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, WEST , c + IntVector(1,0,0), VarLabel::find("newCCWVelocity")));
            }
            if (iseast  && c.x() > min.x()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c - IntVector(1,0,0), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c - IntVector(1,0,0), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, EAST , c - IntVector(1,0,0), VarLabel::find("newCCWVelocity")));
            }
            if (issouth && c.y() < max.y()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, SOUTH, c + IntVector(0,1,0), VarLabel::find("newCCWVelocity")));
            }
            if (isnorth && c.y() > min.y()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c - IntVector(0,1,0), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c - IntVector(0,1,0), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, NORTH, c - IntVector(0,1,0), VarLabel::find("newCCWVelocity")));
            }
            if (isbot   && c.z() < max.z()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, BOTTOM,c + IntVector(0,0,1), VarLabel::find("newCCWVelocity")));
            }
            if (istop   && c.z() > min.z()) {
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c - IntVector(0,0,1), VarLabel::find("newCCUVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c - IntVector(0,0,1), VarLabel::find("newCCVVelocity")));
              cnt->extractCells.push_back(scinew extractCell(exmode, TOP  , c - IntVector(0,0,1), VarLabel::find("newCCWVelocity")));
            }
            break;

          default:
            throw ProblemSetupException("ERROR: containerExtract: not implemented", __FILE__, __LINE__);
            break;

        } //switch extract mode
      } //foreach extractVarLabel
    }//foreach containerPoints
  } // foreach d_containers

  /******************
   * Create the directory / file structure.
   ******************/

  // create the directory structure
  for(unsigned int i = 0; i < d_containers.size(); i++) {
    container* cnt = d_containers[i];
    string udaDir = d_dataArchiver->getOutputLocation();
    string dirPath = udaDir + "/" + cnt->name;

    ostringstream l;
    const Patch* patch = patches->get(0);
    l << patch->getLevel()->getIndex();
    string levelNum = l.str();
    string path = dirPath + "/" + levelNum + "/" ;
    createDirectory(dirPath, levelNum);

    extractCell c;
    vector<extractCell*>::iterator ity;
    for(unsigned int i =0; i < cnt->extractCells.size(); i++) { 
      ostringstream fname;
      fname << path << *(cnt->extractCells[i]);
      string filename = fname.str();
      createFile(filename, *(cnt->extractCells[i]));
    }
  }
}

void containerExtract::restartInitialize()
{
  // need to do something here
  //  new_dw->put(max_vartype(0.0), ps_lb->lastWriteTimeLabel);
}

//______________________________________________________________________
void containerExtract::scheduleDoAnalysis(SchedulerP& sched,
                                     const LevelP& level)
{
  cout_doing << "containerExtract::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("containerExtract::doAnalysis", 
                   this,&containerExtract::doAnalysis);
                     
  t->requires(Task::OldDW, ps_lb->lastWriteTimeLabel);
  
  Ghost::GhostType gac = Ghost::AroundCells;
  for (unsigned int i =0 ; i < d_varLabels.size(); i++) {
    // bulletproofing
    if(d_varLabels[i] == NULL){
      string name = d_varLabels[i]->getName();
      throw InternalError("containerExtract: scheduleDoAnalysis label not found: " 
                          + name , __FILE__, __LINE__);
    }
    t->requires(Task::NewDW,d_varLabels[i], gac, 1);
  }
  t->computes(ps_lb->lastWriteTimeLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void containerExtract::doAnalysis(const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw)
{   
  UintahParallelComponent* DA = dynamic_cast<UintahParallelComponent*>(d_dataArchiver);
  LoadBalancer* lb = dynamic_cast<LoadBalancer*>( DA->getPort("load balancer"));

  const Level* level = getLevel(patches);

  max_vartype writeTime;
  old_dw->get(writeTime, ps_lb->lastWriteTimeLabel);
  double lastWriteTime = writeTime;

  double now = d_dataArchiver->getCurrentTime();
  double nextWriteTime = lastWriteTime + 1.0/d_writeFreq;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    int proc = lb->getPatchwiseProcessorAssignment(patch);
    cout_dbg << Parallel::getMPIRank() << "   working on patch " << patch->getID() << " which is on proc " << proc << endl;
    //__________________________________
    // write data if this processor owns this patch
    // and if it's time to write
    if( proc == pg->myrank() && now >= nextWriteTime){

      cout_doing << pg->myrank() << " " 
        << "Doing doAnalysis (containerExtract)\t\t\t\tL-"
        << level->getIndex()
        << " patch " << patch->getGridIndex()<< endl;
      //__________________________________
      // loop over each of the variables
      // load them into the data vectors
      vector< constCCVariable<int> >      CC_integer_data;
      vector< constCCVariable<double> >   CC_double_data;
      vector< constCCVariable<Vector> >   CC_Vector_data;
      vector< constSFCXVariable<double> > SFCX_double_data;
      vector< constSFCYVariable<double> > SFCY_double_data;
      vector< constSFCZVariable<double> > SFCZ_double_data;

      /* We need to lookup VarLabels to compare them to fluxcells later.
         Since they're not stored in the variable vectors preserve them here. */
      vector< VarLabel* >                  cci_var_labels;
      vector< VarLabel* >                  ccd_var_labels;
      vector< VarLabel* >                  ccv_var_labels;
      vector< VarLabel* >                  sfcx_var_labels;
      vector< VarLabel* >                  sfcy_var_labels;
      vector< VarLabel* >                  sfcz_var_labels;

      constCCVariable<int>    q_CC_integer;      
      constCCVariable<double> q_CC_double;
      constCCVariable<Vector> q_CC_Vector;

      constSFCXVariable<double> q_SFCX_double;      
      constSFCYVariable<double> q_SFCY_double;
      constSFCZVariable<double> q_SFCZ_double;      


      constCCVariable<double> q_radiationFluxEIN;
      constCCVariable<double> q_radiationFluxWIN;
      constCCVariable<double> q_radiationFluxNIN;
      constCCVariable<double> q_radiationFluxSIN;
      constCCVariable<double> q_radiationFluxTIN;
      constCCVariable<double> q_radiationFluxBIN;

      Ghost::GhostType gac = Ghost::AroundCells;
      int indx = d_matl->getDWIndex();

      for (unsigned int i =0 ; i < d_varLabels.size(); i++) {

        // bulletproofing
        if(d_varLabels[i] == NULL){
          string name = d_varLabels[i]->getName();
          throw InternalError("containerExtract: analyze label not found: " 
              + name , __FILE__, __LINE__);
        }

        const Uintah::TypeDescription* td = d_varLabels[i]->typeDescription();
        const Uintah::TypeDescription* subtype = td->getSubType();

        string name;

        switch(td->getType()){
          case Uintah::TypeDescription::CCVariable:      // CC Variables
            switch(subtype->getType()) {

              case Uintah::TypeDescription::double_type:
                new_dw->get(q_CC_double, d_varLabels[i], indx, patch, gac, 1);
                CC_double_data.push_back(q_CC_double);
                ccd_var_labels.push_back(d_varLabels[i]);
                break;

              case Uintah::TypeDescription::Vector:
                new_dw->get(q_CC_Vector, d_varLabels[i], indx, patch, gac, 1);
                CC_Vector_data.push_back(q_CC_Vector);
                ccv_var_labels.push_back(d_varLabels[i]);
                break;
              case Uintah::TypeDescription::int_type:
                new_dw->get(q_CC_integer, d_varLabels[i], indx, patch, gac, 1);
                CC_integer_data.push_back(q_CC_integer);
                cci_var_labels.push_back(d_varLabels[i]);
                break; 
              default:
                throw InternalError("containerExtract: invalid data type", __FILE__, __LINE__); 
            }
            break;
          case Uintah::TypeDescription::SFCXVariable:   // SFCX Variables
            new_dw->get(q_SFCX_double, d_varLabels[i], indx, patch, gac, 1);
            SFCX_double_data.push_back(q_SFCX_double);
            sfcx_var_labels.push_back(d_varLabels[i]);
            break;
          case Uintah::TypeDescription::SFCYVariable:    // SFCY Variables
            new_dw->get(q_SFCY_double, d_varLabels[i], indx, patch, gac, 1);
            SFCY_double_data.push_back(q_SFCY_double);
            sfcy_var_labels.push_back(d_varLabels[i]);
            break;
          case Uintah::TypeDescription::SFCZVariable:   // SFCZ Variables
            new_dw->get(q_SFCZ_double, d_varLabels[i], indx, patch, gac, 1);
            SFCZ_double_data.push_back(q_SFCZ_double);
            sfcz_var_labels.push_back(d_varLabels[i]);
            break;
          default:
            ostringstream warn;
            warn << "ERROR:AnalysisModule:containerExtract: ("<<d_varLabels[i]->getName() << " " 
              << td->getName() << " ) has not been implemented" << endl;
            throw InternalError(warn.str(), __FILE__, __LINE__);
        }
      }            

      for (unsigned int i = 0; i < d_containers.size(); i++) {
        container* cnt = d_containers[i];

        extractCell* exc;
        for(vector<extractCell*>::iterator ity = cnt->extractCells.begin(); ity != cnt->extractCells.end(); ity++) {
          exc = *ity; 

          if (!patch->containsCell(exc->c))
            continue;  // just in case - the point-to-cell logic might throw us off on patch boundaries...

          IntVector c = exc->c;
          ostringstream fname;
          // create the directory structure
          string udaDir = d_dataArchiver->getOutputLocation();
          string dirPath = udaDir + "/" + cnt->name; 

          ostringstream l;
          l << patch->getLevel()->getIndex();
          string levelNum = l.str();
          string path = dirPath + "/" + levelNum + "/" ;
          fname<<path<<*exc;
          string filename = fname.str();
          //cout_dbg << "Fname: "<<filename<<endl;
          //create file and write the file header  
          ifstream test(filename.c_str());
          if (!test){
            createFile(filename, *exc);
          }

          FILE *fp;
          fp = fopen(filename.c_str(), "a");
          if (!fp){
            throw InternalError("\nERROR:dataAnalysisModule:containerExtract:  failed opening file"+filename,__FILE__, __LINE__);
          }

          Point here = patch->cellPosition(c);
          double time = d_dataArchiver->getCurrentTime();
          fprintf(fp,    "%E\t %E\t %E\t %E",here.x(),here.y(),here.z(), time);


          /* Find out which vector contains the variable and print it out. */
          const Uintah::TypeDescription* td = exc->vl->typeDescription();
          const Uintah::TypeDescription* subtype = td->getSubType();

          unsigned int i = 0;

          switch(td->getType()){
            case Uintah::TypeDescription::CCVariable:      // CC Variables
              switch(subtype->getType()) {
                case Uintah::TypeDescription::double_type:
                  for (i = 0; i < CC_double_data.size(); i++)
                    if (exc->vl->getName() == ccd_var_labels[i]->getName()) 
                      fprintf(fp, "    %16E",CC_double_data[i][c]);
                  break;
                case Uintah::TypeDescription::Vector:
                  for (i = 0; i < CC_Vector_data.size(); i++)
                    if (exc->vl->getName() == ccv_var_labels[i]->getName()) 
                      fprintf(fp, "    %16E    %16E    %16E", CC_Vector_data[i][c].x(), CC_Vector_data[i][c].y(), CC_Vector_data[i][c].z());
                  break;
                case Uintah::TypeDescription::int_type:
                  for (i = 0; i < CC_integer_data.size(); i++)
                    if (exc->vl->getName() == cci_var_labels[i]->getName()) 
                      fprintf(fp, "    %16i",CC_integer_data[i][c]);
                  break; 
                default:
                  throw InternalError("containerExtract: invalid data type", __FILE__, __LINE__); 
              }
              break;
            case Uintah::TypeDescription::SFCXVariable:   // SFCX Variables
              for (i = 0; i < SFCX_double_data.size(); i++)
                if (exc->vl->getName() == sfcx_var_labels[i]->getName()) 
                  fprintf(fp, "    %16E",SFCX_double_data[i][c]);
              break;
            case Uintah::TypeDescription::SFCYVariable:    // SFCY Variables
              for (i = 0; i < SFCY_double_data.size(); i++)
                if (exc->vl->getName() == sfcy_var_labels[i]->getName()) 
                  fprintf(fp, "    %16E",SFCY_double_data[i][c]);
              break;
            case Uintah::TypeDescription::SFCZVariable:   // SFCZ Variables
              for (i = 0; i < SFCZ_double_data.size(); i++)
                if (exc->vl->getName() == sfcz_var_labels[i]->getName()) 
                  fprintf(fp, "    %16E",SFCZ_double_data[i][c]);
              break;
            default:
              ostringstream warn;
              warn << "ERROR:AnalysisModule:containerExtract: ("<<exc->vl->getName() << " " 
                << td->getName() << " ) has not been implemented" << endl;
              throw InternalError(warn.str(), __FILE__, __LINE__);
          }

          fprintf(fp,    "\n");
          fclose(fp);

        } //loop over extractCells
      } //loop over d_containers
      lastWriteTime = now;     
    }  // time to write data

    new_dw->put(max_vartype(lastWriteTime), ps_lb->lastWriteTimeLabel); 
  }  // patches
}
//______________________________________________________________________
//  Open the file if it doesn't exist and write the file header
void containerExtract::createFile(string& filename, extractCell& exc)
{ 
  FILE *fp;
  fp = fopen(filename.c_str(), "w");
  fprintf(fp,"X_CC      Y_CC      Z_CC      Time"); 

  fprintf(fp,"    %s", exc.vl->getName().c_str());
  
  fprintf(fp,"\n");
  fclose(fp);
  cout << Parallel::getMPIRank() << " containerExtract:Created file " << filename << endl;
}
//______________________________________________________________________
// create the directory structure   lineName/LevelIndex
//
void
containerExtract::createDirectory(string& lineName, string& levelIndex)
{
  DIR *check = opendir(lineName.c_str());
  if ( check == NULL ) {
    cout << Parallel::getMPIRank() << "containerExtract:Making directory " << lineName << endl;
    MKDIR( lineName.c_str(), 0777 );
  } else {
    closedir(check);
  }
  
  // level index
  string path = lineName + "/" + levelIndex;
  check = opendir(path.c_str());
  if ( check == NULL ) {
    cout << "containerExtract:Making directory " << path << endl;
    MKDIR( path.c_str(), 0777 );
  } else {
    closedir(check);
  }
}
