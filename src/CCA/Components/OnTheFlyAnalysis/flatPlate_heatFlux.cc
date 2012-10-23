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

#include <CCA/Components/OnTheFlyAnalysis/flatPlate_heatFlux.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <Core/Exceptions/InternalError.h>
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
//______________________________________________________________________ 
//  To turn on the output
//  setenv SCI_DEBUG "FLATPLATE_HEATFLUX_DBG_COUT:+" 
static DebugStream cout_doing("FLATPLATE_HEATFLUX_DOING_COUT", false);
static DebugStream cout_dbg("FLATPLATE_HEATFLUX_DBG_COUT", false);
//______________________________________________________________________              
flatPlate_heatFlux::flatPlate_heatFlux(ProblemSpecP& module_spec,
                                       SimulationStateP& sharedState,
                                       Output* dataArchiver)
  : AnalysisModule(module_spec, sharedState, dataArchiver)
{
  d_sharedState  = sharedState;
  d_prob_spec    = module_spec;
  d_dataArchiver = dataArchiver;
  d_matl_set = 0;
  v_lb = scinew total_heatRateLabel();
  M_lb = scinew MPMLabel();
}

//__________________________________
flatPlate_heatFlux::~flatPlate_heatFlux()
{
  cout_doing << " Doing: destorying flatPlate_heatFlux " << endl;
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  VarLabel::destroy(v_lb->total_heatRateLabel);
  delete v_lb;
  delete M_lb;
  
  // delete each plane
  vector<plane*>::iterator iter;
  for( iter  = d_plane.begin();iter != d_plane.end(); iter++){
    delete *iter;
  }
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void flatPlate_heatFlux::problemSetup(const ProblemSpecP& prob_spec,
                                      GridP& grid,
                                      SimulationStateP& sharedState)
{
  cout_doing << "Doing problemSetup \t\t\t\tflatPlate_heatFlux" << endl;
  
  if(!d_dataArchiver){
    throw InternalError("flatPlate_heatFlux:couldn't get output port", __FILE__, __LINE__);
  }
  
  v_lb->total_heatRateLabel = VarLabel::create("total_heatRate",  sum_vartype::getTypeDescription());

  // determine which material index to compute
  d_matl = d_sharedState->parseAndLookupMaterial(d_prob_spec, "material");
  
  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_sub = d_matl_set->getUnion();
  
  ProblemSpecP plane_ps = d_prob_spec->findBlock("plane"); 
  if (!plane_ps){
    throw ProblemSetupException("\n ERROR:flatPlate_heatFlux: Couldn't find <plane> tag \n", __FILE__, __LINE__);    
  }
  Point start, end;
  plane_ps->require("startingPt", start);
  plane_ps->require("endingPt",   end);
  
  
  //__________________________________
  // bullet proofing
  // -plane must be parallel to the coordinate system
  // -plane can't exceed computational domain
  // -define the corner points on the plane

  // plane must be parallel to the coordinate system
  bool X = (start.x() == end.x());
  bool Y = (start.y() == end.y());  // 1 out of 3 of these must be true
  bool Z = (start.z() == end.z());
  
  bool validPlane = false;
  
  if( !X && !Y && Z){
    validPlane = true;
    d_oneOrZero= Vector(1,1,0);
    d_corner_pt[0] = start;
    d_corner_pt[1] = Point(start.x(), end.y(),  start.z());
    d_corner_pt[2] = Point(end.x(),   end.y(),  start.z());
    d_corner_pt[3] = Point(end.x(),   start.y(),start.z());
  }

  if( !X && Y && !Z){
    validPlane = true;
    d_oneOrZero = Vector(1,0,1);
    d_corner_pt[0] = start;
    d_corner_pt[1] = Point(end.x(),   start.y(), start.z());
    d_corner_pt[2] = Point(end.x(),   start.y(), end.z());
    d_corner_pt[3] = Point(start.x(), start.y(), end.z());
  } 
  if( X && !Y && !Z){
    validPlane = true;
    d_oneOrZero = Vector(0,1,1);
    d_corner_pt[0] = start;
    d_corner_pt[1] = Point(start.x(), end.y(),  start.z());
    d_corner_pt[2] = Point(start.x(), end.y(),  end.z());
    d_corner_pt[3] = Point(start.x(), start.y(),end.z());
  }
   
  if(validPlane == false){
    ostringstream warn;
    warn << "\n ERROR:PlaneExtract: the plane that you've specified " << start 
         << " " << end << " is not parallel to the coordinate system. \n" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  //the plane can't exceed computational domain
  BBox compDomain;
  grid->getInteriorSpatialRange(compDomain);

  Point min = compDomain.min();
  Point max = compDomain.max();

  if(start.x() < min.x() || start.y() < min.y() ||start.z() < min.z() ||
     end.x() > max.x()   || end.y() > max.y()   || end.z() > max.z() ){
    ostringstream warn;
    warn << "\n ERROR:flatPlat_heatFlux: a portion of plane that you've specified " << start 
         << " " << end << " lies outside of the computational domain. \n" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  
  if(start.x() > end.x() || start.y() > end.y() || start.z() > end.z() ) {
    ostringstream warn;
    warn << "\n ERROR:flatPlate_heatFlux: the plane that you've specified " << start 
         << " " << end << " the starting point is > than the ending point \n" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  // put the input variables into the global struct
  // only 1 plane for now
  plane* p = scinew plane;
  p->startPt = start;
  p->endPt   = end;
  d_plane.push_back(p);
  
}

//______________________________________________________________________
void flatPlate_heatFlux::scheduleInitialize(SchedulerP& sched,
                                            const LevelP& level)
{
  return;  // do nothing
}

void flatPlate_heatFlux::initialize(const ProcessorGroup*, 
                                    const PatchSubset* patches,
                                    const MaterialSubset*,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{  
}

void flatPlate_heatFlux::restartInitialize()
{
}

//______________________________________________________________________
void flatPlate_heatFlux::scheduleDoAnalysis(SchedulerP& sched,
                                   const LevelP& level)
{
  cout_doing << "flatPlate_heatFlux::scheduleDoAnalysis " << endl;
  Task* t = scinew Task("flatPlate_heatFlux::doAnalysis", 
                   this,&flatPlate_heatFlux::doAnalysis);
  
  Ghost::GhostType gn = Ghost::None;
  
  t->requires(Task::NewDW, M_lb->gHeatFluxLabel, d_matl_sub, gn,0);
  t->computes(v_lb->total_heatRateLabel);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
  
  //cout << " is required " << sched->isVariableRequired(M_lb->gHeatFluxLabel) << endl;
}

//______________________________________________________________________
// Compute the total heatRate field.
void flatPlate_heatFlux::doAnalysis(const ProcessorGroup* pg,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matl_sub ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{       
  const Level* level = getLevel(patches);
  Vector total_heatRate = Vector(0.0);  
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    cout_doing << pg->myrank() << " " 
               << "Doing doAnalysis (flatPlate_heatFlux)\t\t\t\tL-"
               << level->getIndex()
               << " patch " << patch->getGridIndex()<< endl;
                
    Ghost::GhostType gn = Ghost::None;
    int indx = d_matl->getDWIndex();
    
    constNCVariable<Vector> gHeatFlux;
    NCVariable<Vector> gHeatRate;
    
    new_dw->get(gHeatFlux, M_lb->gHeatFluxLabel, indx,patch,gn, 0);
    new_dw->allocateTemporary(gHeatRate,patch);
    gHeatRate.initialize(Vector(0.0));
             
    // find the physical domain and cell index range
    // associated with this plane
    Point start_pt = d_plane[0]->startPt;
    Point end_pt   = d_plane[0]->endPt;
    
    Box patchDomain = patch->getBox();
    // intersection
    start_pt = Max(patchDomain.lower(), start_pt);
    end_pt   = Min(patchDomain.upper(), end_pt);
 
    //__________________________________
    // Find the node iterator for the plane on this patch
    Box box(start_pt,end_pt);
    NodeIterator planeIterLim = patch->getNodeIterator(box);
    
    Vector dx = patch->dCell();
    Vector delta = dx;
      
    // set delta[d] = 1 if that direction is normal to the plane  
    for (int d = 0; d< 3 ; d++ ){  
      if (d_oneOrZero[d] == 0){
        delta[d]=1;
      }
    }
    
    //__________________________________
    // Hit all the cells in the plane including edges and corner cells
    double surfaceArea = delta.x() * delta.y() * delta.z();
    for(NodeIterator iter=planeIterLim; !iter.done();iter++) {
      
      if (!patch->containsCell(*iter))
        continue;  // just in case - the point-to-cell logic might throw us off on patch boundaries...
        
      IntVector n = *iter;
      gHeatRate[n] = surfaceArea * gHeatFlux[n];
    }
    
    //__________________________________
    // Hit the edges of the plane 
    vector<Patch::FaceType>::const_iterator iter;
    vector<Patch::FaceType> bf;
    patch->getBoundaryFaces(bf);

    double edgeSurfaceArea = 0.5 * delta.x() * delta.y() * delta.z();
    
    Box edge[4];
    edge[0]=Box(d_corner_pt[0], d_corner_pt[1]);
    edge[1]=Box(d_corner_pt[1], d_corner_pt[2]);
    edge[2]=Box(d_corner_pt[3], d_corner_pt[2]);
    edge[3]=Box(d_corner_pt[0], d_corner_pt[3]);
    
    for(int e = 0; e <4 ; e++){
      NodeIterator edgeIterLim = patch->getNodeIterator(edge[e]);

      for(NodeIterator iter=edgeIterLim; !iter.done();iter++) {
        IntVector n = *iter;
        gHeatRate[n] = edgeSurfaceArea * gHeatFlux[n];
      }
    }

    //__________________________________
    //  Hit the corner Nodes
    double cornerSurfaceArea = 0.25 * delta.x() * delta.y() * delta.z();
    
    for(int c = 0; c< 4; c++){
      
      if(patch->containsPoint(d_corner_pt[c]) ){
        IntVector n = patch->findClosestNode(d_corner_pt[c]);
        
        gHeatRate[n] = cornerSurfaceArea * gHeatFlux[n];
      }
    }
    
    //__________________________________
    // Hit all the cells in plane
    // and compute the total heatRate
    for(NodeIterator iter=planeIterLim; !iter.done();iter++) {
      IntVector n = *iter;
      total_heatRate += gHeatRate[n];
    }
  }  // patches
  //cout << " total_heatRate: " << total_heatRate << endl;
  new_dw->put(sumvec_vartype(total_heatRate), v_lb->total_heatRateLabel);
}
