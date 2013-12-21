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

#include <CCA/Components/MPM/ParticleCreator/GUVParticleCreator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GUVSphereShellPiece.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
using namespace Uintah;

static DebugStream debug("GUV_Create", false);
static DebugStream debug_loop("GUV_Create_Loop", false);
static DebugStream debug_other("GUV_Create_Other", false);
static DebugStream debug_loop_other("GUV_Create_LO", false);

/////////////////////////////////////////////////////////////////////////
//
// Constructor
//
GUVParticleCreator::GUVParticleCreator(MPMMaterial* matl,
                                       MPMLabel* lb,
                                       MPMFlags* flags)
  : ShellParticleCreator(matl,lb, flags)
{
}

/////////////////////////////////////////////////////////////////////////
//
// Destructor
//
GUVParticleCreator::~GUVParticleCreator()
{
}

/////////////////////////////////////////////////////////////////////////
//
// Actually create particles using geometry
//
ParticleSubset* 
GUVParticleCreator::createParticles(MPMMaterial* matl, 
                                    particleIndex numParticles,
                                    CCVariable<short int>& cellNAPID,
                                    const Patch* patch,
                                    DataWarehouse* new_dw,
                                    MPMLabel* lb,
                                    vector<GeometryObject*>& d_geom_objs)
{
  // Constants
  Vector zero(0.0,0.0,0.0);

  // Get datawarehouse index
  int dwi = matl->getDWIndex();

  // Create a particle subset for the patch
  debug_other << "GUVPartCreator::create:: numParticles = " << numParticles
              << " dwi = " << dwi << " lb = " << lb 
              << " patch = " << patch << " new_dw = " << new_dw << endl;
  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
                                                              dwi, lb, patch,
                                                              new_dw);
  // Create the variables that go with each guv particle
  ParticleVariable<int>     pType;
  ParticleVariable<double>  pThick0, pThick;
  ParticleVariable<Vector>  pNormal0, pNormal;
  new_dw->allocateAndPut(pType,    lb->pTypeLabel,            subset);
  new_dw->allocateAndPut(pThick,   lb->pThickTopLabel,        subset);
  new_dw->allocateAndPut(pThick0,  lb->pInitialThickTopLabel, subset);
  new_dw->allocateAndPut(pNormal,  lb->pNormalLabel,          subset);
  new_dw->allocateAndPut(pNormal0, lb->pInitialNormalLabel,   subset);

  // Initialize the global particle index
  particleIndex start = 0;

  // Loop thru the geometry objects 
  vector<GeometryObject*>::const_iterator obj;
  for (obj = d_geom_objs.begin(); obj != d_geom_objs.end(); ++obj) {  

    // Initialize the per geometryObject particle count
    particleIndex count = 0;

    // If the geometry piece is outside the patch, look
    // for the next geometry piece
    GeometryPiece* piece = (*obj)->getPiece();
    Box b = (piece->getBoundingBox()).intersect(patch->getBox());
    if (b.degenerate()) {
      count = 0;
      continue;
    }
    
    // Find volume of influence of each particle as a
    // fraction of the cell size
    IntVector ppc = (*obj)->getNumParticlesPerCell();
    Vector size(1./((double) ppc.x()), 1./((double) ppc.y()),
                1./((double) ppc.z()));
    
    // If the geometry object is a guv perform special 
    // operations else just treat the geom object in the standard
    // way
    GUVSphereShellPiece* guv = dynamic_cast<GUVSphereShellPiece*>(piece);
    debug << "GUVPartCreator::create::piece = " << piece 
          << " guv = " << guv << endl;

    // Get the GUV data
    if (guv) {

      vector<double>* vol = guv->getVolume();
      debug <<"GUVPartCreator::create::vol = "<< vol
            <<" size = "<<vol->size()<< endl;
      geomvols::key_type volkey(patch,*obj);
      vector<double>::const_iterator voliter = d_vol[volkey].begin();

      vector<int>* type = guv->getType();
      debug <<"GUVPartCreator::create::type = "<< type
            <<" size = "<<type->size()<< endl;
      geomint::key_type typekey(patch,*obj);
      vector<int>::const_iterator typeiter = d_type[typekey].begin();

      vector<double>* thick = guv->getThickness();
      debug <<"GUVPartCreator::create::thick = "<< thick
            <<" size = "<<thick->size()<<endl;
      geomvols::key_type thickkey(patch,*obj);
      vector<double>::const_iterator thickiter = d_thick[thickkey].begin();

      vector<Vector>* norm = guv->getNormal();
      debug <<"GUVPartCreator::create::norm = "<< norm
            <<" size = "<<norm->size()<< endl;
      geomvecs::key_type normkey(patch,*obj);
      vector<Vector>::const_iterator normiter = d_norm[normkey].begin();

      vector<Point>* pos = guv->get_position();
      debug <<"GUVPartCreator::create::pos = "<< pos
            <<" size = "<<pos->size()<< endl;
      geompoints::key_type poskey(patch,*obj);
      vector<Point>::const_iterator positer = d_pos[poskey].begin();

      debug_other << "GUVPartCreator::create:: patch = " << patch
                  << " obj = " << *obj << endl;

      positer = d_pos[poskey].begin();
      debug << "GUVPartCreator::create::Before positer loop : d_pos size " 
            << d_pos[poskey].size() << endl;
      IntVector cell_idx(0,0,0);
      for (int ct = 0; positer != d_pos[poskey].end() ; ++positer,++ct) {

        debug_loop << "GUVPartCreator::create::particle " << ct 
                   << " pt = " << *positer << endl;

        // if the point is not inside the current patch there is a serious
        // memory problem
        Point p = *positer;
        if (!patch->findCell(p, cell_idx)) {
          throw InternalError("GUVPartCreator::createParticles::**ERROR** \
                               Memory Corruption ?");
        }
        debug_other << "GUVPartCreator::create:: Point = " << ct
                    << " coords = " << p 
                    << " cell = " << cell_idx << endl;
        debug_loop << "GUVPartCreator::create::cell_idx = " << cell_idx << endl;
        debug << "GUVPartCreator::create::particle " << ct 
              << " Pos = " << *positer 
              << " Cell = " << cell_idx << endl;

        particleIndex pidx = start+count;
        debug_loop << "GUVPartCreator::create::pidx = " << pidx << endl;

        pvelocity[pidx]=(*obj)->getInitialVelocity();
        ptemperature[pidx]=(*obj)->getInitialTemperature();
        psp_vol[pidx]=1.0/matl->getInitialDensity();
        pdisp[pidx] = zero;
        pexternalforce[pidx] = zero;
        debug_loop << "GUVPartCreator::create::velocity = " << pvelocity[pidx] 
              << " temperature = " << ptemperature[pidx]
              << " sp_vol = " << psp_vol[pidx]
              << " disp = " << pdisp[pidx]
              << " externalforce = " << pexternalforce[pidx] << endl;
                
        position[pidx] = p; 
        psize[pidx] = size;
        pvolume[pidx] = *voliter; ++voliter;
        pmass[pidx] = matl->getInitialDensity()*pvolume[pidx];
        debug_loop << "GUVPartCreator::create::position = " << position[pidx] 
              << " size = " << psize[pidx]
              << " volume = " << pvolume[pidx]
              << " mass = " << pmass[pidx] << endl;

        pType[pidx] = *typeiter; ++typeiter;
        pThick[pidx] = *thickiter; ++thickiter;
        pThick0[pidx] = pThick[pidx]; 
        pNormal[pidx] = *normiter; ++normiter;
        pNormal0[pidx] = pNormal[pidx]; 
        debug_loop_other << "GUVPartCreator::create::Type = " << pType[pidx] 
              << " Thick = " << pThick[pidx]
              << " Thick0 = " << pThick0[pidx]
              << " Normal = " << pNormal[pidx]
              << " Normal0 = " << pNormal0[pidx]
              << " volume = " << pvolume[pidx] << endl;
       
        long64 cellID = ((long64)cell_idx.x() << 16) |
          ((long64)cell_idx.y() << 32) |
          ((long64)cell_idx.z() << 48);
        short int& myCellNAPID = cellNAPID[cell_idx];
        ASSERT(myCellNAPID < 0x7fff);
        myCellNAPID++;
        pparticleID[pidx] = cellID | (long64)myCellNAPID;
        debug_loop << "GUVPartCreator::create::particleID = " 
              << pparticleID[pidx] << endl;
      count++;
      }

    } else {
      cout << "**WARNING** GUV materials cannot interact with other materials."
           << endl;
    }
    start += count; 
  }

  return subset;
}

/////////////////////////////////////////////////////////////////////////
//
// Return number of particles
//
particleIndex 
GUVParticleCreator::countAndCreateParticles(const Patch* patch,
                                            GeometryObject* obj) 
{
  GeometryPiece* piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;

  geompoints::key_type poskey(patch, obj);
  geomvols::key_type volkey(patch, obj);
  geomint::key_type typekey(patch, obj);
  geomvols::key_type thickkey(patch, obj);
  geomvecs::key_type normkey(patch, obj);

  GUVSphereShellPiece* guv = dynamic_cast<GUVSphereShellPiece*>(piece);
  debug << "GUVPartCreator::count:: patch = " << patch
        << " guv = " << guv << endl;

  if (guv) {

    Vector dxpp = patch->dCell()/obj->getNumParticlesPerCell();    
    double dx = Min(Min(dxpp.x(),dxpp.y()), dxpp.z());
    guv->setParticleSpacing(dx);
    debug << "GUVPartCreator::count:: dx = " << dx << endl;

    int numPts = guv->createPoints();
    debug << "GUVPartCreator::count:: numPts = " << numPts << endl;

    vector<Point>* pos = guv->get_position();
    vector<double>* vol = guv->getVolume();
    vector<int>* type = guv->getType();
    vector<double>* thick = guv->getThickness();
    vector<Vector>* norm = guv->getNormal();
    debug <<"GUVPartCreator::count::pos = "<< pos
          <<" size = "<<pos->size()<< endl;
    debug <<"GUVPartCreator::count::vol = "<< vol
          <<" size = "<<vol->size()<< endl;
    debug <<"GUVPartCreator::count::type = "<< type
          <<" size = "<<type->size()<< endl;
    debug <<"GUVPartCreator::count::thick = "<< thick
          <<" size = "<<thick->size()<<endl;
    debug <<"GUVPartCreator::count::norm = "<< norm
          <<" size = "<<norm->size()<< endl;

    Point p;
    IntVector cell_idx;
    int count = 0;
    for (int ii = 0; ii < numPts; ++ii) {
      p = pos->at(ii);
      if (patch->findCell(p, cell_idx)) {
        debug_loop << "GUVPartCreator::count::particle " << count 
              << " Pos = " << p 
              << " Cell = " << cell_idx << endl;

        d_pos[poskey].push_back(p);
        d_vol[volkey].push_back(vol->at(ii));
        d_type[typekey].push_back(type->at(ii));
        d_thick[thickkey].push_back(thick->at(ii));
        d_norm[normkey].push_back(norm->at(ii));

        ++count;
        debug_loop <<"GUVPartCreator::count::count = " << count 
              << " numPts = " << numPts << endl;
      }
    }
  }

  debug_other << "GUVPartCreator::count:: patch = " << patch
              << " obj = " << obj << endl;
  vector<Point>::const_iterator positer = d_pos[poskey].begin();
  IntVector cell(0,0,0);
  for (int ct = 0; positer != d_pos[poskey].end() ; ++positer,++ct) {
    patch->findCell(*positer, cell);
    debug_other << "GUVPartCreator::count:: Point = " << ct
                << " coords = " << *positer 
                << " cell = " << cell << endl;
  }
  debug <<"GUVPartCreator::count::poskey.size = " 
        << d_pos[poskey].size() << endl;
  return d_pos[poskey].size();
}
