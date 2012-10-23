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

#include <CCA/Components/MPM/ParticleCreator/ShellParticleCreator.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Box.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/ShellGeometryPiece.h>
#include <iostream>

using namespace std;
using namespace Uintah;


/////////////////////////////////////////////////////////////////////////
//
// Constructor
//
ShellParticleCreator::ShellParticleCreator(MPMMaterial* matl,
                                           MPMFlags* flags)
  : ParticleCreator(matl,flags)
{
}

/////////////////////////////////////////////////////////////////////////
//
// Destructor
//
ShellParticleCreator::~ShellParticleCreator()
{
}

/////////////////////////////////////////////////////////////////////////
//
// Actually create particles using geometry
//
ParticleSubset* 
ShellParticleCreator::createParticles(MPMMaterial* matl, 
                                      particleIndex numParticles,
                                      CCVariable<short int>& cellNAPID,
                                      const Patch* patch,
                                      DataWarehouse* new_dw,
                                      vector<GeometryObject*>& d_geom_objs)
{
  // Print the physical boundary conditions
  printPhysicalBCs();

  // Get datawarehouse index
  int dwi = matl->getDWIndex();

  // Create a particle subset for the patch
  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
                                                              dwi, patch,
                                                              new_dw);
  // Create the variables that go with each shell particle
  ParticleVariable<double>  pThickTop0, pThickBot0, pThickTop, pThickBot;
  ParticleVariable<Vector>  pNormal0, pNormal;
  new_dw->allocateAndPut(pThickTop,   d_lb->pThickTopLabel,        subset);
  new_dw->allocateAndPut(pThickTop0,  d_lb->pInitialThickTopLabel, subset);
  new_dw->allocateAndPut(pThickBot,   d_lb->pThickBotLabel,        subset);
  new_dw->allocateAndPut(pThickBot0,  d_lb->pInitialThickBotLabel, subset);
  new_dw->allocateAndPut(pNormal,     d_lb->pNormalLabel,          subset);
  new_dw->allocateAndPut(pNormal0,    d_lb->pInitialNormalLabel,   subset);

  // Initialize the global particle index
  particleIndex start = 0;

  // Loop thru the geometry objects 
  vector<GeometryObject*>::const_iterator obj;
  for (obj = d_geom_objs.begin(); obj != d_geom_objs.end(); ++obj) {  

    // Initialize the per geometryObject particle count
    particleIndex count = 0;

    // If the geometry piece is outside the patch, look
    // for the next geometry piece
    GeometryPieceP piece = (*obj)->getPiece();
    Box b = (piece->getBoundingBox()).intersect(patch->getExtraBox());
    if (b.degenerate()) {
      count = 0;
      continue;
    }

    // Find volume of influence of each particle as a
    // fraction of the cell size
    IntVector ppc = (*obj)->getInitialData_IntVector("res");
    Vector dxpp = patch->dCell()/(*obj)->getInitialData_IntVector("res");
    Vector dcorner = dxpp*0.5;
    Matrix3 size(1./((double) ppc.x()),0.,0.,
                 0.,1./((double) ppc.y()),0.,
                 0.,0.,1./((double) ppc.z()));
    
    // If the geometry object is a shell perform special 
    // operations else just treat the geom object in the standard
    // way
    ShellGeometryPiece* shell = dynamic_cast<ShellGeometryPiece*>(piece.get_rep());

    // Create the appropriate particles 
    if (shell) {

      // The position, volume and size variables are from the 
      // ParticleCreator class
      int numP = shell->createParticles(patch, position, pvolume,
                                        pThickTop, pThickBot, pNormal, 
                                        psize, start);

      // Update the other variables that are attached to each particle
      // (declared in the ParticleCreator class)
      for (int idx = 0; idx < numP; idx++) {
        particleIndex pidx = start+idx;
        pvelocity[pidx]=(*obj)->getInitialData_Vector ("velocity");
        ptemperature[pidx]=(*obj)->getInitialData_double("temperature");
        pdisp[pidx] = Vector(0.,0.,0.);
        pfiberdir[pidx] = Vector(0.0,0.0,0.0);

        // Calculate particle mass
        double partMass = matl->getInitialDensity()*pvolume[pidx];
        pmass[pidx] = partMass;

        // The particle can be tagged as a surface particle.
        // If there is a physical BC attached to it then mark with the 
        // physical BC pointer
        if (d_useLoadCurves) 
          pLoadCurveID[pidx] = getLoadCurveID(position[pidx], dxpp);

        // Apply the force BC if applicable
        Vector pExtForce(0,0,0);
        ParticleCreator::applyForceBC(dxpp, position[pidx], partMass, 
                                      pExtForce);
        pexternalforce[pidx] = pExtForce;
                
        // Assign a particle id
        IntVector cell_idx;
        if (patch->findCell(position[pidx],cell_idx)) {
          long64 cellID = ((long64)cell_idx.x() << 16) |
            ((long64)cell_idx.y() << 32) |
            ((long64)cell_idx.z() << 48);
          short int& myCellNAPID = cellNAPID[cell_idx];
          ASSERT(myCellNAPID < 0x7fff);
          myCellNAPID++;
          pparticleID[pidx] = cellID | (long64)myCellNAPID;
        } else {
          double x = position[pidx].x();
          double y = position[pidx].y();
          double z = position[pidx].z();
          if (fabs(x) < 1.0e-15) x = 0.0;
          if (fabs(y) < 1.0e-15) y = 0.0;
          if (fabs(z) < 1.0e-15) z = 0.0;
          double px = patch->getExtraBox().upper().x();
          double py = patch->getExtraBox().upper().y();
          double pz = patch->getExtraBox().upper().z();
          if (fabs(px) < 1.0e-15) px = 0.0;
          if (fabs(py) < 1.0e-15) py = 0.0;
          if (fabs(pz) < 1.0e-15) pz = 0.0;
          if (x == px) x -= 1.0e-10;
          if (y == py) y -= 1.0e-10;
          if (z == pz) z -= 1.0e-10;
          position[pidx] = Point(x,y,z);
          if (!patch->findCell(position[pidx],cell_idx)) {
            cerr << "Pidx = " << pidx << " Pos = " << position[pidx]
                 << " patch BBox = " << patch->getExtraBox()
                 << " cell_idx = " << cell_idx
                 << " low = " << patch->getExtraCellLowIndex()
                 << " high = " << patch->getExtraCellHighIndex()
                 << " : Particle not in any cell." << endl;
            pparticleID[pidx] = 0;
          } else {
            long64 cellID = ((long64)cell_idx.x() << 16) |
              ((long64)cell_idx.y() << 32) |
              ((long64)cell_idx.z() << 48);
            short int& myCellNAPID = cellNAPID[cell_idx];
            ASSERT(myCellNAPID < 0x7fff);
            myCellNAPID++;
            pparticleID[pidx] = cellID | (long64)myCellNAPID;
          }
        }

        // The shell specific variables
        pThickTop0[pidx]  = pThickTop[pidx]; 
        pThickBot0[pidx]  = pThickBot[pidx]; 
        pNormal0[pidx]    = pNormal[pidx]; 

      } // End of loop thry particles per geom-object

    } else {
     
      // Loop thru cells and assign particles
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        Point lower = patch->nodePosition(*iter) + dcorner;
        for(int ix=0;ix < ppc.x(); ix++){
          for(int iy=0;iy < ppc.y(); iy++){
            for(int iz=0;iz < ppc.z(); iz++){
              IntVector idx(ix, iy, iz);
              Point p = lower + dxpp*idx;
              IntVector cell_idx = *iter;
              // If the assertion fails then we may just need to change
              // the format of particle ids such that the cell indices
              // have more bits.
              ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
                     && cell_idx.z() <= 0xffff);
              long64 cellID = ((long64)cell_idx.x() << 16) |
                ((long64)cell_idx.y() << 32) |
                ((long64)cell_idx.z() << 48);
              if(piece->inside(p)){
                particleIndex pidx = start+count; 
                position[pidx]=p;
                pdisp[pidx] = Vector(0.,0.,0.);
                pvolume[pidx]=dxpp.x()*dxpp.y()*dxpp.z();
                pvelocity[pidx]=(*obj)->getInitialData_Vector("velocity");
                ptemperature[pidx]=(*obj)->getInitialData_double("temperature");
                psp_vol[pidx]=1.0/matl->getInitialDensity(); 
                pfiberdir[pidx] = Vector(0.0,0.0,0.0);

                // Calculate particle mass
                double partMass =matl->getInitialDensity()*pvolume[pidx];
                pmass[pidx] = partMass;
                
                // If the particle is on the surface and if there is
                // a physical BC attached to it then mark with the 
                // physical BC pointer
                if (d_useLoadCurves) {
                  if (checkForSurface(piece,p,dxpp)) {
                    pLoadCurveID[pidx] = getLoadCurveID(p, dxpp);
                  } else {
                    pLoadCurveID[pidx] = 0;
                  }
                }

                // Apply the force BC if applicable
                Vector pExtForce(0,0,0);
                ParticleCreator::applyForceBC(dxpp, p, partMass, pExtForce);
                pexternalforce[pidx] = pExtForce;
                
                // Assign particle id
                short int& myCellNAPID = cellNAPID[cell_idx];
                pparticleID[pidx] = cellID | (long64)myCellNAPID;
                psize[pidx] = size;
                ASSERT(myCellNAPID < 0x7fff);
                myCellNAPID++;
                count++;
                
                // Assign dummy values to shell-specific variables
                pThickTop[pidx]   = 1.0;
                pThickTop0[pidx]  = 1.0;
                pThickBot[pidx]   = 1.0;
                pThickBot0[pidx]  = 1.0;
                pNormal[pidx]     = Vector(0,1,0);
                pNormal0[pidx]    = Vector(0,1,0);
              }  // if inside
            }  // loop in z
          }  // loop in y
        }  // loop in x
      } // for
    } // end of else
    start += count; 
  }

  return subset;
}

/////////////////////////////////////////////////////////////////////////
//
// Return number of particles
//
particleIndex 
ShellParticleCreator::countParticles(const Patch* patch,
                                     vector<GeometryObject*>& d_geom_objs) 
{
  return ParticleCreator::countParticles(patch,d_geom_objs);
}

/////////////////////////////////////////////////////////////////////////
//
// Return number of particles
//
particleIndex 
ShellParticleCreator::countAndCreateParticles(const Patch* patch,
                                              GeometryObject* obj) 
{

  GeometryPieceP piece = obj->getPiece();
  ShellGeometryPiece* shell = dynamic_cast<ShellGeometryPiece*>(piece.get_rep());
  if (shell) return shell->returnParticleCount(patch);
  return ParticleCreator::countAndCreateParticles(patch,obj); 
}

/////////////////////////////////////////////////////////////////////////
//
// Register variables for crossing patches
//
void 
ShellParticleCreator::registerPermanentParticleState(MPMMaterial*)
{
  particle_state.push_back(d_lb->pThickTopLabel);
  particle_state.push_back(d_lb->pInitialThickTopLabel);
  particle_state.push_back(d_lb->pThickBotLabel);
  particle_state.push_back(d_lb->pInitialThickBotLabel);
  particle_state.push_back(d_lb->pNormalLabel);
  particle_state.push_back(d_lb->pInitialNormalLabel);

  particle_state_preReloc.push_back(d_lb->pThickTopLabel_preReloc);
  particle_state_preReloc.push_back(d_lb->pInitialThickTopLabel_preReloc);
  particle_state_preReloc.push_back(d_lb->pThickBotLabel_preReloc);
  particle_state_preReloc.push_back(d_lb->pInitialThickBotLabel_preReloc);
  particle_state_preReloc.push_back(d_lb->pNormalLabel_preReloc);
  particle_state_preReloc.push_back(d_lb->pInitialNormalLabel_preReloc);
}
