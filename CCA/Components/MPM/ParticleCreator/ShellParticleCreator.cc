#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ShellParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/ShellGeometryPiece.h>

using namespace Uintah;


/////////////////////////////////////////////////////////////////////////
//
// Constructor
//
ShellParticleCreator::ShellParticleCreator(MPMMaterial* matl,
					   MPMLabel* lb,
					   int n8or27,
					   bool haveLoadCurve,
					   bool doErosion)
  : ParticleCreator(matl,lb,n8or27,haveLoadCurve, doErosion)
{
  // Transfer to the lb's permanent particle state array of vectors
  lb->d_particleState.push_back(particle_state);
  lb->d_particleState_preReloc.push_back(particle_state_preReloc);
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
				      MPMLabel* lb,
				      vector<GeometryObject*>& d_geom_objs)
{
  // Print the physical boundary conditions
  printPhysicalBCs();

  // Get datawarehouse index
  int dwi = matl->getDWIndex();

  // Create a particle subset for the patch
  ParticleSubset* subset = ParticleCreator::allocateVariables(numParticles,
							      dwi, lb, patch,
							      new_dw);
  // Create the variables that go with each shell particle
  ParticleVariable<double>  pThickTop0, pThickBot0, pThickTop, pThickBot;
  ParticleVariable<Vector>  pNormal0, pNormal;
  new_dw->allocateAndPut(pThickTop,   lb->pThickTopLabel,        subset);
  new_dw->allocateAndPut(pThickTop0,  lb->pInitialThickTopLabel, subset);
  new_dw->allocateAndPut(pThickBot,   lb->pThickBotLabel,        subset);
  new_dw->allocateAndPut(pThickBot0,  lb->pInitialThickBotLabel, subset);
  new_dw->allocateAndPut(pNormal,     lb->pNormalLabel,          subset);
  new_dw->allocateAndPut(pNormal0,    lb->pInitialNormalLabel,   subset);

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
    Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
    Vector dcorner = dxpp*0.5;
    Vector size(1./((double) ppc.x()), 1./((double) ppc.y()),
		1./((double) ppc.z()));
    
    // If the geometry object is a shell perform special 
    // operations else just treat the geom object in the standard
    // way
    ShellGeometryPiece* shell = dynamic_cast<ShellGeometryPiece*>(piece);

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
	pvelocity[pidx]=(*obj)->getInitialVelocity();
	ptemperature[pidx]=(*obj)->getInitialTemperature();
	psp_vol[pidx]=1.0/matl->getInitialDensity();
#ifdef FRACTURE
        pdisp[pidx] = Vector(0.,0.,0.);
#endif
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
	  cerr << "cellID is not right" << endl;
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
#ifdef FRACTURE
                pdisp[pidx] = Vector(0.,0.,0.);
#endif
		pvolume[pidx]=dxpp.x()*dxpp.y()*dxpp.z();
		pvelocity[pidx]=(*obj)->getInitialVelocity();
		ptemperature[pidx]=(*obj)->getInitialTemperature();
	        psp_vol[pidx]=1.0/matl->getInitialDensity(); 

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
				     vector<GeometryObject*>& d_geom_objs) const
{
  return ParticleCreator::countParticles(patch,d_geom_objs);
}

/////////////////////////////////////////////////////////////////////////
//
// Return number of particles
//
particleIndex 
ShellParticleCreator::countParticles(GeometryObject* obj,
				     const Patch* patch) const
{

  GeometryPiece* piece = obj->getPiece();
  ShellGeometryPiece* shell = dynamic_cast<ShellGeometryPiece*>(piece);
  if (shell) return shell->returnParticleCount(patch);
  return ParticleCreator::countParticles(obj,patch); 
}

/////////////////////////////////////////////////////////////////////////
//
// Register variables for crossing patches
//
void 
ShellParticleCreator::registerPermanentParticleState(MPMMaterial* matl,
						     MPMLabel* lb)

{
  particle_state.push_back(lb->pThickTopLabel);
  particle_state.push_back(lb->pInitialThickTopLabel);
  particle_state.push_back(lb->pThickBotLabel);
  particle_state.push_back(lb->pInitialThickBotLabel);
  particle_state.push_back(lb->pNormalLabel);
  particle_state.push_back(lb->pInitialNormalLabel);

  particle_state_preReloc.push_back(lb->pThickTopLabel_preReloc);
  particle_state_preReloc.push_back(lb->pInitialThickTopLabel_preReloc);
  particle_state_preReloc.push_back(lb->pThickBotLabel_preReloc);
  particle_state_preReloc.push_back(lb->pInitialThickBotLabel_preReloc);
  particle_state_preReloc.push_back(lb->pNormalLabel_preReloc);
  particle_state_preReloc.push_back(lb->pInitialNormalLabel_preReloc);
}
