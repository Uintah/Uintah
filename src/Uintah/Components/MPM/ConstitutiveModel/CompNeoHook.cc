
#include "ConstitutiveModelFactory.h"
#include "CompNeoHook.h"
#include <SCICore/Malloc/Allocator.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Components/MPM/MPMLabel.h>
#include <SCICore/Malloc/Allocator.h>
#include <fstream>
#include <iostream>

#include <Uintah/Components/MPM/Fracture/Visibility.h>

using std::cerr;
using namespace Uintah::MPM;
using SCICore::Math::Min;
using SCICore::Math::Max;
using SCICore::Geometry::Vector;

CompNeoHook::CompNeoHook(ProblemSpecP& ps)
{
  ps->require("bulk_modulus",d_initialData.Bulk);
  ps->require("shear_modulus",d_initialData.Shear);

  p_cmdata_label = scinew VarLabel("p.cmdata",
                                ParticleVariable<CMData>::getTypeDescription());
  p_cmdata_label_preReloc = scinew VarLabel("p.cmdata+",
                                ParticleVariable<CMData>::getTypeDescription());

  bElBarLabel = scinew VarLabel("p.bElBar",
                ParticleVariable<Matrix3>::getTypeDescription());
 
  bElBarLabel_preReloc = scinew VarLabel("p.bElBar+",
                ParticleVariable<Matrix3>::getTypeDescription());
}

CompNeoHook::~CompNeoHook()
{
  // Destructor

  delete p_cmdata_label;
  delete p_cmdata_label_preReloc;
  delete bElBarLabel;
  delete bElBarLabel_preReloc;
 
}

void CompNeoHook::initializeCMData(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
   // Put stuff in here to initialize each particle's
   // constitutive model parameters and deformationMeasure
   Matrix3 Identity, zero(0.);
   Identity.Identity();

   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<CMData> cmdata;
   new_dw->allocate(cmdata, p_cmdata_label, pset);
   ParticleVariable<Matrix3> deformationGradient;
   new_dw->allocate(deformationGradient, lb->pDeformationMeasureLabel, pset);
   ParticleVariable<Matrix3> pstress;
   new_dw->allocate(pstress, lb->pStressLabel, pset);
   ParticleVariable<Matrix3> bElBar;
   new_dw->allocate(bElBar,  bElBarLabel, pset);

   for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++) {
          cmdata[*iter] = d_initialData;
          deformationGradient[*iter] = Identity;
          bElBar[*iter] = Identity;
          pstress[*iter] = zero;
   }
   new_dw->put(cmdata, p_cmdata_label);
   new_dw->put(deformationGradient, lb->pDeformationMeasureLabel);
   new_dw->put(pstress, lb->pStressLabel);
   new_dw->put(bElBar, bElBarLabel);

   computeStableTimestep(patch, matl, new_dw);

}

void CompNeoHook::addParticleState(std::vector<const VarLabel*>& from,
				   std::vector<const VarLabel*>& to)
{
   from.push_back(p_cmdata_label);
   from.push_back(bElBarLabel);
   to.push_back(p_cmdata_label_preReloc);
   to.push_back(bElBarLabel_preReloc);
}

void CompNeoHook::computeStableTimestep(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& new_dw)
{
   // This is only called for the initial timestep - all other timesteps
   // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int matlindex = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<CMData> cmdata;
  new_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  new_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  new_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     double mu = cmdata[idx].Shear;
     double bulk = cmdata[idx].Bulk;
     c_dil = sqrt((bulk + 4.*mu/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
    }
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel);
}

void CompNeoHook::computeStressTensor(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& old_dw,
                                        DataWarehouseP& new_dw)
{
  Matrix3 velGrad,Shear,fbar,deformationGradientInc;
  double J,p,muBar,IEl,U,W,se=0.;
  double c_dil=0.0,Jinc;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double onethird = (1.0/3.0);
  Matrix3 Identity;

  Identity.Identity();

  Vector dx = patch->dCell();
  double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

  int matlindex = matl->getDWIndex();
  // Create array for the particle position
  ParticleSubset* pset = old_dw->getParticleSubset(matlindex, patch);
  ParticleVariable<Point> px;
  old_dw->get(px, lb->pXLabel, pset);

  // Create array for the particle deformation
  ParticleVariable<Matrix3> deformationGradient;
  old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
  ParticleVariable<Matrix3> bElBar;
  old_dw->get(bElBar, bElBarLabel, pset);

  // Create array for the particle stress
  ParticleVariable<Matrix3> pstress;
  old_dw->get(pstress, lb->pStressLabel, pset);

  // Retrieve the array of constitutive parameters
  ParticleVariable<CMData> cmdata;
  old_dw->get(cmdata, p_cmdata_label, pset);
  ParticleVariable<double> pmass;
  old_dw->get(pmass, lb->pMassLabel, pset);
  ParticleVariable<double> pvolume;
  old_dw->get(pvolume, lb->pVolumeLabel, pset);
  ParticleVariable<Vector> pvelocity;
  old_dw->get(pvelocity, lb->pVelocityLabel, pset);
  
   // As a side-effect of computeStressTensor, pDilationalWaveSpeed
   // are calculated and for delT and saved that will be used later by fracture
  ParticleVariable<double> pDilationalWaveSpeed;
  new_dw->allocate(pDilationalWaveSpeed, lb->pDilationalWaveSpeedLabel, pset);

  NCVariable<Vector> gvelocity;

  new_dw->get(gvelocity, lb->gMomExedVelocityLabel, matlindex,patch,
            Ghost::AroundCells, 1);
  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel);

  ParticleVariable<int> pVisibility;
  if(matl->getFractureModel()) {
    new_dw->get(pVisibility, lb->pVisibilityLabel, pset);
  }

  for(ParticleSubset::iterator iter = pset->begin();
     iter != pset->end(); iter++){
     particleIndex idx = *iter;

     velGrad.set(0.0);
     // Get the node indices that surround the cell
     IntVector ni[8];
     Vector d_S[8];

     if(!patch->findCellAndShapeDerivatives(px[idx], ni, d_S))
           continue;

     Visibility vis;
     if(matl->getFractureModel()) {
  	vis = pVisibility[idx];
      	vis.modifyShapeDerivatives(d_S);
     }

     for(int k = 0; k < 8; k++) {
	 if(vis.visible(k) ) {
            Vector& gvel = gvelocity[ni[k]];
            for (int j = 0; j<3; j++){
              for (int i = 0; i<3; i++) {
                velGrad(i+1,j+1)+=gvel(i) * d_S[k](j) * oodx[j];
              }
            }
	 }
     }

    // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
    // time step and the velocity gradient and the material constants
    double shear = cmdata[idx].Shear;
    double bulk  = cmdata[idx].Bulk;

    // Compute the deformation gradient increment using the time_step
    // velocity gradient
    // F_n^np1 = dudx * dt + Identity
    deformationGradientInc = velGrad * delT + Identity;

    Jinc = deformationGradientInc.Determinant();

    // Update the deformation gradient tensor to its time n+1 value.
    deformationGradient[idx] = deformationGradientInc *
                             deformationGradient[idx];

    // get the volume preserving part of the deformation gradient increment
    fbar = deformationGradientInc * pow(Jinc,-onethird);

    bElBar[idx] = fbar*bElBar[idx]*fbar.Transpose();
    IEl = onethird*bElBar[idx].Trace();

    // Shear is equal to the shear modulus times dev(bElBar)
    Shear = (bElBar[idx] - Identity*IEl)*shear;

    // get the volumetric part of the deformation
    J = deformationGradient[idx].Determinant();

    // get the hydrostatic part of the stress
    p = 0.5*bulk*(J - 1.0/J);

    // compute the total stress (volumetric + deviatoric)
    pstress[idx] = Identity*p + Shear/J;

    // Compute the strain energy for all the particles
    U = .5*bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
    W = .5*shear*(bElBar[idx].Trace() - 3.0);

    pvolume[idx]=Jinc*pvolume[idx];

    se += (U + W)*pvolume[idx]/J;

    // Compute wave speed at each particle, store the maximum
    muBar = IEl * shear;

    if(pmass[idx] > 0){
      c_dil = sqrt((bulk + 4.*shear/3.)*pvolume[idx]/pmass[idx]);
    }
    else{
      c_dil = 0.0;
      pvelocity[idx] = Vector(0.0,0.0,0.0);
    }
    WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
		     Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
		     Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
		     
    pDilationalWaveSpeed[idx] = c_dil;
  }

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel);
  new_dw->put(pstress, lb->pStressLabel_preReloc);
  new_dw->put(deformationGradient, lb->pDeformationMeasureLabel_preReloc);
  new_dw->put(bElBar, bElBarLabel_preReloc);

  // Put the strain energy in the data warehouse
  new_dw->put(sum_vartype(se), lb->StrainEnergyLabel);

  // This is just carried forward
  new_dw->put(cmdata, p_cmdata_label_preReloc);
  // Store updated particle volume
  new_dw->put(pvolume,lb->pVolumeDeformedLabel);

  new_dw->put(pDilationalWaveSpeed, lb->pDilationalWaveSpeedLabel);
}

double CompNeoHook::computeStrainEnergy(const Patch* patch,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
  double se=0;

  return se;
}

void CompNeoHook::addComputesAndRequires(Task* task,
					 const MPMMaterial* matl,
					 const Patch* patch,
					 DataWarehouseP& old_dw,
					 DataWarehouseP& new_dw) const
{
   task->requires(old_dw, lb->pXLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->pDeformationMeasureLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, p_cmdata_label, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pMassLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(old_dw, lb->pVolumeLabel, matl->getDWIndex(),  patch,
                  Ghost::None);
   task->requires(new_dw, lb->gMomExedVelocityLabel, matl->getDWIndex(), patch,
                  Ghost::AroundCells, 1);
   task->requires(old_dw, bElBarLabel, matl->getDWIndex(), patch,
                  Ghost::None);
   task->requires(old_dw, lb->delTLabel);

   if(matl->getFractureModel()) {
      task->requires(new_dw, lb->pVisibilityLabel, matl->getDWIndex(), patch,
			Ghost::AroundNodes, 1 );
   }

   task->computes(new_dw, lb->pStressLabel_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pDeformationMeasureLabel_preReloc, matl->getDWIndex(), patch);
   task->computes(new_dw, bElBarLabel_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, p_cmdata_label_preReloc, matl->getDWIndex(),  patch);
   task->computes(new_dw, lb->pVolumeDeformedLabel, matl->getDWIndex(), patch);
   
   task->computes(new_dw, lb->pDilationalWaveSpeedLabel, matl->getDWIndex(), patch);
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace MPM {

static MPI_Datatype makeMPI_CMData()
{
   ASSERTEQ(sizeof(CompNeoHook::CMData), sizeof(double)*2);
   MPI_Datatype mpitype;
   MPI_Type_vector(1, 2, 2, MPI_DOUBLE, &mpitype);
   return mpitype;
}

const TypeDescription* fun_getTypeDescription(CompNeoHook::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      td = scinew TypeDescription(TypeDescription::Other,
			       "CompNeoHook::CMData", true, &makeMPI_CMData);
   }
   return td;
}
   }
}

// $Log$
// Revision 1.40  2000/09/09 20:23:20  tan
// Replace BrokenCellShapeFunction with particle visibility information in shape
// function computation.
//
// Revision 1.39  2000/09/08 18:23:56  tan
// Added visibility calculation to fracture broken cell shape function
// interpolation.
//
// Revision 1.38  2000/09/08 01:45:59  tan
// Added pDilationalWaveSpeedLabel for fracture and is saved as a
// side-effect of computeStressTensor in each constitutive model class.
//
// Revision 1.37  2000/09/07 21:11:09  tan
// Added particle variable pMicrocrackSize for fracture.
//
// Revision 1.36  2000/09/06 19:45:09  jas
// Changed new to scinew in constitutive models related to crack stuff.
//
// Revision 1.35  2000/09/05 07:46:33  tan
// Applied BrokenCellShapeFunction to constitutive models where fracture
// is involved.
//
// Revision 1.34  2000/08/21 19:01:37  guilkey
// Removed some garbage from the constitutive models.
//
// Revision 1.33  2000/08/14 22:38:10  bard
// Corrected strain energy calculation.
//
// Revision 1.32  2000/08/08 01:32:42  jas
// Changed new to scinew and eliminated some(minor) memory leaks in the scheduler
// stuff.
//
// Revision 1.31  2000/07/27 22:39:44  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.30  2000/07/25 19:10:26  guilkey
// Changed code relating to particle combustion as well as the
// heat conduction.
//
// Revision 1.29  2000/07/07 23:57:21  guilkey
// Added volumetric dilation to particle volume.
//
// Revision 1.28  2000/07/07 23:52:09  guilkey
// Removed some inefficiences in the way the deformed volume was allocated
// and stored, and also added changing particle volume to CompNeoHookPlas.
//
// Revision 1.27  2000/07/05 23:43:33  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.26  2000/06/23 23:15:48  guilkey
// Corrected a ghost cell dependency problem to allow these models to work with
// multiple patches.
//
// Revision 1.25  2000/06/23 22:11:07  guilkey
// Added hack to the wavespeed to avoid floating point exception in case of no particles
// on a patch.
//
// Revision 1.24  2000/06/19 21:22:33  bard
// Moved computes for reduction variables outside of loops over materials.
//
// Revision 1.23  2000/06/16 23:23:39  guilkey
// Got rid of pVolumeDeformedLabel_preReloc to fix some confusion
// the scheduler was having.
//
// Revision 1.22  2000/06/16 05:03:04  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.21  2000/06/15 21:57:04  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.20  2000/06/09 23:52:36  bard
// Added fudge factors to time step calculations.
//
// Revision 1.19  2000/06/09 21:07:32  jas
// Added code to get the fudge factor directly into the constitutive model
// inititialization.
//
// Revision 1.18  2000/06/09 00:20:13  bard
// Removed computes pVolume, now computes only pVolumeDeformed
//
// Revision 1.17  2000/06/08 22:00:29  bard
// Added Time Step control to reflect finite deformations and material velocities.
// Removed fudge factors.
//
// Revision 1.16  2000/06/08 16:50:51  guilkey
// Changed some of the dependencies to account for what goes on in
// the burn models.
//
// Revision 1.15  2000/06/01 23:12:06  guilkey
// Code to store integrated quantities in the DW and save them in
// an archive of sorts.  Also added the "computes" in the right tasks.
//
// Revision 1.14  2000/05/31 22:37:09  guilkey
// Put computation of strain energy inside the computeStressTensor functions,
// and store it in a reduction variable in the datawarehouse.
//
// Revision 1.13  2000/05/30 21:07:02  dav
// delt to delT
//
// Revision 1.12  2000/05/30 20:19:02  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.11  2000/05/30 17:08:26  dav
// Changed delt to delT
//
// Revision 1.10  2000/05/26 21:37:33  jas
// Labels are now created and accessed using Singleton class MPMLabel.
//
// Revision 1.9  2000/05/26 18:15:11  guilkey
// Brought the CompNeoHook constitutive model up to functionality
// with the UCF.  Also, cleaned up all of the working models to
// rid them of the SAMRAI crap.
//
// Revision 1.8  2000/05/11 20:10:13  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.7  2000/05/07 06:02:03  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.6  2000/04/26 06:48:14  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/25 18:42:33  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.4  2000/04/19 21:15:54  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.3  2000/04/14 17:34:42  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.2  2000/03/20 17:17:07  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:53  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:48  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/12/17 22:05:22  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.7  1999/11/17 22:26:35  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.6  1999/11/17 20:08:46  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.5  1999/10/14 16:13:04  guilkey
// Added bElBar to the packStream function
//
// Revision 1.4  1999/09/22 22:49:02  guilkey
// Added data to the pack/unpackStream functions to get the proper data into the
// ghost cells.
//
// Revision 1.3  1999/09/10 19:08:37  guilkey
// Added bElBar to the copy constructor
//
// Revision 1.2  1999/06/18 05:44:51  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:38  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.3  1999/05/31 19:36:12  cgl
// Work in stand-alone version of MPM:
//
// - Added materials_dat.cc in src/constitutive_model to generate the
//   materials.dat file for preMPM.
// - Eliminated references to ConstitutiveModel in Grid.cc and GeometryObject.cc
//   Now only Particle and Material know about ConstitutiveModel.
// - Added reads/writes of Particle start and restart information as member
//   functions of Particle
// - "part.pos" now has identicle format to the restart files.
//   mpm.cc modified to take advantage of this.
//
// Revision 1.2  1999/05/30 02:10:47  cgl
// The stand-alone version of ConstitutiveModel and derived classes
// are now more isolated from the rest of the code.  A new class
// ConstitutiveModelFactory has been added to handle all of the
// switching on model type.  Between the ConstitutiveModelFactory
// class functions and a couple of new virtual functions in the
// ConstitutiveModel class, new models can be added without any
// source modifications to any classes outside of the constitutive_model
// directory.  See csafe/Uintah/src/CD/src/constitutive_model/HOWTOADDANEWMODEL
// for updated details on how to add a new model.
//
// --cgl
//
// Revision 1.1  1999/04/10 00:12:07  guilkey
// Compressible Neo-Hookean hyperelastic constitutive model
//
