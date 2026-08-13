/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/VUMAT.h>
#include <Core/Grid/Patch.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Level.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Matrix3.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <dlfcn.h>

using namespace std;
using namespace Uintah;

// Material Constants are C1, C2 and PR (poisson's ratio).  
// The shear modulus = 2(C1 + C2).

VUMAT::VUMAT(ProblemSpecP& ps, MPMFlags* Mflag) 
  : ConstitutiveModel(Mflag)
{
  ps->require("filename",d_initialData.filename);

  // Require elastic constants for computing wave speed
  ps->require("YoungsModulus",d_initialData.E);
  ps->require("PR",d_initialData.PR);

  // Read in a VUMAT formatted input file pointed to in "filename" above
  readInput(d_initialData.filename.c_str(),
            d_initialData.library,
            d_initialData.function,
            d_initialData.nstatev,
            d_initialData.props);

  const TypeDescription* P_dbl =ParticleVariable<double>::getTypeDescription();

  for(int i = 0; i< d_initialData.nstatev; i++){
   ostringstream vlnum;
   vlnum << i;
   pStateVarLabel.push_back(VarLabel::create("p.statevar" + vlnum.str(),
                                                                        P_dbl));
   pStateVarLabel_preReloc.push_back(VarLabel::create("p.statevar+"+vlnum.str(),
                                                                        P_dbl));
  }
  pEnerInternLabel = VarLabel::create("p.enerIntern", P_dbl);
  pEnerInternLabel_preReloc = VarLabel::create("p.enerIntern+", P_dbl);
  pEnerInelasLabel = VarLabel::create("p.enerInelas", P_dbl);
  pEnerInelasLabel_preReloc = VarLabel::create("p.enerInelas+", P_dbl);  

  // Load up the specific vumat library described by the arguments below
  loadLibrary(d_initialData.library.c_str(), d_initialData.function.c_str());
  
}

VUMAT::~VUMAT()
{
  for(int i = 0; i< d_initialData.nstatev; i++){
    VarLabel::destroy(pStateVarLabel[i]);
    VarLabel::destroy(pStateVarLabel_preReloc[i]);
  }
  VarLabel::destroy(pEnerInternLabel);
  VarLabel::destroy(pEnerInternLabel_preReloc);
  VarLabel::destroy(pEnerInelasLabel);
  VarLabel::destroy(pEnerInelasLabel_preReloc);  
  
  // Close down vumat library
  dlclose(lib_handle);
  
}

// A simple helper to trim whitespace from the beginning and end of a string.
void VUMAT::trim(std::string& s) {
    s.erase(0, s.find_first_not_of(" \t\n\r"));
    s.erase(s.find_last_not_of(" \t\n\r") + 1);
}

  // Simple config parser
int VUMAT::readInput(const char*   filename,
                     std::string & library,
                     std::string & function,
                     int         & nstatev,
                     std::vector<double> & props) {

    std::ifstream inputFile(filename);

    if (!inputFile.is_open()) {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return 1; // Return an empty config on error
    }

    std::string line;
    while (std::getline(inputFile, line)) {
      std::stringstream ss(line);
      std::string key, value;

      // Split the line into key and value at the '=' delimiter
      if (std::getline(ss, key, '=') && std::getline(ss, value)) {
        trim(key);
        trim(value);

        // Check the key and parse the value into the correct struct member
        if (key == "library") {
          library = value;
        } else if (key == "function") {
          function = value;
        } else if (key == "nstatev") {
          try {
            nstatev = std::stoi(value);
          } catch (...) { /* Handle potential conversion error */ }
        } else if (key == "props") {
          std::stringstream propsStream(value);
          std::string propValue;
          props.clear();
          while (std::getline(propsStream, propValue, ',')) {
            props.push_back(std::stod(propValue));
          }
        }
      }
    }
    return 0;
}


// Load the VUMAT function
int VUMAT::loadLibrary(const char* libraryFile,
              const char* functionName) {

    // Load the library containing the VUMAT function
    // NEED RTLD_GLOBAL to use VUAMT's library dependencies
    lib_handle = dlopen(libraryFile, RTLD_LAZY | RTLD_GLOBAL);

    if (!lib_handle) {
      std::cout << "Failed to open library" << std::endl;
      return 1;
    }

    // Get handle to function
    void *func_handle = dlsym(lib_handle,functionName);

    if (func_handle == NULL) {
      std::cout << "Failed to load function" << std::endl;
      return 1;
    }

    vumat_func = reinterpret_cast<vumat_handle>(func_handle);

    return 0;
}

void VUMAT::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","VUMAT");
  }
    
  cm_ps->appendElement("filename",d_initialData.filename);
  cm_ps->appendElement("YoungsModulus",d_initialData.E);
  cm_ps->appendElement("PR",d_initialData.PR);
}

VUMAT* VUMAT::clone()
{
  return scinew VUMAT(*this);
}

void 
VUMAT::initializeCMData(const Patch* patch,
                        const MPMMaterial* matl,
                              DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  std::vector<ParticleVariable<double> > pStateVar(d_initialData.nstatev);
  for(int i = 0; i< d_initialData.nstatev; i++){
    new_dw->allocateAndPut(pStateVar[i], pStateVarLabel[i],   pset);

    // Initialize State Variables to zero
    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      pStateVar[i][*iter] = 0.0;
    }
  }

  //
  // initialize energy state variables and set to zero
  //
  ParticleVariable<double> pEnerIntern;
  new_dw->allocateAndPut(pEnerIntern, pEnerInternLabel, pset);
  ParticleVariable<double> pEnerInelas;
  new_dw->allocateAndPut(pEnerInelas, pEnerInelasLabel, pset);
  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
    pEnerIntern[*iter] = 0.0;
    pEnerInelas[*iter] = 0.0;
  } 
  
  computeStableTimeStep(patch, matl, new_dw);
}

void VUMAT::computeStableTimeStep(const Patch* patch,
                                             const MPMMaterial* matl,
                                             DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  // Retrieve the array of constitutive parameters
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
  double E  = d_initialData.props[0];

  for(ParticleSubset::iterator iter = pset->begin();
      iter != pset->end(); iter++){
     particleIndex idx = *iter;

     // Compute wave speed + particle velocity at each particle, 
     // store the maximum
     c_dil = sqrt(E*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  if(delT_new < 1.e-12)
    new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel, patch->getLevel());
  else
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void VUMAT::computeStressTensor(const PatchSubset* patches,
                                           const MPMMaterial* matl,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Matrix3 Identity,B;
    Identity.Identity();
    double c_dil = 0.0,se=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx=patch->dCell();
    
    int dwi = matl->getDWIndex();

    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> pDeformGrad, pstressOld, pDeformGradOld;
    constParticleVariable<Matrix3> velGrad;
    ParticleVariable<Matrix3> pstress;
    constParticleVariable<double> pmass;
    constParticleVariable<double> pvolume;
    constParticleVariable<double> pvolumeOld;
    constParticleVariable<Vector> pvelocity;
    ParticleVariable<double> pdTdt, p_q;
    constParticleVariable<double> pEnerIntern_old, pEnerInelas_old;
    ParticleVariable<double> pEnerIntern_new, pEnerInelas_new;
    std::vector<constParticleVariable<double> > 
                                           pStateVar_old(d_initialData.nstatev);
    std::vector<ParticleVariable<double> > pStateVar_new(d_initialData.nstatev);
    constParticleVariable<int> pLocalized;
    ParticleVariable<int> pLocalized_new;

    delt_vartype delT;
    simTime_vartype simTime(0);
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(simTime, lb->simulationTimeLabel);
    old_dw->get(pDeformGradOld,  lb->pDeformationMeasureLabel, pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(pstressOld,          lb->pStressLabel,             pset);
    old_dw->get(pvolumeOld,          lb->pVolumeLabel,             pset);
    old_dw->get(pLocalized,          lb->pLocalizedMPMLabel,       pset);

    new_dw->get(pvolume,             lb->pVolumeLabel_preReloc,    pset);
    new_dw->get(pDeformGrad, lb->pDeformationMeasureLabel_preReloc,pset);
    new_dw->get(velGrad,             lb->pVelGradLabel_preReloc,   pset);

    new_dw->allocateAndPut(pstress,  lb->pStressLabel_preReloc,    pset);
    new_dw->allocateAndPut(pdTdt,    lb->pdTdtLabel,               pset);
    new_dw->allocateAndPut(p_q,      lb->p_qLabel_preReloc,        pset);
    new_dw->allocateAndPut(pLocalized_new,      lb->pLocalizedMPMLabel_preReloc,        pset);

    for(int i = 0; i< d_initialData.nstatev; i++){
      old_dw->get(pStateVar_old[i], pStateVarLabel[i],             pset);
      new_dw->allocateAndPut(pStateVar_new[i],
                                      pStateVarLabel_preReloc[i],  pset);
    }
    old_dw->get(pEnerIntern_old,      pEnerInternLabel,             pset);
    old_dw->get(pEnerInelas_old,      pEnerInelasLabel,             pset);
    new_dw->allocateAndPut(pEnerIntern_new, pEnerInternLabel_preReloc,  pset);
    new_dw->allocateAndPut(pEnerInelas_new, pEnerInelasLabel_preReloc,  pset);
    
    double E  = d_initialData.E;
    double PR = d_initialData.PR;

    // 
    // Inputs
    //
    int nblock = pset->numParticles();
    int ndir = 3;
    int nshr = 3;
    int nstatev = d_initialData.nstatev;
    int nfieldv = 0;
    int nprops = d_initialData.props.size();
    int lanneal = 0;
    double stepTime = delT;
    double totalTime = simTime + delT;
    double dt = delT;
    std::vector<double> density(nblock, 0.0);
    std::vector<double> strainInc(nblock * (ndir + nshr), 0.0);
    std::vector<double> stretchOld(nblock * (ndir + nshr), 0.0);
    std::vector<double> defgradOld(nblock * (ndir + 2 * nshr), 0.0);
    std::vector<double> stressOld(nblock * (ndir + nshr), 0.0);
    std::vector<double> stateOld(nblock * nstatev, 0.0);
    std::vector<double> enerInternOld(nblock, 0.0);
    std::vector<double> enerInelasOld(nblock, 0.0);
    std::vector<double> stretchNew(nblock * (ndir + nshr), 0.0);
    std::vector<double> defgradNew(nblock * (ndir + 2 * nshr), 0.0);

    //
    // Not used / unsupported
    // Could be filled in future versions
    //
    char * cmname = nullptr;
    std::vector<double> coordMp(nblock, 0.0);
    std::vector<double> charLength(nblock, 0.0);
    std::vector<double> relSpinInc(nblock * nshr, 0.0);
    std::vector<double> fieldOld(nblock * nfieldv, 0.0);
    std::vector<double> fieldNew(nblock * nfieldv, 0.0);
    std::vector<double> tempOld(nblock, 0.0);
    std::vector<double> tempNew(nblock, 0.0);
    
    //
    // Outputs
    //
    std::vector<double> stressNew(nblock * (ndir + nshr));
    std::vector<double> stateNew(nblock * nstatev);
    std::vector<double> enerInternNew(nblock);
    std::vector<double> enerInelasNew(nblock);
    
    Matrix3 tensorU, tensorR, tensorUOld, tensorROld;

    //
    // iterate over particles and pack data for VUMAT call
    //
    int vumatBlockId = 0;
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++, vumatBlockId++){
      particleIndex idx = *iter;

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);
      pDeformGradOld[idx].polarDecompositionRMB(tensorUOld, tensorROld);

      stretchOld[vumatBlockId + 0 * nblock] = tensorUOld(0,0);
      stretchOld[vumatBlockId + 1 * nblock] = tensorUOld(1,1);
      stretchOld[vumatBlockId + 2 * nblock] = tensorUOld(2,2);
      stretchOld[vumatBlockId + 3 * nblock] = tensorUOld(0,1);
      stretchOld[vumatBlockId + 4 * nblock] = tensorUOld(1,2);
      stretchOld[vumatBlockId + 5 * nblock] = tensorUOld(0,2);

      stretchNew[vumatBlockId + 0 * nblock] = tensorU(0,0);
      stretchNew[vumatBlockId + 1 * nblock] = tensorU(1,1);
      stretchNew[vumatBlockId + 2 * nblock] = tensorU(2,2);
      stretchNew[vumatBlockId + 3 * nblock] = tensorU(0,1);
      stretchNew[vumatBlockId + 4 * nblock] = tensorU(1,2);
      stretchNew[vumatBlockId + 5 * nblock] = tensorU(0,2);
      
      for(int i = 0; i<nstatev; i++){
        stateOld[vumatBlockId + i * nblock] = pStateVar_old[i][idx];
      }

      density[vumatBlockId] = pmass[idx]/pvolume[idx];
      //density[vumatBlockId] = pmass[idx]/(0.5 * (pvolume[idx] + pvolumeOld[idx]));

      // rotate stress old into material coordinates 
      Matrix3 SO = pstressOld[idx];
      SO = (tensorROld.Transpose()) * SO * tensorROld;

      stressOld[vumatBlockId + 0 * nblock] = SO(0,0);
      stressOld[vumatBlockId + 1 * nblock] = SO(1,1);
      stressOld[vumatBlockId + 2 * nblock] = SO(2,2);
      stressOld[vumatBlockId + 3 * nblock] = SO(0,1);
      stressOld[vumatBlockId + 4 * nblock] = SO(1,2);
      stressOld[vumatBlockId + 5 * nblock] = SO(2,0);

      // compute strain increment and rotate into material coordinates
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*0.5 * delT;
      Matrix3 corotD = (tensorR.Transpose()) * D * tensorR;

      strainInc[vumatBlockId + 0 * nblock] = corotD(0,0);
      strainInc[vumatBlockId + 1 * nblock] = corotD(1,1);
      strainInc[vumatBlockId + 2 * nblock] = corotD(2,2);
      strainInc[vumatBlockId + 3 * nblock] = corotD(0,1);
      strainInc[vumatBlockId + 4 * nblock] = corotD(1,2);
      strainInc[vumatBlockId + 5 * nblock] = corotD(2,0);
     
      defgradOld[vumatBlockId + 0 * nblock] = pDeformGradOld[idx](0,0);
      defgradOld[vumatBlockId + 1 * nblock] = pDeformGradOld[idx](1,1);
      defgradOld[vumatBlockId + 2 * nblock] = pDeformGradOld[idx](2,2);
      defgradOld[vumatBlockId + 3 * nblock] = pDeformGradOld[idx](0,1);
      defgradOld[vumatBlockId + 4 * nblock] = pDeformGradOld[idx](1,2);
      defgradOld[vumatBlockId + 5 * nblock] = pDeformGradOld[idx](2,0);
      defgradOld[vumatBlockId + 6 * nblock] = pDeformGradOld[idx](1,0);
      defgradOld[vumatBlockId + 7 * nblock] = pDeformGradOld[idx](2,1);
      defgradOld[vumatBlockId + 8 * nblock] = pDeformGradOld[idx](0,2);

      defgradNew[vumatBlockId + 0 * nblock] = pDeformGrad[idx](0,0);
      defgradNew[vumatBlockId + 1 * nblock] = pDeformGrad[idx](1,1);
      defgradNew[vumatBlockId + 2 * nblock] = pDeformGrad[idx](2,2);
      defgradNew[vumatBlockId + 3 * nblock] = pDeformGrad[idx](0,1);
      defgradNew[vumatBlockId + 4 * nblock] = pDeformGrad[idx](1,2);
      defgradNew[vumatBlockId + 5 * nblock] = pDeformGrad[idx](2,0);
      defgradNew[vumatBlockId + 6 * nblock] = pDeformGrad[idx](1,0);
      defgradNew[vumatBlockId + 7 * nblock] = pDeformGrad[idx](2,1);
      defgradNew[vumatBlockId + 8 * nblock] = pDeformGrad[idx](0,2);

      enerInternOld[vumatBlockId] =  pEnerIntern_old[idx];
      enerInelasOld[vumatBlockId] = pEnerInelas_old[idx];

    }

    //
    // Call VUMAT
    //
    vumat_func(nblock, ndir, nshr, nstatev, nfieldv, nprops, lanneal, 
	       stepTime, totalTime, dt, cmname, &(coordMp[0]), &(charLength[0]), 
	       &(d_initialData.props[0]), &(density[0]), &(strainInc[0]), &(relSpinInc[0]), 
	       &(tempOld[0]), &(stretchOld[0]), &(defgradOld[0]), &(fieldOld[0]), 
	       &(stressOld[0]), &(stateOld[0]), &(enerInternOld[0]), &(enerInelasOld[0]), 
	       &(tempNew[0]), &(stretchNew[0]), &(defgradNew[0]), &(fieldNew[0]), 
	       &(stressNew[0]), &(stateNew[0]), &(enerInternNew[0]), &(enerInelasNew[0]));

    //
    // iterate over particles and unpack VUMAT output
    //
    vumatBlockId = 0;
    for(ParticleSubset::iterator iter = pset->begin();iter!=pset->end();iter++, vumatBlockId++){
      particleIndex idx = *iter;   

      // Compute polar decomposition of F (F = RU)
      pDeformGrad[idx].polarDecompositionRMB(tensorU, tensorR);
      
      for(int i = 0; i< nstatev; i++){
        pStateVar_new[i][idx] = stateNew[vumatBlockId + i * nblock];
      }
      pEnerIntern_new[idx] = enerInternNew[vumatBlockId];
      pEnerInelas_new[idx] = enerInelasNew[vumatBlockId];
      
      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      pstress[idx] = Matrix3(stressNew[vumatBlockId + 0 * nblock],
			     stressNew[vumatBlockId + 3 * nblock],
			     stressNew[vumatBlockId + 5 * nblock],
                             stressNew[vumatBlockId + 3 * nblock],
			     stressNew[vumatBlockId + 1 * nblock],
			     stressNew[vumatBlockId + 4 * nblock],
                             stressNew[vumatBlockId + 5 * nblock],
			     stressNew[vumatBlockId + 4 * nblock],
			     stressNew[vumatBlockId + 2 * nblock]);
      
      // Rotate the stress back to the laboratory coordinates
      pstress[idx] = (tensorR*pstress[idx])*(tensorR.Transpose());

      const double rhoHalfStep =  pmass[idx]/(0.5 * (pvolume[idx] + pvolumeOld[idx]));
      
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*0.5;
      
      // Compute wave speed + particle velocity at each particle, 
      // store the maximum
      c_dil = sqrt(E/rhoHalfStep);
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double bulk = E/(3.*(1. -2.*PR));
	double G = E/(2.*(1. + PR));
	double c_bulk = sqrt((bulk + 4.0/3.0 * G)/rhoHalfStep);

	double vdov = D.Trace();
	
	//
	// compute characteristic length in a different way as the element deforms
	//
	//Vector newExtents = pDeformGrad[idx] * dx;
	//double vol = newExtents[0] * newExtents[1] * newExtents[2];
	//double area1 = newExtents[0] * newExtents[1];
	//double area2 = newExtents[0] * newExtents[2];
	//double area3 = newExtents[1] * newExtents[2];
	//double maxarea = std::max(area1, std::max(area2, area3));
	//dx_ave = vol / maxarea;

	/*
	std::cout << "vdov " << vdov << std::endl;
	std::cout << "ss " << c_bulk << std::endl;
	std::cout << "rho " << rhoHalfStep << std::endl;
	std::cout << "arealg " << dx_ave << std::endl;
	*/
	
        p_q[idx] = artificialBulkViscosity(vdov, c_bulk, rhoHalfStep, dx_ave);
	
	// add qdV to internal energy
	//double dV = pvolume[idx] - pvolumeOld[idx];
	//double qdV = (p_q[idx] * dV) / pmass[idx]; // pEnerIntern is per unit mass 
	double qdV = p_q[idx]*vdov/rhoHalfStep * delT; 
	
	pEnerIntern_new[idx] = pEnerIntern_new[idx] - qdV;
	
      } else {
        p_q[idx] = 0.;
      }

      // check whether to erode the particle
      if (pmass[idx]/pvolume[idx] < 0.0) {
	pLocalized_new[idx] = -999;
      }
      else {
	pLocalized_new[idx] = pLocalized[idx];
      }

      // Compute the strain energy for all the particles
      double e = 0.;  // Fix this
      se += e;
      
    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();

    if(delT_new < 1.e-12)
      new_dw->put(delt_vartype(DBL_MAX), lb->delTLabel);
    else
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),      lb->StrainEnergyLabel);
    }
  }

}

void VUMAT::carryForward(const PatchSubset* patches,
                                    const MPMMaterial* matl,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    // Carry forward the data common to all constitutive models 
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model 
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());
    
    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }
}

         
void VUMAT::addParticleState(std::vector<const VarLabel*>& from,
                             std::vector<const VarLabel*>& to)
{
  for(int i = 0; i< d_initialData.nstatev; i++){
    from.push_back(pStateVarLabel[i]);
    to.push_back(pStateVarLabel_preReloc[i]);
  }
  from.push_back(pEnerInternLabel);
  to.push_back(pEnerInternLabel_preReloc);
  from.push_back(pEnerInelasLabel);
  to.push_back(pEnerInelasLabel_preReloc);
}

void VUMAT::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet* patches ) const
{
  // Add the computes and requires that are common to all explicit 
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  Ghost::GhostType  gnone = Ghost::None;
  for(int i = 0; i< d_initialData.nstatev; i++){
    task->requiresVar(Task::OldDW, pStateVarLabel[i],   matlset, gnone);
    task->computesVar(pStateVarLabel_preReloc[i],       matlset);
  }
  task->requiresVar(Task::OldDW, pEnerInternLabel,   matlset, gnone);
  task->computesVar(pEnerInternLabel_preReloc,       matlset);
  task->requiresVar(Task::OldDW, pEnerInelasLabel,   matlset, gnone);
  task->computesVar(pEnerInelasLabel_preReloc,       matlset);

  task->requiresVar(Task::OldDW, lb->pLocalizedMPMLabel, matlset, gnone);
  task->computesVar(lb->pLocalizedMPMLabel_preReloc,     matlset);
}

//______________________________________________________________________
//
void VUMAT::addInitialComputesAndRequires(Task* task,
                                          const MPMMaterial* matl,
                                          const PatchSet*) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  // StateVar
  for(int i = 0; i< d_initialData.nstatev; i++){
    task->computesVar(pStateVarLabel[i],       matlset);
  }
  task->computesVar(pEnerInternLabel,       matlset);
  task->computesVar(pEnerInelasLabel,       matlset);
}

void 
VUMAT::addComputesAndRequires(Task* ,
                                   const MPMMaterial* ,
                                   const PatchSet* ,
                                   const bool ) const
{
}

double VUMAT::computeRhoMicroCM(double pressure,
                              const double p_ref,
                              const MPMMaterial* matl,
                              double temperature,
                              double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
/*
  double bulk = d_initialData.V1/(3.*(1. -2.*d_initialData.V2));

  double p_gauge = pressure - p_ref;
  double rho_cur;

  rho_cur = rho_orig*(p_gauge/bulk + sqrt((p_gauge/bulk)*(p_gauge/bulk) +1));

  return rho_cur;
*/
  return rho_orig;
  cerr << "No version of computeRhoMicroCM exists yet for VUMAT" << endl;
}

void VUMAT::computePressEOSCM(double rho_cur,double& pressure,
                                         double p_ref,
                                         double& dp_drho, double& tmp,
                                         const MPMMaterial* matl, 
                                         double temperature)
{
/*
  double bulk = d_initialData.V1/(3.*(1. -2.*d_initialData.V2));
  double rho_orig = matl->getInitialDensity();
  double shear = d_initialData.V1/(2.*(1+d_initialData.V2));

  double p_g = .5*bulk*(rho_cur/rho_orig - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = .5*bulk*(rho_orig/(rho_cur*rho_cur) + 1./rho_orig);
  tmp = (bulk + 4.*shear/3.)/rho_cur;  // speed of sound squared
*/

  cerr << "No version of computePressEOSCM exists yet for VUMAT" << endl;
}

double VUMAT::getCompressibility()
{
/*
  double bulk = d_initialData.V1/(3.*(1. -2.*d_initialData.V2));
  return 1.0/bulk;
*/
  cerr << "No version of computePressEOSCM exists yet for VUMAT" << endl;
  return 1.0;
}

namespace Uintah {
} // End namespace Uintah
