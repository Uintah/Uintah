/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI),
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/MPM/ConstitutiveModel/Kayenta.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>

#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Parallel/Parallel.h>

#include <sci_defs/uintah_defs.h>
#include <Core/Math/Weibull.h>

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
////////////////////////////////////////////////////////////////////////////////
// The following functions are found in fortran/*.F
//SUBROUTINE KAYENTA_CALC( NBLK, NINSV, DT, UI, GC, DC,
//   $                                   SIGARG, D, SVARG, USM   )

extern "C"{

#if defined( FORTRAN_UNDERSCORE_END )
#  define KAYENTA_CHK kayenta_chk_
#  define KAYENTA_CALC kayenta_calc_
#  define KAYENTA_RXV kayenta_rxv_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define KAYENTA_CHK kayenta_chk_
#  define KAYENTA_RXV kayenta_rxv_
#  define KAYENTA_CALC kayenta_calc__
#else // NONE
#  define KAYENTA_CHK kayenta_chk
#  define KAYENTA_CALC kayenta_calc
#  define KAYENTA_RXV kayenta_rxv
#endif

//#define KMM_ORTHOTROPIC
//#undef KMM_ORTHOTROPIC
//#define KMM_ANISOTROPIC
//#undef KMM_ANISOTROPIC

   void KAYENTA_CHK( double UI[], double GC[], double DC[] );
   void KAYENTA_CALC(int &nblk, int &ninsv, double &dt,
                     double UI[], double GC[], double DC[], double stress[],
                     double D[], double svarg[], double &USM );
   void KAYENTA_RXV( double UI[], double GC[], double DC[], int &nx,
                     char namea[], char keya[], double rinit[], double rdim[],
                     int iadvct[], int itype[] );
}

// End fortran functions.
////////////////////////////////////////////////////////////////////////////////
using namespace std; using namespace Uintah;

Kayenta::Kayenta(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  // See Kayenta_pnt.Blk to see where these numbers come from
  // User Inputs
  d_NBASICINPUTS=70;
  d_NUMJNTS=0;
  d_NUMJOINTINPUTS=0*d_NUMJNTS;
  d_NTHERMOPLAST=5;
  d_NUIEOSMG=22;
  d_IEOSMGCT=d_NBASICINPUTS+d_NTHERMOPLAST;
  d_NUMEOSINPUTS=d_NUIEOSMG+d_NTHERMOPLAST;
  // Total number of User Inputs
  d_NKMMPROP=d_NBASICINPUTS+d_NUMJOINTINPUTS+d_NUMEOSINPUTS;
  // Global Constants
  d_NKMMGC=0;
  // Derived Constants
  d_NKMMDC=13;
  // Internal State Variables
  // d_NINSV automatically read

  // pre-initialize all of the user inputs to zero.
  for(int i = 0; i<d_NKMMPROP; i++){
     UI[i] = 0.;
     GC[i] = 0.;
  }
  for(int i = 0; i<d_NKMMDC; i++){
     DC[i] = 0.;
  }
  // Read model parameters from the input file
  getInputParameters(ps);

  // Check that model parameters are valid and allow model to change if needed

  //First, print out the UI values specified by the user
  proc0cout << "Original UI values" << endl;
  for(int i = 0; i<d_NKMMPROP; i++){
     proc0cout << "UI[" << i << "] = " << UI[i] << endl;
  }

  KAYENTA_CHK(UI,GC,DC);

  //Now, print out the UI values after alteration by KAYENTA_CHK
  proc0cout << "Modified UI values" << endl;
  for(int i = 0; i<d_NKMMPROP; i++){
     proc0cout << "UI[" << i << "] = " << UI[i] << endl;
  }


  //Create VarLabels for Kayenta internal state variables (ISVs)
  int nx;
  char namea[5000];
  char keya[5000];
  double rdim[700];
  int iadvct[100];
  int itype[100];

  KAYENTA_RXV( UI, GC, DC, nx, namea, keya, rinit, rdim, iadvct, itype );

  //Print out the Derived Constants
//  proc0cout << "Derived Constants" << endl;
//  for(int i = 0; i<d_NKMMDC; i++){
//     proc0cout << "DC[" << i << "] = " << DC[i] << endl;
//  }

  // Print out Internal State Variables
  d_NINSV=nx;
  proc0cout << "Internal State Variables" << endl;
  proc0cout << "# ISVs = " << d_NINSV << endl;
//  for(int i = 0;i<d_NINSV; i++){
//    proc0cout << "ISV[" << i << "] = " << rinit[i] << endl;
//  }
  setErosionAlgorithm();
  initializeLocalMPMLabels();
}

Kayenta::Kayenta(const Kayenta* cm) : ConstitutiveModel(cm)
{
  for(int i=0;i<d_NKMMPROP;i++){
    UI[i] = cm->UI[i];
  }

  wdist.WeibMed    = cm->wdist.WeibMed;
  wdist.WeibMod    = cm->wdist.WeibMod;
  wdist.WeibRefVol = cm->wdist.WeibRefVol;
  wdist.WeibSeed   = cm->wdist.WeibSeed;
  wdist.Perturb    = cm->wdist.Perturb;
  wdist.WeibDist   = cm->wdist.WeibDist;

  d_allowNoTension = cm->d_allowNoTension;
  d_removeMass = cm->d_removeMass;

  //Create VarLabels for Kayenta internal state variables (ISVs)
  initializeLocalMPMLabels();
}

Kayenta::~Kayenta()
{
   for (unsigned int i = 0; i< ISVLabels.size();i++){
     VarLabel::destroy(ISVLabels[i]);
   }
   VarLabel::destroy(peakI1IDistLabel);
   VarLabel::destroy(peakI1IDistLabel_preReloc);
   VarLabel::destroy(pLocalizedLabel);
   VarLabel::destroy(pLocalizedLabel_preReloc);
}

void Kayenta::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","kayenta");
  }
  // Kayenta User Input Variables UI[FortranNumber-1]  // Description (units)
  cm_ps->appendElement("B0",      UI[0]);   // Initial intact elastic builk modulus (stress)
  cm_ps->appendElement("B1",      UI[1]);   // Coefficient in bulk modulus hardening (stress)
  cm_ps->appendElement("B2",      UI[2]);   // Coefficient in bulk modulus softening (stress)
  cm_ps->appendElement("B3",      UI[3]);   // Coefficient in bulk modulus softening (stress)
  cm_ps->appendElement("B4",      UI[4]);   // Power in bulk modulus softening ()
  cm_ps->appendElement("G0",      UI[5]);   // Initial intact elastic shear modulus (stress)
  cm_ps->appendElement("G1",      UI[6]);   // Coefficient in shear modulus hardening ()
  cm_ps->appendElement("G2",      UI[7]);   // Coefficient in shear modulus hardening (1/stress)
  cm_ps->appendElement("G3",      UI[8]);   // Coefficient in shear modulus hardening (stress)
  cm_ps->appendElement("G4",      UI[9]);   // Power in shear modulus softening ()
  cm_ps->appendElement("RJS",     UI[10]);  // Joint spacing (length)
  cm_ps->appendElement("RKS",     UI[11]);  // Joint shear stiffness (stress/length)
  cm_ps->appendElement("RKN",     UI[12]);  // Joint normal stiffness (stress/length)
  cm_ps->appendElement("A1",      UI[13]);  // Shear failure parameter 1 (stress)
  cm_ps->appendElement("A2",      UI[14]);  // Shear failure parameter 2 (1/stress)
  cm_ps->appendElement("A3",      UI[15]);  // Shear failure parameter 3 (stress)
  cm_ps->appendElement("A4",      UI[16]);  // Shear failure parameter 4 ()
  cm_ps->appendElement("P0",      UI[17]);  // Init value of XL for pore collapse (stress)
  cm_ps->appendElement("P1",      UI[18]);  // Pressure-volume parameter 1 (1/stress)
  cm_ps->appendElement("P2",      UI[19]);  // Pressure-volume parameter 2 (1/stress^2)
  cm_ps->appendElement("P3",      UI[20]);  // Compaction volume strain asymptote (strain)
  cm_ps->appendElement("CR",      UI[21]);  // Shear failure shape parameter ()
  cm_ps->appendElement("RK",      UI[22]);  // TXE/TXC strength ratio ()
  cm_ps->appendElement("RN",      UI[23]);  // Initial shear yield offset (stress)
  cm_ps->appendElement("HC",      UI[24]);  // Kinematic hardening parameter (stress)
  cm_ps->appendElement("CTI1",    UI[25]);  // Tension cut-off value of I1 (stress)
  cm_ps->appendElement("CTPS",    UI[26]);  // Tension cut-off of principal stress (stress)
  cm_ps->appendElement("T1",      UI[27]);  // Relaxation time constant 1 (time)
  cm_ps->appendElement("T2",      UI[28]);  // Relaxation time constant 2 (strain-rate)
  cm_ps->appendElement("T3",      UI[29]);  // Relaxation time constant 3 ()
  cm_ps->appendElement("T4",      UI[30]);  // Relaxation time constant 4 (strain-rate)
  cm_ps->appendElement("T5",      UI[31]);  // Relaxation time constant 5 (stress)
  cm_ps->appendElement("T6",      UI[32]);  // Relaxation time constant 6 (time)
  cm_ps->appendElement("T7",      UI[33]);  // Relaxation time constant 7 (stress)
  cm_ps->appendElement("J3TYPE",  UI[34]);  // Octahedral profile shape ID
  cm_ps->appendElement("A2PF",    UI[35]);  // Potential function parameter 1 (1/stress)
  cm_ps->appendElement("A4PF",    UI[36]);  // Potential function parameter 2 ()
  cm_ps->appendElement("CRPF",    UI[37]);  // Potential function parameter 3 ()
  cm_ps->appendElement("RKPF",    UI[38]);  // Potential function parameter 4 ()
  cm_ps->appendElement("SUBX",    UI[39]);  // Subcycle control parameter exponent ()
  cm_ps->appendElement("DEJAVU",  UI[40]);  // =1 if parameters have been checked ()
  cm_ps->appendElement("FAIL0",   UI[41]);  // Failure parameter 1 (time)
  cm_ps->appendElement("FAIL1",   UI[42]);  // Failure parameter 2 ()
  cm_ps->appendElement("FAIL2",   UI[43]);  // Failure parameter 3 ()
  cm_ps->appendElement("FAIL3",   UI[44]);  // Failure parameter 4 ()
  cm_ps->appendElement("FAIL4",   UI[45]);  // Failure parameter 5 ()
  cm_ps->appendElement("FAIL5",   UI[46]);  // Failure parameter 6 ()
  cm_ps->appendElement("FAIL6",   UI[47]);  // Failure parameter 7 ()
  cm_ps->appendElement("FAIL7",   UI[48]);  // Failure parameter 8 ()
  cm_ps->appendElement("FAIL8",   UI[49]);  // Failure parameter 9 ()
  cm_ps->appendElement("FAIL9",   UI[50]);  // Failure parameter 10 ()
  cm_ps->appendElement("PEAKI1I", UI[51]);  // Peak I1 hydrostatic tension strength
  cm_ps->appendElement("STRENI",  UI[52]);  // Peak (high pressure) shear strength
  cm_ps->appendElement("FSLOPEI", UI[53]);  // Initial slope of limit surface at PEAKI1I
  cm_ps->appendElement("PEAKI1F", UI[54]);  // same as PEAKI1I, but for failed surface
  cm_ps->appendElement("STRENF",  UI[55]);  // same as STRENI, but for failed surface
  cm_ps->appendElement("SOFTENING",UI[56]); // failure handling option
  cm_ps->appendElement("FSLOPEF", UI[57]);  // same as FSLOPEI, but for failed surface
  cm_ps->appendElement("FAILSTAT",UI[58]);  // >0= failure statistics
  cm_ps->appendElement("EOSID",   UI[59]);  // equation of state id
  cm_ps->appendElement("USEHOSTEOS",UI[60]);// boolean for using EOS
  cm_ps->appendElement("DILATLIM",UI[61]);  // Limit on plastic dilatation
  cm_ps->appendElement("FREE01",  UI[62]);  //
  cm_ps->appendElement("FREE02",  UI[63]);  //
  cm_ps->appendElement("FREE03",  UI[64]);  //
  cm_ps->appendElement("FREE04",  UI[65]);  //
  cm_ps->appendElement("FREE05",  UI[66]);  //
  cm_ps->appendElement("CTPSF",   UI[67]);  // Fracture cutoff of principal stress (stress)
  cm_ps->appendElement("YSLOPEI", UI[68]);  // Intact high pressure slope ()
  cm_ps->appendElement("YSLOPEF", UI[69]);  // Failed high pressure slope ()
  // Kayenta EOSMG User Inputs
  int IJTHERMPAR =d_NBASICINPUTS+d_NUMJOINTINPUTS;
  cm_ps->appendElement("TMPRXP",  UI[IJTHERMPAR]);
  cm_ps->appendElement("THERM01", UI[IJTHERMPAR + 1]);
  cm_ps->appendElement("THERM02", UI[IJTHERMPAR + 2]);
  cm_ps->appendElement("THERM03", UI[IJTHERMPAR + 3]);
  cm_ps->appendElement("TMPRM0",  UI[IJTHERMPAR + 4]);
  // Kayenta EOSMGCT User Inputs
  cm_ps->appendElement("RHO0",    UI[d_IEOSMGCT]);
  cm_ps->appendElement("TMPR0",   UI[d_IEOSMGCT +  1]);
  cm_ps->appendElement("SNDSP0",  UI[d_IEOSMGCT +  2]);
  cm_ps->appendElement("S1MG",    UI[d_IEOSMGCT +  3]);
  cm_ps->appendElement("GRPAR",   UI[d_IEOSMGCT +  4]);
  cm_ps->appendElement("CV",      UI[d_IEOSMGCT +  5]);
  cm_ps->appendElement("ESFT",    UI[d_IEOSMGCT +  6]);
  cm_ps->appendElement("RP",      UI[d_IEOSMGCT +  7]);
  cm_ps->appendElement("PS",      UI[d_IEOSMGCT +  8]);
  cm_ps->appendElement("PE",      UI[d_IEOSMGCT +  9]);
  cm_ps->appendElement("CE",      UI[d_IEOSMGCT + 10]);
  cm_ps->appendElement("NSUB",    UI[d_IEOSMGCT + 11]);
  cm_ps->appendElement("S2MG",    UI[d_IEOSMGCT + 12]);
  cm_ps->appendElement("TYP",     UI[d_IEOSMGCT + 13]);
  cm_ps->appendElement("RO",      UI[d_IEOSMGCT + 14]);
  cm_ps->appendElement("TO",      UI[d_IEOSMGCT + 15]);
  cm_ps->appendElement("S",       UI[d_IEOSMGCT + 16]);
  cm_ps->appendElement("GRPARO",  UI[d_IEOSMGCT + 17]);
  cm_ps->appendElement("B",       UI[d_IEOSMGCT + 18]);
  cm_ps->appendElement("XB",      UI[d_IEOSMGCT + 19]);
  cm_ps->appendElement("NB",      UI[d_IEOSMGCT + 20]);
  cm_ps->appendElement("PWR",     UI[d_IEOSMGCT + 21]);
  //  ________________________________________________________________________
  //  EOSMG Derived Constants
  cm_ps->appendElement("A1MG",    DC[0]);
  cm_ps->appendElement("A2MG",    DC[1]);
  cm_ps->appendElement("A3MG",    DC[2]);
  cm_ps->appendElement("A4MG",    DC[3]);
  cm_ps->appendElement("A5MG",    DC[4]);
  cm_ps->appendElement("A0MG",    DC[5]);
  cm_ps->appendElement("AEMG",    DC[6]);
  cm_ps->appendElement("FK0",     DC[7]);
  cm_ps->appendElement("AF",      DC[8]);
  cm_ps->appendElement("PF",      DC[9]);
  cm_ps->appendElement("XF",      DC[10]);
  cm_ps->appendElement("CF",      DC[11]);
  cm_ps->appendElement("RMX",     DC[12]);
  //  ________________________________________________________________________
  //  Uintah Variability Variables
  cm_ps->appendElement("peakI1IPerturb", wdist.Perturb);
  cm_ps->appendElement("peakI1IMed",     wdist.WeibMed);
  cm_ps->appendElement("peakI1IMod",     wdist.WeibMod);
  cm_ps->appendElement("peakI1IRefVol",  wdist.WeibRefVol);
  cm_ps->appendElement("peakI1ISeed",    wdist.WeibSeed);
  cm_ps->appendElement("PEAKI1IDIST",    wdist.WeibDist);
}

Kayenta* Kayenta::clone()
{
  return scinew Kayenta(*this);
}

void Kayenta::initializeCMData(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  StaticArray<ParticleVariable<double> > ISVs(d_NINSV+1);

//  proc0cout << "In initializeCMData" << endl;
  for(int i=0;i<d_NINSV;i++){
    new_dw->allocateAndPut(ISVs[i],ISVLabels[i], pset);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++){
      ISVs[i][*iter] = rinit[i];
    }
  }

  ParticleVariable<int>     pLocalized;
  new_dw->allocateAndPut(pLocalized,         pLocalizedLabel, pset);
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    pLocalized[*iter] = 0;
  }

  ParticleVariable<double> peakI1IDist;
  new_dw->allocateAndPut(peakI1IDist, peakI1IDistLabel, pset);
  if ( wdist.Perturb){
      // Make the seed differ for each patch, otherwise each patch gets the
      // same set of random #s.
      int patchID = patch->getID();
      int patch_div_32 = patchID/32;
      patchID = patchID%32;
      unsigned int unique_seed = ((wdist.WeibSeed+patch_div_32+1) << patchID);

    SCIRun::Weibull weibGen(wdist.WeibMed,wdist.WeibMod,wdist.WeibRefVol,
                            unique_seed,wdist.WeibMod);
    proc0cout << "Weibull Variables for PEAKI1I: (initialize CMData)\n"
            << "Median:            " << wdist.WeibMed
            << "\nModulus:         " << wdist.WeibMod
            << "\nReference Vol:   " << wdist.WeibRefVol
            << "\nSeed:            " << wdist.WeibSeed
            << "\nPerturb?:        " << wdist.Perturb << std::endl;

    constParticleVariable<double>pVolume;
    new_dw->get(pVolume, lb->pVolumeLabel, pset);

    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){
       peakI1IDist[*iter] = weibGen.rand(pVolume[*iter]);
    }
  }
  computeStableTimestep(patch, matl, new_dw);
}

void Kayenta::allocateCMDataAddRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        MPMLabel* lb) const
{
  const MaterialSubset* matlset = matl->thisMaterial();

  // Allocate the variables shared by all constitutive models
  // for the particle convert operation
  // This method is defined in the ConstitutiveModel base class.
  addSharedRForConvertExplicit(task, matlset, patches);
  // Add requires local to this model
  for(int i=0;i<d_NINSV;i++){
    task->requires(Task::NewDW,ISVLabels_preReloc[i], matlset, Ghost::None);
  }
  task->requires(Task::NewDW,peakI1IDistLabel_preReloc,matlset,Ghost::None);
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,      matlset,Ghost::None);
}


void Kayenta::allocateCMDataAdd(DataWarehouse* new_dw,
                                ParticleSubset* addset,
                                map<const VarLabel*,
                                ParticleVariableBase*>* newState,
                                ParticleSubset* delset,
                                DataWarehouse* )
{
  // Copy the data common to all constitutive models from the particle to be
  // deleted to the particle to be added.
  // This method is defined in the ConstitutiveModel base class.
  copyDelToAddSetForConvertExplicit(new_dw, delset, addset, newState);

  StaticArray<ParticleVariable<double> > ISVs(d_NINSV+1);
  StaticArray<constParticleVariable<double> > o_ISVs(d_NINSV+1);
  constParticleVariable<double> o_peakI1IDist;
  ParticleVariable<int>     pLocalized;
  constParticleVariable<int>     o_Localized;
  new_dw->get(o_peakI1IDist,peakI1IDistLabel_preReloc,delset);

  ParticleVariable<double> peakI1IDist;
  new_dw->allocateTemporary(peakI1IDist,addset);
  new_dw->allocateTemporary(pLocalized,addset);
  new_dw->get(o_Localized,pLocalizedLabel_preReloc,delset);
  ParticleSubset::iterator o,n = addset->begin();
  for (o=delset->begin(); o != delset->end(); o++, n++) {
          peakI1IDist[*n] = o_peakI1IDist[*o];
  }
  (*newState)[peakI1IDistLabel]=peakI1IDist.clone();

  for(int i=0;i<d_NINSV;i++){
    new_dw->allocateTemporary(ISVs[i], addset);
    new_dw->get(o_ISVs[i],ISVLabels_preReloc[i], delset);

    ParticleSubset::iterator o,n = addset->begin();
    for (o=delset->begin(); o != delset->end(); o++, n++) {
      ISVs[i][*n] = o_ISVs[i][*n];

    }
    (*newState)[ISVLabels[i]]=ISVs[i].clone();
  }

  for (o=delset->begin(); o != delset->end(); o++, n++) {
    pLocalized[*n] = o_Localized[*o];
  }
  (*newState)[pLocalizedLabel]=pLocalized.clone();

}

void Kayenta::addRequiresDamageParameter(Task* task,
                                           const MPMMaterial* matl,
                                           const PatchSet* ) const
{
  const MaterialSubset* matlset = matl->thisMaterial();
  task->requires(Task::NewDW, pLocalizedLabel_preReloc,matlset,Ghost::None);
}


void Kayenta::getDamageParameter(const Patch* patch,
                                   ParticleVariable<int>& damage,
                                   int dwi,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  ParticleSubset* pset = old_dw->getParticleSubset(dwi,patch);
  constParticleVariable<int> pLocalized;
  new_dw->get(pLocalized, pLocalizedLabel_preReloc, pset);

  ParticleSubset::iterator iter;
  for (iter = pset->begin(); iter != pset->end(); iter++) {
    damage[*iter] = pLocalized[*iter];
  }

}


void Kayenta::addParticleState(std::vector<const VarLabel*>& from,
                               std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  for(int i=0;i<d_NINSV;i++){
    from.push_back(ISVLabels[i]);
    to.push_back(ISVLabels_preReloc[i]);
  }
  from.push_back(pLocalizedLabel);
  from.push_back(peakI1IDistLabel);
  to.push_back(pLocalizedLabel_preReloc);
  to.push_back(peakI1IDistLabel_preReloc);
}

void Kayenta::computeStableTimestep(const Patch* patch,
                                    const MPMMaterial* matl,
                                    DataWarehouse* new_dw)
{
  // This is only called for the initial timestep - all other timesteps
  // are computed as a side-effect of computeStressTensor
  Vector dx = patch->dCell();
  int dwi = matl->getDWIndex();
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  constParticleVariable<double> pmass, pvolume;
  constParticleVariable<Vector> pvelocity;

  new_dw->get(pmass,     lb->pMassLabel,     pset);
  new_dw->get(pvolume,   lb->pVolumeLabel,   pset);
  new_dw->get(pvelocity, lb->pVelocityLabel, pset);

  double c_dil = 0.0;
  Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

  double bulk = UI[0];
  double G = UI[5];
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  UI[d_IEOSMGCT     ]=matl->getInitialDensity();      // RHO0
  UI[d_IEOSMGCT +  1]=matl->getRoomTemperature();     // TMPR0
  UI[d_IEOSMGCT +  2]=bulk/matl->getInitialDensity(); // SNDSP0
  UI[d_IEOSMGCT +  5]=matl->getInitialCv();           // CV

  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}


void
Kayenta::setErosionAlgorithm()
{
  d_allowNoTension = false;
  d_removeMass=false;
  if (flag->d_doErosion) {
    if (flag->d_erosionAlgorithm == "AllowNoTension")
      d_allowNoTension = true;
    else if (flag->d_erosionAlgorithm == "RemoveMass")
      d_removeMass = true;
  }
}


void Kayenta::viscousStressUpdate(Matrix3& D, const Matrix3& old_stress, double& rho_orig,const double& old_volume, double& bulk, double& viscosity, double& delT, Matrix3& new_stress, Matrix3& new_defgrad, double& rho_cur, double& new_volume, double& USM, double& c_dil ){
  new_volume  = old_volume;
  rho_cur = rho_orig;
  Matrix3 Identity;
  Identity.Identity();
  // the deformation gradient will be set to the identity
  new_defgrad = Identity;
  double one_third = 1.0/3.0;
  double pressure = one_third*old_stress.Trace();
  double pressure_rate = bulk*D.Trace();
  Matrix3 DPrime = D - Identity*one_third*D.Trace();
  Matrix3 shear = DPrime*(2.*viscosity);

  // first we add the pressure to the old stress tensor:
  pressure = pressure + pressure_rate*delT;
  // check to see if pressure is compresive:
  if (pressure>0){
    pressure=0;
  }
  // now we add the shear and pressure components
  new_stress = Identity*pressure + shear;

  //now we must define USM
  USM = bulk;

  c_dil = sqrt(bulk/rho_cur);



}

void Kayenta::computeStressTensor(const PatchSubset* patches,
                                  const MPMMaterial* matl,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<Vector> d_S(interpolator->size());
    vector<double> S(interpolator->size());

    Matrix3 velGrad,deformationGradientInc,Identity,zero(0.),One(1.);
    double c_dil=0.0,Jinc;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);

    Identity.Identity();

    Vector dx = patch->dCell();
    double oodx[3] = {1./dx.x(), 1./dx.y(), 1./dx.z()};

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Point> px;
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    ParticleVariable<Matrix3> deformationGradient_new;
    constParticleVariable<double> pmass, pvolume, ptemperature, peakI1IDist;
    ParticleVariable<double> pvolume_new, peakI1IDist_new;
    constParticleVariable<Vector> pvelocity, psize;
    constNCVariable<Vector> gvelocity;
    delt_vartype delT;
    constParticleVariable<int> pLocalized;
    ParticleVariable<int>     pLocalized_new;
    constParticleVariable<long64> pParticleID;

    old_dw->get(pLocalized, pLocalizedLabel, pset);
    new_dw->allocateAndPut(pLocalized_new,pLocalizedLabel_preReloc, pset);
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Ghost::GhostType  gac   = Ghost::AroundCells;

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(psize,               lb->pSizeLabel,               pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
    old_dw->get(peakI1IDist,         peakI1IDistLabel,             pset);
    old_dw->get(pParticleID,         lb->pParticleIDLabel,         pset);

    StaticArray<constParticleVariable<double> > ISVs(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],           ISVLabels[i],                 pset);
    }

    new_dw->get(gvelocity,lb->gVelocityStarLabel, dwi,patch, gac, NGN);

    ParticleVariable<double> pdTdt,p_q;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pvolume_new,     lb->pVolumeLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel_preReloc,     pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->allocateAndPut(deformationGradient_new,
                           lb->pDeformationMeasureLabel_preReloc,        pset);
    new_dw->allocateAndPut(peakI1IDist_new, peakI1IDistLabel_preReloc,   pset);

    peakI1IDist_new.copyData(peakI1IDist);

    StaticArray<ParticleVariable<double> > ISVs_new(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
    }

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;

      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;
      // Initialize velocity gradient
      velGrad.set(0.0);

      if(!flag->d_axisymmetric){
        // Get the node indices that surround the cell
        interpolator->findCellAndShapeDerivatives(px[idx],ni,d_S,psize[idx],
                                                  deformationGradient[idx]);

        computeVelocityGradient(velGrad,ni,d_S,oodx,gvelocity);

      } else {  // axi-symmetric kinematics
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx],
                                                      deformationGradient[idx]);
        // x -> r, y -> z, z -> theta
        computeAxiSymVelocityGradient(velGrad,ni,d_S,S,oodx,gvelocity,px[idx]);
      }

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad + velGrad.Transpose())*.5;
      pLocalized_new[idx]=0;

      // New Way using subcycling
      Matrix3 one; one.Identity();
      Matrix3 F=deformationGradient[idx];
      double Lnorm_dt = velGrad.Norm()*delT;
      int num_scs = max(1,2*((int) Lnorm_dt));
      if(num_scs > 1000){
        cout << "NUM_SCS = " << num_scs << endl;
      }
      double dtsc = delT/(double (num_scs));
      Matrix3 OP_tensorL_DT = one + velGrad*dtsc;
      for(int n=0;n<num_scs;n++){
        F=OP_tensorL_DT*F;
      }
      deformationGradient_new[idx]=F;

      // Old First Order Way
//    deformationGradientInc = velGrad * delT + Identity;

//    Jinc = deformationGradientInc.Determinant();

      // Update the deformation gradient tensor to its time n+1 value.
//    deformationGradient_new[idx] = deformationGradientInc *
//                                   deformationGradient[idx];

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();
      double Jold = deformationGradient[idx].Determinant();
      Jinc = J/Jold;

      // Check 1: Look at Jacobian
      if (J<=0.0 || J > d_hugeJ) {
          double Jold = deformationGradient[idx].Determinant();
          cout<<"negative or huge J encountered J="<<J<<", Jold = " << Jold << " deleting particle" << endl;
          cout << "pos = " << px[idx] << endl;

          pLocalized_new[idx]=-999;
          cout<< "localizing (deleting) particle "<<pParticleID[idx]<<endl;
          cout<< "material = " << dwi << endl << "Momentum deleted = "
                                        << pvelocity[idx]*pmass[idx] <<endl;
          deformationGradient_new[idx]=one;
          D=Matrix3(0.);
      }

      pvolume_new[idx]=Jinc*pvolume[idx];

      // Compute the local sound speed
      double rho_cur = rho_orig/J;

      // NEED TO FIND R
      Matrix3 tensorR, tensorU;

      //Comment by KC: Computing tensorR at the beginning of the timestep
      deformationGradient[idx].polarDecompositionAFFinvTran(tensorU, tensorR);

      // This is the previous timestep Cauchy stress
      // unrotated tensorSig=R^T*pstress*R
      Matrix3 tensorSig = (tensorR.Transpose())*(pstress[idx]*tensorR);

      // Load into 1-D array for the fortran code
      double sigarg[6];
      sigarg[0]=tensorSig(0,0);
      sigarg[1]=tensorSig(1,1);
      sigarg[2]=tensorSig(2,2);
      sigarg[3]=tensorSig(0,1);
      sigarg[4]=tensorSig(1,2);
      sigarg[5]=tensorSig(2,0);

      // UNROTATE D: S=R^T*D*R
      D=(tensorR.Transpose())*(D*tensorR);

      // Load into 1-D array for the fortran code
      double Darray[6];
      Darray[0]=D(0,0);
      Darray[1]=D(1,1);
      Darray[2]=D(2,2);
      Darray[3]=D(0,1);
      Darray[4]=D(1,2);
      Darray[5]=D(2,0);
      double svarg[d_NINSV];
      double USM=9e99;
      double dt = delT;
      int nblk = 1;
  
      // Load ISVs into a 1D array for fortran code
      for(int i=0;i<d_NINSV;i++){
        svarg[i]=ISVs[i][idx];
      }
      // 'Hijack' FAIL1 = UI[41] with perturbed value if desired
      // put real value of UI[41] in tmp var just in case
      double TFAIL_tmp = UI[41];

      // Scale FAIL1 according to a characteristic particle length
      UI[41]*=cbrt(pvolume_new[idx]);
      if (wdist.Perturb){
        double tempVar = UI[51];
        // 'Hijack' PEAKI1I = UI[51] with perturbed value if desired
        // put real value of UI[51] in tmp var just in case
        UI[51] = peakI1IDist[idx];
        KAYENTA_CALC(nblk, d_NINSV, dt, UI, GC, DC, sigarg, Darray, svarg, USM);
        UI[51]=tempVar;
      } else {
        KAYENTA_CALC(nblk, d_NINSV, dt, UI, GC, DC, sigarg, Darray, svarg, USM);
      }
      // Put T1 back for now
      UI[41]=TFAIL_tmp;

      // Unload ISVs from 1D array into ISVs_new
      for(int i=0;i<d_NINSV;i++){
        ISVs_new[i][idx]=svarg[i];
      }

      // This is the Cauchy stress, still unrotated
      tensorSig(0,0) = sigarg[0];
      tensorSig(1,1) = sigarg[1];
      tensorSig(2,2) = sigarg[2];
      tensorSig(0,1) = sigarg[3];
      tensorSig(1,0) = sigarg[3];
      tensorSig(2,1) = sigarg[4];
      tensorSig(1,2) = sigarg[4];
      tensorSig(2,0) = sigarg[5];
      tensorSig(0,2) = sigarg[5];

      //Comment by KC : Computing tensorR at the end of the time-step
      deformationGradient_new[idx].polarDecompositionAFFinvTran(tensorU,tensorR);

      // ROTATE pstress_new: S=R*tensorSig*R^T
      pstress_new[idx] = (tensorR*tensorSig)*(tensorR.Transpose());
      c_dil = sqrt(USM/rho_cur);
      // Compute The Strain Energy for all the particles
      Matrix3 AvgStress = (pstress_new[idx] + pstress[idx])*.5;

      double e = (D(0,0)*AvgStress(0,0) +
                  D(1,1)*AvgStress(1,1) +
                  D(2,2)*AvgStress(2,2) +
              2.*(D(0,1)*AvgStress(0,1) +
                  D(0,2)*AvgStress(0,2) +
                  D(1,2)*AvgStress(1,2))) * pvolume_new[idx]*delT;

      se += e;

      // Compute wave speed at each particle, store the maximum
      Vector pvelocity_idx = pvelocity[idx];
      WaveSpeed=Vector(Max(c_dil+fabs(pvelocity_idx.x()),WaveSpeed.x()),
                       Max(c_dil+fabs(pvelocity_idx.y()),WaveSpeed.y()),
                       Max(c_dil+fabs(pvelocity_idx.z()),WaveSpeed.z()));

      // Compute artificial viscosity term
      if (flag->d_artificial_viscosity) {
        double dx_ave = (dx.x() + dx.y() + dx.z())/3.0;
        double c_bulk = sqrt(UI[0]/rho_cur);
        Matrix3 D=(velGrad + velGrad.Transpose())*0.5;
        p_q[idx] = artificialBulkViscosity(D.Trace(), c_bulk, rho_cur, dx_ave);
      } else {
        p_q[idx] = 0.;
      }

    }  // end loop over particles

    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(se),     lb->StrainEnergyLabel);
    }

    delete interpolator;
  }
}



void Kayenta::carryForward(const PatchSubset* patches,
                           const MPMMaterial* matl,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int dwi = matl->getDWIndex();
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

    constParticleVariable<double> peakI1IDist;
    ParticleVariable<double> peakI1IDist_new;
    constParticleVariable<int>     pLocalized;

    old_dw->get(peakI1IDist, peakI1IDistLabel, pset);
    new_dw->allocateAndPut(peakI1IDist_new,
                                 peakI1IDistLabel_preReloc, pset);
    peakI1IDist_new.copyData(peakI1IDist);
    old_dw->get(pLocalized,      pLocalizedLabel,      pset);

    // Carry forward the data common to all constitutive models
    // when using RigidMPM.
    // This method is defined in the ConstitutiveModel base class.
    carryForwardSharedData(pset, old_dw, new_dw, matl);

    // Carry forward the data local to this constitutive model
    StaticArray<constParticleVariable<double> > ISVs(d_NINSV+1);
    StaticArray<ParticleVariable<double> > ISVs_new(d_NINSV+1);
    ParticleVariable<int>          pLocalized_new;

    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],ISVLabels[i], pset);
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
      ISVs_new[i].copyData(ISVs[i]);
  }
    new_dw->allocateAndPut(pLocalized_new, pLocalizedLabel_preReloc, pset);
    // Don't affect the strain energy or timestep size
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),   lb->StrainEnergyLabel);
    }
  }

}

void Kayenta::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  // Other constitutive model and input dependent computes and requires
  for(int i=0;i<d_NINSV;i++){
    task->computes(ISVLabels[i], matlset);
  }
  task->computes(pLocalizedLabel,     matlset);
  task->computes(peakI1IDistLabel, matlset);
}

void Kayenta::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl,
                                     const PatchSet* patches) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();
  addSharedCRForHypoExplicit(task, matlset, patches);

  // Computes and requires for internal state data
  for(int i=0;i<d_NINSV;i++){
    task->requires(Task::OldDW, ISVLabels[i],          matlset, Ghost::None);

    task->computes(             ISVLabels_preReloc[i], matlset);
  }
  task->requires(Task::OldDW, pLocalizedLabel,        matlset, Ghost::None);
  task->requires(Task::OldDW, peakI1IDistLabel, matlset, Ghost::None);
  task->requires(Task::OldDW, lb->pParticleIDLabel,  matlset, Ghost::None);
  task->computes(peakI1IDistLabel_preReloc, matlset);
  task->computes(pLocalizedLabel_preReloc,      matlset);
}

void Kayenta::addComputesAndRequires(Task*,
                                     const MPMMaterial*,
                                     const PatchSet*,
                                     const bool ) const
{
}

double Kayenta::computeRhoMicroCM(double pressure,
                                  const double p_ref,
                                  const MPMMaterial* matl,
                                  double temperature,
                                  double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = UI[0];

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR Kayenta" << endl;
#endif
}

void Kayenta::computePressEOSCM(double rho_cur, double& pressure,
                                double p_ref,
                                double& dp_drho,      double& tmp,
                                const MPMMaterial* matl,
                                double temperature)
{

  double bulk = UI[0];
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 1
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR Kayenta" << endl;
#endif
}

double Kayenta::getCompressibility()
{
  return 1.0/UI[0];
}

void
Kayenta::getInputParameters(ProblemSpecP& ps)
{
  ps->require("B0",             UI[0]);
  ps->getWithDefault("B1",      UI[1],0.0);
  ps->getWithDefault("B2",      UI[2],0.0);
  ps->getWithDefault("B3",      UI[3],0.0);
  ps->getWithDefault("B4",      UI[4],0.0);
  ps->require("G0",             UI[5]);
  ps->getWithDefault("G1",      UI[6],0.0);
  ps->getWithDefault("G2",      UI[7],0.0);
  ps->getWithDefault("G3",      UI[8],0.0);
  ps->getWithDefault("G4",      UI[9],0.0);
  ps->getWithDefault("RJS",     UI[10],0.0);
  ps->getWithDefault("RKS",     UI[11],0.0);
  ps->getWithDefault("RKN",     UI[12],0.0);
  ps->getWithDefault("A1",      UI[13],0.0);
  ps->getWithDefault("A2",      UI[14],0.0);
  ps->getWithDefault("A3",      UI[15],0.0);
  ps->getWithDefault("A4",      UI[16],0.0);
  ps->getWithDefault("P0",      UI[17],0.0);
  ps->getWithDefault("P1",      UI[18],0.0);
  ps->getWithDefault("P2",      UI[19],0.0);
  ps->getWithDefault("P3",      UI[20],0.0);
  ps->getWithDefault("CR",      UI[21],0.0);
  ps->getWithDefault("RK",      UI[22],0.0);
  ps->getWithDefault("RN",      UI[23],0.0);
  ps->getWithDefault("HC",      UI[24],0.0);
  ps->getWithDefault("CTI1",    UI[25],0.0);
  ps->getWithDefault("CTPS",    UI[26],0.0);
  ps->getWithDefault("T1",      UI[27],0.0);
  ps->getWithDefault("T2",      UI[28],0.0);
  ps->getWithDefault("T3",      UI[29],0.0);
  ps->getWithDefault("T4",      UI[30],0.0);
  ps->getWithDefault("T5",      UI[31],0.0);
  ps->getWithDefault("T6",      UI[32],0.0);
  ps->getWithDefault("T7",      UI[33],0.0);
  ps->getWithDefault("J3TYPE",  UI[34],0.0);
  ps->getWithDefault("A2PF",    UI[35],0.0);
  ps->getWithDefault("A4PF",    UI[36],0.0);
  ps->getWithDefault("CRPF",    UI[37],0.0);
  ps->getWithDefault("RKPF",    UI[38],0.0);
  ps->getWithDefault("SUBX",    UI[39],0.0);
  ps->getWithDefault("DEJAVU",  UI[40],0.0);
  ps->getWithDefault("FAIL0",   UI[41],0.0);
  ps->getWithDefault("FAIL1",   UI[42],0.0);
  ps->getWithDefault("FAIL2",   UI[43],0.0);
  ps->getWithDefault("FAIL3",   UI[44],0.0);
  ps->getWithDefault("FAIL4",   UI[45],0.0);
  ps->getWithDefault("FAIL5",   UI[46],0.0);
  ps->getWithDefault("FAIL6",   UI[47],0.0);
  ps->getWithDefault("FAIL7",   UI[48],0.0);
  ps->getWithDefault("FAIL8",   UI[49],0.0);
  ps->getWithDefault("FAIL9",   UI[50],0.0);
  ps->getWithDefault("PEAKI1I", UI[51],0.0);
  ps->getWithDefault("STRENI",  UI[52],0.0);
  ps->getWithDefault("FSLOPEI", UI[53],0.0);
  ps->getWithDefault("PEAKI1F", UI[54],0.0);
  ps->getWithDefault("STRENF",  UI[55],0.0);
  ps->getWithDefault("SOFTENING",UI[56],0.0);
  ps->getWithDefault("FSLOPEF", UI[57],0.0);
  ps->getWithDefault("FAILSTAT",UI[58],0.0);
  ps->getWithDefault("EOSID",   UI[59],0.0);
  ps->getWithDefault("USEHOSTEOS",UI[60],0.0);
  ps->getWithDefault("DILATLIM",UI[61],0.0);
  ps->getWithDefault("FREE01",  UI[62],0.0);
  ps->getWithDefault("FREE02",  UI[63],0.0);
  ps->getWithDefault("FREE03",  UI[64],0.0);
  ps->getWithDefault("FREE04",  UI[65],0.0);
  ps->getWithDefault("FREE05",  UI[66],0.0);
  ps->getWithDefault("CTPSF",   UI[67],0.0);
  ps->getWithDefault("YSLOPEI", UI[68],0.0);
  ps->getWithDefault("YSLOPEF", UI[69],0.0);

  //     ________________________________________________________________________
  //     EOSMG inputs
  int IJTHERMPAR =d_NBASICINPUTS+d_NUMJOINTINPUTS;
  ps->getWithDefault("TMPRXP",  UI[IJTHERMPAR    ],0.0);
  ps->getWithDefault("THERM01", UI[IJTHERMPAR + 1],0.0);
  ps->getWithDefault("THERM02", UI[IJTHERMPAR + 2],0.0);
  ps->getWithDefault("THERM03", UI[IJTHERMPAR + 3],0.0);
  ps->getWithDefault("TMPRM0",  UI[IJTHERMPAR + 4],0.0);
  //     ________________________________________________________________________
  //     EOSMGCT inputs
  ps->getWithDefault("RHO0",    UI[d_IEOSMGCT     ],0.0);
  ps->getWithDefault("TMPR0",   UI[d_IEOSMGCT +  1],0.0);
  ps->getWithDefault("SNDSP0",  UI[d_IEOSMGCT +  2],0.0);
  ps->getWithDefault("S1MG",    UI[d_IEOSMGCT +  3],0.0);
  ps->getWithDefault("GRPAR",   UI[d_IEOSMGCT +  4],0.0);
  ps->getWithDefault("CV",      UI[d_IEOSMGCT +  5],0.0);
  ps->getWithDefault("ESFT",    UI[d_IEOSMGCT +  6],0.0);
  ps->getWithDefault("RP",      UI[d_IEOSMGCT +  7],0.0);
  ps->getWithDefault("PS",      UI[d_IEOSMGCT +  8],0.0);
  ps->getWithDefault("PE",      UI[d_IEOSMGCT +  9],0.0);
  ps->getWithDefault("CE",      UI[d_IEOSMGCT + 10],0.0);
  ps->getWithDefault("NSUB",    UI[d_IEOSMGCT + 11],0.0);
  ps->getWithDefault("S2MG",    UI[d_IEOSMGCT + 12],0.0);
  ps->getWithDefault("TYP",     UI[d_IEOSMGCT + 13],0.0);
  ps->getWithDefault("RO",      UI[d_IEOSMGCT + 14],0.0);
  ps->getWithDefault("TO",      UI[d_IEOSMGCT + 15],0.0);
  ps->getWithDefault("S",       UI[d_IEOSMGCT + 16],0.0);
  ps->getWithDefault("GRPARO",  UI[d_IEOSMGCT + 17],0.0);
  ps->getWithDefault("B",       UI[d_IEOSMGCT + 18],0.0);
  ps->getWithDefault("XB",      UI[d_IEOSMGCT + 19],0.0);
  ps->getWithDefault("NB",      UI[d_IEOSMGCT + 20],0.0);
  ps->getWithDefault("PWR",     UI[d_IEOSMGCT + 21],0.0);
  //    ________________________________________________________________________
  //    EOSMG Derived Constants
  ps->getWithDefault("A1MG",    DC[0],0.0);
  ps->getWithDefault("A2MG",    DC[1],0.0);
  ps->getWithDefault("A3MG",    DC[2],0.0);
  ps->getWithDefault("A4MG",    DC[3],0.0);
  ps->getWithDefault("A5MG",    DC[4],0.0);
  ps->getWithDefault("A0MG",    DC[5],0.0);
  ps->getWithDefault("AEMG",    DC[6],0.0);
  ps->getWithDefault("FK0",     DC[7],0.0);
  ps->getWithDefault("AF",      DC[8],0.0);
  ps->getWithDefault("PF",      DC[9],0.0);
  ps->getWithDefault("XF",      DC[10],0.0);
  ps->getWithDefault("CF",      DC[11],0.0);
  ps->getWithDefault("RMX",     DC[12],0.0);
  //    ________________________________________________________________________
  //    Uintah Variability Variables
  ps->get("PEAKI1IDIST",wdist.WeibDist);
  WeibullParser(wdist);
  ps->getWithDefault("hugeJ", d_hugeJ, 20.0);

//  proc0cout << "Weibull Variables for PEAKI1I (getInputParameters):\n"
//            << "Median:            " << wdist.WeibMed
//            << "\nModulus:         " << wdist.WeibMod
//            << "\nReference Vol:   " << wdist.WeibRefVol
//            << "\nSeed:            " << wdist.WeibSeed << std::endl;
}

void
Kayenta::initializeLocalMPMLabels()
{

  // create a localized variable
  pLocalizedLabel = VarLabel::create("p.localized",
                  ParticleVariable<int>::getTypeDescription());
  pLocalizedLabel_preReloc = VarLabel::create("p.localized+",
                  ParticleVariable<int>::getTypeDescription());
 vector<string> ISVNames;

// These lines of code are added by KC to replace the currently hard-coded
// internal variable allocation with a proper call to KMMRXV routine.
//Create VarLabels for Kayenta internal state variables (ISVs)
  int nx;
  char namea[5000];
  char keya[5000];
  double rinit[100];
  double rdim[700];
  int iadvct[100];
  int itype[100];

  KAYENTA_RXV( UI, GC, DC, nx, namea, keya, rinit, rdim, iadvct, itype );

  char *ISV[d_NINSV];
  ISV[0] = strtok(keya, "|"); // Splits | between words in string
  ISVNames.push_back(ISV[0]);
  proc0cout << "ISV[" << 0 << "] is called " << ISVNames[0] << endl;
  for(int i = 1; i < d_NINSV ; i++) {
// If you specify NULL, by default it will start again from the previous stop.
     ISV[i] = strtok (NULL, "|");
     ISVNames.push_back(ISV[i]);
     proc0cout << "ISV[" << i << "] is called " << ISVNames[i] << endl;
  }
// Code ends here.KC


  for(int i=0;i<d_NINSV;i++){
    ISVLabels.push_back(VarLabel::create(ISVNames[i],
                          ParticleVariable<double>::getTypeDescription()));
    ISVLabels_preReloc.push_back(VarLabel::create(ISVNames[i]+"+",
                          ParticleVariable<double>::getTypeDescription()));
  }
  peakI1IDistLabel = VarLabel::create("peakI1IDist",
                     ParticleVariable<double>::getTypeDescription());
  peakI1IDistLabel_preReloc = VarLabel::create("peakI1IDist+",
                     ParticleVariable<double>::getTypeDescription());
}


// Weibull input parser that accepts a structure of input
// parameters defined as:
//
// bool Perturb        'True' for perturbed parameter
// double WeibMed       Medain distrib. value OR const value
//                         depending on bool Perturb
// double WeibMod       Weibull modulus
// double WeibRefVol    Reference Volume
// int    WeibSeed      Seed for random number generator
// std::string WeibDist  String for Distribution
//
// the string 'WeibDist' accepts strings of the following form
// when a perturbed value is desired:
//
// --Distribution--|-Median-|-Modulus-|-Reference Vol -|- Seed -|
// "    weibull,      45e6,      4,        0.0001,          0"
//
// or simply a number if no perturbed value is desired.

void
Kayenta::WeibullParser(WeibParameters &iP)
{

  // Remove all unneeded characters
  // only remaining are alphanumeric '.' and ','
  for ( int i = iP.WeibDist.length()-1; i >= 0; i--) {
    iP.WeibDist[i] = tolower(iP.WeibDist[i]);
    if ( !isalnum(iP.WeibDist[i]) &&
       iP.WeibDist[i] != '.' &&
       iP.WeibDist[i] != ',' &&
       iP.WeibDist[i] != '-' &&
       iP.WeibDist[i] != EOF) {
         iP.WeibDist.erase(i,1);
    }
  } // End for

  if (iP.WeibDist.substr(0,4) == "weib") {
    iP.Perturb = true;
  } else {
    iP.Perturb = false;
  }

  // ######
  // If perturbation is NOT desired
  // ######
  if ( !iP.Perturb ) {
    bool escape = false;
    int num_of_e = 0;
    int num_of_periods = 0;
    for ( unsigned int i = 0; i < iP.WeibDist.length(); i++) {
      if ( iP.WeibDist[i] != '.'
           && iP.WeibDist[i] != 'e'
           && iP.WeibDist[i] != '-'
           && !isdigit(iP.WeibDist[i]) ) escape = true;

      if ( iP.WeibDist[i] == 'e' ) num_of_e += 1;

      if ( iP.WeibDist[i] == '.' ) num_of_periods += 1;

      if ( num_of_e > 1 || num_of_periods > 1 || escape ) {
        std::cerr << "\n\nERROR:\nInput value cannot be parsed. Please\n"
                     "check your input values.\n" << std::endl;
        exit (1);
      }
    } // end for(int i = 0;....)

    if ( escape ) exit (1);

    iP.WeibMed  = atof(iP.WeibDist.c_str());
  }

  // ######
  // If perturbation IS desired
  // ######
  if ( iP.Perturb ) {
    int weibValues[4];
    int weibValuesCounter = 0;

    for ( unsigned int r = 0; r < iP.WeibDist.length(); r++) {
      if ( iP.WeibDist[r] == ',' ) {
        weibValues[weibValuesCounter] = r;
        weibValuesCounter += 1;
      } // end if(iP.WeibDist[r] == ',')
    } // end for(int r = 0; ...... )

    if (weibValuesCounter != 4) {
      std::cerr << "\n\nERROR:\nWeibull perturbed input string must contain\n"
                   "exactly 4 commas. Verify that your input string is\n"
                   "of the form 'weibull, 45e6, 4, 0.001, 1'.\n" << std::endl;
      exit (1);
    } // end if(weibValuesCounter != 4)

    std::string weibMedian;
    std::string weibModulus;
    std::string weibRefVol;
    std::string weibSeed;

    weibMedian  = iP.WeibDist.substr(weibValues[0]+1,weibValues[1]-weibValues[0]-1);
    weibModulus = iP.WeibDist.substr(weibValues[1]+1,weibValues[2]-weibValues[1]-1);
    weibRefVol  = iP.WeibDist.substr(weibValues[2]+1,weibValues[3]-weibValues[2]-1);
    weibSeed    = iP.WeibDist.substr(weibValues[3]+1);

    iP.WeibMed    = atof(weibMedian.c_str());
    iP.WeibMod    = atof(weibModulus.c_str());
    iP.WeibRefVol = atof(weibRefVol.c_str());
    iP.WeibSeed   = atoi(weibSeed.c_str());
    UI[51]=iP.WeibMed;  // Set this here to satisfy KAYENTA_CHK
  } // End if (iP.Perturb)
}
