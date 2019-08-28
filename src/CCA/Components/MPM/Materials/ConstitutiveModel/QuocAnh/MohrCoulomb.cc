/*
The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/MohrCoulomb.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/ShengMohrCoulomb.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/QuocAnh/ClassicMohrCoulomb.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
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
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <Core/Math/Matrix3.h>
#include <Core/Math/Short27.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//#include <Core/Containers/StaticArray.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>

#include <sci_defs/uintah_defs.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

//sets of external variables for the Sheng Mohr Coulomb algorithm by WTS. Some are redundant.

double ALFACHECK, ALFACHANGE, ALFARATIO, MAXITER, YIELDTOL, TOL_METHOD, INCREMENT_TYPE, BETA_FACT;
double DRIFT_CORRECTION, EULER_ITERATIONS,CRITICAL_STEP_SIZE;
double STEP_MAX, STEP_MIN, ERROR_DEF, USE_ERROR_STEP, MIN_DIVISION_SIZE;
double INTEGRATION_TOL=0.0001;
double LINES_SEC_NO=50;
int MAX_LOOP, SOLUTION_ALGORITHM, ALGORITHM_TYPE;   //MAX_LOOP - number of steps to solve, SOLUTION_ALGORITHM - algorithm to use
double USE_ERROR=0.5, SAVE_FOR_ERROR=1;    //these are values - 1st - when 'use' the additional error to speed up the calculations (here 30%)
// SAVEFORERROR says how greater accuracy the program should use; 1 means no change. 0.5 means 50% greater accuracy. (tol * 0.5) etc
double CHGEPSINTOL=10e-9;
double ADDTOLYIELD=0.8;
double SUCTIONTOL=0.00000001;
double TINY=1e-14;		//value used for example in checking whether we are not dividing by zero in CalcStressElast, volumetric strain must be also larger then tiny
double PMIN=0.0001;			// value of minimum mean stress to calculate K in CalcStressElast
int USE_NICE_SCHEME=0;

////////////////////////////////////////////////////////////////////////////////
// The following functions are found in fortran/*.F
//SUBROUTINE YENTA_CALC( NBLK, NINSV, DT, PROP,
  //   $                                   SIGARG, D, SVARG, USM   )

extern "C"{

#if defined( FORTRAN_UNDERSCORE_END )
#  define DMMCHK dmmchk_
#  define DIAMM_CALC diamm_calc_
#  define DMMRXV dmmrxv_
#elif defined( FORTRAN_UNDERSCORE_LINUX )
#  define DMMCHK dmmchk_
#  define DMMRXV dmmrxv_
#  define DIAMM_CALC diamm_calc__
#else // NONE
#  define DMMCHK dmmchk
#  define DIAMM_CALC diamm_calc
#  define DMMRXV dmmrxv
#endif

//#define DMM_ANISOTROPIC
//#undef DMM_ANISOTROPIC

 //  void DMMCHK( double UI[], double UJ[], double UK[] );

/*
C
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for DMM model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C       UI: user input as read and stored by host code.
C
C       Upon entry, the UI array contains the user inputs EXACTLY
C       as read from the user.  These inputs must be ordered in the
C       UI array as indicated in the file kmmpnt.Blk.
C       See kmmpnt.Blk for parameter definitions and keywords.
C
C       DC: Not used with this model
C
C    Other output
C    ------------
C       GC: Not used with this model
C       DC: Not used with this model
C       Because GC and DC are not used, you may call this routine
C       with a line of the form "CALL DMMCHK(UI,UI,UI)"
*/

  // void DIAMM_CALC( int &nblk, int &ninsv, double &dt,
   //                                 double UI[], double stress[], double D[],
     //                               double svarg[], double &USM );
/*
C***********************************************************************
C
C     Description:
C           Drucker-Prager plasticity model with elastic strain induced
C           anisotropy.
C
C***********************************************************************
C
C     input arguments
C     ===============
C      NBLK       int                   Number of blocks to be processed
C      NINSV      int                   Number of internal state vars
C      DTARG      dp                    Current time increment
C      UI       dp,ar(nprop)            User inputs
C      D          dp,ar(6)              Strain increment
C
C     input output arguments
C     ======================
C      STRESS   dp,ar(6)                stress
C      SVARG    dp,ar(ninsv)            state variables
C
C     output arguments
C     ================
C      USM      dp                      uniaxial strain modulus
C
C***********************************************************************
C
C      stresss and strains, plastic strain tensors
C          11, 22, 33, 12, 23, 13
C
C***********************************************************************
*/

   void DMMRXV( double UI[], double UJ[], double UK[], int &nx, char* namea[],
                char* keya[], double rinit[], double rdim[], int iadvct[],
                int itype[] );
}
/*
C**********************************************************************
C     REQUESTED EXTRA VARIABLES FOR KAYENTA
C
C     This subroutine creates lists of the internal state variables
C     needed for DMM. This routine merely sends a
C     LIST of internal state variable requirements back to the host
C     code.   IT IS THE RESPONSIBILITY OF THE HOST CODE to loop over
C     the items in each list to actually establish necessary storage
C     and (if desired) set up plotting, restart, and advection
C     control for each internal state variable.
C
C     called by: host code after all input data have been checked
C
C     input
C     -----
C          UI = user input array
C          GC = unused for this model (placeholder)
C          DC = unused for this model (placeholder)
C
C     output
C     ------
C          NX = number of extra variables                    [DEFAULT=0]
C       NAMEA = single character array created from a string array
C               (called NAME) used locally in this routine to register
C               a descriptive name for each internal state variable.
C        KEYA = single character array created from a string array
C               (called KEY) used locally in this routine to register
C               a plot keyword for each internal state variable.
C          | Note: NAMEA and KEYA are created from the local variables |
C          | NAME and KEY by calls to the subroutine TOKENS, which     |
C          | is a SERVICE routine presumed to ALREADY exist within the |
C          | host code (we can provide this routine upon request).     |
C          | "NAME" is a fortran array of strings. "NAMEA" is a one    |
C          | dimensional array of single characters. For readability,  |
C          | most of this subroutine writes to the NAME array. Only at |
C          | the very end is NAME converted to NAMEA by calling a      |
C          | the utility routine TOKENS. The KEY array is similarly    |
C          | converted to KEYA.  These conversions are performed       |
C          | because host codes written in C or C++ are unable to      |
C          | process FORTRAN string arrays. Upon request, we can       |
C          | provide a utility routine that will convert BACK to       |
C          | FORTRAN string arrays if your host code is FORTRAN.       |
C          | Likewise, we can provide a C++ routine that will allow    |
C          | parsing the single-character arrays to convert them back  |
C          | to strings if your code is C++. Alternatively, you can    |
C          | simply ignore the NAMEA and KEYA outputs of this routine  |
C          | if your host code does not wish to establish plotting     |
C          | information.                                              |
C
C       RINIT = initial value for each ISV               [DEFAULT = 0.0]
C        RDIM = physical dimension exponents             [DEFAULT = 0.0]
C               This variable is dimensioned RDIM(7,*) for the 7 base
C               dimensions (and * for the number of extra variables):
C
C                      1 --- length
C                      2 --- mass
C                      3 --- time
C                      4 --- temperature
C                      5 --- discrete count
C                      6 --- electric current
C                      7 --- luminous intensity
C
C                Suppose, for example, that an ISV has units of stress.
C                Dimensionally, stress is length^(1) times mass^(-1)
C                times time^(-2). Therefore, this routine would return
C                1.0, -1.0, and -2.0 as the first three values of the
C                RDIM array. Host codes that work only in one unit
C                set (e.g., SI) typically ignore the RDIM output.
C
C      IADVCT = advection option                           [DEFAULT = 0]
C                    = 0 advect by mass-weighted average
C                    = 1 advect by volume-weighted average
C                    = 2 don't advect
C            The advection method will often be ignored by host codes.
C            It is used for Eulerian codes and for Lagrangian codes that
C            re-mesh (and therefore need guidance about how to "mix"
C            internal state variables). Note: a value of 2 implies that
C            the ISV is output only.
C
C        ITYPE = variable type (see migtionary preface)    [DEFAULT = 1]
C                  1=scalar
C                  6=2nd-order symmetric tensor
C        The component ordering for ITYPE=6 is 11, 22, 33, 12, 23, 31.
C        Consequently, the 11 component is the first one to be requested
C        in tensor lists, and its IFLAG is set to 6. To indicate that
C        subsequent ISVs are the remaining components of the same tensor,
C        the next five ISVs are given an IFLAG value of -6.
C        Host codes that don't change basis can ignore ITYPE.
*/
// End fortran functions.
////////////////////////////////////////////////////////////////////////////////
using std::cerr; using namespace Uintah;

MohrCoulomb::MohrCoulomb(ProblemSpecP& ps,MPMFlags* Mflag)
  : ConstitutiveModel(Mflag)
{
  d_NBASICINPUTS=48;
  d_NMGDC=0;

// Total number of properties
  d_NDMMPROP=d_NBASICINPUTS+d_NMGDC;

  // pre-initialize all of the user inputs to zero.
  for(int i = 0; i<d_NDMMPROP; i++){
     UI[i] = 0.;
  }
  // Read model parameters from the input file
  getInputParameters(ps);

  // Check that model parameters are valid and allow model to change if needed


  //DMMCHK(UI,UI,&UI[d_NBASICINPUTS]);
  CheckModel (UI);


  //Create VarLabels for GeoModel internal state variables (ISVs)
  int nx;
     nx=d_NBASICINPUTS;

    for (int i=0;i<nx;i++)
    {
        rinit[i]=UI[i];
        //cerr<<" UI["<<i<<"]="<<UI[i];
    }
  d_NINSV=nx;
  //  cout << "d_NINSV = " << d_NINSV <<l endl;

  initializeLocalMPMLabels();
}

#if 0
MohrCoulomb::MohrCoulomb(const MohrCoulomb* cm) : ConstitutiveModel(cm)
{
  for(int i=0;i<d_NDMMPROP;i++){
    UI[i] = cm->UI[i];
  }

  //Create VarLabels for Diamm internal state variables (ISVs)
  initializeLocalMPMLabels();
}
#endif

MohrCoulomb::~MohrCoulomb()
{
   for (unsigned int i = 0; i< ISVLabels.size();i++){
     VarLabel::destroy(ISVLabels[i]);
   }
}

void MohrCoulomb::outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag)
{
  ProblemSpecP cm_ps = ps;
  if (output_cm_tag) {
    cm_ps = ps->appendChild("constitutive_model");
    cm_ps->setAttribute("type","MohrCoulomb");
  }

  cm_ps->appendElement("G",UI[0]);   //  shear modulus (stress)
  cm_ps->appendElement("K",UI[1]);   //  bulk modulus (stress)
  cm_ps->appendElement("c",UI[2]);   //  cohesion (stress)
  cm_ps->appendElement("Phi",UI[3]);   // friction angle (degrees)
  cm_ps->appendElement("Psi",UI[4]);   // dilation angle (degrees, for non-associated flow rule)
  cm_ps->appendElement("Version",UI[5]);   // Version of the model

   cm_ps->appendElement("suction",UI[6]);
  cm_ps->appendElement("UseWaterRetention",UI[7]);


  cm_ps->appendElement("WR_Param1",UI[8]);
  cm_ps->appendElement("WR_Param2",UI[9]);
  cm_ps->appendElement("WR_Param3",UI[10]);
  cm_ps->appendElement("WR_Param4",UI[11]);

  cm_ps->appendElement("SpecVol",UI[12]);  // meridional yld prof param

  cm_ps->appendElement("PhiB",UI[13]);  // Fredlund shear angle, c=c+suction*tan(phib)
    cm_ps->appendElement("Usetransition",UI[14]); // undrained shear strength transition
  cm_ps->appendElement("A1",UI[15]);	// water influence parameter
  cm_ps->appendElement("B1",UI[16]);	// water influence parameter
  cm_ps->appendElement("W",UI[17]);	// water content
  cm_ps->appendElement("beta",UI[18]);	// strain rate influence parameter
  cm_ps->appendElement("strain_ref",UI[19]); // shear strain rate reference
  cm_ps->appendElement("shear_strain_rate",UI[20]); // shear strain rate
  cm_ps->appendElement("Usemodul",UI[21]); // modul with strain rate
  cm_ps->appendElement("m_modyul",UI[22]); // modul ratio
  cm_ps->appendElement("nuy",UI[23]); // modul ratio
  cm_ps->appendElement("shear_strain",UI[24]);


  cm_ps->appendElement("Use_linear",UI[25]);
  cm_ps->appendElement("a",UI[26]);
  cm_ps->appendElement("y_ref",UI[27]);


  cm_ps->appendElement("strain11",UI[28]);
  cm_ps->appendElement("strain22",UI[29]);
  cm_ps->appendElement("strain33",UI[30]);
  cm_ps->appendElement("strain12",UI[31]);
  cm_ps->appendElement("strain23",UI[32]);
  cm_ps->appendElement("strain13",UI[33]);

  cm_ps->appendElement("Use_softening",UI[34]);
  cm_ps->appendElement("St",UI[35]);
  cm_ps->appendElement("strain_95",UI[36]);

    cm_ps->appendElement("y",UI[37]);
    cm_ps->appendElement("n",UI[38]);

    cm_ps->appendElement("s_xx",UI[39]);
    cm_ps->appendElement("s_yy",UI[40]);
    cm_ps->appendElement("Ko",UI[41]);

    cm_ps->appendElement("Use_regular",UI[42]);
    cm_ps->appendElement("tFE",UI[43]);
    cm_ps->appendElement("tShear",UI[44]);

    cm_ps->appendElement("s_xy",UI[45]);

	cm_ps->appendElement("n_nonlocalMC", UI[46]);
	cm_ps->appendElement("l_nonlocal", UI[47]);

}

MohrCoulomb* MohrCoulomb::clone()
{
  return scinew MohrCoulomb(*this);
}

void MohrCoulomb::initializeCMData(const Patch* patch,
                               const MPMMaterial* matl,
                               DataWarehouse* new_dw)
{
  // Initialize the variables shared by all constitutive models
  // This method is defined in the ConstitutiveModel base class.
  initSharedDataForExplicit(patch, matl, new_dw);

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  std::vector<ParticleVariable<double> > ISVs(d_NINSV+1);

  cout << "In initializeCMData" << endl;
  for(int i=0;i<d_NINSV;i++){
    new_dw->allocateAndPut(ISVs[i],ISVLabels[i], pset);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end(); iter++){
      ISVs[i][*iter] = rinit[i];
    }
  }

  computeStableTimestep(patch, matl, new_dw);
}

void MohrCoulomb::addParticleState(std::vector<const VarLabel*>& from,
                               std::vector<const VarLabel*>& to)
{
  // Add the local particle state data for this constitutive model.
  for(int i=0;i<d_NINSV;i++){
    from.push_back(ISVLabels[i]);
    to.push_back(ISVLabels_preReloc[i]);
  }
}

void MohrCoulomb::computeStableTimestep(const Patch* patch,
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

  double bulk = UI[1];
  double G = UI[0]; //modified: K=UI[1], G=UI[0]
  for(ParticleSubset::iterator iter = pset->begin();iter != pset->end();iter++){
     particleIndex idx = *iter;

     // Compute wave speed at each particle, store the maximum
     c_dil = sqrt((bulk + 4.*G/3.)*pvolume[idx]/pmass[idx]);
     WaveSpeed=Vector(Max(c_dil+fabs(pvelocity[idx].x()),WaveSpeed.x()),
                      Max(c_dil+fabs(pvelocity[idx].y()),WaveSpeed.y()),
                      Max(c_dil+fabs(pvelocity[idx].z()),WaveSpeed.z()));
  }
  //UI[14]=matl->getInitialDensity();
  //UI[15]=matl->getRoomTemperature();
  //UI[14]=bulk/matl->getInitialDensity();  ??tim
  //UI[19]=matl->getInitialCv();
  WaveSpeed = dx/WaveSpeed;
  double delT_new = WaveSpeed.minComponent();
  new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
}

void MohrCoulomb::computeStressTensor(const PatchSubset* patches,
                                  const MPMMaterial* matl,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{

double rho_orig = matl->getInitialDensity();
  for(int p=0;p<patches->size();p++){
    double se = 0.0;
    const Patch* patch = patches->get(p);

    Matrix3 Identity; Identity.Identity();
    double c_dil=0.0;
    Vector WaveSpeed(1.e-12,1.e-12,1.e-12);
    Vector dx = patch->dCell();

    int dwi = matl->getDWIndex();
    // Create array for the particle position
    ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
    constParticleVariable<Matrix3> deformationGradient, pstress;
    ParticleVariable<Matrix3> pstress_new;
    constParticleVariable<Matrix3> deformationGradient_new, velGrad;
    constParticleVariable<double> pmass, pvolume, ptemperature;
    constParticleVariable<double> pvolume_new;
    constParticleVariable<Vector> pvelocity;
    constParticleVariable<Point> px, pxnew;
    delt_vartype delT;

    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    old_dw->get(px,                  lb->pXLabel,                  pset);
    old_dw->get(pstress,             lb->pStressLabel,             pset);
    old_dw->get(pmass,               lb->pMassLabel,               pset);
    old_dw->get(pvolume,             lb->pVolumeLabel,             pset);
    old_dw->get(pvelocity,           lb->pVelocityLabel,           pset);
    old_dw->get(ptemperature,        lb->pTemperatureLabel,        pset);
    old_dw->get(deformationGradient, lb->pDeformationMeasureLabel, pset);
	new_dw->get(pvolume_new,		 lb->pVolumeLabel_preReloc,    pset);
	new_dw->get(pxnew,				 lb->pXLabel_preReloc,		   pset);

	std::vector<constParticleVariable<double> > ISVs(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],           ISVLabels[i],                 pset);
    }

    ParticleVariable<double> pdTdt,p_q;

    new_dw->allocateAndPut(pstress_new,     lb->pStressLabel_preReloc,   pset);
    new_dw->allocateAndPut(pdTdt,           lb->pdTdtLabel,              pset);
    new_dw->allocateAndPut(p_q,             lb->p_qLabel_preReloc,       pset);
    new_dw->get(deformationGradient_new,
                                 lb->pDeformationMeasureLabel_preReloc,  pset);
    new_dw->get(velGrad,         lb->pVelGradLabel_preReloc,             pset);

	std::vector<ParticleVariable<double> > ISVs_new(d_NINSV+1);
    for(int i=0;i<d_NINSV;i++){
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
    }

    for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
      particleIndex idx = *iter;


      // Assign zero internal heating by default - modify if necessary.
      pdTdt[idx] = 0.0;

      // Calculate rate of deformation D, and deviatoric rate DPrime,
      Matrix3 D = (velGrad[idx] + velGrad[idx].Transpose())*.5;

      // get the volumetric part of the deformation
      double J = deformationGradient_new[idx].Determinant();
      // Check 1: Look at Jacobian
      if (!(J > 0.0)) {
        cerr << getpid() ;
        constParticleVariable<long64> pParticleID;
        old_dw->get(pParticleID, lb->pParticleIDLabel, pset);
        cerr << "**ERROR** Negative Jacobian of deformation gradient"
             << " in particle " << pParticleID[idx] << endl;
        cerr << "l = " << velGrad[idx] << endl;
        cerr << "F_old = " << deformationGradient[idx] << endl;
        cerr << "F_new = " << deformationGradient_new[idx] << endl;
        cerr << "J = " << J << endl;
        throw InternalError("Negative Jacobian",__FILE__,__LINE__);
      }

      // Compute the local sound speed
      double rho_cur = rho_orig/J;

      // NEED TO FIND R
      Matrix3 tensorR, tensorU;

      // Look into using Rebecca's PD algorithm
      deformationGradient_new[idx].polarDecompositionRMB(tensorU, tensorR);

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
      double Dlocal[6];
	  Dlocal[0]=D(0,0);
	  Dlocal[1]=D(1,1);
	  Dlocal[2]=D(2,2);
	  Dlocal[3]=D(0,1);
	  Dlocal[4]=D(1,2);
	  Dlocal[5]=D(2,0);
      double svarg[d_NINSV];
      double USM=9e99;
      double dt = delT;
      int nblk = 1;

      // Load ISVs into a 1D array for fortran code
      for(int i=0;i<d_NINSV;i++){
        svarg[i]=ISVs[i][idx];
      }

	  // Undrained increase linearly with depth
	  double n = svarg[38];

	  for (int i = 0; i < n; i++) {
		  svarg[37] = px[idx](1);
		  n = n - 1;
	  }
	  svarg[38] = n;

	  // Compute Ko
	  double s_xx = sigarg[0];
	  double s_yy = sigarg[1];
	  double s_xy = sigarg[3];

	  svarg[39] = s_xx;
	  svarg[40] = s_yy;
	  svarg[45] = s_xy;
	  double Ko = s_xx / s_yy;
	  svarg[41] = Ko;

	  // SHear strain and shear strain rate
	  double shear_strain_local = 0;
	  double strain11 = svarg[28];
	  double strain22 = svarg[29];
	  double strain33 = svarg[30];
	  double strain12 = svarg[31];
	  double strain23 = svarg[32];
	  double strain13 = svarg[33];

	  double e11 = Dlocal[0];
	  double e22 = Dlocal[1];
	  double e33 = Dlocal[2];
	  double e12 = Dlocal[3];
	  double e23 = Dlocal[4];
	  double e13 = Dlocal[5];
	  double shear_strain_rate = UI[20];

	  double Use_regular = UI[42];
	  double tFE = UI[43];
	  double tShear = UI[44];

	  strain11 += e11 * dt;
	  strain22 += e22 * dt;
	  strain33 += e33 * dt;
	  strain12 += e12 * dt;
	  strain23 += e23 * dt;
	  strain13 += e13 * dt;

	  shear_strain_local = 1.0 / 2.0*sqrt(2 * (pow((strain11 - strain22), 2.0) + pow((strain11 - strain33), 2.0) + pow((strain22 - strain33), 2.0)) + 3.0*(pow(strain12, 2.0) + pow(strain13, 2.0) + pow(strain23, 2.0)));  

	  shear_strain_rate = 1.0 / 2.0*sqrt(2 * (pow((e11 - e22), 2) + pow((e11 - e33), 2) + pow((e22 - e33), 2)) + 3 * (pow(e12, 2) + pow(e13, 2) + pow(e23, 2)));  

	  svarg[28] = strain11;
	  svarg[29] = strain22;
	  svarg[30] = strain33;
	  svarg[31] = strain12;
	  svarg[32] = strain23;
	  svarg[33] = strain13;
	
	  /*
// Non local 
  double n_nonlocal = UI[46];
  double l_nonlocal = UI[47];
  
  double domain_nonlocal = l_nonlocal * l_nonlocal;
  double Up = 0;
  double Up1 = 0;
  double Down = 0;
  //double Uptest = 0;

  double weight = 0;
  double rx = 0;
  double ry = 0;
  double rz = 0;
  double r2 = 0;

  for (ParticleSubset::iterator iter1 = pset->begin();
	  iter1 != pset->end(); iter1++) {
	  particleIndex idx1 = *iter1;

	  rx = pxnew[idx1].x() - pxnew[idx].x();
	  ry = pxnew[idx1].y() - pxnew[idx].y();
	  rz = pxnew[idx1].z() - pxnew[idx].z();

	  r2 = rx * rx + ry * ry + rz * rz;

	  if (r2 <= (9*domain_nonlocal)) {

		  weight = sqrt(r2)*exp(-r2/domain_nonlocal)/domain_nonlocal;
	  }
	  Up += shear_strain_local * weight*pvolume_new[idx1];
	  Up1 += shear_strain_rate * weight*pvolume_new[idx1];
	  Down += (weight*pvolume_new[idx1]);
  }

  double shear_strain_nonlocal = 0;
  double shear_strain_rate_nonlocal = 0;
  shear_strain_nonlocal = (1 - n_nonlocal)*shear_strain_local + (n_nonlocal*Up/Down);
  shear_strain_rate_nonlocal = (1 - n_nonlocal)*shear_strain_rate + (n_nonlocal*Up1 / Down);

  
  */

  double elastic_strain = svarg[2] / svarg[0];
  double shear_strain_nonlocal = shear_strain_local;
  double shear_strain_rate_nonlocal = shear_strain_rate;

  if (Use_regular > 0)
  {
	  if (shear_strain_nonlocal > elastic_strain) {
		  shear_strain_nonlocal = shear_strain_nonlocal * tFE / tShear;
		  shear_strain_rate_nonlocal = shear_strain_rate_nonlocal * tFE / tShear;
	  }
  }

  svarg[24] = shear_strain_local;
  svarg[20] = shear_strain_rate;

// Calling the external model here
      CalculateStress (nblk, d_NINSV, dt, UI, sigarg, Dlocal, svarg, USM, shear_strain_nonlocal, shear_strain_rate_nonlocal);

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

      // ROTATE pstress_new: S=R*tensorSig*R^T
      pstress_new[idx] = (tensorR*tensorSig)*(tensorR.Transpose());

	  /*
	// Non local stress
	  double n_nonlocal = UI[46];
	  double l_nonlocal = UI[47];

	  double domain_nonlocal = l_nonlocal * l_nonlocal;
	  double Up0 = 0;
	  double Up1 = 0;
	  double Up2 = 0;
	  double Up3 = 0;
	  double Up4 = 0;
	  double Up5 = 0;
	  double Down = 0;
	  //double Uptest = 0;

	  double weight = 0;
	  double rx = 0;
	  double ry = 0;
	  double rz = 0;
	  double r2 = 0;

	  double s11 = pstress_new[idx](0,0);
	  double s22 = pstress_new[idx](1,1);
	  double s33 = pstress_new[idx](2,2);
	  double s12 = pstress_new[idx](0,1);
	  double s23 = pstress_new[idx](1,2);
	  double s13 = pstress_new[idx](2,0);

	  for (ParticleSubset::iterator iter1 = pset->begin();
		  iter1 != pset->end(); iter1++) {
		  particleIndex idx1 = *iter1;

		  rx = pxnew[idx1].x() - pxnew[idx].x();
		  ry = pxnew[idx1].y() - pxnew[idx].y();
		  rz = pxnew[idx1].z() - pxnew[idx].z();

		  r2 = rx * rx + ry * ry + rz * rz;

		  if (r2 <= (9*domain_nonlocal)) {

			  weight = sqrt(r2)*exp(-r2/domain_nonlocal)/domain_nonlocal;
		  }
		  Up0 += s11 * weight*pvolume_new[idx1];
		  Up1 += s22 * weight*pvolume_new[idx1];
		  Up2 += s33 * weight*pvolume_new[idx1];
		  Up3 += s12 * weight*pvolume_new[idx1];
		  Up4 += s23 * weight*pvolume_new[idx1];
		  Up5 += s13 * weight*pvolume_new[idx1];
		  Down += (weight*pvolume_new[idx1]);

		 //Uptest += idx1 * weight*pvolume_new[idx1];

		 //cerr << weight << ' ' << r2 << ' ' << domain_nonlocal << endl;

	  }

	  double Snonlocal[6];
	  Snonlocal[0] = (1 - n_nonlocal)*s11 + (n_nonlocal*Up0/Down);
	  Snonlocal[1] = (1 - n_nonlocal)*s22 + (n_nonlocal*Up1/Down);
	  Snonlocal[2] = (1 - n_nonlocal)*s33 + (n_nonlocal*Up2/Down);
	  Snonlocal[3] = (1 - n_nonlocal)*s12 + (n_nonlocal*Up3/Down);
	  Snonlocal[4] = (1 - n_nonlocal)*s23 + (n_nonlocal*Up4/Down);
	  Snonlocal[5] = (1 - n_nonlocal)*s13 + (n_nonlocal*Up5/Down);

	  //double test;
      //test = (1 - n_nonlocal) * idx + (n_nonlocal * Uptest / Down);

      //cerr << n_nonlocal << ' ' << Uptest << ' ' << Down << endl;
	  //cerr << test << ' ' << idx <<endl;

	  pstress_new[idx](0,0) = Snonlocal[0];
	  pstress_new[idx](1,2) = Snonlocal[1];
	  pstress_new[idx](2,2) = Snonlocal[2];
	  pstress_new[idx](0,1) = Snonlocal[3];
	  pstress_new[idx](1,0) = Snonlocal[3];
	  pstress_new[idx](1,2) = Snonlocal[4];
	  pstress_new[idx](2,1) = Snonlocal[4];
	  pstress_new[idx](2,0) = Snonlocal[5];
	  pstress_new[idx](0,2) = Snonlocal[5];
	  */

      c_dil = sqrt(USM/rho_cur);

      // Compute the strain energy for all the particles
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
        double c_bulk = sqrt(UI[1]/rho_cur);
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
  }
}

void MohrCoulomb::carryForward(const PatchSubset* patches,
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
	std::vector<constParticleVariable<double> > ISVs(d_NINSV+1);
	std::vector<ParticleVariable<double> > ISVs_new(d_NINSV+1);

    for(int i=0;i<d_NINSV;i++){
      old_dw->get(ISVs[i],ISVLabels[i], pset);
      new_dw->allocateAndPut(ISVs_new[i],ISVLabels_preReloc[i], pset);
      ISVs_new[i].copyData(ISVs[i]);
  }

    // Don't affect the strain energy or timestep size
    new_dw->put(delt_vartype(1.e10), lb->delTLabel, patch->getLevel());

    if (flag->d_reductionVars->accStrainEnergy ||
        flag->d_reductionVars->strainEnergy) {
      new_dw->put(sum_vartype(0.),     lb->StrainEnergyLabel);
    }
  }

}

void MohrCoulomb::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* ) const
{
  // Add the computes and requires that are common to all explicit
  // constitutive models.  The method is defined in the ConstitutiveModel
  // base class.
  const MaterialSubset* matlset = matl->thisMaterial();

  cout << "In add InitialComputesAnd" << endl;

  // Other constitutive model and input dependent computes and requires
  for(int i=0;i<d_NINSV;i++){
    task->computes(ISVLabels[i], matlset);
  }
}

void MohrCoulomb::addComputesAndRequires(Task* task,
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
}

void MohrCoulomb::addComputesAndRequires(Task*,
                                     const MPMMaterial*,
                                     const PatchSet*,
                                     const bool ) const
{
}

double MohrCoulomb::computeRhoMicroCM(double pressure,
                                  const double p_ref,
                                  const MPMMaterial* matl,
                                  double temperature,
                                  double rho_guess)
{
  double rho_orig = matl->getInitialDensity();
  double p_gauge = pressure - p_ref;
  double rho_cur;
  double bulk = UI[1];

  rho_cur = rho_orig/(1-p_gauge/bulk);

  return rho_cur;

#if 1
  cout << "NO VERSION OF computeRhoMicroCM EXISTS YET FOR MohrCoulomb" << endl;
#endif
}

void MohrCoulomb::computePressEOSCM(double rho_cur, double& pressure,
                                double p_ref,
                                double& dp_drho,      double& tmp,
                                const MPMMaterial* matl,
                                double temperature)
{

  double bulk = UI[1];
  double rho_orig = matl->getInitialDensity();

  double p_g = bulk*(1.0 - rho_orig/rho_cur);
  pressure = p_ref + p_g;
  dp_drho  = bulk*rho_orig/(rho_cur*rho_cur);
  tmp = bulk/rho_cur;  // speed of sound squared

#if 1
  cout << "NO VERSION OF computePressEOSCM EXISTS YET FOR MohrCoulomb" << endl;
#endif
}

double MohrCoulomb::getCompressibility()
{
return 1.0/UI[1];

//return 1; //experimental:1 // found to be irrelevant
}

void
MohrCoulomb::getInputParameters(ProblemSpecP& ps)
{
  ps->getWithDefault("G",UI[0],0.0);              // shear modulus
  ps->getWithDefault("K",UI[1],0.0);              // bulk modulus
  ps->getWithDefault("c",UI[2],0.0);              // cohesion
  ps->getWithDefault("Phi",UI[3],0.0);              // Shear angle deg
  ps->getWithDefault("Psi",UI[4],0.0);   // Agle
  ps->getWithDefault("Version",UI[5],0.0);   // Version (1:classic, 3:ShengMohrCoulomb)
  ps->getWithDefault("suction",UI[6],0.0);   // initial suction
  ps->getWithDefault("UseWaterRetention",UI[7],0.0);  // whether water retention curve should be used
  ps->getWithDefault("WR_Param1",UI[8],0.0);  // water retention parameter
  ps->getWithDefault("WR_Param2",UI[9],0.0);  // water retention parameter
  ps->getWithDefault("WR_Param3",UI[10],0.0);  // water retention parameter
  ps->getWithDefault("WR_Param4",UI[11],0.0);  // water retention parameter
  ps->getWithDefault("SpecVol",UI[12],0.0);  // meridional yld prof param (dim. less)
  ps->getWithDefault("PhiB",UI[13],0.0);  //
    ps->getWithDefault("Usetransition",UI[14],0.0); // undrained shear strength transition
  ps->getWithDefault("A1",UI[15],0.0); 	// water influence parameter
  ps->getWithDefault("B1",UI[16],0.0); 	// water influence parameter
  ps->getWithDefault("W",UI[17],0.0); 	// water content
  ps->getWithDefault("beta",UI[18],0.0); 	// strain rate influence parameter
  ps->getWithDefault("strain_ref",UI[19],0.0); 	// strain rate influence parameter
  ps->getWithDefault("shear_strain_rate",UI[20],0.0); 	// strain rate influence parameter
  ps->getWithDefault("Usemodul",UI[21],0.0);
  ps->getWithDefault("m_modul",UI[22],0.0);
  ps->getWithDefault("nuy",UI[23],0.0);
  ps->getWithDefault("shear_strain",UI[24],0.0);

  ps->getWithDefault("Use_linear",UI[25],0.0);
  ps->getWithDefault("a",UI[26],0.0);
  ps->getWithDefault("y_ref",UI[27],0.0);

  ps->getWithDefault("strain11",UI[28],0.0);
  ps->getWithDefault("strain22",UI[29],0.0);
  ps->getWithDefault("strain33",UI[30],0.0);
  ps->getWithDefault("strain12",UI[31],0.0);
  ps->getWithDefault("strain23",UI[32],0.0);
  ps->getWithDefault("strain13",UI[33],0.0);

  ps->getWithDefault("Use_softening",UI[34],0.0);
  ps->getWithDefault("St",UI[35],0.0);
  ps->getWithDefault("strain_95",UI[36],0.0);

  ps->getWithDefault("y",UI[37],0.0);
  ps->getWithDefault("n",UI[38],0.0);

  ps->getWithDefault("s_xx",UI[39],0.0);
  ps->getWithDefault("s_yy",UI[40],0.0);
   ps->getWithDefault("Ko",UI[41],0.0);

    ps->getWithDefault("Use_regular",UI[42],0.0);
    ps->getWithDefault("tFE",UI[43],0.0);
    ps->getWithDefault("tShear",UI[44],0.0);

    ps->getWithDefault("s_xy",UI[45],0.0);

	ps->getWithDefault("n_nonlocalMC", UI[46], 0.0);
	ps->getWithDefault("l_nonlocal", UI[47], 0.0);
}

void
MohrCoulomb::initializeLocalMPMLabels()
{
  vector<string> ISVNames;



  ISVNames.push_back("G");
  ISVNames.push_back("K");
  ISVNames.push_back("c");
  ISVNames.push_back("Phi");
  ISVNames.push_back("Psi");
  ISVNames.push_back("Version");
  ISVNames.push_back("Suction");
  ISVNames.push_back("UseWaterRetention");
  ISVNames.push_back("WR1");
  ISVNames.push_back("WR2");
  ISVNames.push_back("WR3");
  ISVNames.push_back("WR4");
  ISVNames.push_back("SpecificVol");
  ISVNames.push_back("PhiB");
  ISVNames.push_back("Usetransition");
  ISVNames.push_back("A1");
  ISVNames.push_back("B1");
  ISVNames.push_back("W");
  ISVNames.push_back("beta");
  ISVNames.push_back("strain_ref");
  ISVNames.push_back("shear_strain_rate");
  ISVNames.push_back("Usemodul");
  ISVNames.push_back("m_modul");
  ISVNames.push_back("nuy");
  ISVNames.push_back("shear_strain");

  ISVNames.push_back("Use_linear");
  ISVNames.push_back("a");
  ISVNames.push_back("y_ref");

  ISVNames.push_back("strain11");
  ISVNames.push_back("strain22");
  ISVNames.push_back("strain33");
  ISVNames.push_back("strain12");
  ISVNames.push_back("strain23");
  ISVNames.push_back("strain13");

  ISVNames.push_back("Use_softening");
  ISVNames.push_back("St");
  ISVNames.push_back("strain_95");


  ISVNames.push_back("y");
  ISVNames.push_back("n");


  ISVNames.push_back("s_xx");
  ISVNames.push_back("s_yy");
  ISVNames.push_back("Ko");


  ISVNames.push_back("Use_regular");
  ISVNames.push_back("tFE");
  ISVNames.push_back("tShear");
    ISVNames.push_back("s_xy");

	ISVNames.push_back("n_nonlocalMC");
	ISVNames.push_back("l_nonlocal");

	

  for(int i=0;i<d_NINSV;i++){
    ISVLabels.push_back(VarLabel::create(ISVNames[i],
                          ParticleVariable<double>::getTypeDescription()));
    ISVLabels_preReloc.push_back(VarLabel::create(ISVNames[i]+"+",
                          ParticleVariable<double>::getTypeDescription()));
  }
}

//CODE ADDED BY WTS FOR SHENGMOHRCOULOMB BELOW
void MohrCoulomb::CalculateStress (int &nblk, int &ninsv, double &dt,
                                    double UI[], double stress[], double D[],
                                    double svarg[], double &USM, double shear_strain_nonlocal, double shear_strain_rate_nonlocal)


/*
copied from DIAMM, giving the input required
The procedure calculates stress for the Mohr-Coulomb model, with several
additional options. Those include the variation of the flow rule,
several choices of Mohr-Coulomb like yield surfaces and the dependency
on the strain rate. Most of the above-described features are 'work in
progress'

The list of arguments below is from the original DIAMM model. The arguments
are a bit adjusted to the Mohr-Coulomb, though in spirit they remain same.

int &nblk, int &ninsv, double &dt,double UI[], double stress[], double D[],
double svarg[], double &USM
C
C     input arguments
C     ===============
C      NBLK       int                   Number of blocks to be processed
C      NINSV      int                   Number of internal state vars
C      DTARG      dp                    Current time increment
C      UI       dp,ar(nprop)            User inputs
C      D          dp,ar(6)              Strain increment
C
C     input output arguments
C     ======================
C      STRESS   dp,ar(6)                stress
C      SVARG    dp,ar(ninsv)            state variables
C
C     output arguments
C     ================
C      USM      dp                      uniaxial strain modulus
C
C***********************************************************************
C
C      stresss and strains, plastic strain tensors
C          11, 22, 33, 12, 23, 13
C
C***********************************************************************

*/

{

//this is slightly slow as each model used needs to be declared, but on the other hand allows for keeping things clean

ShengMohrCoulomb SMCModel;
ClassicMohrCoulomb CMCModel;

if (nblk!=1) cerr<<"Mohr-Coulomb model may only be used with nblk equal to 1. Results obtained are incorrect."<<endl;

double G=UI[0]; //shear modulus [stress units]
double K=UI[1]; //bulk modulus [stress units]
double c=UI[2]; //cohesion [stress units]

double m_modul=UI[22];
double nuy=UI[23];
double Phi=UI[3]; //friction angle [degrees]
double Psi=UI[4]; //dilation angle [degrees, at the moment unused, assumed Phi]
int Flavour=int(UI[5]);
	double a1=UI[15];
	double b1=UI[16];
	double W=UI[17];
	double beta=UI[18];
	double strain_ref=UI[19];
    
    double Use_softening=UI[34];
    double St=UI[35];
    double strain_95=UI[36];

    double Use_linear=UI[25];
    double a=UI[26];
    double y_ref=UI[27];
    double y=svarg[37];

/*
Flavour
1- classic Mohr - Coulomb,
2 - classic Mohr-Coulomb with tension cut-off (not implemented yet)
3 - ShengMohrCoulomb (rounded MC surface see eq 13 in Sheng D, Sloan SW & Yu HS
Computations Mechanics 26:185-196 (2000)
4 - ShengMohrCoulomb with tension cut-off (not implemented yet)
5 - SloanMohrCoulomb (see Sloan et al... , not implemented yet)
6 - SloanMohrCoulomb with tension cut-off
11- classic Mohr - Coulomb with semi-implicit stress integration
*/

int UseWaterRetention=UI[7];

double Temp;

//cerr<<UI[0]<<UI[1]<<' '<<UI[2]<<' '<<UI[3]<<' '<<UI[4];
//this 2x3 lines are because the components in the added code are assumed in different sequence.
//probably this exchange is of no importance, but it is added for peace of mind
Temp=stress[4];
stress[4]=stress[5];
stress[5]=Temp;

Temp=D[4];
D[4]=D[5];
D[5]=Temp;

BBMPoint InitialPoint;
double StrainIncrement[6];

for (int i=0; i<6; i++)
{
    InitialPoint.stress[i]=-stress[i];
    StrainIncrement[i]=-D[i]*dt;
}

double Usetransition=UI[14];
if (Usetransition>0)
{

	if(shear_strain_rate_nonlocal >strain_ref){
	    c=St*a1*pow(W,-b1)*pow(shear_strain_rate_nonlocal /strain_ref,beta);
	}
	    else{
	    c=St*a1*pow(W,-b1);
	    }
}

// Shear strength linear with depth
if(Use_linear>0)
{
    c = c + a * (y-y_ref);
}

double Usemodul=UI[21];
if(Usemodul>0)
{

    G=m_modul*c/2.0/(1.0+nuy);
    K=m_modul*c/3.0/(1.0-2*nuy);
}


if(Use_softening>0)
{
    if(shear_strain_nonlocal>c/G)
    {
        c = c * (1.0/St+(1.0-1.0/St)*pow(2.71,(-3.0*shear_strain_nonlocal /strain_95)));
    }
}

if (UseWaterRetention>0)
{
double WTRParam[5];
WTRParam[0]=UI[8];
WTRParam[1]=UI[9];
WTRParam[2]=UI[10];
WTRParam[3]=UI[11];
WTRParam[4]=0.0;
double PhiB=UI[13];

double Suction;
double SpecVol;
Suction=svarg[6];
SpecVol=svarg[12];


    double Sr=GetSr(UseWaterRetention,Suction, WTRParam);   //get from current suction value
    double dVolStrain=StrainIncrement[0]+StrainIncrement[1]+StrainIncrement[2];
    double SpecVolNew=SpecVol-dVolStrain*SpecVol; //1/(1-TotalVolume);  //adjustment due to
    double VolWater=Sr*(SpecVol-1)/SpecVol; //constant volume of water - thus we use initial specific volume
    double VolAir=(SpecVolNew-1)/SpecVolNew-VolWater;  //volume of voids - new one, using new specific volume
    double TotalVolume=VolAir+VolWater;
    double SrNew=VolWater/TotalVolume;       //voids will reduce by dvol, so dSr=VolWater/TotalVolume
    if (SrNew>0.99999) SrNew=0.99999;
    if (SrNew<0.0001) SrNew=0.0001;
    double SuctionNew=GetSuction(UseWaterRetention,SrNew,WTRParam);                      //calculate from specified equation (van Genuchten or Gallipoli or...)
    if (SuctionNew<0) SuctionNew=0.0;
    c=c+tan(PhiB*3.1415/180)*SuctionNew;

    svarg[6]=SuctionNew;
    svarg[12]=SpecVolNew;
}
svarg[0]=G;
svarg[1]=K;
svarg[2]=c;


int Region;

switch (Flavour)
{
case 1:
    {
    CMCModel.SetModelParameters(G,K,c,Phi,Psi);
    CMCModel.SetDefaultIntegrationParameters();
    CMCModel.Integrate (StrainIncrement,&InitialPoint);

    }
    break;

case 11:
    {
    CMCModel.SetModelParameters(G,K,c,Phi,Psi);
    CMCModel.SetDefaultIntegrationParameters();
    CMCModel.IntegrateMCIClassic (StrainIncrement,&InitialPoint,&Region);

    }
    break;

case 12:
    {
    CMCModel.SetModelParameters(G,K,c,Phi,Psi);
    CMCModel.SetDefaultIntegrationParameters();
    CMCModel.IntegrateMCIClassic (StrainIncrement,&InitialPoint,&Region);

    }
    break;


case 3:
    {
    SMCModel.SetModelParameters(G,K,c,Phi);
    SMCModel.SetDefaultIntegrationParameters();
    SMCModel.Integrate (StrainIncrement,&InitialPoint);
    }
    break;

default:
    {
    cerr<<"Error: Mohr Coulomb Model Flavour unspecified or the flavour demanded not implemented yet"<<endl;
    cerr<<"Flavour of the model is set to:"<<Flavour<<endl;
    getchar();
    }

}

//stress back for output
for (int i=0; i<6; i++) stress[i]=-InitialPoint.stress[i];

//this 2x3 lines are because the components in the added code are assumed in different sequence.
//probably this exchange is of no importance, but it is added for the peace of mind

Temp=stress[4];
stress[4]=stress[5];
stress[5]=Temp;

Temp=D[4];
D[4]=D[5];
D[5]=Temp;


double Factor=5.0; //factor is  a quick fix, as otherwise the USM is too low and predicted stable time step is way too high

USM=Factor*(G+0.3*K)/3.0;
//without Factor it
//seems that the USM is too low and the analysis fails. Not sure why
//as the elastic wave should be the quickest and it apparently work in diamm
//maybe I missed something important there
}

void MohrCoulomb::CheckModel(double UI[])
{
    if (UI[0]<=0.0) cerr<<"Shear Modulus in the Mohr-Coulomb Model equal to: "<<UI[0]<<" This will cause the code to malfuncion. Any results obtained are invalid."<<endl;
    if (UI[2]<0.0) cerr<<"Cohesion in the Mohr-Coulomb Model is set to negative value: "<<UI[2]<<" This will cause the code to malfuncion. Any results obtained are invalid."<<endl;
    if (UI[3]<0.0) cerr<<"Friction angle in the Mohr-Coulomb Model is set to: "<<UI[3]<<" This will cause the code to malfuncion. Any results obtained are invalid."<<endl;
    if (UI[4]<0.0) cerr<<"Dilation angle in the Mohr-Coulomb Model is set to: "<<UI[4]<<" This will cause the code to malfuncion. Any results obtained are invalid."<<endl;
    if (UI[5]<1.0) cerr<<"Version of the Mohr-Coulomb Model is set to: "<<UI[5]<<" This will cause the code to malfuncion. Any results obtained are invalid."<<endl;

}

double MohrCoulomb::GetSr (double UseWaterRetention, double Suction, double * WTRParam)
{
    if (UseWaterRetention==1.0)
    {
        //VanGenuchten Model
        double m=WTRParam [0];
        double n=WTRParam [1];
        double alpha=WTRParam [2];
        double Sr=alpha*Suction;
        Sr=pow(Sr,n);
        Sr=1/(1+Sr);
        Sr=pow(Sr,m);
        if (Sr>1.0) Sr=1.0; if (Sr<0) Sr=0.0;
        return Sr;
    }
    else return 1.0;
}


double MohrCoulomb::GetSuction (double UseWaterRetention, double Sr, double * WTRParam)
{
    if (UseWaterRetention==1.0)
    {
        //VanGenuchten Model
        double m=WTRParam [0];
        double n=WTRParam [1];
        double alpha=WTRParam [2];
        double Suction=pow(Sr,1/m);
        Suction= (1-Suction)/Suction;
        Suction=pow(Suction,1/n);
        Suction=Suction/alpha;
        if (Suction<0) Suction=0.0;
        return Suction;
    }
    else return 0;
}
