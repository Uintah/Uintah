/*
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC § 105.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* This is an Abaqus UMAT interface to the portable version of the Tonge Ramesh model */

#include "UMATTR.h"
#include "PortableTongeRamesh.h"
#include "PState.h"

#include <sstream>                                      // std::cerr, std::cout, std::endl
#include <stdexcept>                            // std::runtime_error
#include <iostream>                 // std::cerr, std::endl
#include <string>
#include <cstring>              // strncpy(),
#include <cassert>              // assert()
#include <climits>              // ULONG_MAX
#include <stdio.h>              // printf()

// This function is provided to track the temperature in the 'SCD' histroy variable
// instead of the 'TEMP' history variable. Otherwise it is identical to PTR_umat_stressUpdate

void PTR_umat_stressUpdate_SCD(double STRESS[6], double STATEV[], double DDSDDE[6][6],
                           double *SSE, double *SPD, double *SCD, double *RPL,
                           double DDSDDT[6], double DRPLDE[6], double *DRPLDT,
                           const double STRAN[6], const double DSTRAN[6], const double TIME[2],
                           const double *DTIME, double *TEMP, double *DTEMP,
                           const double *PREDEF, const double *DPRED, const double *CMNAME,
                           const int *NDI, const int *NSHR, const int *NTENS, const int *NSTATV,
                           const double PROPS[PTR_NUM_MAT_PARAMS], const int *NPROPS, const double COORDS[3],
                           const double DROT[3][3], double *PNEWDT, const double *CELENT,
                           const double DFGRD0[3][3], const double DFGRD1[3][3],
                           const int *NOEL, const int *NPT, const int *LAYER,
                           const int *KSPT, const int *KSTEP, const int *KINC) {
  double ptemp = *SCD;
  PTR_umat_stressUpdate( STRESS, STATEV, DDSDDE,
                         SSE, SPD, SCD, RPL,
                         DDSDDT, DRPLDE, DRPLDT,
                         STRAN, DSTRAN, TIME,
                         DTIME, &ptemp, DTEMP,
                         PREDEF, DPRED, CMNAME,
                         NDI, NSHR, NTENS, NSTATV,
                         PROPS, NPROPS, COORDS,
                         DROT, PNEWDT, CELENT,
                         DFGRD0, DFGRD1,
                         NOEL, NPT, LAYER,
                         KSPT,   KSTEP,  KINC);
  *SCD = ptemp;
}

/*
 * Wrapper interface description:
 *
 * This wrapper conforms to the standard Abaqus UMAT interface with the following exceptions:
 *   - CMNAME has type double instead of being a character array
 *   - DDSDDE is the instentanious elastic stiffness, attempts to use DDSDDE to converge to an
 *            equlibrium stress are unlikely to succede and almost certanly will not converge
 *            quadratically.
 *   - The driver assumes a full 3D implimentation, attempts to use 2D approximations
 *     will cause the model to crash (due to an assert() statement). Bypassing the assert()
 *     test to run a 2D problem will cause incorrect memory access issues.
 *
 * The function arguments are:
 *IO  STRESS  - Cauchy stress 11,22,33,12,13,23 
 *IO  STATEV  - Array of internal state variables (PTR_FLAWHIST_OFFSET+3*NumFlawFamilies)
 *O   DDSDDE  - Stiffness matrix - This is the ELASTIC stiffness matrix and
 *              does not include the effects of plasticity or damage, it is
 *              unlikely to converge when used in implicit analysis.
 *IO  SSE     - Strain energy per unit mass
 *O   SPD     - Total dissipation due to granular flow and lattice plasticity
 *U   SCD     - Creep dissipation 
 *O   RPL     - Heating rate due to thermoelasticity and plastic work
 *U   DDSDDT  - derivitive of stress with temperature
 *U   DRPLDE  - derivitive of plastic work with strain
 *U   DRPLDT  - derivitive of heating rate with temperature
 *I   STRAN   - Logrithmic strain 
 *I   DSTRAN  - Increment in strain
 *I   TIME    - Beginning and ending time for timestep (only ending is used)
 *I   DTIME   - Time increment for the timestep
 *IO  TEMP    - Temperature
 *O   DTEMP   - Change in temperature
 *U   PREDEF  - Predefined field variables
 *U   DPRED   - Increment in predefined field variables
 *U   CMNAME* - Constutive model name
 *I   NDI     - Number of direct stresses (must be 3)
 *I   NSHR    - Number of shear stress components (also 3)
 *I   NTENS   - Number of tensor components (6)
 *I   NSTATEV - Number of state variables (PTR_FLAWHIST_OFFSET+3*NumFlawFamilies)
 *I   PROPS   - Array of material properties
 *I   NPROPS  - Number of material properties
 *U   COORDS  - Integration point coordinates
 *I   DROT    - Incremental rotation durring the timestep
 *U   PNEWDT  - Suggusted new timestep
 *I   CELENT  - Average size of the element
 *U   DFGRD0  - Deformation gradient at the start of the step
 *U   DFGRD1  - Deformation gradient at the end of the step
 *I   NOEL    - Element number
 *I   NPT     - integration point number
 *I   LAYER   - Layer number
 *I   KSPT    - Section point number
 *U   KSTEP   - Current step number
 *U   KINC    - Current increment number
 *
 * I: Variable is an input, O: Variable is output only, IO: Variable is input and output
 * U: Variable is unused and included only to satisfy the interface requirements.
 *
 * The variables NOEL NPT LAYER KSPT and the initial value of damage can be used
 * to initalize the local flaw distribution based on sampling. This can help
 * mitigate some mesh bias that is common in models that show localization.
 *
 *  PTR_umat_register:
 *  Provide physically reasonable initial values for the state variables and return the names
 *  of the internal state variables.
 */

void PTR_umat_stressUpdate(double STRESS[6], double STATEV[], double DDSDDE[6][6],
                           double *SSE, double *SPD, double *SCD, double *RPL,
                           double DDSDDT[6], double DRPLDE[6], double *DRPLDT,
                           const double STRAN[6], const double DSTRAN[6], const double TIME[2],
                           const double *DTIME, double *TEMP, double *DTEMP,
                           const double *PREDEF, const double *DPRED, const double *CMNAME,
                           const int *NDI, const int *NSHR, const int *NTENS, const int *NSTATV,
                           const double PROPS[PTR_NUM_MAT_PARAMS], const int *NPROPS, const double COORDS[3],
                           const double DROT[3][3], double *PNEWDT, const double *CELENT,
                           const double DFGRD0[3][3], const double DFGRD1[3][3],
                           const int *NOEL, const int *NPT, const int *LAYER,
                           const int *KSPT, const int *KSTEP, const int *KINC) {
  assert(*NPROPS == PTR_NUM_MAT_PARAMS);
  assert(*NDI    == 3);
  assert(*NSHR   == 3);
  assert(*NTENS  == 6);
  // Unpack material properties:
  PTR::Flags                  flags;
  PTR::ModelData              initialData;
  PTR::flawDistributionData   flawDistData;
  PTR::BrittleDamageData      brittle_damage;
  PTR::granularPlasticityData gpData;
  PTR::ArtificialViscosity    artViscData;
  PTR::CMData                 eosData;
  PTR::unpackMatParams(PROPS, &flags, &initialData, &flawDistData,
                       &brittle_damage, &gpData, &artViscData, &eosData);
  const double delT = *DTIME;                     // Time step size
  const double pTemp = *TEMP;                     // Temperature
  const double rho_orig = initialData.rho_orig;   // initial density
  // Assume logrithmic strain and that the incomming strain is at the beginning of the
  // timestep
  const double J_old    = std::exp(STRAN[0] + STRAN[1] + STRAN[2]); // Volume change ratio at the beginning of the step
  const double J        = J_old * std::exp(DSTRAN[0] + DSTRAN[1] + DSTRAN[2]); // Volume change ratio at the end of the step
  // Unpack history variables:
  Matrix3x3 pStress;                              // Cauchy Stress
  Matrix3x3 pStress_qs;                           // Reference stress for multistage viscoplastic granular flow alg.
  Matrix3x3 velGrad;                              // Velocity gradient - computed from the strain increment
  Matrix3x3 Rinc;                                 // Rotation increment over the timestep
  // The Abaqus UMAT specification states that the incomming stress will be rotated
  // prior to being passed to the constutive model.
  pStress.set(0,0, STRESS[0]);
  pStress.set(1,1, STRESS[1]);
  pStress.set(2,2, STRESS[2]);
  pStress.set(1,2, STRESS[5]);
  pStress.set(2,1, STRESS[5]);
  pStress.set(0,2, STRESS[4]);
  pStress.set(2,0, STRESS[4]);
  pStress.set(0,1, STRESS[3]);
  pStress.set(1,0, STRESS[3]);

  velGrad.set(0,0, DSTRAN[0]);
  velGrad.set(1,1, DSTRAN[1]);
  velGrad.set(2,2, DSTRAN[2]);
  // In UMats the shear strains are stored as engineering shear strain
  velGrad.set(1,2, 0.5*DSTRAN[5]);
  velGrad.set(2,1, 0.5*DSTRAN[5]);
  velGrad.set(0,2, 0.5*DSTRAN[4]);
  velGrad.set(2,0, 0.5*DSTRAN[4]);
  velGrad.set(0,1, 0.5*DSTRAN[3]);
  velGrad.set(1,0, 0.5*DSTRAN[3]);

  if(delT>0.0){
    velGrad /= delT;
  }

  for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
      Rinc.set(i,j,DROT[j][i]); // I/O Marticies use FORTRAN ordering not C.
    }
  }
  
  double IEl = STATEV[0];                         // Indicator of volume preserving deformation 
  double damage = STATEV[1];                      // damage variable 
  double JGP    = STATEV[2];                      // Volume change ratio from granular flow
  double GP_strain = STATEV[3];                   // Magnitude of the accumulated deviatoric component of granular flow
  double GP_energy = STATEV[4];                   // Energy dissipated by granular flow
  double plasticStrain = STATEV[5];               // Accumulated plastic strain for volume preserving plasticity
  double plasticEnergy = STATEV[6];               // Energy Dissipated by volume preserving plasticity
  int localized        = static_cast<int>(std::floor(STATEV[9]+0.5)); // Failure indicator
  double epsVGP        = STATEV[10];              // Volume strain from GFMS alg
  double gamGP         = STATEV[11];              // Shear strain from GFMS alg
  double epsVGP_qs     = STATEV[12];              // Volume strain from GFMS alg for reference state
  double gamGP_qs      = STATEV[13];              // Shear strain from GFMS alg for reference state

  pStress_qs.set(0,0, STATEV[14]);
  pStress_qs.set(1,1, STATEV[15]);
  pStress_qs.set(2,2, STATEV[16]);
  pStress_qs.set(1,2, STATEV[17]);
  pStress_qs.set(2,1, STATEV[17]);
  pStress_qs.set(0,2, STATEV[18]);
  pStress_qs.set(2,0, STATEV[18]);
  pStress_qs.set(0,1, STATEV[19]);
  pStress_qs.set(1,0, STATEV[19]);

  pStress_qs = Rinc * (pStress_qs*Rinc.transpose());

  if( (*NSTATV - PTR_FLAWHIST_OFFSET)%3 ){
    std::stringstream msg;
    msg << "Invalid number of history variables found \n"
        << "this constutive model uses " << PTR_FLAWHIST_OFFSET
        << " plus 3 per flaw family. \n";
    msg << "  File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
    msg << "debugging information:\n"
        << "Reported Number of state Variables (NSTATV): " << *NSTATV << "\n";
    throw std::runtime_error(msg.str()); 
  }
  const int numFlaws = (*NSTATV-PTR_FLAWHIST_OFFSET)/3;
  assert(numFlaws == PROPS[13]);

  std::vector<double> wingLength(numFlaws, 0.0);
  std::vector<double> wingLength_new(numFlaws, 0.0);
  std::vector<double> flawNumber(numFlaws, 0.0);
  std::vector<double> flawSize(numFlaws, 0.0);
  for (int i=0; i<numFlaws; ++i){
    flawNumber[i] = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 0];
    flawSize[i]   = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 1];
    wingLength[i] = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 2];
    wingLength_new[i] = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 2];
  }

  // If the initial damage value has been set to an invalid value
  // then compute the initial flaw distribution and the associated
  // damage level.
  if(std::abs(TIME[1]-delT)<0.5*delT && (STATEV[1]<0)){
    // If damage < 0 extract the random bits from that value (it should be in the range
    // 0 > damage > -1.0)
    double urandomSeed = std::abs(STATEV[1]) - std::floor(std::abs(STATEV[1]));
    unsigned long artViscSeed = static_cast<unsigned long>(static_cast<double>(ULONG_MAX)*urandomSeed);
    unsigned long seedArray[6] = {static_cast<unsigned long>(flawDistData.randomSeed), artViscSeed, static_cast<unsigned long>(*NOEL), static_cast<unsigned long>(*NPT), static_cast<unsigned long>(*LAYER), static_cast<unsigned long>(*KSPT)};
    damage = PTR::initalizeFlawDist(flawSize.data(), flawNumber.data(), flawDistData, *CELENT, seedArray, 6);
    damage = brittle_damage.incInitialDamage ? damage : 0.0;
  }

  // Prep call to Portable TongeRamesh
  PTR::PortableMieGruneisenEOSTemperature eos(eosData);
  double damage_dt(delT);                         // Unused
  double pEnergy = (*SSE)*rho_orig;               // Strain energy per unit original volume
  double dx_ave  = *CELENT;                       // Reference element length
  double p_q     = 0;                             // Artificial viscous pressure
  double pdTdt   = 0.0;                           // Time Rate of change of temperature
  double c_dil   = 0.0;                           // Dilatational wave speed
  
  PTR::advanceTimeSigmaL(
                         flags, initialData, flawDistData, brittle_damage, gpData,
                         artViscData, &eos,
                         velGrad, &pStress, &pStress_qs,
                         delT, J_old, J, pTemp, rho_orig, dx_ave,
                         &IEl, &plasticStrain, &plasticEnergy, &damage,
                         &JGP, &GP_strain, &GP_energy, &pEnergy, &damage_dt,
                         &epsVGP, &gamGP, &epsVGP_qs, &gamGP_qs, &localized,
                         &p_q, &pdTdt, &c_dil,
                         &wingLength, &flawNumber, &flawSize, &wingLength_new,
                         true
                         );
  // Update stress and history variables
  *SSE      = pEnergy/rho_orig;
  *SPD      = (plasticEnergy + GP_energy) / rho_orig;
  *TEMP     = pdTdt*delT + pTemp;   // In a standard UMAT the heating rate should be assigned to RPL
  *DTEMP    =  pdTdt*delT;
  *RPL      = pdTdt/eosData.C_v;
  STRESS[0] = pStress.get(0,0);
  STRESS[1] = pStress.get(1,1);
  STRESS[2] = pStress.get(2,2);
  STRESS[3] = 0.5*(pStress.get(0,1) + pStress.get(1,0));
  STRESS[4] = 0.5*(pStress.get(0,2) + pStress.get(2,0));
  STRESS[5] = 0.5*(pStress.get(2,1) + pStress.get(1,2));

  STATEV[0] = IEl;
  STATEV[1] = damage;
  STATEV[2] = JGP;
  STATEV[3] = GP_strain;
  STATEV[4] = GP_energy;
  STATEV[5] = plasticStrain;
  STATEV[6] = plasticEnergy;
  STATEV[7] = p_q;
  STATEV[8] = pdTdt;
  STATEV[9] = localized;
  STATEV[10] = epsVGP;
  STATEV[11] = gamGP;
  STATEV[12] = epsVGP_qs;
  STATEV[13] = gamGP_qs;

  STATEV[14] = pStress_qs.get(0,0);
  STATEV[15] = pStress_qs.get(1,1);
  STATEV[16] = pStress_qs.get(2,2);
  STATEV[17] = 0.5*(pStress_qs.get(1,2) + pStress_qs.get(2,1));
  STATEV[18] = 0.5*(pStress_qs.get(0,2) + pStress_qs.get(2,0));
  STATEV[19] = 0.5*(pStress_qs.get(1,0) + pStress_qs.get(0,1));

  for (int i=0; i<numFlaws; ++i){
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 0] = flawNumber[i];
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 1] = flawSize[i];
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 2] = wingLength_new[i];
  }

  // Calculate the elastic stiffness and put it in ddsdde:
  // Setup the plasticity state variable:
  double JEL = J/JGP;
  double rho_cur  = rho_orig/JEL;
  // Set up the PlasticityState (for t_n+1) --------------------------------
  PState state;
  state.pressure = -pStress.trace()/3.0;
  state.temperature = pTemp;
  state.initialTemperature = eosData.theta_0;
  state.density = rho_cur;
  state.initialDensity = rho_orig;
  state.initialVolume = 1.0;
  state.volume = state.initialVolume*J;
  state.specificHeat = eosData.C_v;
  state.energy = (state.temperature-state.initialTemperature)*state.specificHeat; // This is the internal energy do
  // to the temperature
  // Set the moduli:
  state.initialBulkModulus = eos.computeIsentropicBulkModulus(rho_orig, rho_cur, pTemp);
  state.initialShearModulus = initialData.tauDev;
  state.bulkModulus  = PTR::calculateBulkPrefactor(damage, state, JEL)*state.initialBulkModulus;
  state.shearModulus = PTR::calculateShearPrefactor(damage, state)*state.initialShearModulus;
  double mu = state.shearModulus;
  double lambda = state.bulkModulus - 2*mu/3.0;
  for (int i=0; i<6; ++i){
    for (int j=0; j<6; ++j){
      DDSDDE[i][j] = 0.0;
    }
  }
  DDSDDE[0][0] = lambda + 2.0*mu;
  DDSDDE[0][1] = lambda;
  DDSDDE[0][2] = lambda;
  DDSDDE[1][0] = lambda;
  DDSDDE[1][1] = lambda + 2.0*mu;
  DDSDDE[1][2] = lambda;
  DDSDDE[2][0] = lambda;
  DDSDDE[2][1] = lambda;
  DDSDDE[2][2] = lambda + 2.0*mu;
  DDSDDE[3][3] = mu;
  DDSDDE[4][4] = mu;
  DDSDDE[5][5] = mu;
}

void PTR_umat_repairAdvect(double STRESS[6], double STATEV[], double DDSDDE[6][6],
                           double *SSE, double *SPD, double *SCD, double *RPL,
                           double DDSDDT[6], double DRPLDE[6], double *DRPLDT,
                           double STRAN[6], double DSTRAN[6], double TIME[2],
                           double *DTIME, double *TEMP, double *DTEMP,
                           double *PREDEF, double *DPRED, double *CMNAME,
                           int *NDI, int *NSHR, int *NTENS, int *NSTATV,
                           double PROPS[], int *NPROPS, double COORDS[3],
                           double DROT[3][3], double *PNEWDT, double *CELENT,
                           double DFGRD0[3][3], double DFGRD1[3][3],
                           int *NOEL, int *NPT, int *LAYER,
                           int *KSPT, int *KSTEP, int *KINC)
{
  assert(*NPROPS == PTR_NUM_MAT_PARAMS);
  assert(*NDI    == 3);
  assert(*NSHR   == 3);
  assert(*NTENS  == 6);
  // Unpack material properties:
  PTR::Flags                  flags;
  PTR::ModelData              initialData;
  PTR::flawDistributionData   flawDistData;
  PTR::BrittleDamageData      brittle_damage;
  PTR::granularPlasticityData gpData;
  PTR::ArtificialViscosity    artViscData;
  PTR::CMData                 eosData;
  PTR::unpackMatParams(PROPS, &flags, &initialData, &flawDistData,
                       &brittle_damage, &gpData, &artViscData, &eosData);
  const double delT = *DTIME;
  const double pTemp = *TEMP;
  const double rho_orig = initialData.rho_orig;
  // Assume logrithmic strain and that the incomming strain is at the beginning of the
  // timestep
  const double J    = std::exp(STRAN[0] + STRAN[1] + STRAN[2]);
  // Unpack history variables:
  Matrix3x3 pStress,pStress_qs;
  // The Abaqus UMAT specification states that the incomming stress will be rotated
  // prior to being passed to the constutive model.
  pStress.set(0,0, STRESS[0]);
  pStress.set(1,1, STRESS[1]);
  pStress.set(2,2, STRESS[2]);
  pStress.set(1,2, STRESS[5]);
  pStress.set(2,1, STRESS[5]);
  pStress.set(0,2, STRESS[4]);
  pStress.set(2,0, STRESS[4]);
  pStress.set(0,1, STRESS[3]);
  pStress.set(1,0, STRESS[3]);

  double IEl = STATEV[0];
  double damage = STATEV[1];
  double JGP    = STATEV[2];
  double GP_strain = STATEV[3];
  double GP_energy = STATEV[4];
  double plasticStrain = STATEV[5];
  double plasticEnergy = STATEV[6];
  int localized        = static_cast<int>(std::floor(STATEV[9]+0.5));
  double epsVGP        = STATEV[10];
  double gamGP         = STATEV[11];
  double epsVGP_qs     = STATEV[12];
  double gamGP_qs      = STATEV[13];

  pStress_qs.set(0,0, STATEV[14]);
  pStress_qs.set(1,1, STATEV[15]);
  pStress_qs.set(2,2, STATEV[16]);
  pStress_qs.set(1,2, STATEV[17]);
  pStress_qs.set(2,1, STATEV[17]);
  pStress_qs.set(0,2, STATEV[18]);
  pStress_qs.set(2,0, STATEV[18]);
  pStress_qs.set(0,1, STATEV[19]);
  pStress_qs.set(1,0, STATEV[19]);

  if( (*NSTATV - PTR_FLAWHIST_OFFSET)%3 ){
    std::stringstream msg;
    msg << "Invalid number of history variables found \n"
        << "this constutive model uses " << PTR_FLAWHIST_OFFSET
        << " plus 3 per flaw family. \n";
    msg << "  File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
    msg << "debugging information:\n"
        << "Reported Number of state Variables (NSTATV): " << *NSTATV << "\n";
    throw std::runtime_error(msg.str()); 
  }
  const int numFlaws = (*NSTATV-PTR_FLAWHIST_OFFSET)/3;
  assert(numFlaws == PROPS[13]);

  std::vector<double> wingLength(numFlaws, 0.0);
  std::vector<double> flawNumber(numFlaws, 0.0);
  std::vector<double> flawSize(numFlaws, 0.0);
  for (int i=0; i<numFlaws; ++i){
    flawNumber[i] = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 0];
    flawSize[i]   = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 1];
    wingLength[i] = STATEV[PTR_FLAWHIST_OFFSET + 3*i + 2];
  }

  // Prep call to Portable TongeRamesh
  PTR::PortableMieGruneisenEOSTemperature eos(eosData);
  double damage_dt(delT);
  double pEnergy = (*SSE)*rho_orig;
  double p_q     = 0;
  double pdTdt   = 0.0;
  
  PTR::postAdvectionFixup(
                      flags, initialData, flawDistData, brittle_damage, gpData,
                      artViscData, &eos,
                      &pStress, &pStress_qs,
                      J, pTemp, rho_orig,
                      &IEl, &plasticStrain, &plasticEnergy, &damage,
                      &JGP, &GP_strain, &GP_energy, &pEnergy, &damage_dt,
                      &epsVGP, &gamGP, &epsVGP_qs, &gamGP_qs, &localized,
                      &wingLength, &flawNumber, &flawSize
                      );
  // Update stress and history variables
  *SSE      = pEnergy/rho_orig;
  *SPD      = (plasticEnergy + GP_energy) / rho_orig;
  // Leave TEMP, DTEMP, and RPL
  STRESS[0] = pStress.get(0,0);
  STRESS[1] = pStress.get(1,1);
  STRESS[2] = pStress.get(2,2);
  STRESS[3] = 0.5*(pStress.get(0,1) + pStress.get(1,0));
  STRESS[4] = 0.5*(pStress.get(0,2) + pStress.get(2,0));
  STRESS[5] = 0.5*(pStress.get(2,1) + pStress.get(1,2));

  STATEV[0] = IEl;
  STATEV[1] = damage;
  STATEV[2] = JGP;
  STATEV[3] = GP_strain;
  STATEV[4] = GP_energy;
  STATEV[5] = plasticStrain;
  STATEV[6] = plasticEnergy;
  STATEV[7] = p_q;
  STATEV[8] = pdTdt;
  STATEV[9] = localized;
  STATEV[10] = epsVGP;
  STATEV[11] = gamGP;
  STATEV[12] = epsVGP_qs;
  STATEV[13] = gamGP_qs;

  STATEV[14] = pStress_qs.get(0,0);
  STATEV[15] = pStress_qs.get(1,1);
  STATEV[16] = pStress_qs.get(2,2);
  STATEV[17] = 0.5*(pStress_qs.get(1,2) + pStress_qs.get(2,1));
  STATEV[18] = 0.5*(pStress_qs.get(0,2) + pStress_qs.get(2,0));
  STATEV[19] = 0.5*(pStress_qs.get(1,0) + pStress_qs.get(0,1));

  for (int i=0; i<numFlaws; ++i){
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 0] = flawNumber[i];
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 1] = flawSize[i];
    STATEV[PTR_FLAWHIST_OFFSET + 3*i + 2] = wingLength[i];
  }

  // Calculate the elastic stiffness and put it in ddsdde:
  // Setup the plasticity state variable:
  double JEL = J/JGP;
  double rho_cur  = rho_orig/JEL;
  // Set up the PlasticityState (for t_n+1) --------------------------------
  PState state;
  state.pressure = -pStress.trace()/3.0;
  state.temperature = pTemp;
  state.initialTemperature = eosData.theta_0;
  state.density = rho_cur;
  state.initialDensity = rho_orig;
  state.initialVolume = 1.0;
  state.volume = state.initialVolume*J;
  state.specificHeat = eosData.C_v;
  state.energy = (state.temperature-state.initialTemperature)*state.specificHeat; // This is the internal energy do
  // to the temperature
  // Set the moduli:
  state.initialBulkModulus = eos.computeIsentropicBulkModulus(rho_orig, rho_cur, pTemp);
  state.initialShearModulus = initialData.tauDev;
  state.bulkModulus  = PTR::calculateBulkPrefactor(damage, state, JEL)*state.initialBulkModulus;
  state.shearModulus = PTR::calculateShearPrefactor(damage, state)*state.initialShearModulus;
  double mu = state.shearModulus;
  double lambda = state.bulkModulus - 2*mu/3.0;
  for (int i=0; i<6; ++i){
    for (int j=0; j<6; ++j){
      DDSDDE[i][j] = 0.0;
    }
  }
  DDSDDE[0][0] = lambda + 2.0*mu;
  DDSDDE[0][1] = lambda;
  DDSDDE[0][2] = lambda;
  DDSDDE[1][0] = lambda;
  DDSDDE[1][1] = lambda + 2.0*mu;
  DDSDDE[1][2] = lambda;
  DDSDDE[2][0] = lambda;
  DDSDDE[2][1] = lambda;
  DDSDDE[2][2] = lambda + 2.0*mu;
  DDSDDE[3][3] = mu;
  DDSDDE[4][4] = mu;
  DDSDDE[5][5] = mu;
}


// assign name or initial value
void PTR_umat_register(const int nHistVar, const char *tag,
                       char **characterData, int *integerData,
                       double *histVar) {
  if (strcmp(tag, "varname")==0) {
    PTR_umat_getStateVarNameArray(nHistVar, characterData);
  }
  else if (strcmp(tag, "advection")==0) {
    // Accept the default volume weighted advection method
    // continue;
  }
  else if (strcmp(tag, "initialvalue")==0) {
    PTR_umat_getInitialValuesNoProps(nHistVar, histVar);
  }
  else if (strcmp(tag, "alias")==0) {
  }
}

void PTR_umat_getInitialValuesNoProps(const int nHistVar, double histVar[]){
  for (int i=0; i<nHistVar; ++i){
    switch (i){
    case 0:
      histVar[0] = 1.0;
      break;
    case 1:
      // Assign -1 to the initial damage value to force flaw distribution
      // calculation during the first timestep. Assigning the local flaw
      // distribution requires knowledge of the material parameters.
      histVar[1] = -1.0;
      break;
    case 2:
      histVar[2] = 1.0;
      break;
    default:
      histVar[i] = 0.0;
      break;
    }
  }
}

void PTR_umat_getInitialValues(const int nHistVar, double histVar[], const int nProps,
                               const double props[], const double dx_ave, const int numSeedVals,
                               const unsigned long seedArray[]){
  for (int i=0; i<PTR_FLAWHIST_OFFSET; ++i){
    switch (i){
    case 0:                   // Iel
      histVar[0] = 1.0;
      break;
    case 2:                   // JGP
      histVar[2] = 1.0;
      break;
    default:
      histVar[i] = 0.0;
      break;
    }
  }
  // Now deal with the flaw distribution:
  // Unpack material properties:
  PTR::Flags                  flags;
  PTR::ModelData              initialData;
  PTR::flawDistributionData   flawDistData;
  PTR::BrittleDamageData      brittle_damage;
  PTR::granularPlasticityData gpData;
  PTR::ArtificialViscosity    artViscData;
  PTR::CMData                 eosData;
  PTR::unpackMatParams(props, &flags, &initialData, &flawDistData,
                       &brittle_damage, &gpData, &artViscData, &eosData);
  int expNumHistParam = PTR_umat_getNumStateVars(nProps, props);
  if( !( expNumHistParam == nHistVar) ){
    std::stringstream msg;
    msg << "Incorrect number of history variables detected in PTR_umat_getInitialValues()\n"
        << "\tFile: " << __FILE__ << ", Line: " << __LINE__ << "\n"
        << "\tYou MUST reserve space for " << PTR_FLAWHIST_OFFSET << "+"
        << "(" << PTR_NUM_FLAW_HIST_PARAM << "*" << "the number of flaw families)"
        << "history variables\n"
        << "With the current set of material properties that number is: "
        << expNumHistParam << "\n";
    throw std::runtime_error(msg.str());
  }
  std::vector<double> flawSize(flawDistData.numCrackFamilies);
  std::vector<double> flawNumber(flawDistData.numCrackFamilies);
  std::vector<unsigned long> seedArray2(numSeedVals+1);
  for(int i=0; i<numSeedVals; ++i){
    seedArray2[i] = seedArray[i];
  }
  seedArray2[numSeedVals] = flawDistData.randomSeed;
  double initialDamage = PTR::initalizeFlawDist(flawSize.data(), flawNumber.data(),
                                                flawDistData, dx_ave, seedArray2.data(), numSeedVals+1);
  if(brittle_damage.incInitialDamage){
    histVar[1] = initialDamage;
  }
  // Check here to make sure that nHistVar == PTR_FLAWHIST_OFFSET+3*flawSize.size()
  for (unsigned int i=0; i<flawSize.size(); ++i){
    histVar[PTR_FLAWHIST_OFFSET+3*i+0] = flawNumber[i];
    histVar[PTR_FLAWHIST_OFFSET+3*i+1] = flawSize[i];
    histVar[PTR_FLAWHIST_OFFSET+3*i+2] = 0.0;
  }
}

void PTR_umat_getStateVarName(const int varNum, char histVarName[80]){
  std::string ISVname = PTR::getHistVarName(varNum);
  strncpy(histVarName,ISVname.c_str(),80);
}

void PTR_umat_getStateVarNameArray(const int nHistVar, char **characterData){
  for (int i=0; i<nHistVar;++i) {
    PTR_umat_getStateVarName(i, characterData[i]);
  }
}

int PTR_umat_getNumStateVars(const int nProps, const double props[]){
  return PTR_FLAWHIST_OFFSET+3*props[PTR_NUM_FLAW_BIN_IDX];
}


