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

/*
 * This is an Abaqus UMAT interface to the portable version of the Tonge-Ramesh
 * micromechanics based failure model for brittle materials. If you are interested
 * in this model please contact the developer (Andrew Tonge andrew.l.tonge.civ@mail.mil)
 * for copies of the relivent paper and user manuals. If there is an issue with the
 * physics in the model you may contact the developer and he will try to assist
 * particularly with determining if the model is in error or if the host code is
 * driving the model to a place that is physically unreasonable.
 *
 */

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
 * PTR_umat_stressUpdate:
 *  Advance stress and history variables to the end of the timestep with a strain increment
 *  DSSTRAN
 *  
 * PTR_umat_stressUpdate_SCD:
 *  Calls PTR_umat_stressUpdate but tracks the temperature in the variable SCD.
 *
 * PTR_umat_repairAdvect:
 *  Try to repair the material state after an advection step in either ALE or Eularian codes.
 *  Damage and pressure are recomputed, non-negative history variables that are less than 0
 *  are reset to 0, and the reference stress for the single surface visco-plastic granular
 *  flow model is set to the projection of the stress on to the yield surface.
 *  
 * PTR_umat_register:
 *  Provide initial values and a descriptive name for each of the state variables.
 *  Inputs:
 *    n: Number of state variables
 *    tag: one of:
 *       - "varname" - return the variable names
 *       - "advection" - assign advection procedure (does nothing)
 *       - "initialvalue" - assign the initial value of the history variables
 *  Outputs:
 *    characterData: A buffer to place the history variable names in. The buffer
 *       must hold 80 characters per variable name. Currently there are no variable
 *       names that require all of this space.
 *    doubleData: Array for placing initial values of internal state variables.
 *  Unused:
 *    integerData
 *
 * PTR_umat_getInitialValues
 *  Prefered method to initalize the history variables. Called for each element/integration
 *  point.
 *  Inputs:
 *   nHistVar - Number of history variables
 *   nProps - Number of material properties
 *   props[nProps] - array of material properties
 *   dx_ave - Element edge length, used to compute the element volume
 *   numSeedVals - length of the seedArray
 *   seedArray - Unique array of numbers used for randomizing the initial flaw distribution.
 *  Output:
 *   histVars[nHistVar] - Array of history variables
 *
 * PTR_umat_getInitialValuesNoProps   
 *  Place default initial values in the array histVar. These values will be overwritten
 *  during the first timestep. If possible, it is better to use PTR_umat_getInitialValues.
 *  Input:
 *   nHistVar - Number of history variables
 *  Output:
 *   histVar[nHistVar] - Array of history variables initalized to default values
 *
 * PTR_umat_getStateVarName:
 *  Copy the name of state variable varNum into the character array histVarName
 *  Input:
 *   varNum - The variable number of interest
 *  Ouput:
 *   histVarName - name of the history variable
 *
 * PTR_umat_getStateVarNameArray:
 *  Given the number of state variables (nHistVar) sequentially copy them into
 *  a the characterData array. characterData has the shape [nHistVar][80]
 *
 * PTR_umat_getNumStateVars:
 *  Compute the number of state variables that are carried in the STATV array.
 *  This depends on the number of flaw bins that the user specifies.
 *  
 */

#ifndef PTR_UMAT_H
#define PTR_UMAT_H

extern "C" {
  void PTR_umat_stressUpdate(double STRESS[6], double STATEV[], double DDSDDE[6][6],
                             double *SSE, double *SPD, double *SCD, double *RPL,
                             double DDSDDT[6], double DRPLDE[6], double *DRPLDT,
                             const double STRAN[6], const double DSTRAN[6], const double TIME[2],
                             const double *DTIME, double *TEMP, double *DTEMP,
                             const double *PREDEF, const double *DPRED, const double *CMNAME,
                             const int *NDI, const int *NSHR, const int *NTENS, const int *NSTATV,
                             const double PROPS[], const int *NPROPS, const double COORDS[3],
                             const double DROT[3][3], double *PNEWDT, const double *CELENT,
                             const double DFGRD0[3][3], const double DFGRD1[3][3],
                             const int *NOEL, const int *NPT, const int *LAYER,
                             const int *KSPT, const int *KSTEP, const int *KINC);

  void PTR_umat_stressUpdate_SCD(double STRESS[6], double STATEV[], double DDSDDE[6][6],
                                   double *SSE, double *SPD, double *SCD, double *RPL,
                                   double DDSDDT[6], double DRPLDE[6], double *DRPLDT,
                                   const double STRAN[6], const double DSTRAN[6], const double TIME[2],
                                   const double *DTIME, double *TEMP, double *DTEMP,
                                   const double *PREDEF, const double *DPRED, const double *CMNAME,
                                   const int *NDI, const int *NSHR, const int *NTENS, const int *NSTATV,
                                   const double PROPS[], const int *NPROPS, const double COORDS[3],
                                   const double DROT[3][3], double *PNEWDT, const double *CELENT,
                                   const double DFGRD0[3][3], const double DFGRD1[3][3],
                                   const int *NOEL, const int *NPT, const int *LAYER,
                                   const int *KSPT, const int *KSTEP, const int *KINC);
  
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
                             int *KSPT, int *KSTEP, int *KINC) ;

  void PTR_umat_register(const int n, const char *tag,
                  char **characterData, int *integerData,
                  double *doubleData) ;

  void PTR_umat_getInitialValues(const int nHistVar, double histVars[], const int nProps,
                                 const double props[], const double dx_ave,
                                 const int numSeedVals,
                                 const unsigned long seedArray[]);

  void PTR_umat_getInitialValuesNoProps(const int nHistVar, double histVar[]);

  void PTR_umat_getStateVarName(const int varNum, char histVarName[80]);
  
  void PTR_umat_getStateVarNameArray(const int nHistVar, char **characterData);

  int PTR_umat_getNumStateVars(const int nProps, const double props[]);
  
} /* end extern "C" */
#endif
