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

#ifndef PTR_DEFS_H
#define PTR_DEFS_H

#define PTR_NUM_MAT_PARAMS 76
#define PTR_NUM_SCALAR_ISV 14
#define PTR_NUM_TENS_ISV 1
#define PTR_FLAWHIST_OFFSET (PTR_NUM_SCALAR_ISV + 6*PTR_NUM_TENS_ISV)
#define PTR_NUM_FLAW_DIST_PARAM 12
#define PTR_NUM_FLAW_HIST_PARAM 3
#define PTR_BULKMOD_IDX 6
#define PTR_SHEARMOD_IDX 7
#define PTR_DESITY_IDX 8
#define PTR_LOCALIZED_IDX 9
#define PTR_NUM_FLAW_BIN_IDX 13
const char * const PTR_MatParamNames[PTR_NUM_MAT_PARAMS] = {
  "useDamage",
  "usePlasticity",
  "useGranularPlasticity",
  "useOldStress",
  "artificialViscosity",
  "artificialViscousHeating",
  "BulkModulus",
  "ShearModulus",
  "rho_orig",
  "FlowStress",
  "HardeningModulus",
  "InitialPlasticStrain",
  "J2RelaxationTime",
  "NumCrackFamilies",
  "MeanFlawSize",
  "FlawDensity",
  "StdFlawSize",
  "FlawDistType",
  "MinFlawSize",
  "MaxFlawSize",
  "FlawDistExponent",
  "RandomizeFlawDist",
  "RandomSeed",
  "RandomizeMethod",
  "BinBias",
  "KIc",
  "FlawFriction",
  "FlawAngle",
  "FlawGrowthExponent",
  "FlawGrowthAlpha",
  "CriticalDamage",
  "MaxDamage",
  "MicroMechPlaneStrain",
  "MaxDamageInc",               /* Remove? */
  "UseDamageTimestep",          /* Remove */
  "dt_increaseFactor",          /* Remove */
  "IncInitDamage",
  "DoFlawInteraction",
  "GPTimeConst",
  "JLoc",
  "GPGranularSlope",
  "GPCohesion",
  "GPYieldSurfType",            /* Merge with 46? */
  "GPPc",
  "GPJref",
  "GPPref",
  "GPSurfaceType",              /* 0-Two surface, 1- Single surface */
  "AbsToll",
  "RelToll",
  "MaxIter",
  "MaxLevels",
  "GFMSm0",
  "GFMSm1",
  "GFMSm2",
  "GFMSp0",
  "GFMSp1",
  "GFMSp2",
  "GFMSp3",
  "GFMSp4",
  "GFMSa1",
  "GFMSa2",
  "GFMSa3",
  "GFMSBeta",
  "GFMSPsi",
  "GFMSJ3Type",
  "ArtVisc1",
  "ArtVisc2",
  "MGC0",
  "MGGamma0",
  "MGS1",
  "MGS2",
  "MGS3",
  "MGCv",
  "MGTheta_0",
  "JMin",                       /* Remove */
  "MGNPts"                      /* Remove */
  };

const char * const PTR_FlawDistParamNames[PTR_NUM_FLAW_DIST_PARAM] = {
  "NumCrackFamilies",
  "MeanFlawSize",
  "FlawDensity",
  "StdFlawSize",
  "FlawDistType",
  "MinFlawSize",
  "MaxFlawSize",
  "FlawDistExponent",
  "RandomizeFlawDist",
  "RandomSeed",
  "RandomizeMethod",
  "BinBias"
};

/* The vector of history variables consists of two parts, a fixed length portion
   representing the variable names in PTR_StateVarNames followed by a variable
   length portion that contains the flaw distribution and wing crack growth
   information. This variable length portion has the ordering shown in
   PTR_FlawDistStateBaseNames*/
const char * const PTR_StateVarNames[PTR_FLAWHIST_OFFSET] = {
  "Iel",
  "damage",
  "JGP",
  "GP_strain",
  "GP_energy",
  "plasticStrain",
  "plasticEnergy",
  "artViscPres",
  "volHeating",
  "localized",
  "epsVGP",
  "gamGP",
  "epsVGP_qs",
  "gamGP_qs",
  "sig_qs_11",
  "sig_qs_22",
  "sig_qs_33",
  "sig_qs_23",
  "sig_qs_13",
  "sig_qs_12"
};

const char * const PTR_FlawDistStateBaseNames[PTR_NUM_FLAW_HIST_PARAM] = {
  "flawNumber",
  "starterFlawSize",
  "wingLength"
};

#endif // end ifdef PTR_DEFS_H
