/*
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC § 105.
 *
 * However, because the project utilizes code licensed from contributors and other
 * third parties, it therefore is licensed under the MIT License.
 * http://opensource.org/licenses/mit-license.php.
 *
 * Under that license, permission is granted free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the conditions that any
 * appropriate copyright notices and this permission notice are
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef TONGERAMESHPORTABLE_H
#define TONGERAMESHPORTABLE_H

#include "PortableMieGruneisenEOSTemperature.h"
#include "Vector3.h"
#include "Matrix3x3.h"
#include "PState.h"
#include <cmath>
#include <vector>               // std::vector<double>

#include "GFMS_full.h"          // GFMS::solParam GFMS::matParam
#include "PTR_defs.h"
namespace PTR                   // Portable TongeRamesh
{

  // Constants
  const double onethird = (1.0/3.0);
  const double sqtwthds = sqrt(2.0/3.0);

  struct Flags{
	bool implicit;
	bool useDamage;
	bool usePlasticity;
	bool useGranularPlasticity;
	bool with_color;
	bool useOldStress;
	bool artificialViscosity;
	bool artificialViscosityHeating;
	
	// Erosion Algorithms
	bool doErosion;
	bool allowNoTension;
	bool allowNoShear;
	
	bool setStressToZero;
  };

  struct ArtificialViscosity{
	double coeff1;
	double coeff2;
  };

  // Create a datatype for the flaw distribution:
  struct flawDistributionData {
	int numCrackFamilies;     // Number of families to discritize the distribution into
	double meanFlawSize;      // Mean size of the flaws
	double flawDensity;       // Mean flaw density in the sample
	double stdFlawSize;       // Standard deviation of the flaw size
	std::string type;              // Type of the distribution (delta, normal, pareto)
	double minFlawSize;       // Minimum flaw for Pareto dist
	double maxFlawSize;       // Maximum flaw for Pateto dist
	double exponent;          // Exponent for Pateto dist
	bool   randomizeDist;     // Make each particle have a unique distribution
	int    randomSeed;        // Seed for random number generation
	int    randomMethod;      // Method for selecting bin size and location
	double binBias;           // Exponent for flaw distribution bin bias (1.0 for no bias)
	bool   useEtaField;       // Flag for using a fourier field to define the local flaw density
	std::string etaFilename;       // File name containing fourier data for flaw density
	bool   useSizeField;      // Flag for using a fourier field to define the local flaw size shift
	std::string sizeFilename;       // File name containing fourier data for flaw size
  };

  //Create datatype for brittle damage
  struct BrittleDamageData {
	bool printDamage;    /* Flag to print damage */

	// used for Bhasker's damage model:
	double KIc;               // Critical stress intensity factor
	double mu;                // Crack face friction coefficient
	double phi;               // angle between crack normal and max comp stress
	double cgamma;            // Exponent for crack growth speed
	double alpha;             // Multiplier for max crack velocity
	double criticalDamage;    // Damage level to start granular flow or mark as failed
	double maxDamage;         // Damage level to stop damage evolution
	// Use plane strain assumption for SCM calculation
	bool usePlaneStrain;

	// Control damage evolution timestepping:
	double maxDamageInc; /* Maximum damage increment in a time step */
	bool   useDamageTimeStep; // Control the global timestep with the damage timestep
	bool   useOldStress;      // Compute the damage based on the stress from the previous timestep
	double dt_increaseFactor;
	bool   incInitialDamage; // Include the initial flaw as a part of the damage level
	bool   doFlawInteraction; // do the ellipse calculation for flaw interactions
  };

  typedef enum GPModel {TwoSurface, SingleSurface} GPModel;
  struct granularPlasticityData {
	double timeConstant;      // Time constant for viscoplastic update (0 is rate independent)
	double JGP_loc;           // Value of JGP to trigure localized particles

	// Parameters that define the Granular plastic yeild surface:
	double A;                 // Damaged scale parameter
	double B;                 // Damaged hydrostatic tensile strength
	int    yeildSurfaceType;  // 1 for cone with hemispherical cap, 2 for parabola

	// P-alpha compaction model parameters:
	double Pc;                // Pressure (+ compression) for full compaction
	double alpha_e;           // Distension corrisponding to elastic compaction pressure
	double Pe;                // Pressure required to start compaction at J^{GP}=\alpha_e
    GPModel GPModelType;        // TwoSurface for the original implimentation, SingleSurface for the
    // new and imporved model.
    GFMS::solParam GFMSsolParams;
    GFMS::matParam GFMSmatParams;
  };

  // Create datatype for storing model parameters
  struct ModelData {
    double Bulk;
    double tauDev;
    double rho_orig;
    // For Plasticity
    double FlowStress;
    double K;
    double Alpha;
    double timeConstant;
  };

  void computeIncStress(	const BrittleDamageData brittle_damage,
                            const double eta3d,
                            const double matrixStress[2],
                            double incStress[3],
                            const double wingDamage,
                            const double parentDamage,
                            const PState state);

  double calculateDamageGrowth(	const BrittleDamageData brittle_damage,
                                const Matrix3x3 stress,
                                const double N[],
                                const double s[],
                                const double old_L[],
                                const double currentDamage,
                                double new_L[],
                                double new_Ldot[],
                                const double dt,
                                const int Localized,
                                const PState state,
                                const int nBins
                                );

  void calcGranularFlow( const Flags flags, // input flags
                         const BrittleDamageData brittle_damage, // Damage flags
                         const granularPlasticityData gpData,
                         const PortableMieGruneisenEOSTemperature *eos, // passed through to computePressure
                         const double delT,
                         const double pDamage_new,
                         const double J,
                         const double pGPJ_old,
                         Matrix3x3 *bElBar_new, // Input and output (deviatoric elastic strain)
                         PState *state,
                         double *pGPJ,
                         double *pGP_strain,
                         double *pGP_energy,
                         double *pdTdt
                         );
  
  double computePressure(const PortableMieGruneisenEOSTemperature *eos, 
                         const Matrix3x3 F, 
                         const PState state, 
                         const double currentDamage
                         );

  double calculateBulkPrefactor( const double currentDamage, 
                                 const PState state,
                                 const double J = 1.0
                                 );

  double calculateShearPrefactor(	const double currentDamage, 
                                    const PState state
                                    );

  double calc_yeildFunc_g_gs_gp(	const granularPlasticityData gpData,
                                    const double sigma_s,
                                    const double sigma_p,
                                    double *gs, 
                                    double *gp);

  double artificialBulkViscosity(	const double Dkk, 
                                    const double c_bulk, 
                                    const double rho,
                                    const double dx,
                                    const ArtificialViscosity av
                                    );

  double computeStableTimestep(	const ModelData initialData,
                                const Vector3 pVelocity,
                                const Vector3 dx,
                                const double pMass,
                                const double pVolume);
								
  void ComputeStressTensorInnerLoop(
                                    // Data Structures
                                    const Flags flags,
                                    const ModelData initialData,
                                    const flawDistributionData flawDistData,
                                    const BrittleDamageData brittle_damage,
                                    const granularPlasticityData gpData,
                                    const ArtificialViscosity artificialViscosity,
                                    const PortableMieGruneisenEOSTemperature *eos,
                
                                    // Input Matrices
                                    const Matrix3x3 pDefGrad,
                                    const Matrix3x3 pDefGrad_new,
                                    const Matrix3x3 pVelGrad,

                                    // Output Matrix
                                    Matrix3x3 *pDeformRate,
                                    Matrix3x3 *bElBar,
                                    Matrix3x3 *pStress,
                                    Matrix3x3 *pStress_qs,

                                    // Input Vector3
                                    const Vector3 pVelocity,
                                    const Vector3 dx,

                                    // Output Vector3
                                    Vector3 *WaveSpeed,

                                    // Input double
                                    const double pGPJ_old,
                                    const double RoomTemperature,
                                    const double pTemperature,
                                    const double rho_orig,
                                    const double pVolume_new,
                                    const double pMass,
                                    const double SpecificHeat,
                                    const double pDamage,
                                    const double K,
                                    const double flow,
                                    const double delT,
                
                                    // Output double
                                    double *pGP_strain,
                                    double *pPlasticStrain,
                                    double *pPlasticEnergy,
                                    double *pDamage_new,
                                    double *pGPJ,
                                    double *pGP_energy,
                                    double *pEnergy_new,
                                    double *damage_dt,
                                    double *p_q,
                                    double *se,
                                    double *pdTdt,
                                    double *pepsV,
                                    double *pgam,
                                    double *pepsV_qs,
                                    double *pgam_qs,

                                    // Input int
                                    const int pLocalized,
                                    const long long pParticleID,
                
                                    // Output int
                                    long long *totalLocalizedParticle,
                                    int *pLocalized_new,
                
                                    // Input std::vector
                                    const std::vector<double> *pWingLength_array,
                                    const std::vector<double> *pFlawNumber_array,
                                    const std::vector<double> *pflawSize_array,

                                    // Output std::vector
                                    std::vector<double> *pWingLength_array_new
                                    );

  void advanceTimeSigmaL(
                         // Data Structures
                         const Flags flags,
                         const ModelData initialData,
                         const flawDistributionData flawDistData,
                         const BrittleDamageData brittle_damage,
                         const granularPlasticityData gpData,
                         const ArtificialViscosity artificialViscosity,
                         const PortableMieGruneisenEOSTemperature *eos,
                         // Input Matrix:
                         const Matrix3x3 velGrad,
                         // Input/OutputMatrix:
                         Matrix3x3 *pStress,
                         Matrix3x3 *pStress_qs,
                         // Input double
                         const double delT,
                         const double J_old,
                         const double J,
                         const double pTemperature,
                         const double rho_orig,
                         const double dx_ave, /* mean spatial dimension for art visc calc */
                         // Input/Output double
                         double *pIEl,
                         double *pPlasticStrain,
                         double *pPlasticEnergy,
                         double *pDamage,
                         double *pGPJ,
                         double *pGP_strain,
                         double *pGP_energy,
                         double *pEnergy,
                         double *damage_dt,
                         double *pepsV,
                         double *pgam,
                         double *pepsV_qs,
                         double *pgam_qs,
                         int *pLocalized,
                         // Output only double:
                         double *p_q_out,
                         double *pdTdt_out,
                         double *c_dil_out,
                         // Input std::vector
                         const std::vector<double> *pWingLength_array,
                         const std::vector<double> *pFlawNumber_array,
                         const std::vector<double> *pflawSize_array,
                         // Output std::vector
                         std::vector<double> *pWingLength_array_new,
                         const bool assumeRotatedTensors=false
                         );

  void postAdvectionFixup(
                          // Data Structures
                          const Flags flags,
                          const ModelData initialData,
                          const flawDistributionData flawDistData,
                          const BrittleDamageData brittle_damage,
                          const granularPlasticityData gpData,
                          const ArtificialViscosity artificialViscosity,
                          const PortableMieGruneisenEOSTemperature *eos,
                          // Input/OutputMatrix:
                          Matrix3x3 *pStress, // Recalculate
                          Matrix3x3 *pStress_qs, // Unused
                          const double J,        // input
                          const double pTemperature, // input
                          const double rho_orig,     // input
                          // Input/Output double
                          double *pIEl, // update
                          double *pPlasticStrain, // Unused
                          double *pPlasticEnergy, 
                          double *pDamage,
                          double *pGPJ,
                          double *pGP_strain,
                          double *pGP_energy,
                          double *pEnergy,
                          double *damage_dt,
                          double *pepsV,
                          double *pgam,
                          double *pepsV_qs,
                          double *pgam_qs,
                          int *pLocalized,
                          // Input/Output std::vector
                          std::vector<double> *pWingLength_array,
                          std::vector<double> *pFlawNumber_array,
                          std::vector<double> *pflawSize_array
                          );


  std::string getHistVarName(const int histVarNum);
  std::string getMatParamName(const int paramNumber);
  void parseFlawDistData(double flawDistParam[PTR_NUM_FLAW_DIST_PARAM],
                         const std::string inputFileName);
  void parseMatParamData(double matParam[PTR_NUM_MAT_PARAMS],
                         const std::string inputFileName);

  void unpackMatParams(const double matParamArray[PTR_NUM_MAT_PARAMS],
                       Flags *flags,
                       ModelData *initialData,
                       flawDistributionData *flawDistData,
                       BrittleDamageData *brittle_damage,
                       granularPlasticityData *gpData,
                       ArtificialViscosity *artificialViscosity,
                       CMData *eosData
                       );
  flawDistributionData unpackFlawDistData( const double flawDistArray[PTR_NUM_FLAW_DIST_PARAM]);
  double initalizeFlawDist( double flawSize[],
                            double flawNumber[],
                            const flawDistributionData flawDistData,
                            const double dx_ave, unsigned long seedArray[],
                            unsigned long nSeedValues
                            );
}	// end namespace PTR

#endif /* TONGERAMESHPORTABLE_H */ 
