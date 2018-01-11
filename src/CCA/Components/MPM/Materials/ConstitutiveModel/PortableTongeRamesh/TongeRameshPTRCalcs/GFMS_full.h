/* Function declarations for performing multistage return to a pressure dependent yield surface */
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

#ifndef SYM_MATRIX3_H
#define SYM_MATRIX3_H

#ifndef EQN_EPS
#define EQN_EPS 1e-12
#endif

#ifndef IsZero
#define IsZero( x ) ((x) > -EQN_EPS && (x) < EQN_EPS)
#endif

#include <ostream>
#include <cmath>		// sqrt

/* Define a symmetric 3x3 matrix, elements are stored
  as: [00 11 22 sqrt(2)*12 sqrt(2)*02 sqrt(2)*01]
     00 01 02
  A= 10 11 12
     20 21 22
  This matrix is represented using Mandel notation
*/

namespace SymMat3{
class SymMatrix3
{
 public:
  SymMatrix3();
  SymMatrix3(const double val);
  SymMatrix3(const bool isIdentity);
  SymMatrix3(	const double v0,
                const double v1,
                const double v2,
                const double v3,
                const double v4,
                const double v5
                );

  virtual ~SymMatrix3();
		
  void identity();
  double determinant() const;
  double trace() const;
  double normSquared() const;
  inline double norm() const
  {
    return sqrt(normSquared()); 
  }

  /* const SymMatrix3 inverse() const; */
  double get(const int i, const int j) const;
  double get(const int i) const;
  void set(const int i, const int j, const double val);
  void set(const int i, const double val);
		
  void swap(SymMatrix3 *rhs);

  void calcRZTheta(double *r, double *z, double *theta,
                   SymMatrix3* er, SymMatrix3 *ez, SymMatrix3 *etheta
                   ) const ;
		
  SymMatrix3& operator+= (const SymMatrix3 rhs);
  const SymMatrix3 operator+(const SymMatrix3 rhs) const;
  SymMatrix3& operator-= (const SymMatrix3 rhs);
  const SymMatrix3 operator-(const SymMatrix3 rhs) const;
  SymMatrix3& operator*= (const double rhs);
  const SymMatrix3 operator*(const double rhs) const;
  SymMatrix3& operator/= (const double rhs);
  const SymMatrix3 operator/(const double rhs) const;
  const SymMatrix3 operator*(const SymMatrix3 rhs) const;

  inline double Contract(const SymMatrix3 mat) const
  {
    // Return the contraction of this matrix with another 

    double contract = 0.0;

    for (int i = 0; i< 3; i++) {
      for(int j=0;j<3;j++){
        contract += get(i, j)*(mat.get(i, j));
      }
    }
    return contract;
  }

  inline SymMatrix3 isotropic() const
  {
    double sigma_h=trace()/3.0;
    return SymMatrix3(sigma_h,sigma_h,sigma_h,0.0,0.0,0.0);
  }

  inline SymMatrix3 deviatoric() const
  {
    SymMatrix3 retMatrix(*this);
    retMatrix -= retMatrix.isotropic();
    /* return (*this - isotropic()); */
    return retMatrix;
  }
		
 private:
  double _values[6];
  /* add your private declarations */
};
}
std::ostream& operator<<(std::ostream& out, const SymMat3::SymMatrix3 rhs);

#endif /* SYM_MATRIX3_H */ 
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

#ifndef GRANULAR_FLOW_DEFS
#define GRANULAR_FLOW_DEFS

#define NUM_SOL_PARAM 4
#define NUM_MAT_PARAM 17
#define NUM_HIST_VAR 16

/* solParams.absToll   = solParamVector[0]; */
/* solParams.relToll   = solParamVector[1]; */
/* solParams.maxIter   = int(floor(solParamVector[2]+0.5)); */
/* solParams.maxLevels = int(floor(solParamVector[3]+0.5)); */

/* sigma_0.set(0,0,histVector[0]); */
/* sigma_0.set(1,1,histVector[1]); */
/* sigma_0.set(2,2,histVector[2]); */
/* sigma_0.set(1,2,histVector[3]); */
/* sigma_0.set(0,2,histVector[4]); */
/* sigma_0.set(0,1,histVector[5]); */
/* sigma_0qs.set(0,0,histVector[6]); */
/* sigma_0qs.set(1,1,histVector[7]); */
/* sigma_0qs.set(2,2,histVector[8]); */
/* sigma_0qs.set(1,2,histVector[9]); */
/* sigma_0qs.set(0,2,histVector[10]); */
/* sigma_0qs.set(0,1,histVector[11]); */
/* epsv0 = histVector[12]; */
/* gam0  = histVector[13]; */


/* matParams.bulkMod        = matParamVector[0]; */
/* matParams.shearMod       = matParamVector[1]; */
    
/* matParams.m0             = matParamVector[2]; */
/* matParams.m1             = matParamVector[3]; */
/* matParams.m2             = matParamVector[4]; */
    
/* matParams.p0             = matParamVector[5]; */
/* matParams.p1             = matParamVector[6]; */
/* matParams.p2             = matParamVector[7]; */
/* matParams.p3             = matParamVector[8]; */
/* matParams.p4             = matParamVector[9]; */
    
/* matParams.a1             = matParamVector[10]; */
/* matParams.a2             = matParamVector[11]; */
/* matParams.a3             = matParamVector[12]; */
    
/* matParams.beta           = matParamVector[13]; */
/* matParams.psi            = matParamVector[14]; */
/* int J3Type = matParamVector[15]>2.25 || matParamVector[15] < 0.5 ? 0 : */
/*   int(floor(matParamVector[15]+0.5)); */
/* switch (J3Type){ */
/* case GFMS::DruckerPrager: */
/*   matParams.psi = 1.0; */
/*   matParams.J3Type = GFMS::DruckerPrager; */
/*   break; */
/* case GFMS::Gudehus: */
/*   if(matParams.psi < 7.0/9.0){ */
/*     matParams.psi = 7.0/9.0; */
/*   } */
/*   if(matParams.psi > 9.0/7.0){ */
/*     matParams.psi = 9.0/7.0; */
/*   } */
/*   matParams.J3Type = GFMS::Gudehus; */
/*   break; */
/* case GFMS::WilliamWarnke: */
/*   if(matParams.psi < 0.5){ */
/*     matParams.psi = 0.5; */
/*   } */
/*   if(matParams.psi > 2.0){ */
/*     matParams.psi = 2.0; */
/*   } */
/*   matParams.J3Type = GFMS::WilliamWarnke; */
/*   break; */
/* default: */
/*   matParams.J3Type = GFMS::DruckerPrager; */
/*   break; */
/* } */
/* matParams.relaxationTime = matParamVector[16]; */

#endif // GRANULAR_FLOW_DEFS
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

#ifndef GFMS_H
#define GFMS_H

//#include "SymMatrix3.h"
//#include "GranularFlowDefs.h"

namespace GFMS{
  typedef enum GammaForm {DruckerPrager,
                          Gudehus,
                          WilliamWarnke} GammaForm;

  typedef struct solParam{
    double absToll;
    double relToll;
    int    maxIter;
    int    maxLevels;
  } solParam;

  typedef struct matParam{
    double bulkMod;          // bulk modulus
    double shearMod;         // shear modulus
    // Friction
    double m0;               // Initial coeficient of friction
    double m1;                         // final coeficient of friction
    double m2;                          // coeficient of friction decay rate
    // Consolidation
    double p0;                        // 
    double p1;                       // 
    double p2;                      // 
    double p3;                       // 
    double p4;                        // Ratio between X and kappa (kappa/X)
    // Frictional yield surface
    double a1;                         // Assemtotic hydristatic tensile strength 
    double a2;                        // Hydrostatic tension curvature
    double a3;                          // Hydrostatic tension curvature
    double beta;                      // Non-associativity 0 <= beta <= 1.0
    double psi;                       //  Ratio between triaxial extension strength and triaxial compressive strength
    GammaForm J3Type;                   // form for J3 dependence (0-DP, 1-Gudenes, 2-Willam-Wenke, 3-MC)
    double relaxationTime;              /* Granular flow relaxation time */
  } matParam;

  double computeMu(const double gam0,  // deviatoric component of the accumulated plastic strain
                   const matParam *matParams, // material parameters
                   double *dm_dg);

  double computeKappa(const double epsv0,  // volumetric component of the accumulated plastic strain
                      const matParam *matParams, // material parameters
                      double *X,
                      double *dX_depsv,
                      double *dK_dX);

  double computeFf(const double z,      // Input z=tr(sigma)/sqrt(3)
                   const double gam,  // deviatoric component of the accumulated plastic strain
                   const matParam *matParams, // material parameters
                   double *dFf_dz,
                   double *dFf_dgam);

  double computeFc(const double z,      // Input z=tr(sigma)/sqrt(3)
                   const double epsv,  // volumetric component of the accumulated plastic strain
                   const matParam *matParams, // material parameters
                   double *dFc_dz,
                   double *dFc_depsv);

  

  // 2D helper functions in the reduced r-z space:
  double calcYieldFunc2D(const double r, // Input r=sqrt(s:s) s=sigma-z*eZ
                         const double z, // Input z=tr(sigma)/sqrt(3)
                         const double epsv0, // Volumetric component of the accumulated plastic strain
                         const double gam0,  // deviatoric component of the accumulated plastic strain
                         const matParam *matParams, // material parameters
                         double *df_dr,           // gradient of the yield function with respect to r
                         double *df_dz            // gradient of the yield function with respect to z
                         );

  void calcFlowDir2D(const double r,
                     const double z,
                     const double epsv0,
                     const double gam0,
                     const matParam *matParams,
                     double *Mr,
                     double *Mz);

  double calc_vertex_t(const double gam0, const matParam *matParams, const solParam *solParams);
  double calcR(const double epsv0, const double gam0, const double z, const matParam *matParams, double *dr_dz);
  double calc_dr_dz(const double r, const double epsv0, const double gam0, const matParam *matParams);
  void calcRMax(const double epsv0, const double gam0,
                const matParam *matParams,
                const solParam *solParams,
                double *rMax,
                double *zAtMax);
  int doVertexTest(const double z, const double r,
                   const double epsv0, const double gam0,
                   const matParam  *matParams, const solParam *solParams,
                   double *zVert );
  void doInnerReturn2D(const double z, const double r, const double epsv0, const double gam0,
                       const matParam *matParams, const solParam *solParams,
                       double *zRet, double *rRet
                       );
  int doLineSearch2D(const double z, const double r, const double zTarget, const double rTarget,
                     const double epsv0, const double gam0,
                     const matParam *matParams, const solParam *solParams,
                     double *zRet, double *rRet
                     );

  int doOuterReturn2D(const double r_tr, const double z_tr, const double epsv0, const double gam0,
                      const matParam *matParams, const solParam *solParams,
                      double *rRet, double *zRet, double *epsv, double *gam);

  double calcGamma(const double theta, const matParam *matParams,
                   double *dg_dTheta);
  double calcYieldFunc(const double r, const double z, const double theta,
                       const double epsv0, const double gam0,
                       const matParam *matParams,
                       double *df_dr,
                       double *df_dz,
                       double *df_dtheta_rinv // df/dtheta * 1/r
                       );

  double calcYieldFunc(const SymMat3::SymMatrix3 *sigma,
                       const double epsv0, const double gam0,
                       const matParam *matParams);
   
  void calcFlowDir(const double r, const double z, const double theta,
                   const double epsv0, const double gam0,
                   const matParam *matParams,
                   double *Mr,
                   double *Mz,
                   double *Mtheta_rinv
                   );
  void doInnerReturn(const SymMat3::SymMatrix3 *sigma, const double epsv0, const double gam0,
                     const matParam *matParams, const solParam *solParams,
                     SymMat3::SymMatrix3 *sigmaRet
                     );
  int doReturnOuter(const SymMat3::SymMatrix3 *sigma_tr, const double epsv0, const double gam0,
                    const matParam *matParams,
                    const solParam *solParams,
                    SymMat3::SymMatrix3 *sigmaRet,
                    double *epsv,
                    double *gam
                    );
  int doReturnStressWithHardening(const SymMat3::SymMatrix3 *sigma, const double epsv0, const double gam0,
                                  const matParam *matParams, const solParam *solParams,
                                  SymMat3::SymMatrix3 *sigmaRet,
                                  double *epsv,
                                  double *gam
                                  );
  double computeEnsambleHardeningModulus(const SymMat3::SymMatrix3 *M, const SymMat3::SymMatrix3 *F, const double z,
                                         const double epsv0, const double gam0,
                                         const matParam *matParams);

  int advanceTime(const double delT,
                  const SymMat3::SymMatrix3 *D,
                  const SymMat3::SymMatrix3 *sigma_0,
                  const double epsv_0,
                  const double gam_0,
                  const matParam *matParams,
                  const solParam *solParams,
                  SymMat3::SymMatrix3 *sigma_1,
                  double *epsv_1,
                  double *gam_1
                  );
  double integrateStressWithRefinement(const double delT,
                                     const SymMat3::SymMatrix3 *D,
                                     const SymMat3::SymMatrix3 *sigma_0,
                                     const double epsv_0,
                                     const double gam_0,
                                     const int steps,
                                     const int level,
                                     const matParam *matParams,
                                     const solParam *solParams,
                                     SymMat3::SymMatrix3 *sigma_1,
                                     double *epsv_1,
                                     double *gam_1
                                     );
  double integrateRateDependent(const double delT,
                              const SymMat3::SymMatrix3 *D,
                              const SymMat3::SymMatrix3 *sigma_0,
                              const SymMat3::SymMatrix3 *sigma_qs0,
                              const double epsv_0,
                              const double gam_0,
                              const double epsv0_qs,
                              const double gam0_qs,
                              const matParam *matParams,
                              const solParam *solParams,
                              SymMat3::SymMatrix3 *sigma_1,
                              SymMat3::SymMatrix3 *sigma_qs1,
                              double *epsv_1,
                              double *gam_1,
                              double *epsv1_qs,
                              double *gam1_qs
                              );
  
  void packHistoryVariables(const SymMat3::SymMatrix3 *sigma,
                            const SymMat3::SymMatrix3 *sigma_qs,
                            const double epsv,
                            const double gam,
                            const double epsv_qs,
                            const double gam_qs,
                            double histVector[NUM_HIST_VAR]
                            );
  
  void unpackHistoryVariables(const double histVector[NUM_HIST_VAR],
                              SymMat3::SymMatrix3 *sigma_0,
                              SymMat3::SymMatrix3 *sigma_0qs,
                              double *epsv0,
                              double *gam0,
                              double *epsv0_qs,
                              double *gam0_qs
                              );

  matParam unpackMaterialParameters(const float matParamVector[NUM_MAT_PARAM]
                                    );
  solParam unpackSolutionParameters(const float solParamVector[NUM_SOL_PARAM]
                                    );
  
} /* end of namespace GFMS */

#endif  /* GFMS_H */
