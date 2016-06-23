/* Function bodies for performing multistage return to a pressure dependent yield surface */
#include "GFMS_full.h"
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

//#include "SymMatrix3.h"
#include <cstring> 		// memcpy
#include <cmath>		// fabs
#include <iomanip>		// std::setw
#include <stdexcept>
#include <iostream>

using namespace SymMat3;
namespace SymMat3{
SymMatrix3::SymMatrix3()
{
  _values[0] = 0.0; //_values[5] = 0.0; _values[4] = 0.0;
  _values[5] = 0.0; _values[1] = 0.0; //_values[3] = 0.0;
  _values[4] = 0.0; _values[3] = 0.0; _values[2] = 0.0;
}

SymMatrix3::SymMatrix3(const double val){
	for (int i = 0; i < 6; i++){
		_values[i] = val;
	}
}

SymMatrix3::SymMatrix3(const bool isIdentity){
  _values[0] = 0.0; //_values[5] = 0.0; _values[4] = 0.0;
  _values[5] = 0.0; _values[1] = 0.0; //_values[3] = 0.0;
  _values[4] = 0.0; _values[3] = 0.0; _values[2] = 0.0;

  if (isIdentity){
    _values[0] = 1.0;
    _values[1] = 1.0;
    _values[2] = 1.0;
  }
}

SymMatrix3::SymMatrix3(	const double v0,
						const double v1,
						const double v2,
						const double v3,
						const double v4,
						const double v5)
{
	_values[0] = v0;
	_values[1] = v1;
	_values[2] = v2;
	_values[3] = v3;
	_values[4] = v4;
	_values[5] = v5;
}

SymMatrix3::~SymMatrix3()
{
}

void SymMatrix3::identity()
{
  _values[0] = 1.0; //_values[5] = 0.0; _values[4] = 0.0;
  _values[5] = 0.0; _values[1] = 1.0; //_values[3] = 0.0;
  _values[4] = 0.0; _values[3] = 0.0; _values[2] = 1.0;
}

void SymMatrix3::swap(SymMatrix3 *rhs){
	double cpy[6];
	size_t n = 6 * sizeof(double);
	memcpy(cpy, rhs->_values, n);
	memcpy(rhs->_values, _values, n);
	memcpy(_values, cpy, n);
}

SymMatrix3& SymMatrix3::operator+= (const SymMatrix3 rhs)
{
	for (int i = 0; i < 6; i++){
		_values[i] += rhs._values[i];
	}
	
	return *this;
}

const SymMatrix3 SymMatrix3::operator+(const SymMatrix3 rhs) const
{
	SymMatrix3 ans(*this);
	ans += rhs;
	return ans;
}

SymMatrix3& SymMatrix3::operator-= (const SymMatrix3 rhs)
{
	for (int i = 0; i < 6; i++){
		_values[i] -= rhs._values[i];
	}
	
	return *this;
}

const SymMatrix3 SymMatrix3::operator-(const SymMatrix3 rhs) const
{
	SymMatrix3 ans(*this);
	ans -= rhs;
	return ans;
}

SymMatrix3& SymMatrix3::operator*= (const double rhs)
{
	for (int i = 0; i < 6; i++){
		_values[i] *= rhs;
	}
	
	return *this;
}

const SymMatrix3 SymMatrix3::operator*(const double rhs) const
{
	SymMatrix3 ans(*this);
	ans *= rhs;
	return ans;
}

SymMatrix3& SymMatrix3::operator/= (const double rhs)
{
	if (fabs(rhs) < 1.e-12){
		throw std::domain_error("SymMatrix3::operator/=: divide by 0");
	}
	for (int i = 0; i < 6; i++){
		_values[i] /= rhs;
	}
	
	return *this;
}

const SymMatrix3 SymMatrix3::operator/(const double rhs) const
{
	SymMatrix3 ans(*this);
	ans /= rhs;
	return ans;
}

const SymMatrix3 SymMatrix3::operator*(const SymMatrix3 rhs) const
{
  SymMatrix3 ans(0.0);
  int i, j, k;
  for (i = 0; i < 3; i++){
    for (j = i; j < 3; j++){
      double sum(0.0);
      for (k = 0; k < 3; k++){
        sum += get(i, k) * rhs.get(k, j);
      }
      ans.set(i,j,sum);
    }
  }
  return ans;
}

double SymMatrix3::determinant() const
{
	double ans = (_values[0] * _values[1] * _values[2]);
	ans       += (_values[5] * _values[3] * _values[4])/(sqrt(2.0)); // This shows up twice
	ans       -= (_values[1] * _values[4] * _values[4])/2.0;
	ans       -= (_values[2] * _values[5] * _values[5])/2.0;
	ans       -= (_values[0] * _values[3] * _values[3])/2.0;
	return ans;
}

double SymMatrix3::get(const int i, const int j) const{
  if (i < 0 || j < 0 || i > 2 || j > 2){
    throw std::out_of_range("SymMatrix3::get: both indices must be in range [0,2]");
  }

  double val = i==j ? _values[i] : _values[6-(i+j)]/sqrt(2.0);
	
  return val;
}

double SymMatrix3::get(const int i) const{
  if( (i < 0) || (i > 5) ){
    throw std::out_of_range("SymMatrix3::get: Index must be in the range [0,5]");
  }
  return _values[i];
}
  
void SymMatrix3::set(const int i, const int j, const double val)
{
  if (i < 0 || j < 0 || i > 2 || j > 2){
    throw std::out_of_range("SymMatrix3::get: both indices must be in range [0,2]");
  }

  if( i==j ){
    _values[i] = val;
  } else {
    _values[6-(i+j)] = val*sqrt(2.0);
  }
}

void SymMatrix3::set(const int i, const double val)
{
  if ( (i < 0 ) || (i>5) ){
    throw std::out_of_range("SymMatrix3::get: both single index must be in range [0,5]");
  }
  
  _values[i] = val;
}

double SymMatrix3::trace() const
{
  double ans;
  ans = _values[0] + _values[1] + _values[2];
  return ans;
}

double SymMatrix3::normSquared() const
{
  int i;
  double ans = 0;
  for (i = 0; i < 6; i++){
    ans += _values[i] * _values[i];
  }
  return ans;
}

void SymMatrix3::calcRZTheta(double *r, double *z, double *theta,
                             SymMatrix3 *er, SymMatrix3 *ez, SymMatrix3 *etheta
                             ) const {
  const double pi = 3.141592653589793;
  *z = trace()/sqrt(3.0);
  *ez = SymMatrix3(1.0,1.0,1.0,0,0,0)/sqrt(3.0);
  SymMatrix3 sigma_dev(this->deviatoric());
  *r = sigma_dev.norm();
  if(*r>1e-12){
    *er = sigma_dev/(*r);
    double deter(er->determinant());
    double sin3theta(3.0*sqrt(6.0)*deter);
    if (std::abs(sin3theta)<1.0){
      double cos3theta(sqrt(1-sin3theta*sin3theta));
      *theta = asin(sin3theta)/3.0;
      SymMatrix3 THat(( (*er)*(*er) - (*ez)/sqrt(3.0))*sqrt(6.0));
      *etheta = (THat - ((*er) * sin3theta))/cos3theta;
    } else {
      *theta = sin3theta < 0 ? -pi/6.0 : pi/6.0;
      *etheta = SymMatrix3(0.0);
    }
  } else {
    *er     = SymMatrix3(0.0);
    *r      = 0.0;
    *etheta = SymMatrix3(0.0);
    *theta  = 0.0;
  }
}

std::ostream& operator<<(std::ostream& out, const SymMatrix3 rhs){
  out << std::endl;
  int i, j;
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++){
      out << "\t" << std::fixed << std::setw( 11 ) << rhs.get(i, j);
    }
    out << std::endl;
  }
  return out;
}

}
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

//#include "SymMatrix3.h"
//#include "GranularFlowMultiStage.h"
#include <cmath>
#include <iostream>
#include <iomanip>              // std::setw std:setprecision
#include <assert.h>             // assert()

using namespace SymMat3;
using namespace GFMS;
namespace GFMS{
  double computeMu(const double gam0,         // deviatoric component of the accumulated plastic strain
                   const matParam *matParams, // material parameters
                   double *dm_dg){
    const double m0(matParams->m0);
    const double m1(matParams->m1);
    const double m2(matParams->m2);
    double expm2gam=exp(-m2*gam0);
    *dm_dg = -m2*(m0-m1)*expm2gam;
    return (m1 + (m0 - m1) * expm2gam);
  }

  double computeKappa(const double epsv0,        // volumetric component of the accumulated plastic strain
                      const matParam *matParams, // material parameters
                      double *X,
                      double *dX_depsv,
                      double *dK_dX){
    const double p0(matParams->p0);
    const double p1(matParams->p1);
    const double p2(matParams->p2);
    const double p3(matParams->p3);
    const double p4(matParams->p4);
    *X=p1 * (p0 + (log(p3+p2*epsv0) - std::abs(log(p3+p2*epsv0)))/2.0);
    *dX_depsv = (p3+p2*epsv0) < 1.0 ? p1*p2/(p3 + p2*epsv0) : 0.0;
    *dK_dX = p4;
    return p4*(*X);
  }

  double computeFf(const double z,            // Input z=tr(sigma)/sqrt(3)
                   const double gam,          // deviatoric component of the accumulated plastic strain
                   const matParam *matParams, // material parameters
                   double *dFf_dz,
                   double *dFf_dgam){
    const double a1(matParams->a1);
    const double a2(matParams->a2);
    const double a3(matParams->a3);
    const double m0(matParams->m0);
    
    double I1 = z*sqrt(3.0);
    double dmu_dg,dFf_dI1, Ff;
    if (a2*I1 > 100.0){
      Ff = a1-a3*exp(100.0) - m0 * I1;
      dFf_dI1 = -a2*a3 * exp(100.0) - m0;
    } else {
      Ff = a1-a3*exp(a2*I1) - m0 * I1;
      dFf_dI1 = -a2*a3 * exp(a2*I1) - m0;
    }
    *dFf_dgam = 0.0;
    if(m0>0){
      double mu = computeMu(gam, matParams,&dmu_dg);
      Ff /= m0;
      *dFf_dgam = Ff*dmu_dg;
      Ff *= mu;
      dFf_dI1 *= mu/m0;
    }
    *dFf_dz   = dFf_dI1*sqrt(3.0);
    return Ff;
  }

  double computeFc(const double z,      // Input z=tr(sigma)/sqrt(3)
                          const double epsv,  // volumetric component of the accumulated plastic strain
                          const matParam *matParams, // material parameters
                          double *dFc_dz,
                          double *dFc_depsv){

    double I1 = z*sqrt(3.0);
    double X,dFc_dI1, Fc, dX_deV, dK_dX;
    double kappa = computeKappa(epsv, matParams, &X, &dX_deV, &dK_dX);
    if (I1 < kappa){
      Fc = 1 - ( (kappa - I1) * (kappa - I1) ) / ( (kappa-X) * (kappa-X) ) ;
      dFc_dI1 = -2.0 * (I1 - kappa) / ( (kappa-X)*(kappa-X) );
      *dFc_dz  = dFc_dI1*sqrt(3.0);
      double kmI1=kappa-I1;
      double kmI1sq=kmI1*kmI1;
      // double kmI1cu=kmI1sq*kmI1;
      double kmX=kappa-X;
      double kmXsq=kmX*kmX;
      double kmXcu=kmXsq*kmX;
      double dFc_dX = -(-2*kmI1sq/kmXcu + 2*kmI1/kmXcu )*dK_dX - 2*kmI1sq/kmXcu;
      *dFc_depsv = dFc_dX * dX_deV;
    } else {
      Fc = 1.0;
      dFc_dI1   = 0.0;
      *dFc_dz    = 0.0;
      *dFc_depsv = 0.0;
    }
    return Fc;
  }
  
  // 2D helper functions in the reduced r-z space:
  double calcYieldFunc2D(const double r, // Input r=sqrt(s:s) s=sigma-z*eZ
                         const double z, // Input z=tr(sigma)/sqrt(3)
                         const double epsv0, // Volumetric component of the accumulated plastic strain
                         const double gam0,  // deviatoric component of the accumulated plastic strain
                         const matParam *matParams, // material parameters
                         double *df_dr,            // gradient of the yield function with respect to r
                         double *df_dz             // gradient of the yield function with respect to z
                         )
  {
    double Ff,dFf_dz,dFf_dgam;
    double Fc,dFc_dz,dFc_depsv;
    Ff = computeFf(z, gam0,  matParams, &dFf_dz, &dFf_dgam);
    Fc = computeFc(z, epsv0, matParams, &dFc_dz, &dFc_depsv);
    double f = 0.5*r*r - (Ff * std::abs(Ff)) * Fc;
    if(Ff >= 0){
      *df_dz = -(( 2.0*Ff*dFf_dz)*Fc + Ff*Ff*dFc_dz);
    } else {
      *df_dz = -((-2.0*Ff*dFf_dz)*Fc + Ff*Ff*dFc_dz);
    }
    *df_dr = r;
    return f;
  }

  void calcFlowDir2D(const double r,
                     const double z,
                     const double epsv0,
                     const double gam0,
                     const matParam *matParams,
                     double *Mr,
                     double *Mz){
    double tmpMr,tmpMz;
    double Ff,dFf_dz,dFf_dgam;
    Ff = computeFf(z, gam0,  matParams, &dFf_dz, &dFf_dgam);
    double Fc,dFc_dz,dFc_depsv;
    Fc = computeFc(z, epsv0, matParams, &dFc_dz, &dFc_depsv);
    if (Fc < 1.0){
      tmpMr = r;
      if(Ff >= 0){
        tmpMz = (- (( 2.0*Ff*dFf_dz)*Fc + Ff*Ff*dFc_dz) );
      } else {
        tmpMz = (- ((-2.0*Ff*dFf_dz)*Fc + Ff*Ff*dFc_dz) );
      }
    } else {
      tmpMr = 1/sqrt(2.0);
      tmpMz = -dFf_dz;
    }
    if (tmpMz>0.0){
      tmpMz *= matParams->beta;
    }
    double mMag = sqrt(tmpMr*tmpMr+tmpMz*tmpMz);
    *Mz = tmpMz/mMag;
    *Mr = tmpMr/mMag;
  }

  double calc_vertex_t(const double gam0, const matParam *matParams, const solParam *solParams){
    double r, dr_dz;
    double zmin(0.0);
    r = calcR(0.0, gam0, zmin, matParams, &dr_dz); // we set epsv0 to 0 because we are only interested in the
                                                   // tension part of the yield surface
    double zVert;
    if (dr_dz < 0){
      double zmax(-r/dr_dz);
      double rVert;       // TO DO: Do something smarter, use a mixed NR and bisection algorithm
                                // to follow the yield surface to the intersection
      doLineSearch2D(1.01*zmax,0.0,solParams->relToll*zmax,0.0,0.0,gam0,matParams,solParams,&zVert,&rVert);
    } else {
      zVert = 1e4*matParams->bulkMod; // return a very large tensile stress
    }
    
    return zVert;
  }

  double calcR(const double epsv0, const double gam0, const double z, const matParam *matParams,
               double *dr_dz)
  {
    double Ff,dFf_dz,dFf_dgam;
    Ff = computeFf(z, gam0,  matParams, &dFf_dz, &dFf_dgam);
    double Fc,dFc_dz,dFc_depsv;
    Fc = computeFc(z, epsv0, matParams, &dFc_dz, &dFc_depsv);
    double sqrt2Fc = Fc < 0.0 ? -sqrt(-2.0*Fc) : sqrt(2.0*Fc);
    double r = Ff*sqrt2Fc;
    if(std::abs(sqrt2Fc) < 1e-12){
      sqrt2Fc = sqrt2Fc > 0.0 ? 1e-12 : -1e-12;
    }
    *dr_dz = sqrt2Fc*dFf_dz + Ff/sqrt2Fc * dFc_dz;
    return r;
  }

  double calc_dr_dz(const double z, const double epsv0, const double gam0, const matParam *matParams){
    double dr_dz;
    calcR(epsv0, gam0, z, matParams, &dr_dz);
    return dr_dz;
  }

  void calcRMax(const double epsv0, const double gam0,
                const matParam *matParams,
                const solParam *solParams,
                double *rMax,
                double *zAtMax){
    double a,b,c,d,e,fa,fb,fc;
    double kappa,X,dK_depsv,dX_depsv;
    const double absToll(solParams->absToll);
    const int maxIter(solParams->maxIter);

    kappa = computeKappa(epsv0, matParams, &X, &dX_depsv, &dK_depsv);
    // To do: Algorithm?? Switch to Ridder's method (also reduce calls to this function)
    a = kappa/sqrt(3.0);
    b = X/sqrt(3.0);
    fa = calc_dr_dz(a,epsv0, gam0, matParams);
    fb = calc_dr_dz(b,epsv0, gam0, matParams);
    c = b;
    fc = fb;
    d = b-a;
    e = d;
    int i=0;
    for (; i<maxIter; ++i){
      if ( (fb > 0 && fc > 0) || (fb < 0 && fc < 0) ){
        c = a;
        fc = fa;
        d = b-a;
        e = d;
      }
      if ( std::abs(fc) < std::abs(fb) ) {
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      }
      double xm = 0.5*(c-b);
      if ( std::abs(xm) < absToll || std::abs(fb) < absToll){
        break;              // b contains the minmum
      }
      if (std::abs(e) > absToll && std::abs(fa) > std::abs(fb)){
        double s = fb/fa;           // quadratic interpolation
        double p,q;
        if (a == c){
          p = 2.0*xm*s;
          q = 1.0-s;
        } else {
          q = fa/fc;
          double r = fb/fc;
          p = s*(xm*q*(q-r)-(b-a)*(r-1.0));
          q = (q-1.0)*(r-1.0)*(s-1.0);
        }
        if (p > 0.0){       // check the bounds
          q = -q;
        }
        p = std::abs(p);
        double min1 = 3.0*xm*q - std::abs(absToll*q);
        double min2 = std::abs(e*q);
        if( 2.0*p < std::min(min1,min2)){
          e = d;           // accept the interpolation
          d = p/q;
        } else {
          d = xm ;         // use bisection (interpolation no good)
          e = d;
        }
      } else{           // bounds are not decreasing slow enough use bisection
        d = xm;
        e = d;
      }
      a = b;            // move last best guess to a
      fa = fb;
      if (std::abs(d) > absToll){ // evaluate new trial root
        b += d;
      } else {
        b += copysign(absToll, xm);
      }
      fb = calc_dr_dz(b,epsv0, gam0, matParams);
    } // end of while loop the zero is in b
    *zAtMax = b;
    double dr_dz;
    *rMax = calcR(epsv0, gam0, *zAtMax, matParams, &dr_dz);
  }

  // Return 1 if the stress should return to the vertex,
  // return 0 otherwise
  int doVertexTest(const double z, const double r,
                   const double epsv0, const double gam0,
                   const matParam  *matParams, const solParam *solParams,
                   double *zVert ){
    double dr_dz;
    double rTest = calcR(epsv0, gam0, z, matParams, &dr_dz);
    if(rTest>0.0){return 0;}
    const double G = matParams->shearMod;
    const double K = matParams->bulkMod;
    *zVert = calc_vertex_t(gam0,matParams,solParams);
    double Mr,Mz;
    calcFlowDir2D(0.0, *zVert, epsv0, gam0, matParams, &Mr, &Mz);
    // Check to see if the trial stress is beyond the vertex
    double s1p = z-(*zVert);
    double s1s = r;
    double eta = 2.0*G/(3.0*K);
    double A_p=Mz;
    double A_s=eta*Mr;
    double Amag = sqrt(A_p*A_p + A_s*A_s);
    A_p /= Amag;
    A_s /= Amag;
    double s_dot_A = s1p*A_p + s1s*A_s;
    return (s1p >= (A_p * s_dot_A));
  }

  void doInnerReturn2D(const double z, const double r, const double epsv0, const double gam0,
                       const matParam *matParams, const solParam *solParams,
                       double *zRet, double *rRet
                       ){
    *zRet = z;
    double dr_dz;
    *rRet = calcR(epsv0, gam0, z, matParams, &dr_dz);
    if(z >= 0.0 && dr_dz < 0.0){
      doLineSearch2D(z, r,
                     0.0, 0.0,
                     epsv0, gam0, matParams, solParams,
                     zRet, rRet);
    } else if (dr_dz>0.0 || (*rRet)<0.0){
      double rMax,zMax;
      calcRMax(epsv0, gam0, matParams, solParams, &rMax, &zMax);
      doLineSearch2D(z, r,
                     zMax, 0.5*std::min(r,rMax),
                     epsv0, gam0, matParams, solParams,
                     zRet, rRet);
    }
  }

    
  int doLineSearch2D(const double z, const double r, const double zTarget, const double rTarget,
                     const double epsv0, const double gam0,
                     const matParam *matParams, const solParam *solParams,
                     double *zRet, double *rRet
                     ){
    const double relToll(solParams->relToll);
    const double absToll(solParams->absToll);
    const int    maxIter(200*solParams->maxIter);
    double M_s = r - rTarget;
    double M_p = z - zTarget;
    double mag = sqrt(M_p*M_p + M_s*M_s);
    M_s /= mag;
    M_p /= mag;
    int n = 0;
    double sig_p = z;
    double sig_s = r;
    double x1 = 0;
    double x2 = mag;
    double xh = x1; // # a return of 0 results in f(sigma)> 0
    double xl = x2; // # returning to kappa,0 is f(sigma) < 0
    double f,df_dx,df_dr,df_dz,rts,dx,dxold;
    double fl,fh;
    // Solution using Newton's method with a fall back to bisection if
    // the guess is out of bounds or the method is not converging fast
    // enough.
    fl = calcYieldFunc2D(r-(M_s*mag), z-(M_p*mag), epsv0, gam0, matParams, &df_dr, &df_dz);
    fh = calcYieldFunc2D(r, z, epsv0, gam0, matParams, &df_dr, &df_dz);
    if (fh <= 0.0){
      for (int i=1; i<10; i++){
        x1 = -pow(2.0,i)*mag;
        fh = calcYieldFunc2D(r-(x1*M_s), z-(x1*M_p), epsv0, gam0, matParams, &df_dr, &df_dz);
        if (fh > 0.0){
          xh=x1;
          break;
        }
      }
    }
    if(!(fl < 0.0 && fh > 0.0) ){
      assert(fl < 0.0 && fh>0);
    }
    rts = 0.5*(x1 + x2);
    dxold = std::abs(x1-x2);
    dx = dxold;
    sig_p = z - rts * M_p;
    sig_s = r - rts * M_s;
    f = calcYieldFunc2D(sig_s, sig_p, epsv0, gam0, matParams, &df_dr, &df_dz);
    df_dx = (df_dr*M_s) + (df_dz*M_p);
    for (;n<maxIter;++n){
      if ( ( ((rts-xh)*df_dx-f) * ((rts-xl)*df_dx-f) > 0) || // ensure that the root remains bounded
           (std::abs(2.0*f) > std::abs(dxold * df_dx)) ){    // Make sure that it is converging fast enough
        dxold = dx;
        dx = 0.5 * (xh - xl);
        rts = xl + dx;
      } else{
        dxold = dx;
        dx = f/df_dx;
        rts -= dx;
      }
      sig_p = z - rts * M_p;
      sig_s = r - rts * M_s;
      if( std::abs(dx) < absToll or std::abs(dx/mag) < relToll){ // Convergence check
        break;
      }
      f = calcYieldFunc2D(sig_s, sig_p, epsv0, gam0, matParams, &df_dr, &df_dz);
      if(std::abs(f) < absToll){
        break;
      }
      df_dx = (df_dr*M_s) + (df_dz*M_p);
      if(f<0){
        xl = rts;
      }else{
        xh = rts;
      }
    }
    *zRet = sig_p;
    *rRet = sig_s;
    return n;
  }

  int doOuterReturn2D(const double r_tr, const double z_tr, const double epsv0, const double gam0,
                      const matParam *matParams, const solParam *solParams,
                      double *rRet, double *zRet, double *epsv, double *gam
                      ){
    const double bulkMod(matParams->bulkMod);
    const double shearMod(matParams->shearMod);
    const double relToll(solParams->relToll);
    const int maxIter(solParams->maxIter);

    int n = 0;
    double sig_f_r(r_tr), sig_f_z(z_tr);
    double Mr,Mz,Fr,Fz;
    double f_tr = calcYieldFunc2D(sig_f_r, sig_f_z, epsv0, gam0, matParams,
                                  &Fr, &Fz);
    if(f_tr<0){
      *rRet = r_tr;
      *zRet = z_tr;
      *epsv = epsv0;
      *gam  = gam0;
      return 0;
    }
    double dFf_dz,dFf_dgam;
    if(computeFf(z_tr,gam0,matParams, &dFf_dz, &dFf_dgam) <= 0.0){
      double zVert;
      if(doVertexTest(z_tr,r_tr,epsv0,gam0,matParams,solParams,&zVert)){
        *rRet = 0.0;
        *zRet = zVert;
        *epsv = epsv0 + (z_tr - zVert)/(3.0*bulkMod);
        *gam  = gam0  + (r_tr)/(2.0*shearMod);
        return -1;
      }
    }
    double fTest=f_tr;
    for(n=0; n<maxIter; ++n){
      calcFlowDir2D(sig_f_r, sig_f_z, epsv0, gam0, matParams,
                    &Mr, &Mz);
      double eta=2.0*shearMod/(3.0*bulkMod);
      double Az = Mz;
      double Ar = eta*Mr;
      double scaleFactor = sqrt( (sig_f_r*sig_f_r + sig_f_z*sig_f_z)/
                                 (Az*Az+Ar*Ar)
                                 );
      Az *= scaleFactor;
      Ar *= scaleFactor;
      double beta_next = -fTest/(Fr*Ar+Fz*Az);
      sig_f_r += beta_next*Ar;
      sig_f_z += beta_next*Az;
      if ( std::fabs(beta_next) < relToll){
        // Quality check:
        double delR = r_tr-sig_f_r;
        double delZ = z_tr-sig_f_z;
        if( (delR*Fr + delZ*Fz) > 0.0 ) {
          *epsv = epsv0 + (z_tr-sig_f_z)/(3.0*bulkMod);
          *gam  = gam0  + (r_tr-sig_f_r)/(2.0*shearMod);
          *rRet = sig_f_r;
          *zRet = sig_f_z;
          return n;
        } else {
          double Fmag(sqrt(Fz*Fz+Fr*Fr));
          double Nr(Fr/Fmag), Nz(Fz/Fmag);
          double delSigMag = std::min( sqrt(delR*delR + delZ*delZ),
                                       std::abs(Nz*sig_f_z+ Nr*sig_f_r)
                                       );
          const double shiftMag = relToll;
          double zTarget = sig_f_z-Nz*shiftMag*delSigMag;
          double rTarget = sig_f_r-Nr*shiftMag*delSigMag;
          doLineSearch2D(z_tr, r_tr, zTarget, rTarget,
                         epsv0, gam0, matParams, solParams, &sig_f_z, &sig_f_r);
        }
      } 
      fTest = calcYieldFunc2D(sig_f_r, sig_f_z, epsv0, gam0, matParams,
                              &Fr, &Fz);
    }
    doInnerReturn2D(sig_f_z,sig_f_r, epsv0, gam0, matParams, solParams,
                    &sig_f_z, &sig_f_r);
    *epsv = epsv0 + (z_tr-sig_f_z)/(3.0*bulkMod);
    *gam  = gam0  + (r_tr-sig_f_r)/(2.0*shearMod);
    *rRet = sig_f_r;
    *zRet = sig_f_z;
    return n;
  }

  double calcGamma(const double theta, const matParam *matParams,
                   double *dg_dTheta){
    double psi = matParams->psi;
    GammaForm J3Type = matParams->J3Type;
    double theta_bar = -theta;
    double Gamma(1.0);
    switch (J3Type) {
    case (DruckerPrager):
      *dg_dTheta = 0.0;
      Gamma = 1.0;
      break;
    case (Gudehus) :
      // 7/9 < psi < 9/7
      Gamma = 0.5* (1+sin(3.0*theta_bar) + 1.0/psi * (1-sin(3.0*theta_bar)));
      *dg_dTheta = -1.5 * cos(3.0*theta_bar) * (1.0 - 1.0/psi);
      break;
    case (WilliamWarnke):
      // 0.5 < psi < 2.0
      const double pi = 3.141592653589793;
      double alphaStar = pi/6.0 + theta_bar;
      double A1 = 1.0-psi*psi;
      double A2 = (2.0*psi-1.0);
      double A3 = 5.0*psi*psi - 4.0 * psi;
      double B  = cos(alphaStar);
      double C  = (4.0*A1*B*B + A2*A2);
      double D  = 2.0*A1*B + A2*sqrt(4.0*A1*B*B + A3);
      Gamma = C / D;
      double dC_dB = 8.0*A1*B;
      double dD_dB = 2*A1+(A2/(2.0 * sqrt(4.0 * A1 * B*B + A3)))*(8.0*A1*B);
      double dg_dB = -C*dD_dB/(D*D) + dC_dB/D;
      double dB_dalphaStar = -sin(alphaStar);
      double dalphaStar_dtheta = -1.0;
      *dg_dTheta = dg_dB * dB_dalphaStar * dalphaStar_dtheta;
      break;
    }
    return Gamma;
  }

  double calcYieldFunc(const double r, const double z, const double theta,
                       const double epsv0, const double gam0,
                       const matParam *matParams,
                       double *df_dr,
                       double *df_dz,
                       double *df_dtheta_rinv // df/dtheta * 1/r
                       ){
    double gamma,dg_dtheta,f,df_dgr;
    gamma=calcGamma(theta,matParams, &dg_dtheta);
    f         = calcYieldFunc2D(r*gamma, z, epsv0, gam0, matParams, &df_dgr, df_dz);
    *df_dr     = df_dgr*gamma;
    *df_dtheta_rinv = df_dgr*dg_dtheta;
    return f;
  }

  double calcYieldFunc(const SymMatrix3 *sigma,
                       const double epsv0, const double gam0,
                       const matParam *matParams){
    double r,z,theta,df_dr,df_dz,df_dtheta_rinv;
    SymMatrix3 er,ez,etheta;
    sigma->calcRZTheta(&r,&z,&theta,&er,&ez,&etheta);
    return calcYieldFunc(r,z,theta,epsv0,gam0,matParams,&df_dr,&df_dz,&df_dtheta_rinv);
  }
  
  void calcFlowDir(const double r, const double z, const double theta,
                   const double epsv0, const double gam0,
                   const matParam *matParams,
                   double *Mr,
                   double *Mz,
                   double *Mtheta_rinv
                   ){
    // This flow function should only be evaluated when the yield funciton is 0
    double gamma,dg_dtheta,df_dgr;
    gamma=calcGamma(theta,matParams,&dg_dtheta);
    calcFlowDir2D(gamma*r, z, epsv0, gam0, matParams, &df_dgr, Mz);
    *Mr = df_dgr*gamma;
    *Mtheta_rinv = df_dgr * dg_dtheta;
    double mMag = sqrt((*Mr)*(*Mr) + (*Mz)*(*Mz) + (*Mtheta_rinv) * (*Mtheta_rinv));
    *Mr /= mMag;
    *Mz /= mMag;
    *Mtheta_rinv /= mMag;
  }
  void doInnerReturn(const SymMatrix3 *sigma, const double epsv0, const double gam0,
                     const matParam *matParams, const solParam *solParams,
                     SymMatrix3 *sigmaRet
                     ){
    SymMatrix3 er,ez,etheta;
    double rtr,ztr,thetatr,rret,zret;
    sigma->calcRZTheta(&rtr, &ztr, &thetatr, &er, &ez, &etheta);
    double dg_dtheta;
    double gamma = calcGamma(thetatr,matParams,&dg_dtheta);
    doInnerReturn2D(ztr, gamma*rtr, epsv0, gam0, matParams, solParams, &zret, &rret);
    rret /= gamma;
    *sigmaRet = ez*zret + er*rret;
  }

  // Returns the number of iterations used in the loop:
  int doReturnOuter(const SymMatrix3 *sigma_tr, const double epsv0, const double gam0,
                    const matParam *matParams,
                    const solParam *solParams,
                    SymMatrix3 *sigmaRet,
                    double *epsv,
                    double *gam
                    ){
    const double bulkMod(matParams->bulkMod);
    const double shearMod(matParams->shearMod);
    const double relToll(solParams->relToll);
    const double absToll(solParams->absToll);
    const int maxIter(solParams->maxIter);

    const double p2(matParams->p2);
    const double p3(matParams->p3);
    const double epsVmin(p2>0 ? -(p3)/p2 : -1e12);

    double z_tr,r_tr,theta_tr;
    SymMatrix3 ez,er,etheta;
    sigma_tr->calcRZTheta(&r_tr, &z_tr, &theta_tr, &er, &ez, &etheta);
    {
      double df_dr,df_dz,df_dtheta_rinv;
      double f_tr = calcYieldFunc(r_tr, z_tr, theta_tr, epsv0, gam0, matParams,
                                  &df_dr, &df_dz, &df_dtheta_rinv);
      if(f_tr <= 0.0){
        *sigmaRet = *sigma_tr;
        *epsv     = epsv0;
        *gam      = gam0;
        return 0;
      }
    }
    if(z_tr>0){
      double a;
      double gamma=calcGamma(theta_tr, matParams, &a);
      double zVert;
      if( doVertexTest(z_tr, gamma*r_tr, epsv0, gam0, matParams, solParams, &zVert) ){
        // return the stress to the vertex:
        double sigma_m=zVert/sqrt(3.0);
        sigmaRet->set(0,sigma_m);
        sigmaRet->set(1,sigma_m);
        sigmaRet->set(2,sigma_m);
        sigmaRet->set(3,0.0);
        sigmaRet->set(4,0.0);
        sigmaRet->set(5,0.0);
        *epsv = epsv0 + (z_tr-zVert)/(sqrt(3.0)*bulkMod);
        *gam  = gam0  + r_tr/(2.0*shearMod);
        return -1;
      }
    }

    // If the trial stress is outside of the bounding DP yield surface, then first return to
    // the DP surface, then do the 6D projection. This is more robust
    SymMatrix3 sig_e(*sigma_tr),sig_f(*sigma_tr), sig_p(*sigma_tr),
      F(0.0), M(0.0), P(0.0);
    int nOuter(0);
    int n(0);
    double gamma_n(0.0),gamma_np1(0.0);
    SymMatrix3 Fold, Mold,Pold, sig_f_old;
    for(;n<(maxIter+1);++n){
      Fold = F;
      Mold = M;
      Pold = P;
      sig_f_old = sig_f;
      doInnerReturn(&sig_p, epsv0, gam0, matParams, solParams, &sig_f);
      double r,z,theta;
      sig_f.calcRZTheta(&r, &z, &theta, &er, &ez, &etheta);
      {                         // Calculate M
        double Mr,Mz,Mtheta_rinv;
        calcFlowDir(r, z, theta, epsv0, gam0, matParams, &Mr, &Mz, &Mtheta_rinv);
        M = er*Mr    + ez*Mz    + etheta*Mtheta_rinv;
        M /= M.norm();
      }
      if(n>0 && Mold.Contract(M) < 0){
        // The previous iteration placed sig_p beyond a corner, put the stress at the corner
        SymMatrix3 t1 = sig_p-sig_f_old;
        double t1Mag    = t1.norm();
        t1 /= t1Mag;
        t1Mag = ( M.Contract(sig_f) - M.Contract(sig_f_old) ) / M.Contract(t1);
        sig_p = sig_f_old + t1*t1Mag; // This is an improved tangent projection stress, but may not satidfy f=0
        doInnerReturn(&sig_p, epsv0, gam0, matParams, solParams, &sig_f);
        // Return sig_f as the converged stress.
        SymMatrix3 stressInc = *sigma_tr - sig_f;
        double sig_h_inc = stressInc.trace()/3.0;
        *epsv = epsv0 + sig_h_inc/bulkMod;
        if( !(*epsv >= epsVmin) ){
          SymMatrix3 identity(true);
          *epsv = epsVmin;
          sig_h_inc = (epsVmin - epsv0)/bulkMod;
          sig_f = sig_f.deviatoric() + (sigma_tr->isotropic()-identity*sig_h_inc);
          stressInc = stressInc.deviatoric() + identity*sig_h_inc;
        }
        SymMatrix3 devStrainInc = stressInc.deviatoric()/(2.0*shearMod);
        *gam = gam0 + devStrainInc.norm();
        *sigmaRet = sig_f;
        return n;
      }
      {                         // Calculate F
        double df_dr,df_dz,df_dtheta_rinv;
        calcYieldFunc(r, z, theta, epsv0, gam0, matParams, &df_dr, &df_dz, &df_dtheta_rinv);
        F = er*df_dr + ez*df_dz + etheta*df_dtheta_rinv;
      }
      P = M.isotropic()*(3.0*bulkMod) + M.deviatoric()*(2.0*shearMod);
      gamma_np1 = F.Contract(sig_e-sig_f)/F.Contract(P);
      sig_p     = sig_e - P*gamma_np1;
      if ( std::abs( (gamma_np1-gamma_n)/gamma_np1 ) < relToll ||
           std::abs( gamma_np1-gamma_n ) < absToll ){
        // place the stress exactly on the yield surface one last time:
        doInnerReturn(&sig_p,epsv0,gam0,matParams,solParams,&sig_f);
        SymMatrix3 stressInc = *sigma_tr - sig_f;
        double sig_h_inc = stressInc.trace()/3.0;
        *epsv = epsv0 + sig_h_inc/bulkMod;
        if( !(*epsv >= epsVmin) ){
          SymMatrix3 identity(true);
          *epsv = epsVmin;
          sig_h_inc = (epsVmin - epsv0)/bulkMod;
          sig_f = sig_f.deviatoric() + (sigma_tr->isotropic()-identity*sig_h_inc);
          stressInc = stressInc.deviatoric() + identity*sig_h_inc;
        }
        SymMatrix3 devStrainInc = stressInc.deviatoric()/(2.0*shearMod);
        *gam = gam0 + devStrainInc.norm();
        *sigmaRet = sig_f;
        return std::max(n,nOuter);
      } else if (z_tr > 0.0 && sig_p.trace()>0.0){
        // Do a vertex check:
        double zVert,r_p,z_p,theta_p;
        SymMatrix3 er_p,ez_p,etheta_p;
        sig_p.calcRZTheta(&r_p, &z_p, &theta_p, &er_p, &ez_p, &etheta_p);
        double dg_dtheta;
        double gamma = calcGamma(theta_p, matParams, &dg_dtheta);
        if( doVertexTest(z_p, gamma*r_p, epsv0, gam0, matParams, solParams, &zVert) ){
          // return the stress to the vertex:
          double sigma_m=zVert/sqrt(3.0);
          sigmaRet->set(0,sigma_m);
          sigmaRet->set(1,sigma_m);
          sigmaRet->set(2,sigma_m);
          sigmaRet->set(3,0.0);
          sigmaRet->set(4,0.0);
          sigmaRet->set(5,0.0);
          *epsv = epsv0 + (z_tr-zVert)/(sqrt(3.0)*bulkMod);
          *gam  = gam0  + r_tr/(2.0*shearMod);
          return -(std::max(n,nOuter));
        }
      }
      gamma_n = gamma_np1;
    } // end of iteration loop
    { // Loop exited without convergence, put the return stress in a reasonable place:
      SymMatrix3 stressInc = *sigma_tr - sig_f;
      double sig_h_inc = stressInc.trace()/3.0;
      *epsv = epsv0 + sig_h_inc/bulkMod;
      if( !(*epsv >= epsVmin) ){
        SymMatrix3 identity(true);
        *epsv = epsVmin;
        sig_h_inc = (epsVmin - epsv0)/bulkMod;
        sig_f = sig_f.deviatoric() + (sigma_tr->isotropic()-identity*sig_h_inc);
        stressInc = stressInc.deviatoric() + identity*sig_h_inc;
      }
      SymMatrix3 devStrainInc = stressInc.deviatoric()/(2.0*shearMod);
      *gam = gam0 + devStrainInc.norm();
      *sigmaRet = sig_f;
    }
    return std::max(n,nOuter);
  }

  int doReturnStressWithHardening(const SymMatrix3 *sigma_tr, const double epsv0, const double gam0,
                                  const matParam *matParams, const solParam *solParams,
                                  SymMatrix3 *sigmaRet,
                                  double *epsv,
                                  double *gam
                                  ){
    SymMatrix3 sig_p(0.0);
    const double bulkMod(matParams->bulkMod);
    const double shearMod(matParams->shearMod);

    int n = doReturnOuter(sigma_tr, epsv0, gam0, matParams, solParams, &sig_p, epsv, gam);
    
    if(n<=0){
      // The loading step was either elastic or the stress returned to the
      // hydrostatic tension vertex, the softening calculation will not behave well.
      *sigmaRet = sig_p;
      return n;
    }
    
    double r_p,z_p,theta_p,df_dr,df_dz,df_dtheta_rinv;
    SymMatrix3 er_p,ez_p,etheta_p,dSigma(*sigma_tr-sig_p),F,N,M,P;
    sig_p.calcRZTheta(&r_p, &z_p, &theta_p, &er_p, &ez_p, &etheta_p);
    calcYieldFunc(r_p,z_p,theta_p,epsv0,gam0,matParams, &df_dr, &df_dz, &df_dtheta_rinv);
    F = er_p*df_dr + ez_p*df_dz + etheta_p*df_dtheta_rinv;
    if (F.norm() > solParams->relToll){
      N = F/F.norm();
    } else {
      N = ez_p;
    }
    double Mr, Mz, Mtheta_rinv;
    calcFlowDir(r_p, z_p, theta_p, epsv0, gam0, matParams, &Mr, &Mz, &Mtheta_rinv);
    M = er_p*Mr    + ez_p*Mz    + etheta_p*Mtheta_rinv;
    M /= M.norm();
    P = M.isotropic()*(3.0*bulkMod) + M.deviatoric()*(2.0*shearMod);
    double H = computeEnsambleHardeningModulus(&M, &F, z_p,epsv0,gam0,matParams);
    if( std::abs(H)<solParams->absToll ){
      *sigmaRet = sig_p;
      return n;
    }
    double PdotN = P.Contract(N);
    if(PdotN + H <= 0){
      double epsv1(*epsv),gam1(*gam);
      SymMatrix3 sig_p2(0.0);
      int n2 = doReturnOuter(&sig_p, epsv1, gam1, matParams, solParams, &sig_p2, epsv, gam);
      *sigmaRet = sig_p2;
      return (-(100+n+floor(std::abs((double)n2))));
    }
    double GammaStar = N.Contract(dSigma)/PdotN;
    double Gamma     = PdotN/(PdotN+H)*GammaStar;
    double dgam_dlambda  = (M.deviatoric()).norm();
    double depsV_dlambda = M.trace();
    double dgam          = dgam_dlambda * Gamma;
    double depsV         = depsV_dlambda * Gamma;
    double dgam_max      = r_p/(2.0*shearMod);
    if (dgam > dgam_max){
      double epsv1(*epsv),gam1(*gam);
      SymMatrix3 sig_p2(0.0);
      int n2 = doReturnOuter(&sig_p,epsv1,gam1,matParams,solParams,&sig_p2,epsv,gam);
      *sigmaRet = sig_p2;
      return (-(100+n+floor(std::abs((double)n2))));
    }
    sig_p = *sigma_tr - P*Gamma;
    *sigmaRet = sig_p;
    *epsv = epsv0 + depsV;
    *gam  = gam0 + dgam;
    return n;
  }

  double computeEnsambleHardeningModulus(const SymMatrix3 *M, const SymMatrix3 *F, const double z,
                                         const double epsv0, const double gam0,
                                         const matParam *matParams){
    double dgam_dlambda  = (M->deviatoric()).norm();
    double depsV_dlambda = M->trace();

    double Ff,dFf_dz,dFf_dgam;
    Ff = computeFf(z, gam0,  matParams, &dFf_dz, &dFf_dgam);
    double Fc,dFc_dz,dFc_depsv;
    Fc = computeFc(z, epsv0, matParams, &dFc_dz, &dFc_depsv);

    double H = -(-2*Ff*Fc * dFf_dgam * dgam_dlambda - Ff*Ff * dFc_depsv*depsV_dlambda) / F->norm();
    return H;
  }
                                      
  int advanceTime(const double delT,
                  const SymMatrix3 *D,
                  const SymMatrix3 *sigma_0,
                  const double epsv_0,
                  const double gam_0,
                  const matParam *matParams,
                  const solParam *solParams,
                  SymMatrix3 *sigma_1,
                  double *epsv_1,
                  double *gam_1
                  ){
    const double bulkMod(matParams->bulkMod);
    const double shearMod(matParams->shearMod);
    SymMatrix3 sigma_tr((*sigma_0) +
                        ( D->isotropic()*(3.0*bulkMod) + D->deviatoric()*(2.0*shearMod) ) * delT
                        );
    // int n = doReturnStressWithHardening(&sigma_tr,epsv_0,gam_0,matParams,solParams,
    //                                     sigma_1,epsv_1,gam_1
    //                                     );
    int n = doReturnOuter(&sigma_tr,epsv_0,gam_0,matParams,solParams,
                                        sigma_1,epsv_1,gam_1
                                        );
    return n;
  }

  double integrateStressWithRefinement(const double delT,
                                     const SymMatrix3 *D,
                                     const SymMatrix3 *sigma_0,
                                     const double epsv_0,
                                     const double gam_0,
                                     const int steps,
                                     const int level,
                                     const matParam *matParams,
                                     const solParam *solParams,
                                     SymMatrix3 *sigma_1,
                                     double *epsv_1,
                                     double *gam_1
                                     ){
    SymMatrix3 sigma_n(*sigma_0);
    SymMatrix3 sigma_np1(*sigma_0);
    double epsV_n=epsv_0;
    double epsV_np1=epsv_0;
    double gamma_n=gam_0;
    double gamma_np1=gam_0;
    double plasWork = 0.0;
    for (int i=0; i<steps; ++i){
      int n = advanceTime(delT,D,&sigma_n,epsV_n,gamma_n,matParams,solParams, // inputs
                          &sigma_np1, &epsV_np1, &gamma_np1);
      double incPlasWork = 0.0;
      if(n != 0){
        const double bulkMod(matParams->bulkMod);
        const double shearMod(matParams->shearMod);
        SymMatrix3 delSigma = sigma_np1 - sigma_n;
        SymMatrix3 delEpsE  = delSigma.isotropic()/(3.0*bulkMod) + delSigma.deviatoric()/(2.0*shearMod);
        SymMatrix3 delEpsP  = (*D)*delT - delEpsE;
        incPlasWork = (sigma_np1 + sigma_n).Contract(delEpsP)*0.5;
        if(incPlasWork < 0){
          incPlasWork = sigma_np1.Contract(delEpsP);
        }
        if(incPlasWork < 0){
          incPlasWork = sigma_n.Contract(delEpsP);
        }
      }
      if ( (!(incPlasWork>=0 && std::abs((double)n)<solParams->maxIter))
           && level<solParams->maxLevels){
        // refine the timestep by a factor of 10
        incPlasWork = integrateStressWithRefinement(0.1*delT,D,&sigma_n,epsV_n,gamma_n, 10, level+1,
                                      matParams, solParams,
                                      &sigma_np1, &epsV_np1, &gamma_np1);
        sigma_n = sigma_np1;
        epsV_n  = epsV_np1;
        gamma_n = gamma_np1;
      } else {
        sigma_n = sigma_np1;
        epsV_n  = epsV_np1;
        gamma_n = gamma_np1;
      }
      plasWork += incPlasWork;
    }
    *sigma_1 = sigma_np1;
    *epsv_1  = epsV_np1;
    *gam_1 = gamma_np1;
    return plasWork;
  }

  double integrateRateDependent(const double delT,
                              const SymMatrix3 *D,
                              const SymMatrix3 *sigma_0,
                              const SymMatrix3 *sigma_qs0,
                              const double epsv_0,
                              const double gam_0,
                              const double epsv0_qs,
                              const double gam0_qs,
                              const matParam *matParams,
                              const solParam *solParams,
                              SymMatrix3 *sigma_1,
                              SymMatrix3 *sigma_qs1,
                              double *epsv_1,
                              double *gam_1,
                              double *epsv1_qs,
                              double *gam1_qs
                              ){
    // Advance the quasistatic solution without accounting for viscosity:
    // To Do: Make sure that sigma_qs0 is either on or inside the yield
    // surface (to accomidate advection errors)
    double incPlasWork(0.0);
    if ( !(matParams->relaxationTime > 0.0) ){
      incPlasWork = integrateStressWithRefinement(delT,D,sigma_0,epsv_0,gam_0,1,0,
                                    matParams,solParams,
                                    sigma_1,epsv_1,gam_1);
      *sigma_qs1 = SymMat3::SymMatrix3(false);
      *epsv1_qs  = 0.0;
      *gam1_qs   = 0.0;
    } else {
      const double bulkMod(matParams->bulkMod);
      const double shearMod(matParams->shearMod);
      SymMatrix3 sigma_h = *sigma_0 +
        ( D->isotropic()*(3.0*bulkMod) + D->deviatoric()*(2.0*shearMod) ) * delT;
      if ( calcYieldFunc(&sigma_h, epsv_0, gam_0, matParams) <= 0.0 ){
        // The trial stress is elastic
        *sigma_1   = sigma_h;
        *sigma_qs1 = sigma_h;
        *gam_1      = gam_0;
        *epsv_1     = epsv_0;
        *gam1_qs    = gam_0;
        *epsv1_qs   = epsv_0;
      } else {
        incPlasWork = integrateStressWithRefinement(delT,D,sigma_qs0,epsv0_qs,gam0_qs,1,0,
                                      matParams,solParams,
                                      sigma_qs1,epsv1_qs,gam1_qs);
        // Following equation 6.10 of the Kayenta user manual:
        double dt_tauInv = delT/matParams->relaxationTime;
        double exp_dt_tauInv = exp(-dt_tauInv);
        double Rh = (1.0-exp_dt_tauInv)/dt_tauInv;
        double rh = exp_dt_tauInv - Rh; // rh<0 in the Kayenta manual, reversed in Plasticity chapter.
        if(Rh>1.0){
          Rh=1.0;
          exp_dt_tauInv = 1.0;
          rh = 0.0;
        }
        *sigma_1 = (*sigma_qs1) + (sigma_h-(*sigma_qs1))*Rh + ((*sigma_0)-(*sigma_qs0))*rh;
        // include rh in the kinematic history variables b/c they respond as fast as stress
        *epsv_1  = *epsv1_qs + Rh*(epsv_0-(*epsv1_qs)) + rh*(epsv_0 - epsv0_qs);
        *gam_1   = *gam1_qs  + Rh*(gam_0-(*gam1_qs))   + rh*(gam_0 - gam0_qs);
      }
    }
    return incPlasWork;
  }

  // Utility translation functions:
  void unpackHistoryVariables(const double histVector[NUM_HIST_VAR],
                              SymMatrix3 *sigma_0,
                              SymMatrix3 *sigma_0qs,
                              double *epsv0,
                              double *gam0,
                              double *epsv0_qs,
                              double *gam0_qs
                              ){
    sigma_0->set(0,0,histVector[0]);
    sigma_0->set(1,1,histVector[1]);
    sigma_0->set(2,2,histVector[2]);
    sigma_0->set(1,2,histVector[3]);
    sigma_0->set(0,2,histVector[4]);
    sigma_0->set(0,1,histVector[5]);

    sigma_0qs->set(0,0,histVector[6]);
    sigma_0qs->set(1,1,histVector[7]);
    sigma_0qs->set(2,2,histVector[8]);
    sigma_0qs->set(1,2,histVector[9]);
    sigma_0qs->set(0,2,histVector[10]);
    sigma_0qs->set(0,1,histVector[11]);

    *epsv0 = histVector[12];
    *gam0  = histVector[13];

    *epsv0_qs = histVector[14];
    *gam0_qs  = histVector[15];
  }

  void packHistoryVariables(const SymMatrix3 *sigma,
                            const SymMatrix3 *sigma_qs,
                            const double epsv,
                            const double gam,
                            const double epsv_qs,
                            const double gam_qs,
                            double histVector[NUM_HIST_VAR]
                            ){
    histVector[0] = sigma->get(0,0);
    histVector[1] = sigma->get(1,1);
    histVector[2] = sigma->get(2,2);
    histVector[3] = sigma->get(1,2);
    histVector[4] = sigma->get(0,2);
    histVector[5] = sigma->get(0,1);

    histVector[6]  = sigma_qs->get(0,0);
    histVector[7]  = sigma_qs->get(1,1);
    histVector[8]  = sigma_qs->get(2,2);
    histVector[9]  = sigma_qs->get(1,2);
    histVector[10] = sigma_qs->get(0,2);
    histVector[11] = sigma_qs->get(0,1);

    histVector[12] = epsv;
    histVector[13] = gam;
    
    histVector[14] = epsv_qs;
    histVector[15] = gam_qs;
  }

  matParam unpackMaterialParameters(const float matParamVector[NUM_MAT_PARAM]
                                    )
  {
    matParam matParams;
    matParams.bulkMod        = matParamVector[0];
    matParams.shearMod       = matParamVector[1];
    
    matParams.m0             = matParamVector[2];
    matParams.m1             = matParamVector[3];
    matParams.m2             = matParamVector[4];
    
    matParams.p0             = matParamVector[5];
    matParams.p1             = matParamVector[6];
    matParams.p2             = matParamVector[7];
    matParams.p3             = matParamVector[8];
    matParams.p4             = matParamVector[9];
    
    matParams.a1             = matParamVector[10];
    matParams.a2             = matParamVector[11];
    matParams.a3             = matParamVector[12];
    
    matParams.beta           = matParamVector[13];
    matParams.psi            = matParamVector[14];
    int J3Type = matParamVector[15]>2.25 || matParamVector[15] < 0.5 ? 0 :
      int(floor(matParamVector[15]+0.5));
    switch (J3Type){
    case GFMS::DruckerPrager:
      matParams.psi = 1.0;
      matParams.J3Type = GFMS::DruckerPrager;
      break;
    case GFMS::Gudehus:
      if(matParams.psi < 7.0/9.0){
        matParams.psi = 7.0/9.0;
      }
      if(matParams.psi > 9.0/7.0){
        matParams.psi = 9.0/7.0;
      }
      matParams.J3Type = GFMS::Gudehus;
      break;
    case GFMS::WilliamWarnke:
      if(matParams.psi < 0.5){
        matParams.psi = 0.5;
      }
      if(matParams.psi > 2.0){
        matParams.psi = 2.0;
      }
      matParams.J3Type = GFMS::WilliamWarnke;
      break;
    default:
      matParams.J3Type = GFMS::DruckerPrager;
      break;
    }
    matParams.relaxationTime = matParamVector[16];
    return matParams;
  }
  
  solParam unpackSolutionParameters(const float solParamVector[NUM_SOL_PARAM]
                                    )
  {
    solParam solParams;
    solParams.absToll   = solParamVector[0];
    solParams.relToll   = solParamVector[1];
    solParams.maxIter   = int(floor(solParamVector[2]+0.5));
    solParams.maxLevels = int(floor(solParamVector[3]+0.5));
    // Basic error checking,
    // To do: Add output messages and warnings??
    if (solParams.maxIter < 1){
      solParams.maxIter = 1;
    }
    if (solParams.maxLevels < 0){
      solParams.maxLevels = 0;
    }
    return solParams;
  }
}
