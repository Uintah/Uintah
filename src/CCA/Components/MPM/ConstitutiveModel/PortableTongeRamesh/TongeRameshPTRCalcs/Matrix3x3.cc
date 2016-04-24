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
 * 
 * Matrix3x3.cc -
 *
 * This file is based on a file by a similar name in the Uintah 
 * project (uintah.utah.edu). 
 * 
 */


#include "Matrix3x3.h"
#include <cstring> 		// memcpy
#include <cmath>		// fabs
#include <iomanip>		// std::setw
#include <stdexcept>
#include <sstream>					// std::cerr, std::cout, std::endl
#include <iostream>

Matrix3x3::Matrix3x3()
{
	_values[0] = 0.0; _values[1] = 0.0; _values[2] = 0.0;
	_values[3] = 0.0; _values[4] = 0.0; _values[5] = 0.0;
	_values[6] = 0.0; _values[7] = 0.0; _values[8] = 0.0;
}

Matrix3x3::Matrix3x3(const double val){
	for (int i = 0; i < 9; i++){
		_values[i] = val;
	}
}

Matrix3x3::Matrix3x3(const bool isIdentity){
	_values[0] = 0.0; _values[1] = 0.0; _values[2] = 0.0;
	_values[3] = 0.0; _values[4] = 0.0; _values[5] = 0.0;
	_values[6] = 0.0; _values[7] = 0.0; _values[8] = 0.0;

	if (isIdentity){
		_values[0] = 1.0;
		_values[4] = 1.0;
		_values[8] = 1.0;
	}
}

Matrix3x3::Matrix3x3(	const double v0,
						const double v1,
						const double v2,
						const double v3,
						const double v4,
						const double v5,
						const double v6,
						const double v7,
						const double v8)
{
	_values[0] = v0;
	_values[1] = v1;
	_values[2] = v2;
	_values[3] = v3;
	_values[4] = v4;
	_values[5] = v5;
	_values[6] = v6;
	_values[7] = v7;
	_values[8] = v8;
}

Matrix3x3::~Matrix3x3()
{
}

void Matrix3x3::identity()
{
	_values[0] = 1.0; _values[1] = 0.0; _values[2] = 0.0;
	_values[3] = 0.0; _values[4] = 1.0; _values[5] = 0.0;
	_values[6] = 0.0; _values[7] = 0.0; _values[8] = 1.0;
}

Matrix3x3 Matrix3x3::transpose() const
{
	Matrix3x3 ans(*this);
	ans._values[1] = _values[3];
	ans._values[2] = _values[6];
	ans._values[3] = _values[1];
	ans._values[5] = _values[7];
	ans._values[6] = _values[2];
	ans._values[7] = _values[5];
	return ans;
}

void Matrix3x3::swap(Matrix3x3 *rhs){
	double cpy[9];
	size_t n = 9 * sizeof(double);
	memcpy(cpy, rhs->_values, n);
	memcpy(rhs->_values, _values, n);
	memcpy(_values, cpy, n);
}

Matrix3x3 Matrix3x3::operator+= (const Matrix3x3 rhs)
{
	for (int i = 0; i < 9; i++){
		_values[i] += rhs._values[i];
	}
	
	return *this;
}

const Matrix3x3 Matrix3x3::operator+(const Matrix3x3 rhs) const
{
	Matrix3x3 ans(*this);
	ans += rhs;
	return ans;
}

Matrix3x3 Matrix3x3::operator-= (const Matrix3x3 rhs)
{
	for (int i = 0; i < 9; i++){
		_values[i] -= rhs._values[i];
	}
	
	return *this;
}

const Matrix3x3 Matrix3x3::operator-(const Matrix3x3 rhs) const
{
	Matrix3x3 ans(*this);
	ans -= rhs;
	return ans;
}

Matrix3x3 Matrix3x3::operator*= (const double rhs)
{
	for (int i = 0; i < 9; i++){
		_values[i] *= rhs;
	}
	
	return *this;
}

const Matrix3x3 Matrix3x3::operator*(const double rhs) const
{
	Matrix3x3 ans(*this);
	ans *= rhs;
	return ans;
}

Matrix3x3 Matrix3x3::operator/= (const double rhs)
{
	if (fabs(rhs) < 1.e-12){
		throw std::domain_error("Matrix3x3::operator/=: divide by 0");
	}
	for (int i = 0; i < 9; i++){
		_values[i] /= rhs;
	}
	
	return *this;
}

const Matrix3x3 Matrix3x3::operator/(const double rhs) const
{
	Matrix3x3 ans(*this);
	ans /= rhs;
	return ans;
}

const Matrix3x3 Matrix3x3::operator*(const Matrix3x3 rhs) const
{
	Matrix3x3 ans(0.0);
	int i, j, k;
	for (i = 0; i < 3; i++){
		for (j = 0; j < 3; j++){
			for (k = 0; k < 3; k++){
				ans._values[i*3 + j] += get(i, k) * rhs.get(k, j);
			}
		}
	}
	return ans;
}

double Matrix3x3::determinant() const
{
	double ans = (_values[0] * _values[4] * _values[8]);
	ans       += (_values[1] * _values[5] * _values[6]);
	ans       += (_values[2] * _values[3] * _values[7]);
	ans       -= (_values[2] * _values[4] * _values[6]);
	ans       -= (_values[1] * _values[3] * _values[8]);
	ans       -= (_values[0] * _values[5] * _values[7]);
	return ans;
}

const Matrix3x3 Matrix3x3::inverse() const
{
	//  0 1 2
	//  3 4 5
	//  6 7 8
	Matrix3x3 ans;
	double det = determinant();
	
	if (det == 0.0){
		throw std::domain_error("Matrix3x3::inverse: Matrix is singular");
	}
	
	ans._values[0] = ((_values[4] * _values[8]) - (_values[5] * _values[7]));
	ans._values[1] = ((_values[2] * _values[7]) - (_values[1] * _values[8]));
	ans._values[2] = ((_values[1] * _values[5]) - (_values[2] * _values[4]));
	ans._values[3] = ((_values[5] * _values[6]) - (_values[3] * _values[8]));
	ans._values[4] = ((_values[0] * _values[8]) - (_values[2] * _values[6]));
	ans._values[5] = ((_values[2] * _values[3]) - (_values[0] * _values[5]));
	ans._values[6] = ((_values[3] * _values[7]) - (_values[4] * _values[6]));
	ans._values[7] = ((_values[1] * _values[6]) - (_values[0] * _values[7]));
	ans._values[8] = ((_values[0] * _values[4]) - (_values[1] * _values[3]));
	
	ans *= (1/det);
	return ans;
}

double Matrix3x3::get(const int i, const int j) const{
	if (i < 0 || j < 0 || i > 2 || j > 2){
		throw std::out_of_range("Matrix3x3::get: both indices must be in range [0,2]");
	}
	
	int idx = 3*i + j;
	return _values[idx];
}

void Matrix3x3::set(const int i, const int j, const double val)
{
	if (i < 0 || j < 0 || i > 2 || j > 2){
		throw std::out_of_range("Matrix3x3::get: both indices must be in range [0,2]");
	}
	
	int idx = 3*i + j;
	_values[idx] = val;
	
}

double Matrix3x3::trace() const
{
	double ans;
	ans = _values[0] + _values[4] + _values[8];
	return ans;
}

double Matrix3x3::normSquared() const
{
	int i;
	double ans = 0;
	for (i = 0; i < 9; i++){
		ans += _values[i] * _values[i];
	}
	return ans;
}

std::ostream& operator<<(std::ostream& out, const Matrix3x3 rhs){
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

int Matrix3x3::getEigenValues(double *e1, double *e2, double *e3) const
{
  // eigen values will be roots of the following cubic polynomial
  // x^3 + b*x^2 + c*x + d
  double c[4];
  c[3] = 1;
  c[2] = -trace();

  c[1] = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = i+1; j < 3; j++)
      c[1] += get(i, i)*get(j, j) - get(i, j)*get(j,i);
  }
  
  c[0] = -determinant();
  double r[3];

  // num_values will be either 1, 2, or 3
  int num_values = SolveCubic(c, r);

  // sort the eigen values
  if (num_values == 3) {
    int greatest = 1;
    *e1 = r[0];    
    if (r[1] > *e1) {
      *e1 = r[1];
      greatest = 2;
    }
    if (r[2] > *e1) {
      *e1 = r[2];
      greatest = 3;
    }
    *e2 = (greatest != 2) ? r[1] : r[0];
    *e3 = (greatest != 3) ? r[2] : r[0];
    if (*e2 < *e3){
      std::swap(*e2, *e3);
    }
  }
  else if (num_values == 2)
    if (r[0] > r[1]) {
      *e1 = r[0];
      *e2 = r[1];
    } else {
      *e1 = r[1];
      *e2 = r[0];
    }
  else // num_values == 1
    *e1 = r[0];

  return num_values;
}

int SolveLinear(double *c, double *s) {
    if (c[ 1 ] < 0.0000001 && c[ 1 ] > -0.0000001)
         return 0;
    s[0] = -c[0]/c[1]; 
    return 1;
}

int SolveQuadratic(double *c, double *s) {
   double  A, B, C;
   if (c[ 2 ] < 0.0000001 && c[ 2 ] > -0.0000001)
      return SolveLinear(c, s);

   /* Ax^2 + Bx + C = 0 */
   A = c[2];
   B = c[1];
   C = c[0];
     
   double disc = B*B - 4*A*C;
   if (disc < 0) return 0;

   disc = sqrt(disc);
   s[0] = (-B - disc) / (2*A);
   s[1] = (-B + disc) / (2*A);
   return 2;
}

int SolveCubic 	( 	double c[4],
					double s[3]	 
				)
{
    int     i, num;
    double  sub;
    double  A, B, C;
    double  sq_A, p, q;
    double  cb_p, D;

    if (c[ 3 ] < 0.0000001 && c[ 3 ] > -0.0000001)
        return SolveQuadratic(c, s);

    /* normal form: x^3 + Ax^2 + Bx + C = 0 */

    A = c[ 2 ] / c[ 3 ];
    B = c[ 1 ] / c[ 3 ];
    C = c[ 0 ] / c[ 3 ];

    /*  substitute x = y - A/3 to eliminate quadric term:
        x^3 +px + q = 0 */

    sq_A = A * A;
    p = 1.0/3 * (- 1.0/3 * sq_A + B);
    q = 1.0/2 * (2.0/27 * A * sq_A - 1.0/3 * A * B + C);

    /* use Cardano's formula */

    cb_p = p * p * p;
    D = q * q + cb_p;

    if (IsZero(D))
    {
        if (IsZero(q)) /* one triple solution */
        {
            s[ 0 ] = 0;
            num = 1;
        }
        else /* one single and one double solution */
        {
            double u = cbrt(-q);
            s[ 0 ] = 2 * u;
            s[ 1 ] = - u;
            num = 2;
        }
    }
    else if (D < 0) /* Casus irreducibilis: three real solutions */
    {
        double phi = 1.0/3 * acos(-q / sqrt(-cb_p));
        double t = 2 * sqrt(-p);

        s[ 0 ] =   t * cos(phi);
        s[ 1 ] = - t * cos(phi + M_PI / 3);
        s[ 2 ] = - t * cos(phi - M_PI / 3);
        num = 3;
    }
    else /* one real solution */
    {
        double sqrt_D = sqrt(D);
        double u = cbrt(sqrt_D - q);
        double v = - cbrt(sqrt_D + q);

        s[ 0 ] = u + v;
        num = 1;
    }

    /* resubstitute */

    sub = 1.0/3 * A;

    for (i = 0; i < num; ++i)
        s[ i ] -= sub;

    return num;
}

void Matrix3x3::polarDecompositionRMB(Matrix3x3 *U,
                                    Matrix3x3 *R) const
{
  Matrix3x3 F = *this;

  // Get the rotation
  F.polarRotationRMB(R);

  // Stretch: U=R^T*F
  *U=R->transpose()*F;
}

void Matrix3x3::polarRotationRMB(Matrix3x3 *R) const
{

  //     PURPOSE: This routine computes the tensor [R] from the polar
  //     decompositon (a special case of singular value decomposition),
  //
  //            F = RU = VR
  //
  //     for a 3x3 invertible matrix F. Here, R is an orthogonal matrix
  //     and U and V are symmetric positive definite matrices.
  //
  //     This routine determines only [R].
  //     After calling this routine, you can obtain [U] or [V] by
  //       [U] = [R]^T [F]          and        [V] = [F] [R]^T
  //     where "^T" denotes the transpose.
  //
  //     This routine is limited to 3x3 matrices specifically to
  //     optimize its performance. You will notice below that all matrix
  //     operations are written out long-hand because explicit
  //     (no do-loop) programming avoids concerns about some compilers
  //     not optimizing loop and other array operations as well as they
  //     could for small arrays.
  //
  //     This routine returns a proper rotation if det[F]>0,
  //     or an improper orthogonal tensor if det[F]<0.
  //
  //     This routine uses an iterative algorithm, but the iterations are
  //     continued until the error is minimized relative to machine
  //     precision. Therefore, this algorithm should be as accurate as
  //     any purely analytical method. In fact, this algorithm has been
  //     demonstrated to be MORE accurate than analytical solutions because
  //     it is less vulnerable to round-off errors.
  //
  //     Reference for scaling method:
  //     Brannon, R.M. (2004) "Rotation and Reflection" Sandia National
  //     Laboratories Technical Report SAND2004-XXXX (in process).
  //
  //     Reference for fixed point iterator:
  //     Bjorck, A. and Bowie, C. (1971) "An iterative algorithm for
  //     computing the best estimate of an orthogonal matrix." SIAM J.
  //     Numer. Anal., vol 8, pp. 358-364.

  // input
  // -----
  //    F: the 3x3 matrix to be decomposed into the form F=RU
  //
  // output
  // -----
  //    R: polar rotation tensor (3x3 orthogonal matrix)

  Matrix3x3 F = *this;
  Matrix3x3 I;
  I.identity();
  double det = F.determinant();
  if ( det <= 0.0 ) {
    std::stringstream msg;
    msg << "Singular matrix in polar decomposition..." << "\n"
        << "F = " << F << "\n"
        << "det = " << det << "\n"
        << "  File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
    throw std::domain_error(msg.str());
  }

  //Step 1: Compute [C] = Transpose[F] . [F] (Save into E for now)
  Matrix3x3 E = F.transpose()*F;


  // Step 2: To guarantee convergence, scale [F] by multiplying it by
  //         Sqrt[3]/magnitude[F]. This is allowable because this routine
  //         finds ONLY the rotation tensor [R]. The rotation for any
  //         positive multiple of [F] is the same as the rotation for [F]
  //         Scaling [F] by a factor sqrt(3)/mag[F] requires replacing the
  //         previously computed [C] matrix by a factor 3/squareMag[F],
  //         where squareMag[F] is most efficiently computed by trace[C].
  //         Complete computation of [E]=(1/2)([C]-[I]) with [C] now
  //         being scaled.
  double S=3.0/E.trace();
  E=(E*S-I)*0.5;

  // Step 3: Replace S with Sqrt(S) and set the first guess for [R] equal
  //         to the scaled [F] matrix,   [A]=Sqrt[3]F/magnitude[F]

  S=sqrt(S);
  Matrix3x3 A=F*S;

  // Step 4. Compute error of this first guess.
  //     The matrix [A] equals the rotation if and only if [E] equals [0]
  double ERRZ = E.get(0,0)*E.get(0,0) + E.get(1,1)*E.get(1,1) + E.get(2,2)*E.get(2,2) 
    + 2.0*(E.get(0,1)*E.get(0,1) + E.get(1,2)*E.get(1,2) + E.get(2,0)*E.get(2,0));

  // Step 5.  Check if scaling ALONE was sufficient to get the rotation.
  //     This occurs whenever the stretch tensor is isotropic.
  //     A number X is zero to machine precision if (X+1.0)-1.0 evaluates
  //     to zero. Typically, machine precision is around 1.e-16.
  bool converged=false;

  if(ERRZ+1.0 == 1.0){
    converged=true;
  }

  Matrix3x3 X;
  int num_iters=0;
  while(converged==false){

    // Step 6: Improve the solution.
    //
    //     Compute a helper matrix, [X]=[A][I-E].
    //     Do *not* overwrite [A] at this point.
    X=A*(I-E);

    A=X;

    // Step 7: Using the improved solution compute the new
    //         error tensor, [E] = (1/2) (Transpose[A].[A]-[I])

    E = (A.transpose()*A-I)*.5;

    // Step 8: compute new error
    double ERR  = E.get(0,0)*E.get(0,0) + E.get(1,1)*E.get(1,1) + E.get(2,2)*E.get(2,2) 
      + 2.0*(E.get(0,1)*E.get(0,1) + E.get(1,2)*E.get(1,2) + E.get(2,0)*E.get(2,0));

    // Step 9:
    // If new error is smaller than old error, then keep on iterating.
    // If new error equals or exceeds old error, we have reached
    // machine precision accuracy.
    if(ERR>=ERRZ || ERR+1.0 == 1.0){
      converged = true;
    }
    double old_ERRZ=ERRZ;
    ERRZ=ERR;

    if(num_iters==200){
      std::stringstream msg;
      msg.precision(15);
      msg << "Matrix3x3::polarRotationRMB not converging with Matrix:" << "\n"
          << F << "\n"
          << "ERR = " << ERR << "\n"
          << "ERRZ = " << old_ERRZ << "\n"
          << "  File: " << __FILE__ << ", Line: " << __LINE__ << "\n";
      throw std::domain_error(msg.str());
    }
    num_iters++;
  }  // end while

  // Step 10:
  // Load converged rotation into R;
  *R=A;
}
