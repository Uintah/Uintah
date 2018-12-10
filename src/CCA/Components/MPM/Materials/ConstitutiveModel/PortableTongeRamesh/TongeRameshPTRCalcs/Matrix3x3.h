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
 * Matrix3x3.h -
 *
 * This file is based on a file by a similar name in the Uintah 
 * project (uintah.utah.edu). 
 * 
 */


#ifndef MATRIX3X3_H
#define MATRIX3X3_H

#ifndef EQN_EPS
#define EQN_EPS 1e-12
#endif

#ifndef IsZero
#define IsZero( x ) ((x) > -EQN_EPS && (x) < EQN_EPS)
#endif

#include <ostream>
#include <cmath>                // sqrt

class Matrix3x3
{
        public:
                Matrix3x3();
                Matrix3x3(const double val);
                Matrix3x3(const bool isIdentity);
                Matrix3x3(      const double v0,
                                        const double v1,
                                        const double v2,
                                        const double v3,
                                        const double v4,
                                        const double v5,
                                        const double v6,
                                        const double v7,
                                        const double v8);

                virtual ~Matrix3x3();
                
                void identity();
                Matrix3x3 transpose() const;

                double determinant() const;
                double trace() const;
                double normSquared() const;
                inline double norm() const
                {
                        return sqrt(normSquared()); 
                }

                const Matrix3x3 inverse() const;
                double get(const int i, const int j) const;
                void set(const int i, const int j, const double val);
                
                void swap(Matrix3x3 *rhs);
                
                Matrix3x3 operator+= (const Matrix3x3 rhs);
                const Matrix3x3 operator+(const Matrix3x3 rhs) const;
                Matrix3x3 operator-= (const Matrix3x3 rhs);
                const Matrix3x3 operator-(const Matrix3x3 rhs) const;
                Matrix3x3 operator*= (const double rhs);
                const Matrix3x3 operator*(const double rhs) const;
                Matrix3x3 operator/= (const double rhs);
                const Matrix3x3 operator/(const double rhs) const;
                const Matrix3x3 operator*(const Matrix3x3 rhs) const;
                int getEigenValues(double *e1, double *e2, double *e3) const;
  
                inline double Contract(const Matrix3x3 mat) const
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

        void polarRotationRMB(Matrix3x3 *R) const;
        void polarDecompositionRMB(Matrix3x3 *U, Matrix3x3 *R) const;
        private:
                double _values[9];
                /* add your private declarations */
};

std::ostream& operator<<(std::ostream& out, const Matrix3x3 rhs);

int SolveLinear(double *c, double *s);
int SolveQuadratic(double *c, double *s);
int SolveCubic  (       double c[4],
                                        double s[3]      
                                );

#endif /* MATRIX3X3_H */ 
