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
 * Vector3.h -
 *
 * This file is based on a file by a similar name in the Uintah 
 * project (uintah.utah.edu). 
 * 
 */

#ifndef VECTOR3_H
#define VECTOR3_H

class Vector3
{
	public:
		Vector3();
		Vector3(const double val);
		Vector3(const double v1, const double v2, const double v3);
		virtual ~Vector3();

        /* throws std::out_of_range if idx <0 || idx>2 */
		double& operator[] (const int idx);
		double  operator[] (const int idx) const;
		
		inline void operator -= (const Vector3 rhs){
			_values[0] -= rhs._values[0];
			_values[1] -= rhs._values[1];
			_values[2] -= rhs._values[2];
		}
		
		inline Vector3 operator -(const Vector3 rhs){
			Vector3 ans(*this);
			ans -= rhs;
			return ans;
			
		}
		
		inline void operator += (const Vector3 rhs){
			_values[0] += rhs._values[0];
			_values[1] += rhs._values[1];
			_values[2] += rhs._values[2];
		}
		
		inline Vector3 operator+ (const Vector3 rhs){
			Vector3 ans(*this);
			ans += rhs;
			return ans;
			
		}
		
		inline void operator *= (const double rhs){
			_values[0] *= rhs;
			_values[1] *= rhs;
			_values[2] *= rhs;
		}
		
		inline Vector3 operator * (const double rhs){
			Vector3 ans(*this);
			ans *= rhs;
			return ans;
		}
		
		inline double x() const {return _values[0];}
		inline double y() const {return _values[1];}
		inline double z() const {return _values[2];}

		double min() const;

	private:
		double _values[3];
		/* add your private declarations */
};

#endif /* VECTOR3_H */ 
