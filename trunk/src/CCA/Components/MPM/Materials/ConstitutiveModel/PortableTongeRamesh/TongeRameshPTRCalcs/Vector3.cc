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
 * Vector3.cc -
 *
 * This file is based on a file by a similar name in the Uintah 
 * project (uintah.utah.edu). 
 * 
 */


#include "Vector3.h"
#include <stdexcept>    // std::out_of_range

Vector3::Vector3()
{
        _values[0] = _values[1] = _values[2] = 0.0;
}

Vector3::Vector3(const double val){
        _values[0] = _values[1] = _values[2] = val;
}

Vector3::Vector3(const double v1, const double v2, const double v3){
        _values[0] = v1;
        _values[1] = v2;
        _values[2] = v3;
}


Vector3::~Vector3()
{
}

double& Vector3::operator[] (const int idx)
{
        if (idx < 0 || idx > 2)
        {
                throw std::out_of_range("Vector3::operator[]: Index must be in [0,2]");
        }
        return _values[idx];
}

double Vector3::operator[] (const int idx) const{
        if (idx < 0 || idx > 2)
        {
                throw std::out_of_range("Vector3::operator[]: Index must be in [0,2]");
        }
        return _values[idx];
}

double Vector3::min() const
{
        double ans = _values[0];
        if (_values[1] < ans)
        {
                ans = _values[1];
        }
        if (_values[2] < ans)
        {
                ans = _values[2];
        }
        return ans;
}
