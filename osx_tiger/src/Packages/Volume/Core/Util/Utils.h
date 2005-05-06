//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Utils.h
//    Author : Milan Ikits
//    Date   : Fri Jul 16 02:24:37 2004

#ifndef Volume_Utils_h
#define Volume_Utils_h

#include <vector>

namespace Volume {

//---------------------------------------------------------------------------
// Power of 2 functions
//---------------------------------------------------------------------------

bool IsPowerOf2(uint n);
uint NextPowerOf2(uint n);
uint LargestPowerOf2(uint n);

//---------------------------------------------------------------------------
// Sorting functions
//---------------------------------------------------------------------------

// Wire-sort three numbers
template <class T> inline
void SortIndex(T v[3], int i[3])
{
  T v_tmp; int i_tmp;
  i[0] = 0; i[1] = 1; i[2] = 2;
  if(v[0] > v[1]) {
    v_tmp = v[0]; v[0] = v[1]; v[1] = v_tmp;
    i_tmp = i[0]; i[0] = i[1]; i[1] = i_tmp;
  }
  if(v[1] > v[2]) {
    v_tmp = v[1]; v[1] = v[2]; v[2] = v_tmp;
    i_tmp = i[1]; i[1] = i[2]; i[2] = i_tmp;
  }
  if(v[0] > v[1]) {
    v_tmp = v[0]; v[0] = v[1]; v[1] = v_tmp;
    i_tmp = i[0]; i[0] = i[1]; i[1] = i_tmp;
  }
}

// Bubble-sort in increasing order -- vector form
template <typename T, typename U> inline
void Sort(std::vector<T>& domain, std::vector<U>& range)
{
  for(unsigned int i=0; i<domain.size(); i++) {
    for(unsigned int j=i+1; j<domain.size(); j++) {
      if(domain[j] < domain[i]) {
        T domain_tmp = domain[i];
        domain[i] = domain[j];
        domain[j] = domain_tmp;
        U range_tmp = range[i];
        range[i] = range[j];
        range[j] = range_tmp;
      }
    }
  }
}

// Bubble-sort in increasing order -- pointer form
template <typename T, typename U> inline
void Sort(T* domain, U* range, int size)
{
  for(int i=0; i<size; i++) {
    for(int j=i+1; j<size; j++) {
      if(domain[j] < domain[i]) {
        T domain_tmp = domain[i];
        domain[i] = domain[j];
        domain[j] = domain_tmp;
        int range_tmp = range[i];
        range[i] = range[j];
        range[j] = range_tmp;
      }
    }
  }
}

// Bubble-sort in increasing order -- no sort index returned
template <class T> inline
void Sort(T* val, int n)
{
  for(int i=0; i<n; i++) {
    for(int j=i+1; j<n; j++) {
      if(val[j] < val[i]) {
        T val_tmp = val[i];
        val[i] = val[j];
        val[j] = val_tmp;
      }
    }
  }
}

//---------------------------------------------------------------------------
// Misc Utilities
//---------------------------------------------------------------------------

template <typename T> inline
T Clamp(T x, T l, T u)
{
  if (x < l) return l;
  if (x > u) return u;
  return x;
}

template <typename T> inline
int MinIndex(T x, T y, T z)
{
  return x < y ? (x < z ? 0 : 2) : (y < z ? 1 : 2);
}

} // namespace Volume

#endif // Volume_Utils_h

