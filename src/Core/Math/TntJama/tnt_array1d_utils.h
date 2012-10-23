/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
*
* Template Numerical Toolkit (TNT)
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/

#ifndef TNT_ARRAY1D_UTILS_H
#define TNT_ARRAY1D_UTILS_H

/*#include <cstdlib> */
/*#include <cassert> */

namespace TNT
{


template <class T>
std::ostream& operator<<(std::ostream &s, const Array1D<T> &A)
{
    int N=A.dim1();

#ifdef TNT_DEBUG
	s << "addr: " << (void *) &A[0] << "\n";
#endif
    s << N << "\n";
    for (int j=0; j<N; j++)
    {
       s << A[j] << "\n";
    }
    s << "\n";

    return s;
}

template <class T>
std::istream& operator>>(std::istream &s, Array1D<T> &A)
{
	int N;
	s >> N;

	Array1D<T> B(N);
	for (int i=0; i<N; i++)
		s >> B[i];
	A = B;
	return s;
}



template <class T>
Array1D<T> operator+(const Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() != n )
		return Array1D<T>();

	else
	{
		Array1D<T> C(n);

		for (int i=0; i<n; i++)
		{
			C[i] = A[i] + B[i];
		}
		return C;
	}
}



template <class T>
Array1D<T> operator-(const Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() != n )
		return Array1D<T>();

	else
	{
		Array1D<T> C(n);

		for (int i=0; i<n; i++)
		{
			C[i] = A[i] - B[i];
		}
		return C;
	}
}


template <class T>
Array1D<T> operator*(const Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() != n )
		return Array1D<T>();

	else
	{
		Array1D<T> C(n);

		for (int i=0; i<n; i++)
		{
			C[i] = A[i] * B[i];
		}
		return C;
	}
}


template <class T>
Array1D<T> operator/(const Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() != n )
		return Array1D<T>();

	else
	{
		Array1D<T> C(n);

		for (int i=0; i<n; i++)
		{
			C[i] = A[i] / B[i];
		}
		return C;
	}
}









template <class T>
Array1D<T>&  operator+=(Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() == n)
	{
		for (int i=0; i<n; i++)
		{
				A[i] += B[i];
		}
	}
	return A;
}




template <class T>
Array1D<T>&  operator-=(Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() == n)
	{
		for (int i=0; i<n; i++)
		{
				A[i] -= B[i];
		}
	}
	return A;
}



template <class T>
Array1D<T>&  operator*=(Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() == n)
	{
		for (int i=0; i<n; i++)
		{
				A[i] *= B[i];
		}
	}
	return A;
}




template <class T>
Array1D<T>&  operator/=(Array1D<T> &A, const Array1D<T> &B)
{
	int n = A.dim1();

	if (B.dim1() == n)
	{
		for (int i=0; i<n; i++)
		{
				A[i] /= B[i];
		}
	}
	return A;
}






} // namespace TNT

#endif
