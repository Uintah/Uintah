/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

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


#ifndef TNT_ARRAY2D_UTILS_H
#define TNT_ARRAY2D_UTILS_H

/*#include <cstdlib> */
/*#include <cassert> */

namespace TNT
{


template <class T>
std::ostream& operator<<(std::ostream &s, const Array2D<T> &A)
{
    int M=A.dim1();
    int N=A.dim2();

    s << M << " " << N << "\n";

    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            s << A[i][j] << " ";
        }
        s << "\n";
    }


    return s;
}

template <class T>
std::istream& operator>>(std::istream &s, Array2D<T> &A)
{

    int M, N;

    s >> M >> N;

	Array2D<T> B(M,N);

    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
        {
            s >>  B[i][j];
        }

	A = B;
    return s;
}


template <class T>
Array2D<T> operator+(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] + B[i][j];
		}
		return C;
	}
}

template <class T>
Array2D<T> operator-(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] - B[i][j];
		}
		return C;
	}
}


template <class T>
Array2D<T> operator*(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] * B[i][j];
		}
		return C;
	}
}




template <class T>
Array2D<T> operator/(const Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() != m ||  B.dim2() != n )
		return Array2D<T>();

	else
	{
		Array2D<T> C(m,n);

		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				C[i][j] = A[i][j] / B[i][j];
		}
		return C;
	}
}





template <class T>
Array2D<T>&  operator+=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] += B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator-=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] -= B[i][j];
		}
	}
	return A;
}



template <class T>
Array2D<T>&  operator*=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] *= B[i][j];
		}
	}
	return A;
}





template <class T>
Array2D<T>&  operator/=(Array2D<T> &A, const Array2D<T> &B)
{
	int m = A.dim1();
	int n = A.dim2();

	if (B.dim1() == m ||  B.dim2() == n )
	{
		for (int i=0; i<m; i++)
		{
			for (int j=0; j<n; j++)
				A[i][j] /= B[i][j];
		}
	}
	return A;
}

/**
    Matrix Multiply:  compute C = A*B, where C[i][j]
    is the dot-product of row i of A and column j of B.


    @param A an (m x n) array
    @param B an (n x k) array
    @return the (m x k) array A*B, or a null array (0x0)
        if the matrices are non-conformant (i.e. the number
        of columns of A are different than the number of rows of B.)


*/
template <class T>
Array2D<T> matmult(const Array2D<T> &A, const Array2D<T> &B)
{
    if (A.dim2() != B.dim1())
        return Array2D<T>();

    int M = A.dim1();
    int N = A.dim2();
    int K = B.dim2();

    Array2D<T> C(M,K);

    for (int i=0; i<M; i++)
        for (int j=0; j<K; j++)
        {
            T sum = 0;

            for (int k=0; k<N; k++)
                sum += A[i][k] * B [k][j];

            C[i][j] = sum;
        }

    return C;

}

} // namespace TNT

#endif
