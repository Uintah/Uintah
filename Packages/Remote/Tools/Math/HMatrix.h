/////////////////////////////////////////////////////////////
// Matrix.h -  Stuff for working with matrices large and small.
// by David K. McAllister, 1998.

#ifndef _matrix_h
#define _matrix_h

#include <Packages/Remote/Tools/Util/Assert.h>

#include <string.h>
#include <string>

namespace Remote {
using namespace std;
class HVector;

class Matrix
{
	int r, c;
	double *data;

public:
	inline Matrix()
	{
		r = c = 0;
		data = NULL;
	}

	inline Matrix(const int _c, const int _r)
	{
		if(_r * _c > 0)
			data = new double[_r * _c];
		else
			data = NULL;

		r = _r;
		c = _c;
	}

	inline Matrix(const Matrix &A)
	{
		if(A.data)
		{
			data = new double[A.c * A.r];
			memcpy(data, A.data, A.c * A.r * sizeof(double));
			c = A.c;
			r = A.r;
		}
		else
		{
			data = NULL;
			c = r = 0;
		}
	}

	inline ~Matrix()
	{
		if(data)
			delete [] data;
	}

	Matrix operator+(const Matrix &A) const
	{
		ASSERT((data != NULL));
		ASSERT(A.c == c	&& A.r == r);

		Matrix C(c, r);

		for(int i=0; i<c*r; i++)
		{
			C.data[i] = data[i] + A.data[i];
		}

		return C;
	}

	Matrix & operator+=(const Matrix &A)
	{
		ASSERT(data != NULL);
		ASSERT(A.c == c && A.r == r);

		for(int i=0; i<c*r; i++)
		{
			data[i] += A.data[i];
		}

		return *this;
	}

	Matrix & operator/=(const double d)
	{
		ASSERT(data != NULL);
		ASSERT(c > 0 && r > 0);

		for(int i=0; i<c*r; i++)
		{
			data[i] /= d;
		}

		return *this;
	}

	double & operator()(const int x, const int y)
	{
		ASSERT(data != NULL);
		ASSERT(x >= 0 && x < c);
		ASSERT(y >= 0 && y < r);

		return data[y*c+x];
	}
	
	inline Matrix & zero()
	{
		ASSERT(data != NULL);
		ASSERT(c > 0 && r > 0);
		
		memset(data, 0, c * r * sizeof(double));
		
		return *this;
	}
	
	inline string print() const
	{
		ASSERT(data != NULL);
		
		string s;
		char ii[15];
		
		for(int i=0; i<r; i++)
		{
			for(int j=0; j<c; j++)
			{
				s += gcvt(data[i*c+j], 15, ii);
				s += " ";
			}
			s += "\n";
		}
		
		return s;
	}
};

// Defined in HVector.cpp
ostream& operator<<(ostream& os, const Matrix& m);

} // End namespace Remote


#endif
