//////////////////////////////////////////////////////////////////////
// HVector.h - interface for the HVector class.
// by David McAllister, 1998.

#ifndef _hvector_h
#define _hvector_h

#include <Packages/Remote/Tools/Util/Assert.h>

#include <Packages/Remote/Tools/Math/HMatrix.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

namespace Remote {
class HVector
{
	double *data;
	int r;
public:


	inline HVector()
	{
		data = NULL;
		r = 0;
	}

	inline HVector(const int _r)
	{
		if(_r > 0)
			data = new double[_r];
		else
			data = NULL;
		r = _r;
	}

	inline HVector(const HVector &A)
	{
		if(A.data)
		{
			data = new double[A.r];
			memcpy(data, A.data, A.r * sizeof(double));
			r = A.r;
		}
		else
		{
			data = NULL;
			r = 0;
		}
	}

	virtual inline ~HVector()
	{
		if(data)
			delete [] data;
	}

	inline double & operator[](const int i)
	{
		ASSERT(data != NULL);
		
		return data[i];
	}
	
	inline HVector & operator=(const HVector &A)
	{
		if(A.data && A.r)
		{
			if(A.r > r)
			{
				delete [] data;
				data = new double[A.r];
			}
			memcpy(data, A.data, A.r * sizeof(double));
			r = A.r;
		}
		else
		{
			data = NULL;
			r = 0;
		}
		
		return *this;
	}

	// Operators on doubles.

	inline HVector operator*(const double d) const
	{
		ASSERT(data != NULL);

		HVector b(r);

		for(int i=0; i<r; i++)
			b[i] = data[i] * d;

		return b;
	}

	inline HVector & operator*=(const double d)
	{
		ASSERT(data != NULL);

		for(int i=0; i<r; i++)
			data[i] = data[i] * d;

		return *this;
	}

	inline HVector & operator/=(const double d)
	{
		ASSERT(data != NULL);

		for(int i=0; i<r; i++)
			data[i] = data[i] / d;

		return *this;
	}

	inline HVector & operator+=(const double d)
	{
		ASSERT(data != NULL);

		for(int i=0; i<r; i++)
			data[i] = data[i] + d;

		return *this;
	}

	inline HVector & operator-=(const double d)
	{
		ASSERT(data != NULL);

		for(int i=0; i<r; i++)
			data[i] = data[i] - d;

		return *this;
	}

	// Add a constant to each element.
	inline HVector operator+(const double d) const
	{
		ASSERT(data != NULL);

		HVector b(r);

		for(int i=0; i<r; i++)
			b[i] = data[i] + d;

		return b;
	}

	// Operators on HVectors.

	inline HVector operator+(const HVector &A) const
	{
		ASSERT(data != NULL);
		ASSERT(A.data != NULL);
		ASSERT(A.r == r);

		HVector b(r);

		for(int i=0; i<r; i++)
			b[i] = data[i] + A.data[i];

		return b;
	}

	inline HVector & operator+=(const HVector &A)
	{
		ASSERT(data != NULL);
		ASSERT(A.data != NULL);
		ASSERT(A.r == r);

		for(int i=0; i<r; i++)
			data[i] += A.data[i];

		return *this;
	}

	inline HVector operator-(const HVector &A) const
	{
		ASSERT(data != NULL);
		ASSERT(A.data != NULL);
		ASSERT(A.r == r);

		HVector b(r);

		for(int i=0; i<r; i++)
			b[i] = data[i] - A.data[i];

		return b;
	}

	friend inline double Dot(const HVector &A, const HVector &B)
	{
		ASSERT(A.data != NULL);
		ASSERT(B.data != NULL);
		ASSERT(A.r == B.r);

		double dot;
		for(int i=0; i<A.r; i++)
			dot += A.data[i] * B.data[i];

		return dot;
	}

	// Swap the data in these two vectors.
	inline void swap(const HVector &A)
	{
		ASSERT(data != NULL);
		ASSERT(A.data != NULL);
		ASSERT(A.r == r);

		double tmp;
		for(int i=0; i<r; i++)
		  {
			tmp = data[i]; data[i] = A.data[i]; A.data[i] = tmp;
		  }
	}

	// Set it to a new size.
	// Warning: Doesn't preserve data on growth.
	inline void size(int sz)
	{
		ASSERT(sz >= 0);

		if(sz <= r && data)
		{
			// We're shrinking it but not deleting it.
			r = sz;
			return;
		}

		if(data)
			delete [] data;

		r = sz;
		if(r > 0)
			data = new double[r];
		else
			data = NULL;
	}

	inline int size() const
	{
		return r;
	}

	inline HVector & zero()
	{
		ASSERT(data != NULL);
		ASSERT(r > 0);

		memset(data, 0, r * sizeof(double));

		return *this;
	}

	// Set all elements to the given value.
	inline HVector & set(double val)
	{
		ASSERT(data != NULL);
		ASSERT(r > 0);

		for(int i=0; i<r; i++)
			data[r] = val;

		return *this;
	}

	//Return a string with this vector printed in it.
	string print() const;

	// Fill the vector with rands on 0.0 -> 1.0.
	HVector & rand();

	// Fill the vector with a normal distribution of random numbers,
	// with the given standard deviation.
	HVector & nrand(double stdev=0.0);

	inline double length2() const
	{
		ASSERT(data != NULL && r > 0);

		double l2 = 0;
		for(int i=0; i<r; i++)
		{
			l2 += Sqr(data[i]);
		}

		return l2;
	}

	inline double length() const
	{
		return Sqrt(length2());
	}

	friend Matrix OuterProd(const HVector &A, const HVector &B);

	// Takes an array of HVectors, all the same size.
	friend Matrix Covariance(const HVector *VA, const int count);

	friend HVector Mean(const HVector *VA, const int count);
};

ostream& operator<<(ostream& os, const HVector& p);
istream& operator>>(istream& os, HVector& p);

} // End namespace Remote


#endif
