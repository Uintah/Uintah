// HVector.cpp: implementation of the HVector class.
//////////////////////////////////////////////////////////////////////

#include <Packages/Remote/Tools/Math/HVector.h>

namespace Remote {
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

string HVector::print() const
{
	ASSERT(data != NULL);

	string s;
	char ii[20];

	for(int i=0; i<r; i++)
	{
		s += gcvt(data[i], 15, ii);
		s += " ";
	}
	
    return s;
}

ostream& operator<<(ostream& os, const HVector& h)
{
   os << h.print();
   return os;
}

ostream& operator<<(ostream& os, const Matrix& m)
{
   os << m.print();
   return os;
}

HVector & HVector::rand()
{
	for(int i=0; i<r; i++)
		data[i] = DRand();

	return *this;
}

HVector & HVector::nrand(double stdev)
{
	for(int i=0; i<r; i++)
		data[i] = NRand();
	
	return *this;
}

// Takes an array of HVectors.
HVector Mean(const HVector *VA, const int count)
{
	ASSERT(count > 0);
	
	int dim = VA[0].size();
	
	HVector Mu(dim);
	Mu.zero();
	
	for(int i=0; i<count; i++)
	{
		ASSERT(VA[i].size() == dim);

		Mu += VA[i];
	}

	Mu /= count;

	return Mu;
}

Matrix OuterProd(const HVector &A, const HVector &B)
{
	ASSERT(A.data != NULL);
	ASSERT(B.data != NULL);
	ASSERT(A.r == B.r);
	
	Matrix C(A.r, A.r);
	
	for(int i=0; i<A.r; i++)
		for(int j=0; j<A.r; j++)
		{
			C(i, j) = A.data[i] * B.data[j];
		}
		
	return C;
}

// Compute the covariance matrix of this array of vectors.
Matrix Covariance(const HVector *VA, const int count)
{
	ASSERT(count > 0);
	int i;
	int dim = VA[0].size();
	Matrix Cov(dim, dim);
	Cov.zero();
	
	HVector Mu = Mean(VA, count);
	HVector *VL  = new HVector[count];
	
	for(i=0; i<count; i++)
	{
		ASSERT(VA[i].size() == dim);
		VL[i] = VA[i] - Mu;
	}
	
	for(i=0; i<dim; i++)
	{
		for(int j=0; j<dim; j++)
		{
			double t = 0;
			for(int k=0; k<count; k++)
				t += VL[k][i] * VL[k][j];

			Cov(i, j) += t;
		}
	}

	Cov /= double(count);

	delete [] VL;

	return Cov;
}
} // End namespace Remote


