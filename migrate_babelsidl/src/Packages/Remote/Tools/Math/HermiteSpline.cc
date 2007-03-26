/*
 *  HermiteSpline.cc:
 *
 *  Written by:
 *   David K. McAllister
 *   Department of Computer Science
 *   University of North Carolina
 *   August 1997
 *
 *  Copyright (C) 1997 David McAllister
 */

#include <Packages/Remote/Tools/Math/HermiteSpline.h>
#include <Packages/Remote/Tools/Util/Assert.h>

namespace Remote {
#if 0
template<class T>
HermiteSpline<T>::HermiteSpline()
: d(0), p(0), nset(0), NumBasisSamples(0)
{
}

template<class T>
HermiteSpline<T>::HermiteSpline( const Array1<T>& data )
: d(data), p(data.size()), nset(data.size()), NumBasisSamples(0)
{
}

template<class T>
HermiteSpline<T>::HermiteSpline( const Array1<T>& data, const Array1<T>& tang )
: d(data), p(tang), nset(data.size()), NumBasisSamples(0)
{
  ASSERT(data.size() == tang.size());
}

template<class T>
HermiteSpline<T>::HermiteSpline( const int n )
: d(n), nset(n), NumBasisSamples(0)
{
}

template<class T>
HermiteSpline<T>::HermiteSpline( const HermiteSpline<T>& s )
: d(s.d), p(s.p), nset(s.nset), NumBasisSamples(0)
{
  ASSERT(d.size() == p.size());
  ASSERT(d.size() == nset);
}

template<class T>
void HermiteSpline<T>::setData( const Array1<T>& data )
{
   d = data;
   nset = data.size();
   p.resize(nset);
}

template<class T>
void HermiteSpline<T>::setData( const Array1<T>& data, const Array1<T>& tang )
{
   d = data;
   p = tang;
   nset = data.size();
   ASSERT(data.size() == tang.size());
}

template<class T>
void HermiteSpline<T>::add( const T& obj )
{
   d.add(obj);
   p.grow(1);
   nset++;
}

template<class T>
void HermiteSpline<T>::add( const T& obj, const T& tan )
{
   d.add(obj);
   p.add(tan);
   nset++;
}

template<class T>
void HermiteSpline<T>::insertData( const int idx, const T& obj )
{
   d.insert(idx, obj);
   T tmp;
   p.insert(idx, tmp);
   nset++;
}

template<class T>
void HermiteSpline<T>::insertData( const int idx, const T& obj, const T& tan )
{
   d.insert(idx, obj);
   p.insert(idx, tan);
   nset++;
}

template<class T>
void HermiteSpline<T>::removeData( const int idx )
{
   d.remove(idx);
   p.remove(idx);
   nset--;
}

// Sample at intervals specified by SampleBasisFuncs
template<class T>
T HermiteSpline<T>::sample( double x ) const
{
   int i = int(x);
   int iP1 = i+1;
   int k = int((x - double(i)) * double(NumBasisSamples));

   ASSERT(nset >= 2);
   ASSERT(iP1 < nset);
   ASSERT(i >= 0);
   ASSERT(NumBasisSamples > k);

   return (d[i] * h00[k] + d[iP1] * h10[k] + p[i] * h01[k] + p[iP1] * h11[k]);
}

// Sample at an arbitrary t value
template<class T>
T HermiteSpline<T>::operator()( double t ) const
{
   int i = int(t);
   int iP1 = i+1;
   double x = t - double(i);

   // cerr << "Howdy! i=" << i << " x=" << x << " t=" << t << endl;
   ASSERT(nset >= 2);
   ASSERT(iP1 < nset);
   ASSERT(i >= 0);
   // cerr << "d.size() = " << d.size() << " p.size() = " << p.size() << endl;

   return (d[i] * ((2.0 * x + 1.0) * (x - 1.0) * (x - 1.0)) +
	   d[iP1] * ((-2.0 * x + 3.0) * x * x) +
	   p[i] * (x * (x - 1.0) * (x - 1.0)) +
	   p[iP1] * (x * x * (x - 1.0)));
}

// Read / write control points.
template<class T>
T& HermiteSpline<T>::operator[]( const int idx )
{
   return d[idx];
}

// Make the Hermite basis functions.
template<class T>
void HermiteSpline<T>::SampleBasisFuncs(int NumSamples)
{
  int i = 0;
  double dt = 1. / NumBasisSamples;
  NumBasisSamples = NumSamples;

  h00.resize(NumBasisSamples);
  h01.resize(NumBasisSamples);
  h10.resize(NumBasisSamples);
  h11.resize(NumBasisSamples);

  for (double x = 0.0; i < NumBasisSamples; i++, x += dt)
    {
      h00[i] = (2.0 * x + 1.0) * (x - 1.0) * (x - 1.0);
      h01[i] = x * (x - 1.0) * (x - 1.0);
      h10[i] = (-2.0 * x + 3.0) * x * x;
      h11[i] = x * x * (x - 1.0);
    }
}

// This function solves the tridiagonal matrix for the complete
// parametric spline. Solve the (t,x) and (t,y) systems in
// simultaneously.
template<class T>
void HermiteSpline<T>::CompleteSpline(bool GenEndTangents)
{
  ASSERT(nset == d.size());
  ASSERT(nset >= 2);

  Array1<double> a(nset), c(nset), D(nset);
  Array1<T> b(nset);
  double scale;
  int i, iM1, iP1, nM1 = nset - 1;

  /* Fill in the known data. */
  T E = d[1] - d[0];

  a[0] = 0.0;
  D[0] = 1.0;
  c[0] = 0.0;
  if(GenEndTangents)
    p[0] = d[1] - d[0];
  b[0] = p[0];

  for (i = 1, iM1 = 0, iP1 = 2; i < nM1; iM1 = i, i = iP1, iP1++)
    {
      a[i] = 1.0;
      c[i] = 1.0;
      D[i] = 4.0;

      T EM1 = E;
      E = d[iP1] - d[i];
      b[i] = (EM1 + E) * 3.0;
    }
  a[nset - 1] = 0.0;
  c[nset - 1] = 0.0;
  D[nset - 1] = 1.0;
  if(GenEndTangents)
    p[nset - 1] = d[nset - 1] - d[nset - 2];
  b[nset - 1] = p[nset - 1];

  /* Do forward elimination. */
  for (i = 1, iM1 = 0; i < nset; iM1 = i, i++)
    {
      scale = a[i] / D[iM1];
      D[i] -= scale * c[iM1];
      b[i] -= b[iM1] * scale;
    }

  /* Do back substitution. */
  T prev = p[nset - 1];
  for (i = nset - 2; i >= 0; i--)
    prev = p[i] = (b[i] - prev * c[i]) / D[i];
}
#endif
} // End namespace Remote


