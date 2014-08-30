/*
 * Copyright (c) 2014 The University of Utah
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

#ifndef BracketHelper_h
#define BracketHelper_h

#include <cmath>   // for ceil
#include <vector>
using std::vector;

/**
 *  @class BracketHelper
 *  @brief Facilitate rapid searching of an ordered set of objects.
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  The BracketHelper class is a templated class that provides
 *  hashed search capability.  The template objects must define
 *  the following operators:  >=  <=  >  <  ==
 *
 *  The list of values which are sent into the BracketHelper
 *  constructor must be sorted in ascending order!
 */
template <class T>
class BracketHelper{

 public:

  /**
   *  @brief Construct a BracketHelper object.
   *
   *  @param npts : number of points to use in hash table.  Nominally, this
   *                should be a fairly large number for efficient lookups.
   *  @param vals : set of dependent values that are to be hashed.  These
   *                must be sorted in ASCENDING order!
   */
  BracketHelper( const int & npts,
		 const std::vector<T> & vals );

  ~BracketHelper();

  /**
   *   @brief Bracket a given value using the hash table followed by bisection.
   *
   *   @param val  : INPUT  value which should be bracketed
   *   @param ilo  : OUTPUT index of the lower value which brackets val
   *   @param ihi  : OUTPUT index of the upper value which brackets val
   */
  void bracket( const T & val,
		int & ilo,
		int & ihi );

 private:

  BracketHelper();

  struct IxBounds{ int ilo, ihi; };

  void set_up_hash();   // set up the hash table
  void narrow_bounds( const T & target,
		      int & ilo,
		      int & ihi );

  const int nEntries_; // number of points in hash table

  const vector<T> vals_;

  const T  loBound_, hiBound_;

  T  spacing_;         // hash table spacing (assumed uniform)

  IxBounds * ixBnds_;  // the indices bracketing each point in the hash table

  bool hashReady_;     // flag set if the hash table is ready for use

};

/* ================================================================ */

template <class T>
BracketHelper<T>::BracketHelper( const int & npts,
				 const std::vector<T> & vals )
  :  nEntries_( npts ),

     vals_( vals ),

     loBound_( vals.front() ),
     hiBound_( vals.back() ),

     ixBnds_( NULL ),

     hashReady_( false )
{
  // abandon hashing if we don't have any points in the table
  // resort to full bisection.
  if( nEntries_ <= 1 ) return;

  //
  // be sure that the underlying values are in ascending order
  // if they are not, then nothing will work, because we are using
  // bisection as the bracketing algorithm.
  //
  // If you require "bracketing" on an unordered list, you will need
  // something fancier than this!
  //
  for( int i=1; i<(int)vals_.size(); i++ ){
    assert( vals_[i] > vals_[i-1] );
    if ( vals_[i] < vals_[i-1] ){
      std::abort();
    }
  }
  //
  // set spacing (assuming linear spacing)
  //
  assert( nEntries_ > 1 );
  spacing_ = (hiBound_ - loBound_) / T(nEntries_-1);
  assert( spacing_ > 0.0 );
  //
  // allocate storage for hash table indices
  //
  ixBnds_ = new IxBounds[nEntries_];
  //
  // set up the hash table
  //
  set_up_hash();
}

/* ================================================================ */

template <class T>
BracketHelper<T>::~BracketHelper()
{
  if( NULL != ixBnds_ ) delete [] ixBnds_;
}

/* ================================================================ */

template <class T>
void
BracketHelper<T>::set_up_hash()
{
  T thisVal = loBound_;
  for (int i=0; i<nEntries_; i++){

    int itmp;

    bracket( thisVal,          ixBnds_[i].ilo, itmp           );
    bracket( thisVal+spacing_, itmp,           ixBnds_[i].ihi );

    assert( ixBnds_[i].ihi > ixBnds_[i].ilo );

    thisVal += spacing_;
  }
  hashReady_ = true;
}

/* ================================================================ */

template <class T>
void
BracketHelper<T>::bracket( const T & targetval,
			   int & ilo,
			   int & ihi )
{
  // are we outside the range of values?
  if( targetval < vals_[0] ){
    ilo = 0; ihi = 1;
    return;
  }
  else if( targetval > vals_[vals_.size()-1] ){
    ihi = (int)vals_.size()-1;
    ilo = ihi - 1;
    return;
  }


  if ( hashReady_ ){
    // use the hash table to narrow the search region,
    // then finish with bisection
    narrow_bounds( targetval, ilo, ihi );
    if ( (ihi-ilo) <= 1 ) return;
  }
  else {
    // hash table not available, so just set up for basic bisection
    ilo = 0;
    ihi = vals_.size() - 1;
  }

  T valHi = vals_[ihi];
  T valLo = vals_[ilo];

  // did we hit the value?
  if ( targetval >= valHi ) { ilo = ihi-1; return;}
  if ( targetval <= valLo ) { ihi = ilo+1; return;}

  bool more=1;
  int count=0;
  while( more && ( count <= (int)vals_.size() ) ) {
    int itmp = int(ceil(float(ilo+ihi)*0.5));
    T tmpval = vals_[itmp];

    if(tmpval > targetval){ // select lower interval
      valHi = tmpval;
      ihi = itmp;
    }
    else { // select upper interval
      valLo = tmpval;
      ilo = itmp;
    }
    if( (ihi-ilo) <= 1 ) more=false;
    count++;
  }
  assert( targetval >= valLo   &&   targetval <= valHi );
}

/* ================================================================ */

template <class T>
void
BracketHelper<T>::narrow_bounds( const T & target, int & ilo, int & ihi )
{
  // determine index in uniform coarse mesh (assumes linear spacing)
  const int i = int( floor((target-loBound_)/spacing_) );
  ilo = ixBnds_[i].ilo;
  ihi = ixBnds_[i].ihi;
}

/* ================================================================ */

#endif // BracketHelper_h
