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

#include <Core/Grid/BoundaryConditions/BCData.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>

#include <Core/Grid/BoundaryConditions/BoundCond.h> // testing, remove this FIXME

#include <iostream>
#include <algorithm>
#include <typeinfo> // for typeid

using namespace Uintah;
using namespace SCIRun;
using namespace std;

////////////////////////////////////////////////////////////////////////

const int UNINITIALIZED = -2;

////////////////////////////////////////////////////////////////////////

BCData::BCData() :
  d_matl( UNINITIALIZED )
{
}

BCData::BCData(const BCData& rhs)
{
  throw ProblemSetupException( "Error, broken BCData() copy constructor called...", __FILE__, __LINE__ );
#if 0
  This copy constructor appears to be broken... but don't see how...
However, I do know that using it causes the resultant BCData object
to be slightly corrupted, specifically, the BoundCondBase *s in 
d_BCBs somehow sort of lose their parent class pointer... :(
  cout << "!!!!!In the BCData copy constructor!!!!!!!!\n";

  vector<BoundCondBase*>::const_iterator i;
  for (i = rhs.d_BCBs.begin(); i != rhs.d_BCBs.end(); ++i) {
    d_BCBs.push_back((*i)->clone());
  }
#endif
}

BCData::~BCData()
{
  cout << "in ~BCData() for " << this << "\n";
  // FIXME... may need to make the d_BCBs hold smart pointers so they will clean
  // themselves up...
}

////////////////////////////////////////////////////////////////////////

BCData&
BCData::operator= ( const BCData & rhs )
{
  throw ProblemSetupException( "BCData.cc: Error, don't call operator=.", __FILE__, __LINE__ );

#if 0
  if (this == &rhs)
    return *this;

  // Delete the current lhs (if it exists)
  vector<BoundCondBase*>::const_iterator itr;
  for (itr = d_BCBs.begin(); itr != d_BCBs.end(); ++itr) {
    delete (*itr);
  }

  d_BCBs.clear();

  // Now copy the rhs to the lhs...
  //
  for( vector<BoundCondBase*>::const_iterator i = rhs.d_BCBs.begin(); i != rhs.d_BCBs.end(); ++i ) {
  
    d_BCBs.push_back( (*i)->clone() );
  }
    
#endif
  return *this;
}

void 
BCData::addBC( const BoundCondBase * bc )
{
  if( d_matl == UNINITIALIZED ) {
    d_matl = bc->getMatl();
  }
  else if( bc->getMatl() != d_matl ) {
    throw ProblemSetupException( "BCData::addBC(): bc has wrong matl index...", __FILE__, __LINE__ );
  }

  if( exists( bc->getVariable() ) ) {
    throw ProblemSetupException( "BCData::addBC(): bc already exists...", __FILE__, __LINE__ );
  }
  d_BCBs.push_back( bc );
}

const BoundCondBase*
BCData::getBCValue( const string & var_name ) const
{
  for( unsigned int index = 0; index < d_BCBs.size(); index++ ) {

    const BoundCondBase * bcb = d_BCBs[ index ];

#if 0 // FIXME
    static int count = 0;
    cout << "here: "<< index << ", " << count++ << "\n";
    if( var_name == "mixture_fraction" && count >= 4331 ) {
      cout << "in here\n";
      this->print(3);
      cout << "  bcb is: "; bcb->debug(); cout << "\n";
    }
#endif

    if( bcb->getVariable() == var_name ) {
      return bcb;
    }
  }
  return NULL;
}

bool
BCData::exists( const string & var_name ) const
{
  return ( getBCValue( var_name ) != NULL );
}

bool
BCData::exists( const string & bc_type, const string & bc_variable ) const
{
  const BoundCondBase * bc = getBCValue( bc_variable );

  if( bc ) {
    if( bc->getType() == bc_type ) {
      return true;
    }
  }
  return false; 
}

void
BCData::combine( const BCData & from )
{
  for( vector<const BoundCondBase*>::const_iterator itr = from.d_BCBs.begin(); itr != from.d_BCBs.end(); ++itr ) {
    addBC( *itr );
  }
}

void
BCData::print( int depth ) const
{

  static bool verbose = false; // FIXME debugging helper var 

  if( verbose ) { // FIXME debugging output
    string indentation( depth*2, ' ' );
    string indent2( (depth+1)*2, ' ' );
  
    cout << indentation << "BCData for matl " << d_matl << ", contains " << d_BCBs.size() << " BoundCondBases) [" << this << "]:\n";

    for( unsigned int pos = 0; pos < d_BCBs.size(); pos++ ) {
      const BoundCondBase * bcb = d_BCBs[ pos ];
      cout << indent2 << "BC = " << bcb->getVariable() << ", type = " << bcb->getType() << ", matl = " << bcb->getMatl() << "\n";
    }
  }
}
