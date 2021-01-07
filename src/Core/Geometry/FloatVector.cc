/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
 *  FloatVector.cc: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 */

#include <Core/Geometry/FloatPoint.h>
#include <Core/Geometry/FloatVector.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Assert.h>
#include <Core/Util/XMLUtils.h>

#include <iostream>
#include <cstdio>

#include <stdlib.h>

using namespace Uintah;
using namespace std;

namespace Uintah {

string
FloatVector::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", x_, y_, z_);
    return buf;
}

FloatVector
FloatVector::fromString( const string & source ) // Creates a FloatVector from a string that looksl like "[Num, Num, Num]".
{
  FloatVector result;

  // Parse out the [num,num,num]
  // Now pull apart the source string.

  string::size_type i1 = source.find("[");
  string::size_type i2 = source.find_first_of(",");
  string::size_type i3 = source.find_last_of(",");
  string::size_type i4 = source.find("]");
    
  string x_val(source,i1+1,i2-i1-1);
  string y_val(source,i2+1,i3-i2-1);
  string z_val(source,i3+1,i4-i3-1);
    
  UintahXML::validateType( x_val, UintahXML::FLOAT_TYPE ); 
  UintahXML::validateType( y_val, UintahXML::FLOAT_TYPE );
  UintahXML::validateType( z_val, UintahXML::FLOAT_TYPE );
    
  result.x( atof(x_val.c_str()) );
  result.y( atof(y_val.c_str()) );
  result.z( atof(z_val.c_str()) );

  return result;
}

void
FloatVector::find_orthogonal(FloatVector& v1, FloatVector& v2) const
{
    FloatVector v0(Cross(*this, FloatVector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, FloatVector(0,1,0));
    }
    v1=Cross(*this, v0);
    v1.normalize();
    v2=Cross(*this, v1);
    v2.normalize();
}

bool
FloatVector::check_find_orthogonal(FloatVector& v1, FloatVector& v2) const
{
    FloatVector v0(Cross(*this, FloatVector(1,0,0)));
    if(v0.length2() == 0){
	v0=Cross(*this, FloatVector(0,1,0));
    }
    v1=Cross(*this, v0);
    float length1 = v1.length();
    if(length1 == 0)
       return false;
    v1 *= 1./length1;
    v2=Cross(*this, v1);
    float length2 = v2.length();
    if(length2 == 0)
       return false;
    v2 *= 1./length2;
    return true;
}

FloatVector
FloatVector::normal() const
{
   FloatVector v(*this);
   v.normalize();
   return v;			// 
}

ostream& operator<<( ostream& os, const FloatVector& v )
{
  os << ' ' << v.x() << ' ' << v.y() << ' ' << v.z() << ' ';
  return os;
}

istream& operator>>( istream& is, FloatVector& v)
{
  float x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=FloatVector(x,y,z);
  return is;
}

int
FloatVector::operator== ( const FloatVector& v ) const
{
    return v.x_ == x_ && v.y_ == y_ && v.z_ == z_;
}

int FloatVector::operator!=(const FloatVector& v) const
{
    return v.x_ != x_ || v.y_ != y_ || v.z_ != z_;
}


void
FloatVector::rotz90(const int c)
{
    // Rotate by c*90 degrees counter clockwise
    switch(c%4){
    case 0:
	// 0 degrees, do nothing
	break;
    case 1:
	// 90 degrees
	{
	    float newx=-y_;
	    y_=x_;
	    x_=newx;
	}
	break;
    case 2:
	// 180 degrees
	x_=-x_;
	y_=-y_;
	break;
    case 3:
	// 270 degrees
	{
	    float newy=-x_;
	    x_=y_;
	    y_=newy;
	}
	break;
    }
}



} // End namespace Uintah


