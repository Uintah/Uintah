/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  BBox.cc: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Persistent/Persistent.h>
#include <stdlib.h>

namespace SCIRun {

BBox::BBox()
{
    have_some=0;
}

BBox::BBox(const Point& min, const Point& max):
  have_some(1), cmin(min), cmax(max)
{
}

BBox::BBox(const BBox& copy)
: have_some(copy.have_some), cmin(copy.cmin), cmax(copy.cmax) 
{
}
    
BBox::~BBox()
{
}

void BBox::reset()
{
    have_some=0;
}

void BBox::extend(const Point& p)
{
    if(have_some){
	cmin=Min(p, cmin);
	cmax=Max(p, cmax);
    } else {
	cmin=p;
	cmax=p;
	have_some=1;
    }
}

void BBox::extend(const Point& p, double radius)
{
    Vector r(radius,radius,radius);
    if(have_some){
	cmin=Min(p-r, cmin);
	cmax=Max(p+r, cmax);
    } else {
	cmin=p-r;
	cmax=p+r;
	have_some=1;
    }
}

void BBox::extend(const BBox& b)
{
    if(b.valid()){
	extend(b.min());
	extend(b.max());
    }
}

void BBox::extend_disc(const Point& cen, const Vector& normal, double r)
{
    if (normal.length2() < 0.000001) { extend(cen); return; }
    Vector n(normal.normal());
    double x=Sqrt(1-n.x())*r;
    double y=Sqrt(1-n.y())*r;
    double z=Sqrt(1-n.z())*r;
    extend(cen+Vector(x,y,z));
    extend(cen-Vector(x,y,z));
}

Point BBox::center() const
{
    ASSERT(have_some);
    return Interpolate(cmin, cmax, 0.5);
}

void BBox::translate(const Vector &v) {
    cmin+=v;
    cmax+=v;
}

void BBox::scale(double s, const Vector&o) {
    cmin-=o;
    cmax-=o;
    cmin*=s;
    cmax*=s;
    cmin+=o;
    cmax+=o;
}

double BBox::longest_edge()
{

    ASSERT(have_some);
    Vector diagonal(cmax-cmin);
    return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

Point BBox::min() const
{
    ASSERT(have_some);
    return cmin;
}

Point BBox::max() const
{
    ASSERT(have_some);
    return cmax;
}

Vector BBox::diagonal() const
{
    ASSERT(have_some);
    return cmax-cmin;
}

void
BBox::PrepareIntersect( const Point& eye )
{

  double d1 = Abs( eye.x() - cmin.x() );
  double d2 = Abs( eye.x() - cmax.x() );

  if( d1 > d2 )
      bcmin.x( cmax.x() );
  else
      bcmin.x( cmin.x() );
  
 d1 = Abs( eye.y() - cmin.y() );
 d2 = Abs( eye.y() - cmax.y() );

  if( d1 > d2 )
      bcmin.y( cmax.y() );
  else
      bcmin.y( cmin.y() );
  
 d1 = Abs( eye.z() - cmin.z() );
 d2 = Abs( eye.z() - cmax.z() );

  if( d1 > d2 )
      bcmin.z( cmax.z() );
  else
      bcmin.z( cmin.z() );

  extracmin = bcmin;

  // the inb{x,y,z} vars tell if the eyepoint is in between
  // the x, y, z bbox planes
  
  if ( eye.x() < cmax.x() && eye.x() > cmin.x() ||
       eye.x() > cmax.x() && eye.x() < cmin.x()   )
    inbx = 1;
  else
    inbx = 0;
  
  if ( eye.y() < cmax.y() && eye.y() > cmin.y() ||
       eye.y() > cmax.y() && eye.y() < cmin.y()   )
    inby = 1;
  else
    inby = 0;
  
  if ( eye.z() < cmax.z() && eye.z() > cmin.z() ||
       eye.z() > cmax.z() && eye.z() < cmin.z()   )
    inbz = 1;
  else
    inbz = 0;

  // assign the epsilon bigger bbox

  if ( cmin.x() < cmax.x() )
    {
      bcmin.x( cmin.x() - EEpsilon );
      bcmax.x( cmax.x() + EEpsilon );
    }
  else
    {
      bcmin.x( cmin.x() + EEpsilon );
      bcmax.x( cmax.x() - EEpsilon );
    }
    
  if ( cmin.y() < cmax.y() )
    {
      bcmin.y( cmin.y() - EEpsilon );
      bcmax.y( cmax.y() + EEpsilon );
    }
  else
    {
      bcmin.y( cmin.y() + EEpsilon );
      bcmax.y( cmax.y() - EEpsilon );
    }
    
  if ( cmin.z() < cmax.z() )
    {
      bcmin.z( cmin.z() - EEpsilon );
      bcmax.z( cmax.z() + EEpsilon );
    }
  else
    {
      bcmin.z( cmin.z() + EEpsilon );
      bcmax.z( cmax.z() - EEpsilon );
    }
    
  
}


int
BBox::Intersect( const Point& e, const Vector& v, Point& hitNear )
{
  double tx, ty, tz;
  int worked = 0;

  tx = ( extracmin.x() - e.x() ) / v.x();
  ty = ( extracmin.y() - e.y() ) / v.y();
  tz = ( extracmin.z() - e.z() ) / v.z();

  if ( inbx && inby && inbz )
    {
      hitNear = e;
      return 1;
    }

  // is it correct to assume that if tz < 0 then i should return 0?
  if ( inbx && inby )
    {
      if ( tz > 0 )
	return(  TestTz( e, v, tz, hitNear ) );
      else
	return 0;
    }

  if ( inbx && inbz )
    {
      if ( ty > 0 )
	return( TestTy( e, v, ty, hitNear ));
      else
	return 0;
    }

  if ( inby && inbz )
    {
      if ( tx > 0 )
	return( TestTx( e, v, tx, hitNear ));
      else
	return 0;
    }


  // NEXT BUNCH

  if ( inbx )
    if ( ty > 0 || tz > 0 )
      if ( ty > tz )
	{
	  worked = TestTy( e, v, ty, hitNear );

	  if ( worked )
	    return 1;
	  else if ( tz > 0 )
	    return( TestTz( e, v, tz, hitNear ) );
	  else
	    return 0;
	}
      else
	{
	  worked = TestTz( e, v, tz, hitNear );
	  
	  if ( worked )
	    return 1;
	  else if ( ty > 0 )
	    return( TestTy( e, v, ty, hitNear ) );
	  else
	    return 0;
	}
    else
      return 0;

  if ( inby )
    if ( tx > 0 || tz > 0 )
      if ( tx > tz )
	{
	  worked = TestTx( e, v, tx, hitNear );
	  
	  if ( worked )
	    return 1;
	  else if ( tz > 0 )
	    return( TestTz( e, v, tz, hitNear ) );
	  else
	    return 0;
	}
      else
	{
	  worked = TestTz( e, v, tz, hitNear );
	  
	  if ( worked )
	    return 1;
	  else if ( tx > 0 )
	    return( TestTx( e, v, tx, hitNear ) );
	  else
	    return 0;
	}
    else
      return 0;

  if ( inbz )
    if ( ty > 0 || tx > 0 )
      if ( ty > tx )
	{
	  worked = TestTy( e, v, ty, hitNear );
	  
	  if ( worked )
	    return 1;
	  else if ( tx > 0 )
	    return( TestTx( e, v, tx, hitNear ) );
	  else
	    return 0;
	}
      else
	{
	  worked = TestTx( e, v, tx, hitNear );
	  
	  if ( worked )
	    return 1;
	  else if ( ty > 0 )
	    return( TestTy( e, v, ty, hitNear ) );
	  else
	    return 0;
	}
    else
      return 0;

  // the case when inb{x,y,z} are all false
  // TEMP  DO I NEED TO CHECK for tx > 0?

  if (   tx >= ty   &&   tx >= tz   &&   tx > 0  )
    { // tx greatest

      worked = TestTx( e, v, tx, hitNear );
      
      if  ( worked )
	return 1;
      else
	if ( ty >= tz )
	  {
	    if ( ! (ty > 0) )
	      return 0;
	    
	    worked = TestTy( e, v, ty, hitNear );
	    
	    if ( worked )
	      return 1;
	    else if ( tz > 0 )
	      return( TestTz( e, v, tz, hitNear ) );
	    else
	      return 0;
	  }
	else
	  {
	    if ( ! (tz > 0) )
	      return 0;

	    worked = TestTz( e, v, tz, hitNear );

	    if ( worked )
	      return 1;
	    else if ( ty > 0 )
	      return( TestTy( e, v, ty, hitNear ) );
	    else
	      return 0;
	  }
    }
  

  if (   ty >= tx   &&   ty >= tz   &&   ty > 0  )
    { // ty greatest

      worked = TestTy( e, v, ty, hitNear );
      
      if  ( worked )
	return 1;
      else
	if ( tx >= tz )
	  {
	    if ( ! (tx > 0) )
	      return 0;
	    
	    worked = TestTx( e, v, tx, hitNear );
	    
	    if ( worked )
	      return 1;
	    else if ( tz > 0 )
	      return( TestTz( e, v, tz, hitNear ) );
	    else
	      return 0;
	  }
	else
	  {
	    if ( ! (tz > 0) )
	      return 0;

	    worked = TestTz( e, v, tz, hitNear );

	    if ( worked )
	      return 1;
	    else if ( tx > 0 )
	      return( TestTx( e, v, tx, hitNear ) );
	    else
	      return 0;
	  }
    }
  

  if (   tz >= ty   &&   tz >= tx   &&   tz > 0  )
    { // tz greatest

      worked = TestTz( e, v, tz, hitNear );
      
      if  ( worked )
	return 1;
      else
	if ( ty >= tx )
	  {
	    if ( ! (ty > 0) )
	      return 0;
	    
	    worked = TestTy( e, v, ty, hitNear );
	    
	    if ( worked )
	      return 1;
	    else if ( tx > 0 )
	      return( TestTx( e, v, tx, hitNear ) );
	    else
	      return 0;
	  }
	else
	  {
	    if ( ! (tx > 0) )
	      return 0;

	    worked = TestTx( e, v, tx, hitNear );

	    if ( worked )
	      return 1;
	    else if ( ty > 0 )
	      return( TestTy( e, v, ty, hitNear ) );
	    else
	      return 0;
	  }
    }

  return 0;
}

int
BBox::TestTx( const Point& e, const Vector& v, double tx, Point& hitNear )
{
  hitNear = e + v * tx;

  if ( ( hitNear.z() >= bcmin.z() && hitNear.z() <= bcmax.z()  ||
	hitNear.z() <= bcmin.z() && hitNear.z() >= bcmax.z() )  &&
      ( hitNear.y() >= bcmin.y() && hitNear.y() <= bcmax.y()  ||
       hitNear.y() <= bcmin.y() && hitNear.y() >= bcmax.y() )      )
    return 1;

  return 0;
}

int
BBox::TestTy( const Point& e, const Vector& v, double ty, Point& hitNear )
{
  hitNear = e + v * ty;

  if ( ( hitNear.x() >= bcmin.x() && hitNear.x() <= bcmax.x()  ||
	hitNear.x() <= bcmin.x() && hitNear.x() >= bcmax.x() )    &&
      ( hitNear.z() >= bcmin.z() && hitNear.z() <= bcmax.z()  ||
       hitNear.z() <= bcmin.z() && hitNear.z() >= bcmax.z() )     )
    return 1;
  
  return 0;
}

int
BBox::TestTz( const Point& e, const Vector& v, double tz, Point& hitNear )
{
  hitNear = e + v * tz;

  if ( ( hitNear.x() >= bcmin.x() && hitNear.x() <= bcmax.x()  ||
	hitNear.x() <= bcmin.x() && hitNear.x() >= bcmax.x() )    &&
      ( hitNear.y() >= bcmin.y() && hitNear.y() <= bcmax.y()  ||
       hitNear.y() <= bcmin.y() && hitNear.y() >= bcmax.y() )     )
    return 1;
  
  return 0;
}

bool 
BBox::Overlaps( const BBox & bb)
{
  if( bb.cmin.x() > cmax.x() || bb.cmax.x() < cmin.x())
    return false;
  else if( bb.cmin.y() > cmax.y() || bb.cmax.y() < cmin.y())
    return false;
  else if( bb.cmin.z() > cmax.z() || bb.cmax.z() < cmin.z())
    return false;

  return true;
}


void Pio(Piostream & stream, BBox & box) {


    stream.begin_cheap_delim();
    Pio(stream, box.have_some);
    if (box.have_some) {
	Pio(stream, box.cmin);
	Pio(stream, box.cmax);
    }
    stream.end_cheap_delim();
}

} // End namespace SCIRun

