
#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Classlib/Assert.h>
#include <Classlib/Exceptions.h>
#include <Math/MinMax.h>
#include <Classlib/Persistent.h>
#include <stdlib.h>
#include <iostream.h>

BBox::BBox()
{
    have_some=0;
}

BBox::BBox(const BBox& copy)
: have_some(copy.have_some), cmin(copy.cmin), cmax(copy.cmax) 
{
  int i;

  for ( i = 0; i < 6; i++ )
    sides[i] = copy.sides[i];
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

void BBox::extend_cyl(const Point& cen, const Vector& normal, double r)
{
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

Point BBox::corner(int i) const
{
    ASSERT(have_some);
    switch(i){
    case 0:
	return Point(cmin.x(), cmin.y(), cmin.z());
    case 1:
	return Point(cmin.x(), cmin.y(), cmax.z());
    case 2:
	return Point(cmin.x(), cmax.y(), cmin.z());
    case 3:
	return Point(cmin.x(), cmax.y(), cmax.z());
    case 4:
	return Point(cmax.x(), cmin.y(), cmin.z());
    case 5:
	return Point(cmax.x(), cmin.y(), cmax.z());
    case 6:
	return Point(cmax.x(), cmax.y(), cmin.z());
    case 7:
	return Point(cmax.x(), cmax.y(), cmax.z());
    default:
	EXCEPTION(General("Bad corner requested from BBox"));
    }
    return Point(0,0,0);
}

void Pio(Piostream &stream, BBox& box) {
    stream.begin_cheap_delim();
    Pio(stream, box.have_some);
    if (box.have_some) {
	Pio(stream, box.cmin);
	Pio(stream, box.cmax);
    }
    stream.end_cheap_delim();
}



void
BBox::SetupSides( )
{
  sides[0].ChangeBBoxSide( corner(0), corner(1), corner(3), corner(2) );
  
  sides[1].ChangeBBoxSide( corner(0), corner(1), corner(5), corner(4) );

  sides[2].ChangeBBoxSide( corner(0), corner(2), corner(6), corner(4) );
  
  sides[3].ChangeBBoxSide( corner(4), corner(6), corner(7), corner(5) );
  
  sides[4].ChangeBBoxSide( corner(1), corner(3), corner(7), corner(5) );
  
  sides[5].ChangeBBoxSide( corner(2), corner(3), corner(7), corner(6) );
}

/*******************************************************************
 *
 * finds 2 points of intersection of a line and the bbox.
 * if no intersection is found, the routine returns false.
 *
*******************************************************************/

int
BBox::Intersect( Point s, Vector v, Point& hitNear, Point& hitFar )
{
  int i;
  Point tmp;
  double near, far;
  int flag;

  near = 10000;
  far = 0;
  flag = 0;

  for ( i = 0; i < 6; i++ )
    {
      if ( sides[i].Intersect( s, v, tmp ) )
	{
	  if ( ( s - tmp ).length() < near )
	    {
	      hitNear = tmp;
	      near = ( s - tmp ).length();
	      flag = 1;
	    }
	  if ( ( s - tmp ).length() > far )
	    {
	      hitFar = tmp;
	      far = ( s - tmp ).length();
	      flag = 1;
	    }
	}
    }

  return( flag );
}


BBoxSide::BBoxSide( )
{
}



BBoxSide::BBoxSide( Point a, Point b, Point c, Point d )
{
  perimeter[0][0] = b - a;
  perimeter[0][1] = c - b;
  perimeter[0][2] = a - c;
  perimeter[1][0] = d - c;
  perimeter[1][1] = a - d;
  perimeter[1][2] = c - a;

  pl.ChangePlane( a, b, c );

  pts[0][0] = a;
  pts[0][1] = b;
  pts[0][2] = c;
  
  pts[1][0] = c;
  pts[1][1] = d;
  pts[1][2] = a;
  
  next = NULL;
}

void
BBoxSide::ChangeBBoxSide( Point a, Point b, Point c, Point d )
{
  perimeter[0][0] = b - a;
  perimeter[0][1] = c - b;
  perimeter[0][2] = a - c;
  perimeter[1][0] = d - c;
  perimeter[1][1] = a - d;
  perimeter[1][2] = c - a;

  pl.ChangePlane( a, b, c );
  
  pts[0][0] = a;
  pts[0][1] = b;
  pts[0][2] = c;
  
  pts[1][0] = c;
  pts[1][1] = d;
  pts[1][2] = a;
  
  next = NULL;
}

int
BBoxSide::Intersect( Point s, Vector v, Point& hit )
{
  if ( pl.Intersect( s, v, hit ) )
    {
      if ( Dot( hit - pts[0][0], perimeter[0][0] ) >= 0 &&
	  Dot( hit - pts[0][1], perimeter[0][1] ) >= 0 &&
	  Dot( hit - pts[0][2], perimeter[1][0] ) >= 0 &&
	  Dot( hit - pts[1][1], perimeter[1][1] ) >= 0 )
	return 1;
    }
  return 0;
}
