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
BBox::Intersect2( Point s, Vector v, Point& hitNear, Point& hitFar )
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


void
BBox::remap( const Point& eye )
{
  double d1 = Abs( eye.x() - cmin.x() );
  double d2 = Abs( eye.x() - cmax.x() );

  if( d1 > d2 )
    {
      bcmin.x( cmax.x() );
      bcmax.x( cmin.x() );
    }
  else
    {
      bcmin.x( cmin.x() );
      bcmax.x( cmax.x() );
    }
  
 d1 = Abs( eye.y() - cmin.y() );
 d2 = Abs( eye.y() - cmax.y() );

  if( d1 > d2 )
    {
      bcmin.y( cmax.y() );
      bcmax.y( cmin.y() );
    }
  else
    {
      bcmin.y( cmin.y() );
      bcmax.y( cmax.y() );
    }
  
 d1 = Abs( eye.z() - cmin.z() );
 d2 = Abs( eye.z() - cmax.z() );

  if( d1 > d2 )
    {
      bcmin.z( cmax.z() );
      bcmax.z( cmin.z() );
    }
  else
    {
      bcmin.z( cmin.z() );
      bcmax.z( cmax.z() );
    }

  extracmin = bcmin;
  extracmax = bcmax;

  cerr << "Extramin is " << extracmin << endl;
  cerr << "extramax is " << extracmax << endl;

  if ( eye.x() < bcmax.x() && eye.x() > bcmin.x() ||
       eye.x() > bcmax.x() && eye.x() < bcmin.x()   )
    inbx = 1;
  else
    inbx = 0;
  
  if ( eye.y() < bcmax.y() && eye.y() > bcmin.y() ||
       eye.y() > bcmax.y() && eye.y() < bcmin.y()   )
    inby = 1;
  else
    inby = 0;
  
  if ( eye.z() < bcmax.z() && eye.z() > bcmin.z() ||
       eye.z() > bcmax.z() && eye.z() < bcmin.z()   )
    inbz = 1;
  else
    inbz = 0;

  if ( bcmin.x() < bcmax.x() )
    {
      bcmin.x( bcmin.x() - EEpsilon );
      bcmax.x( bcmax.x() + EEpsilon );
    }
  else
    {
      bcmin.x( bcmin.x() + EEpsilon );
      bcmax.x( bcmax.x() - EEpsilon );
    }
    
  if ( bcmin.y() < bcmax.y() )
    {
      bcmin.y( bcmin.y() - EEpsilon );
      bcmax.y( bcmax.y() + EEpsilon );
    }
  else
    {
      bcmin.y( bcmin.y() + EEpsilon );
      bcmax.y( bcmax.y() - EEpsilon );
    }
    
  if ( bcmin.z() < bcmax.z() )
    {
      bcmin.z( bcmin.z() - EEpsilon );
      bcmax.z( bcmax.z() + EEpsilon );
    }
  else
    {
      bcmin.z( bcmin.z() + EEpsilon );
      bcmax.z( bcmax.z() - EEpsilon );
    }
    
  
}

int
BBox::OnCube( const Point& p )
{
  double x = p.x();
  double cx = cmin.x();
  double dx = cmax.x();
  double y = p.y();
  double cy = cmin.y();
  double dy = cmax.y();
  double z = p.z();
  double cz = cmin.z();
  double dz = cmax.z();

  cerr << "ONCUBE " << p <<endl;
  cerr << cmin << cmax << endl;

  if ( ( x < cx+EEpsilon && x > dx-EEpsilon  ||
	x > cx-EEpsilon && x < dx+EEpsilon ) &&
      ( y < cy+EEpsilon && y > dy-EEpsilon  ||
	y > cy-EEpsilon && y < dy+EEpsilon ) &&
      ( z < cz+EEpsilon && z > dz-EEpsilon  ||
	z > cz-EEpsilon && z < dz+EEpsilon ) )
    return 1;
  else
    return 0;
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
	    if ( ! ty > 0 )
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
	    if ( ! tz > 0 )
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
	    if ( ! tx > 0 )
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
	    if ( ! tz > 0 )
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
	    if ( ! ty > 0 )
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
	    if ( ! tx > 0 )
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

  cerr << "I SHOULD BE OUT OF THIS PROCEDURE!!!!\n\n\n\n\n\n\n\n";
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

#if 0
int
BBox::Intersect( const Point& e, const Vector& v, Point& hitNear,
		Point& hitFar )
{
  double tmin, tmax;
  double t[3];
  int oncube[3], order[3];

  cerr << " vvvvvvv " << v << endl;
  
  t[0] = ( cmin.x() - e.x() ) / v.x();
  t[1] = ( cmin.y() - e.y() ) / v.y();
  t[2] = ( cmin.z() - e.z() ) / v.z();

  int i;
  
  for( i = 0; i < 3; i++ )
    if ( t[i] < 0 )
	oncube[i] = 0;
    else
	oncube[i] = OnCube( e+v*t[i] );

  for( i = 0; i<3;i++)
    cerr << " Oncube " << oncube[i];
  cerr << endl;
  

  if ( t[0] > t[1] )
    if ( t[1] > t[2] )
      {
	order[0] = 0;
	order[1] = 1;
	order[2] = 2;
      }
    else
      {
	if ( t[2] > t[0] )
	  {
	    order[0] = 2;
	    order[1] = 0;
	    order[2] = 1;
	  }
	else
	  {
	    order[0] = 0;
	    order[1] = 2;
	    order[2] = 1;
	  }
      }
  else
    {
      if( t[2] > t[1] )
	{
	  order[0] = 2;
	  order[1] = 1;
	  order[2] = 0;
	}
      else
	{
	  order[0] = t[1];
	  if ( t[2] > t[1] )
	    {
	      order[1] = 2;
	      order[2] = 1;
	    }
	  else
	    {
	      order[1] = 1;
	      order[2] = 2;
	    }
	}
    }

  if ( oncube[ order[0] ] )
    tmin = t[order[0]];
  else if ( oncube[ order[1] ] )
    tmin = t[order[1]];
  else if ( oncube[ order[2] ] )
    tmin = t[order[2]];
  else
    {
      cerr << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      exit( -1 );
    }
	    
  cerr << t[0] << "hello " << e+v*t[0] << endl;
  cerr << t[1] << "hello " << e+v*t[1] << endl;
  cerr << t[2] << "hello " << e+v*t[2] << endl;
  
  t[0] = ( cmax.x() - e.x() ) / v.x();
  t[1] = ( cmax.y() - e.y() ) / v.y();
  t[2] = ( cmax.z() - e.z() ) / v.z();

  if ( t[0] < t[1] )
    {
      if ( t[0] < t[2] )
	tmax = t[0];
      else
	tmax = t[2];
    }
  else
    {
      if ( t[1] < t[2] )
	tmax = t[1];
      else
	tmax = t[2];
    }

  int joyous = Intersect2( e, v, hitNear, hitFar );
  
  Point ntmp( e + v * tmin );
  Point xtmp( e + v * tmax );

  int toreturn;
  
  if ( tmin > tmax )
    toreturn = 0;
  else
    toreturn = 1;
  
  cerr << t[0] << "hello " << e+v*t[0] << endl;
  cerr << t[1] << "hello " << e+v*t[1] << endl;
  cerr << t[2] << "hello " << e+v*t[2] << endl;
  
  if ( ! ( joyous        &&       toreturn &&
	  hitNear.InInterval( ntmp, 1e-5 ) &&
	  hitFar.InInterval( xtmp, 1e-5 ) ) )
    {
      if ( joyous || toreturn )
	{
	  cerr << "PROBLEM WITH INTERSECTION\n";

	  if ( ! hitNear.InInterval( ntmp, 1e-5 ) )
	    cerr << "\n\nEXTRA ERROR\n\n\n";
	  
	  cerr << " hitNears are: " <<  hitNear << ntmp << endl;
	  cerr << " hitFars are:  " << hitFar <<  xtmp << endl;
	  cerr << "Intersect2 returned " << Intersect2( e, v,
						       hitNear,
						       hitFar ) << endl;
	  cerr << "toreturn " << toreturn << endl;
	  cerr << tmin << "  " << tmax << endl <<endl;
	}
    }

  hitNear = ntmp;
  hitFar = xtmp;
  
  return toreturn;
}
#endif


#if 0
int
BBox::Intersect( const Point& e, const Vector& v, Point& hitNear,
		Point& hitFar )
{
  double t[6];
  int i,j;
  double b;

  t[0] = ( cmin.x() - e.x() ) / v.x();
  t[1] = ( cmin.y() - e.y() ) / v.y();
  t[2] = ( cmin.z() - e.z() ) / v.z();

  t[3] = ( cmax.x() - e.x() ) / v.x();
  t[4] = ( cmax.y() - e.y() ) / v.y();
  t[5] = ( cmax.z() - e.z() ) / v.z();

  // insertion sort...
  
  for( i = 1; i < 6; i++ )
    {
      b = t[i];
      j = i;
      while( j > 0 &&  t[j-1] > b )
	{
	  t[j] = t[j-1];
	  j--;
	}
      t[j] = b;
    }
  
  cerr << "The following is the sorted array:\n";
  for( i = 0; i < 6; i++ )
    cerr << " " << t[i] << " ";
  cerr << endl << endl;

  Point ntmp( e + v * t[2] );
  Point xtmp( e + v * t[3] );

  Intersect2( e, v, hitNear, hitFar );

  cerr << " hitNears are: " <<  hitNear << ntmp << endl;
  cerr << " hitFars are:  " << hitFar <<  xtmp << endl;

  hitNear = ntmp;
  hitFar = xtmp;

  return 1;
}
#endif


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
