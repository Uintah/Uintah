#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Classlib/Assert.h>
#include <Classlib/Exceptions.h>
#include <Math/MinMax.h>
#include <Classlib/Persistent.h>
#include <stdlib.h>

BBox::BBox()
{
    have_some=0;
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


#include <Math/MusilRNG.h>

void BBox::test_rigorous(RigorousTest* __test){
  BBox b;
  
  int Mx,My,Mz,mx,my,mz;
  Mx=My=Mz=0;
  mx=my=mz=1;

  MusilRNG r(23423);

  int t=1;
  
  for(int c=1;c<=100;++c){
    b.reset();
    TEST(!b.valid());
    Mx=My=Mz=mx=my=mz=0;
    
    t=1; //Used for determining Min points

    for (int i=1;i<=1000;++i){
      int x = int(r()*1000);
      int y = int(r()*1000);
      int z = int(r()*1000);
    
      if(r()<0.3)
	x=-x;
      if(r()<0.6&&r()>=0.3)
	y=-y;
      if(r()<=1&&r()>=0.6)
	z=-z;   
      
      if(t){
	mx=x;
	my=y;
	mz=z;
	Mx=x;
	My=y;
	Mz=z;
	t=0;
      }
      
      //Track the Min and Max Points of the BBox

      //Min()/Max() Tests
      
      Mx=Max(Mx,x);
      mx=Min(mx,x);
    
      My=Max(My,y);
      my=Min(my,y);
      
      Mz=Max(Mz,z);
      mz=Min(mz,z);
      
      b.extend(Point(x,y,z));
      
    
      Vector diag = Point(Mx,My,Mz)-Point(mx,my,mz);

      
      //Min/Max and extend(Point) Tests
      TEST(b.max()==Point(Mx,My,Mz));
      TEST(b.min()==Point(mx,my,mz));

      //diagonal() Tests
      TEST(b.diagonal()==diag);
      
    }
  }
  
  //extend(const Point& p, double radius) Tests

  MusilRNG rnd(32412234);

  BBox b3;
  
  for(int i=0;i<=1000;++i){
    double x,y,z;
    x = double(rnd()*1000);
    if(r()>.5)
      x = -x;
    y = double(rnd()*1000);
    if(r()<.5)
      y=-y;
    z = double(rnd()*1000);
    if(r()>.5)
      z=-z;
    
    Point curr_pt(x,y,z);   //Create a random point in space
    BBox b2;

    TEST(!b2.valid());

    double radius;
    Vector r_vector;
  
    for(int r=0;r<=50;++r){
      radius=double(rnd()*1000);
      if(rnd()>=.5)
	radius=-radius;
		  
      r_vector.x(radius);
      r_vector.y(radius);
      r_vector.z(radius);

      b2.extend(curr_pt,radius);


      //Test to be sure that the Max/Min Values are equal to
      //The bounds of a sphere with a center at position r_vector
      if(!b2.valid()){
	TEST(b2.max()==curr_pt+r_vector);
	TEST(b2.min()==curr_pt-r_vector);
      }
      else{
	TEST(b2.max()==Max(curr_pt+r_vector,b2.max()));
	TEST(b2.min()==Min(curr_pt-r_vector,b2.min()));
      }
    }
      //Copy Constructor Tests
      b3=b2;      
      TEST(b3.max()==b2.max());
      TEST(b3.min()==b2.min());

      //diagonal() Tests
      TEST(b2.diagonal()==(b2.max()-b2.min()));
      TEST(b3.diagonal()==(b3.max()-b3.min()));
      
      //center() Tests
      TEST(b2.center()==(Interpolate(b2.max(),b2.min(),.5)));
      TEST(b3.center()==(Interpolate(b3.max(),b3.min(),.5)));
      
      //longest_edge() Tests
      Vector d=(b2.max()-b2.min());
      TEST(b2.longest_edge()==Max(d.x(),d.y(),d.z()));

      //inside() tests
      for(int i=0;i<20;++i){
	Point pnt(rnd()*1000,rnd()*1000,rnd()*1000);

	Point pmax=b2.max();
	Point pmin=b2.min();
	
	if(b2.valid() && pnt.x()>=pmin.x() && pnt.y()>=pmin.y() && pnt.z()>=pmin.z() && pnt.x()<=pmax.x() && pnt.y()<=pmax.y() && pnt.z()<=pmax.z())
	    TEST(b2.inside(pnt));

      }
    //Translate Tests

    BBox mybox;  //Define a random box
    mybox.extend(Point(x,y,z));
    mybox.extend(Point(z,x,y));
    
    BBox mybox1=mybox;
    
    Vector v(y,z,x);
    mybox.translate(v);

    TEST(mybox.max()==(mybox1.max()+v));
    TEST(mybox.min()==(mybox1.min()+v));

    
    Vector d1=mybox.diagonal();
    Vector d2=mybox.diagonal();

    TEST(d1.length2()==d2.length2());
    
    //Scale Tests
    mybox.reset();
    mybox1.reset();
    TEST(!mybox.valid());
    TEST(!mybox1.valid());
    mybox.extend(Point(x,y,z));
    mybox1=mybox;

    for(int ct=1;ct<=100;ct++){  //Use different scales, and the minpoint
      mybox.scale(double(ct*.1),mybox.min().vector());
      TEST(mybox.center()==mybox1.min());
      d1=mybox.diagonal();
      d2=mybox1.diagonal();
      TEST(d1.length2()==d2.length2()); 
    }

    mybox1=mybox;
    
    for(ct=1;ct<=100;ct++){  //Use different scales, and the maxpoint
      mybox.scale(double(ct*.1),mybox.max().vector());
      TEST(mybox.center()==mybox1.max());
      d1=mybox.diagonal();
      d2=mybox1.diagonal();
      //cout << "D1: " << mybox.min() << " D2: " << mybox.max() << endl;
      TEST(d1.length2()==d2.length2());
    }

    for(ct=1;ct<=100;++ct){
      Vector rv=((mybox.min()+((mybox.max()-mybox.min())*rnd())).vector());
      mybox1=mybox;
      TEST(mybox.min()==mybox1.min());
      TEST(mybox.max()==mybox1.max());
      
      mybox.scale(double(ct*rnd()),mybox.min().vector());  
      TEST(mybox.center()==mybox.min());
      TEST((mybox.diagonal().length2())==(mybox.diagonal().length2())*(ct*rnd()));
      mybox.scale(double(ct*rnd()),rv);
      TEST(mybox.center()==rv.point());
      
    }    
  }
}


   
