#if 1

#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Classlib/Assert.h>
#include <Classlib/Persistent.h>
#include <Classlib/String.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <iostream.h>
#include <stdio.h>

Point Interpolate(const Point& p1, const Point& p2, double w)
{
    return Point(
	Interpolate(p1._x, p2._x, w),
	Interpolate(p1._y, p2._y, w),
	Interpolate(p1._z, p2._z, w));
}

clString Point::string() const
{
#if 0
    return clString("[")
	+to_string(_x)+clString(", ")
	    +to_string(_y)+clString(", ")
		+to_string(_z)+clString("]");
#endif
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
    return clString(buf);
}

int Point::operator==(const Point& p) const
{
    return p._x == _x && p._y == _y && p._z == _z;
}

int Point::operator!=(const Point& p) const
{
    return p._x != _x || p._y != _y || p._z != _z;
}

Point::Point(double x, double y, double z, double w)
{
    if(w==0){
	cerr << "degenerate point!" << endl;
	_x=_y=_z=0;
    } else {
	_x=x/w;
	_y=y/w;
	_z=z/w;
    }
#if SCI_ASSERTION_LEVEL >= 4
    uninit=0;
#endif
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2)
{
    return Point(p1._x*w1+p2._x*w2,
		 p1._y*w1+p2._y*w2,
		 p1._z*w1+p2._z*w2);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3)
{
    return Point(p1._x*w1+p2._x*w2+p3._x*w3,
		 p1._y*w1+p2._y*w2+p3._y*w3,
		 p1._z*w1+p2._z*w2+p3._z*w3);
}

Point AffineCombination(const Point& p1, double w1,
			const Point& p2, double w2,
			const Point& p3, double w3,
			const Point& p4, double w4)
{
    return Point(p1._x*w1+p2._x*w2+p3._x*w3+p4._x*w4,
		 p1._y*w1+p2._y*w2+p3._y*w3+p4._y*w4,
		 p1._z*w1+p2._z*w2+p3._z*w3+p4._z*w4);
}

ostream& operator<<( ostream& os, const Point& p )
{
   os << p.string();
   return os;
}

#if 1 
istream& operator>>( istream& is, Point& v)
{
  double x, y, z;
  char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Point(x,y,z);
  return is;
}
#endif

void Pio(Piostream& stream, Point& p)
{
    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}


int
Point::Overlap( double a, double b, double e )
{
  double hi, lo, h, l;
  
  hi = a + e;
  lo = a - e;
  h  = b + e;
  l  = b - e;

  if ( ( hi > l ) && ( lo < h ) )
    return 1;
  else
    return 0;
}
  
int
Point::InInterval( Point a, double epsilon )
{
  if ( Overlap( _x, a.x(), epsilon ) &&
      Overlap( _y, a.y(), epsilon )  &&
      Overlap( _z, a.z(), epsilon ) )
    return 1;
  else
    return 0;
}


#include <Geometry/Vector.h>

void Point::test_rigorous(RigorousTest* __test)
{	
    //Basic Point Tests

    Point P(0,0,0);
    Point Q(0,0,0);

    for(int x=0;x<=100;++x){
	for(int y=0;y<=100;++y){
	    for(int z=0;z<=100;++z){
		P.x(x);
		P.y(y);
		P.z(z);
		
		TEST(P.x()==x);
		TEST(P.y()==y);
		TEST(P.z()==z);
		
		Q=P;	
		TEST(Q.x()==x);
		TEST(Q.y()==y);
		TEST(Q.z()==z);
		TEST(P==Q);
		Q.x(x+1);
		Q.y(y+1);
		Q.z(z+1);
		TEST(P!=Q);
	    }
	}
    }
    
    //with doubles

    for(x=0;x<=100;++x){
	for(int y=0;y<=100;++y){
	    for(int z=0;z<=100;++z){
		P.x((double)x-.00007);
		P.y((double)y-.00007);
		P.z((double)z-.00007);
		
		TEST(P.x()==(double)x-.00007);
		TEST(P.y()==(double)y-.00007);
		TEST(P.z()==(double)z-.00007);
		
		Q=P;	
		TEST(Q.x()==P.x());
		TEST(Q.y()==P.y());
		TEST(Q.z()==P.z());
		TEST(P==Q);
		Q.x(x+1);
		Q.y(y+1);
		Q.z(z+1);
		TEST(P!=Q);
	    }
	}
    }
   
    //Basic Vector Tests

    Vector V(0,0,0);
    Vector V2(0,0,0);

    for(x=0;x<=100;++x){
	for(int y=0;y<=100;++y){
	    for(int z=0;z<=100;++z){
		V.x(x);
		V.y(y);
		V.z(z);

		TEST(V.x()==x);
		TEST(V.y()==y);
		TEST(V.z()==z);
    
		V2=V;
		TEST(V2.x()==x);
		TEST(V2.y()==y);
		TEST(V2.z()==z);
		TEST(V==V2);
		
		V2.x(-1);
		TEST(!(V==V2));
		V2.x(x);
		
		TEST(V==V2);
		V2.y(-1);
		TEST(!(V==V2));
		V2.y(y);

             	TEST(V==V2);
		V2.z(z+1);
		TEST(!(V==V2));
	    }
	}
    }
    
    //with doubles

    for(x=0;x<=100;++x){
	for(int y=0;y<=100;++y){
	    for(int z=0;z<=100;++z){
		V.x((double)x-.00007);
		V.y((double)y-.00007);
		V.z((double)z-.00007);

		TEST(V.x()==(double)x-.00007);
		TEST(V.y()==(double)y-.00007);
		TEST(V.z()==(double)z-.00007);
		V2=V;
		TEST(V.x()==V2.x());
		TEST(V.y()==V2.y());
		TEST(V.z()==V2.z());
		TEST(V==V2);
		
		V.x(x);
	        TEST(!(V==V2));
		
		V.x((double)x-.00007);
		
	       
		TEST(V==V2);
	    }	
	}
    }

    //Operator tests

    Point p1(0,0,0);
    Point p2(0,0,0);
    Point p3(0,0,0);

    Vector v1(0,0,0);
    Vector v2(0,0,0);
    Vector v3(0,0,0);
    int X,Y,Z;

    for (x=0;x<10;++x){
	for (int y=0;y<10;++y){
	    for (int z=0;z<10;++z){
		p1.x(x);
		p1.y(y);
		p1.z(z);
		TEST((p1.x()==x)&&(p1.y()==y)&&(p1.z()==z));
		
		p2.x(z);
		p2.y(x);
		p2.z(y);
		TEST((p2.x()==z)&&(p2.y()==x)&&(p2.z()==y));

		v1.x(x);
		v1.y(y);
		v1.z(z);
		TEST((v1.x()==x)&&(v1.y()==y)&&(v1.z()==z));
		
		v2.x(z);
		v2.y(x);
		v2.z(y);
		TEST((v2.x()==z)&&(v2.y()==x)&&(v2.z()==y));
		
		v3=p2-p1;
		X=p2.x()-p1.x();
		Y=p2.y()-p1.y();
		Z=p2.z()-p1.z();

		TEST(v3.x()==X);
		TEST(v3.y()==Y);
		TEST(v3.z()==Z);
		
		
		p2=p1+v1;
		X=p1.x()+v1.x();
		Y=p1.y()+v1.y();
		Z=p1.z()+v1.z();

		TEST(p2.x()==X);
		TEST(p2.y()==Y);
		TEST(p2.z()==Z);
		
		p1.x(x);
		p1.y(y);
		p1.z(z);
		TEST((p1.x()==x)&&(p1.y()==y)&&(p1.z()==z));
		v1.x(z);
		v1.y(x);
		v1.z(y);
		TEST((v1.x()==z)&&(v1.y()==x)&&(v1.z()==y));

		X=p1.x()-v1.x();
		Y=p1.y()-v1.y();
		Z=p1.z()-v1.z();
	   
		p3=p1-v1;
		TEST(p3.x()==X);
		TEST(p3.y()==Y);
		TEST(p3.z()==Z);

		double D = x*.12345;
		X=p1.x()*D;
		Y=p1.y()*D;
		Z=p1.z()*D;

	/*	p3=D*p1;
		TEST(p3.x()==X);
		TEST(p3.y()==Y);
		TEST(p3.z()==Z);
		
	*/	

	    }
	}
    }
    


}

#endif
