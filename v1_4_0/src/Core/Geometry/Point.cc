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
 *  Point.cc: ?
 *
 *  Written by:
 *   Author ?
 *   Department of Computer Science
 *   University of Utah
 *   Date ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Tester/RigorousTest.h>
#include <iostream>
#include <stdio.h>

using std::cerr;
using std::endl;
using std::istream;
using std::ostream;

namespace SCIRun {


Point Interpolate(const Point& p1, const Point& p2, double w)
{
    return Point(
	Interpolate(p1._x, p2._x, w),
	Interpolate(p1._y, p2._y, w),
	Interpolate(p1._z, p2._z, w));
}

string Point::get_string() const
{
    char buf[100];
    sprintf(buf, "[%g, %g, %g]", _x, _y, _z);
    return buf;
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
   os << p.get_string();
   return os;
}

istream& operator>>( istream& is, Point& v)
{
    double x, y, z;
    char st;
  is >> st >> x >> st >> y >> st >> z >> st;
  v=Point(x,y,z);
  return is;
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


#include <Core/Geometry/Vector.h>

void Point::test_rigorous(RigorousTest* __test)
{	


    //Basic Point Tests

    Point P(0,0,0);
    Point Q(0,0,0);

    int x;
    for(x=0;x<=100;++x){
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
    double Xd,Yd,Zd;
    const double cnst = 1234.5678;      
    Point m_A(1,2,3);
    Point m_A_prime(1,2,3);
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
		
		//Point-Point=Vector Tests
		
		v3=p2-p1;
		X=(int)(p2.x()-p1.x());
		Y=(int)(p2.y()-p1.y());
		Z=(int)(p2.z()-p1.z());

		TEST(v3.x()==X);
		TEST(v3.y()==Y);
		TEST(v3.z()==Z);
		
		//Point+Vector=Point Tests
		
		p2=p1+v1;
		X=(int)(p1.x()+v1.x());
		Y=(int)(p1.y()+v1.y());
		Z=(int)(p1.z()+v1.z());

		TEST(p2.x()==X);
		TEST(p2.y()==Y);
		TEST(p2.z()==Z);
		
		//Point+=Vector Tests
		p1.x(x);
		p1.y(y);
		p1.z(z);
		TEST((p1.x()==x)&&(p1.y()==y)&&(p1.z()==z));
		v1.x(z);
		v1.y(x);
		v1.z(y);
		TEST((v1.x()==z)&&(v1.y()==x)&&(v1.z()==y));
		X = (int)(p1.x()+v1.x());
		Y = (int)(p1.y()+v1.y());
		Z = (int)(p1.z()+v1.z());

		p1+=v1;
		TEST(p1.x()==X);
		TEST(p1.y()==Y);
		TEST(p1.z()==Z);
	      
		
		//Point-Vector=Point Tests
		
		p1.x(x);
		p1.y(y);
		p1.z(z);
		TEST((p1.x()==x)&&(p1.y()==y)&&(p1.z()==z));
		v1.x(z);
		v1.y(x);
		v1.z(y);
		TEST((v1.x()==z)&&(v1.y()==x)&&(v1.z()==y));
		
		X=(int)(p1.x()-v1.x());
		Y=(int)(p1.y()-v1.y());
		Z=(int)(p1.z()-v1.z());
	  
		
		p3=p1-v1;
		TEST(p3.x()==X);
		TEST(p3.y()==Y);
		TEST(p3.z()==Z);

		//Point-=Vector Tests
		X = (int)(p1.x()-v1.x());
		Y = (int)(p1.y()-v1.y());
		Z = (int)(p1.z()-v1.z());
		
		p1-=v1;
		
		TEST(p1.x()==X);
		TEST(p1.y()==Y);
		TEST(p1.z()==Z);
		
		//Constant*Point=Point Tests

		m_A_prime=m_A*cnst;
		
		Xd = m_A.x()*cnst;
		Yd = m_A.y()*cnst;
		Zd = m_A.z()*cnst;

		TEST (m_A_prime.x()==Xd);
		TEST (m_A_prime.y()==Yd);
		TEST (m_A_prime.z()==Zd);
		
		//Point*=Point Tests
		
		m_A_prime.x(double(x));
		m_A_prime.y(double(y));
		m_A_prime.z(double(z));

		TEST(m_A_prime.x()==x&&m_A_prime.y()==y&&m_A_prime.z()==z);
		
		m_A_prime=m_A_prime*cnst;

		Xd=m_A_prime.x()*cnst;
		Yd=m_A_prime.y()*cnst;
		Zd=m_A_prime.z()*cnst;

		m_A_prime*=cnst;

		TEST(m_A_prime.x()==Xd);
		TEST(m_A_prime.y()==Yd);
		TEST(m_A_prime.z()==Zd);



		//Point/Constant=Point Tests

		m_A_prime=m_A/cnst;
		
		Xd = m_A.x()/cnst;
		Yd = m_A.y()/cnst;
		Zd = m_A.z()/cnst;

		TEST (m_A_prime.x()==Xd);
		TEST (m_A_prime.y()==Yd);
		TEST (m_A_prime.z()==Zd);
		
		//Point/=Constant Tests

		Xd = m_A_prime.x()/cnst;
		Yd = m_A_prime.y()/cnst;
		Zd = m_A_prime.z()/cnst;

		m_A_prime/=cnst;
		
		TEST(m_A_prime.x()==Xd);
		TEST(m_A_prime.y()==Yd);
		TEST(m_A_prime.z()==Zd);
	
#if 0
		//Point-Constant Tests
		
		v1.x(x);
		v1.y(y);
		v1.z(z);

		v2.x(0);
		v2.y(0);
		v2.z(0);

		X = v1.x()-cnst;
		Y = v1.y()-cnst;
		Z = v1.z()-cnst;
		
		v2=v1-cnst;
		
		TEST(v2.x()==X);
		TEST(v2.y()==Y);
		TEST(v2.z()==Z);
#endif	
		
	    }
	}
    }  
}

void Pio(Piostream& stream, Point& p)
{

    stream.begin_cheap_delim();
    Pio(stream, p._x);
    Pio(stream, p._y);
    Pio(stream, p._z);
    stream.end_cheap_delim();
}

const string& 
Point::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

const TypeDescription* get_type_description(Point*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Point", Point::get_h_file_path(), 
				"SCIRun");
  }
  return td;
}

} // End namespace SCIRun




