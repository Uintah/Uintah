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

void Point::test_rigorous(RigorousTest* __test)
{	

    //Basic Point Tests

    Point P(0,0,0);
    Point Q(0,0,0);

    for(int x=-100;x<=100;++x){
	for(int y=-100;y<=100;++y){
	    for(int z=-100;z<=100;++z){
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

    for(x=-100;x<=100;++x){
	for(int y=-100;y<=100;++y){
	    for(int z=-100;z<=100;++z){
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
   
    //Point Testss

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

    
    for (x=-10;x<10;++x){
	for (int y=-10;y<10;++y){
	    for (int z=-10;z<10;++z){
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
		X=p2.x()-p1.x();
		Y=p2.y()-p1.y();
		Z=p2.z()-p1.z();

		TEST(v3.x()==X);
		TEST(v3.y()==Y);
		TEST(v3.z()==Z);
		
		//Point+Vector=Point Tests
		
		p2=p1+v1;
		X=p1.x()+v1.x();
		Y=p1.y()+v1.y();
		Z=p1.z()+v1.z();

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
		X = p1.x()+v1.x();
		Y = p1.y()+v1.y();
		Z = p1.z()+v1.z();

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
		
		X=p1.x()-v1.x();
		Y=p1.y()-v1.y();
		Z=p1.z()-v1.z();
	  
		
		p3=p1-v1;
		TEST(p3.x()==X);
		TEST(p3.y()==Y);
		TEST(p3.z()==Z);

		//Point-=Vector Tests
		X = p1.x()-v1.x();
		Y = p1.y()-v1.y();
		Z = p1.z()-v1.z();
		
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
	
		//Point operator- const Tests
		
		Point pt(x,y,z);
		X = pt.x()-(2*pt.x());
		Y = pt.y()-(2*pt.y());
		Z = pt.z()-(2*pt.z());

		pt = -pt;
		
		TEST(pt.x()==X);
		TEST(pt.y()==Y);
		TEST(pt.z()==Z);
		
	        //Dot Multiplication Tests
		
		Point a1(x,y,z);
		Point a2(z,x,y);
		
		Vector b1(x,y,z);
		Vector b2(z,x,y);

		TEST((a1.x()==x)&&(a1.y()==y)&&(a1.z()==z));
		TEST((a2.x()==z)&&(a2.y()==x)&&(a2.z()==y));

		TEST((b1.x()==x)&&(b1.y()==y)&&(b1.z()==z));
		TEST((b2.x()==z)&&(b2.y()==x)&&(b2.z()==y));
		

		//Dot(Point,Point) Test
		TEST((Dot(a1,a2))==((a1.x()*a2.x())+(a1.y()*a2.y())+(a1.z()*a2.z())));

		//Dot(Point,Vector) Test
		TEST((Dot(a1,b2))==((a1.x()*b2.x())+(a1.y()*b2.y())+(a1.z()*b2.z())));

		//Dot(Vector,Point) Test
		TEST((Dot(b1,a2))==((b1.x()*a2.x())+(b1.y()*a2.y())+(b1.z()*a2.z())));

		//Dot(Vector,Vector) Test
		TEST((Dot(b1,b2))==((b1.x()*b2.x())+(b1.y()*b2.y())+(b1.z()*b2.z())));
		
		//Vector() Tests
		Point p(x,y,z);		
		TEST((p+p.vector())==(p*2));	//Make sure p.vector() is a vector
		
		
		//Max Tests
		Point pa(x,y,z); //These Points are also used for the AffineCombination() Tests
		Point pb(z,x,y);
		Point pc(y,z,x);
		Point pd((z-10),(x+15),(y-7));
		
		Point max=Max(pa,pb);

		if(pa.x()>=pb.x())
		  TEST(max.x()==pa.x());
		if(pa.x()<pb.x())
		  TEST(max.x()==pb.x());

		if(pa.y()>=pb.y())
		  TEST(max.y()==pa.y());
		if(pa.y()<pb.y())
		  TEST(max.y()==pb.y());

		if(pa.z()>=pb.z())
		  TEST(max.z()==pa.z());
		if(pa.z()<pb.z())
		  TEST(max.z()==pb.z());

		//Min Tests
		Point min=Min(pa,pb);

		if(pa.x()<=pb.x())
		  TEST(min.x()==pa.x());
		if(pa.x()>pb.x())
		  TEST(min.x()==pb.x());

		if(pa.y()<=pb.y())
		  TEST(min.y()==pa.y());
		if(pa.y()>pb.y())
		  TEST(min.y()==pb.y());

		if(pa.z()<=pb.z())
		  TEST(min.z()==pa.z());
		if(pa.z()>pb.z())
		  TEST(min.z()==pb.z());
				  
		


		//AffineCombination Tests

		double c=0;
		double d=0;
		double e=0;
		double f=0;
		Point a;
		
		for(int i=1;i<=10;++i){
		  TEST(pa.x()==x);
		  
		  c = .1*double(i);
		  d = 1-c;

		  a = AffineCombination(pa,c,pb,d);
		  TEST(a.x()==((c*pa.x())+(d*pb.x())));
		  TEST(a.y()==((c*pa.y())+(d*pb.y())));
		  TEST(a.z()==((c*pa.z())+(d*pb.z())));
		  
		}
		    
		for(i=0;i<=10;++i){
		  c=.1*double(i);
		  for(int i2=0;i2<=(10-i);++i2){
		    d = .1*double(i2);
		    e = (1-(c+d));
		    
		    TEST(c+d+e==1);

		    a = AffineCombination(pa,c,pb,d,pc,e);
		    TEST((a.x())==((c*pa.x())+(d*pb.x())+(e*pc.x())));
		    TEST((a.y())==((c*pa.y())+(d*pb.y())+(e*pc.y())));
		    TEST((a.z())==((c*pa.z())+(d*pb.z())+(e*pc.z())));   
		  }
		}	


		for(i=0;i<=10;++i){
		  c=.1*double(i);
		  for(int i2=0;i2<=(10-i);++i2){
		    d=.1*double(i2);
		    for(int i3=0;i3<=(10-(i+i2));++i3){
		      e=.1*double(i3);
		      f=1-(c+d+e);
		      TEST(c+d+e+f==1);
		      a = AffineCombination(pa,c,pb,d,pc,e,pd,f);
		      TEST((a.x())==((c*pa.x())+(d*pb.x())+(e*pc.x())+(f*pd.x())));
		     
		      
		    }
		  }
		}
		    
		//Interpolate() Tests
		
		for(i=1;i<=10;++i){
		  c=.1*double(i);
		  a=Interpolate(pa,pb,c);
		  TEST((a.x())==(pb.x()*c+pa.x()*(1.0-c)));
		  TEST((a.y())==(pb.y()*c+pa.y()*(1.0-c)));
		  TEST((a.z())==(pb.z()*c+pa.z()*(1.0-c)));
		}
		
		//InInterval Tests
		

		double epsilon,dist;

		for(i=1;i<=100;++i){
		  epsilon=i;
		  X=(pa.x()-pb.x());
		  Y=(pa.y()-pb.y());
		  Z=(pa.z()-pb.z());
		  dist=Max(Abs(pa.x()-pb.x()),Abs(pa.y()-pb.y()),Abs(pa.z()-pb.z()));
		  dist/=2;	
		  if(dist<epsilon){
		    TEST(pa.InInterval(pb,epsilon));
		   

		  }
		  if(dist>=epsilon+1.e-6)
		    TEST(!(pa.InInterval(pb,epsilon)));
		}
	    }    
	}
    }


    //Vector Tests

    
    for(x=-100;x<=100;++x){
	for(int y=-100;y<=100;++y){
	    for(int z=-100;z<=100;++z){
	      Vector v1(x,y,z);
	      TEST(v1.x()==x);
	      TEST(v1.y()==y);
	      TEST(v1.z()==z);

	      Vector v2=v1;
	      TEST(v2==v1);
	      TEST(v1.x()==v2.x());
	      TEST(v1.y()==v2.y());
	      TEST(v1.z()==v2.z());

	      Vector vx(1,1,1);
	      
	      v1+=vx;
	      TEST(!(v2==v1));
	      TEST(!(v1.x()==v2.x()));
	      TEST(!(v1.y()==v2.y()));
	      TEST(!(v1.z()==v2.z()));
	      

	    }
	}
    }
		
    //Dot() is tested above.
   
    
    for(x=-10;x<=10;++x){
      for(int y=-10;y<=10;++y){
	for(int z=-10;z<=10;++z){
	  Vector v1(x,y,z);
	  TEST((v1.length())==(sqrt(x*x+y*y+z*z)));
	  TEST((v1.length2())==(x*x+y*y+z*z));

	  //Vector*double Tests
	  const double c = 123.4567;
	  Vector v2=v1*c;
	  TEST(v2.x()==(v1.x()*c));
	  TEST(v2.y()==(v1.y()*c));
	  TEST(v2.z()==(v1.z()*c));

	  //Vector*=double Tests
	  v2=v1;
	  v1*=c;
	  TEST(v1.x()==(v2.x()*c));
	  TEST(v1.y()==(v2.y()*c));
	  TEST(v1.z()==(v2.z()*c));

	  //Vector/double Tests
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  v2 = v1/c;
	  TEST(v2.x()==(v1.x()/c));
	  TEST(v2.y()==(v1.y()/c));
	  TEST(v2.z()==(v1.z()/c));
	  
	  //Vector/Vector Tests

	  Vector v3(0,0,0);
	  
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  v2.x(z);
	  v2.y(x);
	  v2.z(y);

	  v3 = v1/v2;

	  TEST((v3.x()==(v1.x()/v2.x()))||(v2.x()==0));

	  //Vector+Vector Tests
	  v3=v1+v2;
	  TEST(v3.x()==(v1.x()+v2.x()));
	  TEST(v3.y()==(v1.y()+v2.y()));
	  TEST(v3.z()==(v1.z()+v2.z()));

	  //Vector+=Vector Tests
	  v3=v1;
	  TEST(v3==v1);
	  v1+=v2;
	  TEST(v1.x()==(v3.x()+v2.x()));
	  TEST(v1.y()==(v3.y()+v2.y()));
	  TEST(v1.z()==(v3.z()+v2.z()));
	  
	  //-Vector Tests

	  v2= -v1;
	  TEST(v2.x()==-(v1.x()));
	  TEST(v2.y()==-(v1.y()));
	  TEST(v2.z()==-(v1.z()));

	  //Vector-Vector Tests
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  v2.x(y);
	  v2.y(z);
	  v2.z(x);
	  
	  v3=v1-v2;

	  TEST(v3.x()==v1.x()-v2.x());


	  //Vector-=Vector Tests
	  v3=v1;
	  
	  v1-=v2;

	  TEST(v1.x()==v3.x()-v2.x());
	  TEST(v1.y()==v3.y()-v2.y());
	  TEST(v1.z()==v3.z()-v2.z());
				      
	  //Cross-Product Tests
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);
	  
	  v2.x(z);
	  v2.y(x);
	  v2.z(y);

	  v3 = Cross(v1,v2);

	  TEST(v3.x()==v1.y()*v2.z()-v1.z()*v2.y());
	  TEST(v3.y()==v1.z()*v2.x()-v1.x()*v2.z());
	  TEST(v3.z()==v1.x()*v2.y()-v1.y()*v2.x());
	  
	  //Absolute Value Tests
	  
	  v3=Abs(v1);

	  TEST(v3.x()==abs(v1.x()));
	  TEST(v3.y()==abs(v1.y()));
	  TEST(v3.z()==abs(v1.z()));

	  //rotz90() tests

	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  //Rotate 0 Degrees
	  v1.rotz90(0);
	  TEST(v1.x()==x);
	  TEST(v1.y()==y);
	  TEST(v1.z()==z);

	  //Rotate 90 Degrees
	  v1.rotz90(1);
	  TEST(v1.x()==(-y));
	  TEST(v1.y()==x);
	  TEST(v1.z()==z);

	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  //Rotate 180 Degrees
	  v1.rotz90(2);
	  TEST(v1.x()==-x);
	  TEST(v1.y()==-y);
	  TEST(v1.z()==z);
	
	  //Rotate 270 Degrees
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);
	  
	  v1.rotz90(3);

	  TEST(v1.x()==y);
	  TEST(v1.y()==-x);
	  TEST(v1.z()==z);

	  //Vector.point() Tests

	  Point pt = v1.point();

	  TEST(pt.x()==v1.x());
	  TEST(pt.y()==v1.y());
	  TEST(pt.z()==v1.z());

	  //Interpolate Tests

	  v1.x(x);
	  v1.y(y);
	  v1.z(z);

	  v2.x(z);
	  v2.y(x);
	  v2.z(y);

	  for(int a=1;c<=10;++a){
	    double cnst = a*.1;
	    v3 = Interpolate(v1,v2,cnst);
	    TEST(v3.x()==(Abs(v2.x()-v1.x())*c));
	    TEST(v3.y()==(Abs(v2.y()-v1.y())*c));
	    TEST(v3.z()==(Abs(v2.z()-v1.z())*c));
	  }

	
	  //asPoint() Tests
	  v1.x(x);
	  v1.y(y);
	  v1.z(z);
	  Point npt=v1.asPoint();

	  TEST(npt.x()==x);
	  TEST(npt.y()==y);
	  TEST(npt.z()==z);
			   
	  //minComponent() Tests
	  if(x<=y&&x<=z)
	    TEST(v1.minComponent()==double(x));
	  if(y<=x&&y<=z)
	    TEST(v1.minComponent()==double(y));
	  if(z<=x&&z<=y)
	    TEST(v1.minComponent()==double(z));

	  //maxComponent() Tests
	  if(x>=y&&x>=z)
	    TEST(v1.maxComponent()==double(x));
	  if(y>=x&&y>=z)
	    TEST(v1.maxComponent()==double(y));
	  if(z>=x&&z>=y)
	    TEST(v1.maxComponent()==double(z));
	  
	  
	  
	  
	    
	    
	  
	}
      }
    } 
           
    //(+,+,+) octant only tests

    //Vector.normal() and Vector.normalize() Tests
    
    for(x=1;x<=10;++x){
      for(int y=1;y<=10;++y){
	for(int z=1;z<=10;++z){

	  Vector va(x,y,z);
	  va.normalize();
	  Vector vb,vc;
	  va.find_orthogonal(vb,vc);
	  TEST(Abs(Dot(va,vb)) <1.e-6);
	  TEST(Abs(Dot(va,vc)) < 1.e-6);
	  TEST(Abs(Dot(vb,vc)) < 1.e-6);

	  TEST(vb.length()>.999999 && vb.length() < 1.0000001);
	  TEST(vc.length()>.999999 && vc.length() < 1.0000001);

	  

	  
	}
      }
    }    



    
    //Point.string() Tests
    Point pst1(1,2,3);
    TEST(pst1.string()=="[1, 2, 3]");

    Point pst2(1.2,3.4,5.6);
    TEST(pst2.string()=="[1.2, 3.4, 5.6]");
    
    Point pst3(12.3,45.6,78.9);
    TEST(pst3.string()=="[12.3, 45.6, 78.9]");

    //Vector.string() Tests

    Vector vst1(1,2,3);
    TEST(vst1.string()=="[1, 2, 3]");

    Vector vst2(1.2,3.4,5.6);
    TEST(vst2.string()=="[1.2, 3.4, 5.6]");
    
    Vector vst3(12.3,45.6,78.9);
    TEST(vst3.string()=="[12.3, 45.6, 78.9]");
    
    


}




#endif

