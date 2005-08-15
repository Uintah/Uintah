
#include <Packages/rtrt/Core/Noise.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Spline.h>
#include <Packages/rtrt/Core/Random.h>

using namespace rtrt;

Noise::Noise(const Noise &n)
{
    tablesize=n.tablesize;
    bitmask=n.bitmask;
    noise_tab=n.noise_tab;
    scramble_tab=n.scramble_tab;
}

Noise::Noise(int seed, int _tablesize)
{
    tablesize=_tablesize;
    bitmask=tablesize-1;
    RandomTable r1(seed, tablesize+15);
    noise_tab=r1.random();
    RandomTable r2(seed, tablesize);
    scramble_tab=r2.scramble();
}

Noise::~Noise()
{
    
}

inline int Noise::get_index(int x, int y, int z)
{
    return scramble_tab[scramble_tab[scramble_tab[x]^y]^z];
}

inline double Noise::lattice(int x,int y,int z)
{
    return noise_tab[get_index(x&bitmask, y&bitmask, z&bitmask)];
}

double Noise::operator()(const Vector& v)
{
    double x=v.x()+10000.0;if(x<0)x=-x;
    double y=v.y()+10000.0;if(y<0)y=-y;
    double z=v.z()+10000.0;if(z<0)z=-z;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    
    // Spline in x
    double p[4][4];
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    p[i][j]=CRSpline(fx, lattice(ix+0, iy+i, iz+j),
			     lattice(ix+1, iy+i, iz+j),
			     lattice(ix+2, iy+i, iz+j),
			     lattice(iz+3, iy+i, iz+j));
	}
    }
    
    // Spline in y
    double pp[4];
    for(int j=0;j<4;j++){
	pp[j]=CRSpline(fy, p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    
    // Spline in z
    double ppp=CRSpline(fz, pp[0], pp[1], pp[2], pp[3]);
    return ppp;
}

double Noise::operator()(double z)
{
    z+=10000.0;if(z<0)z=-z;
    int iz=(int)z;
    double fz=z-iz;
    double weight=CRSpline(fz, lattice(0,0,iz), lattice(0,0,iz+1),
			   lattice(0,0,iz+2), lattice(0,0,iz+3));
    return weight;
}

Vector Noise::dnoise(const Vector& v, double& n)
{
    double x=v.x()+10000.0;if(x<0)x=-x;
    double y=v.y()+10000.0;if(y<0)y=-y;
    double z=v.z()+10000.0;if(z<0)z=-z;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double p[4][4];
    double pp[4];
    
    // Interpolate in y,z, take derivative in x
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    p[i][j]=CRSpline(fy, lattice(ix+j, iy+0, iz+i),
			     lattice(ix+j, iy+1, iz+i),
			     lattice(ix+j, iy+2, iz+i),
			     lattice(ix+j, iy+3, iz+i));
	}
    }
    for(int j=0;j<4;j++){
	pp[j]=CRSpline(fz, p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    double dx=dCRSpline(fx, pp[0], pp[1], pp[2], pp[3]);
    
    // Interpolate in z,x, take derivative in y
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    p[i][j]=CRSpline(fz, lattice(ix+i, iy+j, iz+0),
			     lattice(ix+i, iy+j, iz+1),
			     lattice(ix+i, iy+j, iz+2),
			     lattice(ix+i, iy+j, iz+3));
	}
    }
    for(int j=0;j<4;j++){
	pp[j]=CRSpline(fx, p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    double dy=dCRSpline(fy, pp[0], pp[1], pp[2], pp[3]);
    
    // Interpolate in x,y, take derivative in z
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    p[i][j]=CRSpline(fx, lattice(ix+0, iy+i, iz+j),
			     lattice(ix+1, iy+i, iz+j),
			     lattice(ix+2, iy+i, iz+j),
			     lattice(ix+3, iy+i, iz+j));
	}
    }
    for(int j=0;j<4;j++){
	pp[j]=CRSpline(fy, p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    double dz=dCRSpline(fz, pp[0], pp[1], pp[2], pp[3]);
    
    // Calculate the noise too, since we are almost there
    n=CRSpline(fz, pp[0], pp[1], pp[2], pp[3]);
    return Vector(dx,dy,dz);
}

Vector Noise::dnoise(const Vector& p)
{
    double tmp;
    return dnoise(p, tmp);
}


namespace SCIRun {
void Pio(SCIRun::Piostream& str, rtrt::Noise& obj)
{
  str.begin_cheap_delim();
  Pio(str, obj.tablesize);
  Pio(str, obj.bitmask);

  if (str.reading()) {
    // seed is not stored with the class, so hack an int to seed with. 
    // if we want the same class, we should pio the seed as well.
    RandomTable r1(obj.tablesize, obj.tablesize+15);
    obj.noise_tab=r1.random();
    RandomTable r2(obj.tablesize, obj.tablesize);
    obj.scramble_tab=r2.scramble();
  }
  str.end_cheap_delim();
}
} // end namespace SCIRun
