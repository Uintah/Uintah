
#include <Packages/rtrt/Core/FastNoise.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MiscMath.h>
#include <Packages/rtrt/Core/MusilRNG.h>

using namespace rtrt;
using namespace SCIRun;

FastNoise::FastNoise(int seed, int tablesize)
    : Noise(seed, tablesize)
{
}

FastNoise::~FastNoise()
{

}

inline int FastNoise::get_index(int x, int y, int z, int)
{
    //return scramble_tab[scramble_tab[scramble_tab[x]^y]^z]+o;
    return scramble_tab[x^y^z];
}


inline double FastNoise::smooth(double x)
{
    // -2x^3+3x^2
    // Derivative is zero at zero and 1
    return x*x*(3.0-2.0*x);
}

inline double FastNoise::interpolate(int idx, double xx, double yy,
				     double zz, double sc)
{
	return sc*(
		   noise_tab[idx]*0.5
		   +noise_tab[idx+1]*xx
		   +noise_tab[idx+2]*yy
		   +noise_tab[idx+3]*zz);
}

double FastNoise::operator()(const Vector& v)
{
    double x=v.x()+10000.0;
    double y=v.y()+10000.0;
    double z=v.z()+10000.0;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
#if 0
    double tx=fx-1.0;
    double ty=fy-1.0;
    double tz=fz-1.0;
#endif

    // Smooth makes the derivative zero at the endpoints
    double smx=smooth(fx);
    double smy=smooth(fy);
    double smz=smooth(fz);
#if 0
    double smx1=1.0-smx;
    double smy1=1.0-smy;
    double smz1=1.0-smz;
#endif

    ix&=bitmask;
    iy&=bitmask;
    iz&=bitmask;
    ix1&=bitmask;
    iy1&=bitmask;
    iz1&=bitmask;

#if 0
    int index=get_index(ix,iy,iz,0);
    double weight=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,0);
    weight+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,0);
    weight+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,0);
    weight+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,0);
    weight+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,0);
    weight+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,0);
    weight+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,0);
    weight+=interpolate(index,tx,ty,tz,(smx*smy*smz));
#endif
    double v000 = noise_tab[get_index(ix,iy,iz,0)];
    double v001 = noise_tab[get_index(ix,iy,iz1,0)];
    double v00 = v000 * (1-smz) + v001*smz;
    double v010 = noise_tab[get_index(ix,iy1,iz,0)];
    double v011 = noise_tab[get_index(ix,iy1,iz1,0)];
    double v01 = v010 * (1-smz) + v011*smz;
    double v100 = noise_tab[get_index(ix1,iy,iz,0)];
    double v101 = noise_tab[get_index(ix1,iy,iz1,0)];
    double v10 = v100 * (1-smz) + v101*smz;
    double v110 = noise_tab[get_index(ix1,iy1,iz,0)];
    double v111 = noise_tab[get_index(ix1,iy1,iz1,0)];
    double v11 = v110 * (1-smz) + v111*smz;
    double v0 = v00 * (1-smy) + v01 * smy;
    double v1 = v10 * (1-smy) + v11 * smy;
    double weight = v0 * (1-smx) + v1 * smx;
    return weight;
}

double FastNoise::operator()(double z)
{
    z+=10000.0;
    int iz=(int)z;
    int iz1=iz+1;
    double fz=z-iz;
    double smz=smooth(fz);
    iz&=bitmask;
    iz1&=bitmask;
    int index=scramble_tab[iz];
    int index1=scramble_tab[iz1];
    double weight=Interpolate(noise_tab[index], noise_tab[index1], smz);
    return 1.0-weight;
}

Vector FastNoise::dnoise(const Vector& v)
{
    double x=v.x()+10000.0;
    double y=v.y()+10000.0;
    double z=v.z()+10000.0;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double tx=fx-1.0;
    double ty=fy-1.0;
    double tz=fz-1.0;
    double smx=smooth(fx);
    double smy=smooth(fy);
    double smz=smooth(fz);
    double smx1=1.0-smx;
    double smy1=1.0-smy;
    double smz1=1.0-smz;

    ix&=bitmask;
    iy&=bitmask;
    iz&=bitmask;
    ix1&=bitmask;
    iy1&=bitmask;
    iz1&=bitmask;

    int index=get_index(ix,iy,iz,4);
    double dx=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,4);
    dx+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,4);
    dx+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,4);
    dx+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,4);
    dx+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,4);
    dx+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,4);
    dx+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,4);
    dx+=interpolate(index,tx,ty,tz,(smx*smy*smz));

    index=get_index(ix,iy,iz,8);
    double dy=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,8);
    dy+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,8);
    dy+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,8);
    dy+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,8);
    dy+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,8);
    dy+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,8);
    dy+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,8);
    dy+=interpolate(index,tx,ty,tz,(smx*smy*smz));

    index=get_index(ix,iy,iz,12);
    double dz=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,12);
    dz+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,12);
    dz+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,12);
    dz+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,12);
    dz+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,12);
    dz+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,12);
    dz+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,12);
    dz+=interpolate(index,tx,ty,tz,(smx*smy*smz));
    return Vector(dx,dy,dz);
}

Vector FastNoise::dnoise(const Vector& v, double& n)
{
    double x=v.x()+10000.0;
    double y=v.y()+10000.0;
    double z=v.z()+10000.0;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double tx=fx-1.0;
    double ty=fy-1.0;
    double tz=fz-1.0;
    double smx=smooth(fx);
    double smy=smooth(fy);
    double smz=smooth(fz);
    double smx1=1.0-smx;
    double smy1=1.0-smy;
    double smz1=1.0-smz;

    int index=get_index(ix,iy,iz,0);
    n=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,0);
    n+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,0);
    n+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,0);
    n+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,0);
    n+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,0);
    n+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,0);
    n+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,0);
    n+=interpolate(index,tx,ty,tz,(smx*smy*smz));

    index=get_index(ix,iy,iz,4);
    double dx=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,4);
    dx+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,4);
    dx+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,4);
    dx+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,4);
    dx+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,4);
    dx+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,4);
    dx+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,4);
    dx+=interpolate(index,tx,ty,tz,(smx*smy*smz));

    index=get_index(ix,iy,iz,8);
    double dy=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,8);
    dy+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,8);
    dy+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,8);
    dy+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,8);
    dy+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,8);
    dy+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,8);
    dy+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,8);
    dy+=interpolate(index,tx,ty,tz,(smx*smy*smz));

    index=get_index(ix,iy,iz,12);
    double dz=interpolate(index,fx,fy,fz,(smx1*smy1*smz1));
    index=get_index(ix,iy,iz1,12);
    dz+=interpolate(index,fx,fy,tz,(smx1*smy1*smz));
    index=get_index(ix,iy1,iz,12);
    dz+=interpolate(index,fx,ty,fz,(smx1*smy*smz1));
    index=get_index(ix,iy1,iz1,12);
    dz+=interpolate(index,fx,ty,tz,(smx1*smy*smz));
    index=get_index(ix1,iy,iz,12);
    dz+=interpolate(index,tx,fy,fz,(smx*smy1*smz1));
    index=get_index(ix1,iy,iz1,12);
    dz+=interpolate(index,tx,fy,tz,(smx*smy1*smz));
    index=get_index(ix1,iy1,iz,12);
    dz+=interpolate(index,tx,ty,fz,(smx*smy*smz1));
    index=get_index(ix1,iy1,iz1,12);
    dz+=interpolate(index,tx,ty,tz,(smx*smy*smz));
    return Vector(dx,dy,dz);
}

namespace SCIRun {
void Pio(SCIRun::Piostream& str, rtrt::FastNoise& obj)
{
  str.begin_cheap_delim();
  Pio(str, (Noise&)obj);
  str.end_cheap_delim();
}
} // end namespace SCIRun
