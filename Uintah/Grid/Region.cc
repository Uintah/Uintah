
#include "Region.h"
#include "NodeSubIterator.h"
#include "Primes.h"
#include "SubRegion.h"
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;
using SCICore::Geometry::Max;
using SCICore::Geometry::Min;
#include <SCICore/Math/MiscMath.h>
using SCICore::Math::Floor;
#include <values.h>

Region::Region(const Point& lower, const Point& upper,
	       int nx, int ny, int nz)
    : lower(lower), upper(upper), nx(nx), ny(ny), nz(nz)
{
}

Region::~Region()
{
}

Vector Region::dCell() const
{
    Vector diag = upper-lower;
    return Vector(diag.x()/nx, diag.y()/ny, diag.z()/nz);
}

void Region::findCell(const Vector& pos, int& ix, int& iy, int& iz) const
{
    Vector cellpos = (pos-lower.asVector())*Vector(nx, ny, nz)/(upper-lower);
    ix = Floor(cellpos.x());
    iy = Floor(cellpos.y());
    iz = Floor(cellpos.z());
}

bool Region::findCellAndWeights(const Vector& pos,
				Array3Index ni[8], double S[8]) const
{
    Vector cellpos = (pos-lower.asVector())*Vector(nx, ny, nz)/(upper-lower);
    int ix = Floor(cellpos.x());
    int iy = Floor(cellpos.y());
    int iz = Floor(cellpos.z());
    ni[0] = Array3Index(ix, iy, iz);
    ni[1] = Array3Index(ix, iy, iz+1);
    ni[2] = Array3Index(ix, iy+1, iz);
    ni[3] = Array3Index(ix, iy+1, iz+1);
    ni[4] = Array3Index(ix+1, iy, iz);
    ni[5] = Array3Index(ix+1, iy, iz+1);
    ni[6] = Array3Index(ix+1, iy+1, iz);
    ni[7] = Array3Index(ix+1, iy+1, iz+1);
    double fx = cellpos.x() - ix;
    double fy = cellpos.y() - iy;
    double fz = cellpos.z() - iz;
    double fx1 = 1-fx;
    double fy1 = 1-fy;
    double fz1 = 1-fz;
    S[0] = fx1 * fy1 * fz1;
    S[1] = fx1 * fy1 * fz;
    S[2] = fx1 * fy * fz1;
    S[3] = fx1 * fy * fz;
    S[4] = fx * fy1 * fz1;
    S[5] = fx * fy1 * fz;
    S[6] = fx * fy * fz1;
    S[7] = fx * fy * fz;
    return ix>= 0 && iy>=0 && iz>=0 && ix<nx && iy<ny && iz<nz;
}

bool Region::findCellAndShapeDerivatives(const Vector& pos,
					 Array3Index ni[8],
					 Vector d_S[8]) const
{
    Vector cellpos = (pos-lower.asVector())*Vector(nx, ny, nz)/(upper-lower);
    int ix = Floor(cellpos.x());
    int iy = Floor(cellpos.y());
    int iz = Floor(cellpos.z());
    ni[0] = Array3Index(ix, iy, iz);
    ni[1] = Array3Index(ix, iy, iz+1);
    ni[2] = Array3Index(ix, iy+1, iz);
    ni[3] = Array3Index(ix, iy+1, iz+1);
    ni[4] = Array3Index(ix+1, iy, iz);
    ni[5] = Array3Index(ix+1, iy, iz+1);
    ni[6] = Array3Index(ix+1, iy+1, iz);
    ni[7] = Array3Index(ix+1, iy+1, iz+1);
    double fx = cellpos.x() - ix;
    double fy = cellpos.y() - iy;
    double fz = cellpos.z() - iz;
    double fx1 = 1-fx;
    double fy1 = 1-fy;
    double fz1 = 1-fz;
    d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
    d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
    d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
    d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
    d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
    d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
    d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
    d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
    return ix>= 0 && iy>=0 && iz>=0 && ix<nx && iy<ny && iz<nz;
}

void decompose(int numProcessors, int sizex, int sizey, int sizez,
	       int& numProcessors_x, int& numProcessors_y,
	       int& numProcessors_z)
{
    Primes::FactorType factors;
    int nump = Primes::factorize(numProcessors, factors);
    
    int axis[Primes::MaxFactors];
    for(int i=0;i<nump;i++)
	axis[i]=0;
    numProcessors_x=numProcessors_y=numProcessors_z=-1;
    double min = MAXDOUBLE;
    for(;;){
	// Compute area
	int p[3];
	p[0]=p[1]=p[2]=1;
	for(int i=0;i<nump;i++)
	    p[axis[i]] *= factors[i];
	double area = 2*(double(p[0]-1)*sizey*sizez
			 +double(p[1]-1)*sizex*sizez
			 +double(p[2]-1)*sizex*sizey);

	if(area < min){
	    min = area;
	    numProcessors_x=p[0];
	    numProcessors_y=p[1];
	    numProcessors_z=p[2];
	}

	// Go to next combination
	int i;
	for(i=0;i<nump;i++){
	    axis[i]++;
	    if(axis[i]>=3){
		axis[i]=0;
	    } else {
		break;
	    }
	}
	if(i==nump)
	    break;
    }
}

void Region::subregionIteratorPair(int i, int n,
				   NodeSubIterator& iter,
				   NodeSubIterator& end) const
{
    int npx, npy, npz;
    int nodesx = nx+1;
    int nodesy = ny+1;
    int nodesz = nz+1;
    decompose(n, nodesx, nodesy, nodesz, npx, npy, npz);
    int ipz = i%npz;
    int ipy = (i/npz)%npy;
    int ipx = i/npz/npy;
    int sx = ipx*nodesx/npx;
    int ex = (ipx+1)*nodesx/npx;
    int sy = ipy*nodesy/npy;
    int ey = (ipy+1)*nodesy/npy;
    int sz = ipz*nodesz/npz;
    int ez = (ipz+1)*nodesz/npz;
    iter = NodeSubIterator(sx, sy, sz, ex, ey, ez);
    end = NodeSubIterator(ex, ey, ez, ex, ey, ez);
}

SubRegion Region::subregion(int i, int n) const
{
    int npx, npy, npz;
    int nodesx = nx+1;
    int nodesy = ny+1;
    int nodesz = nz+1;
    decompose(n, nodesx, nodesy, nodesz, npx, npy, npz);
    int ipz = i%npz;
    int ipy = (i/npz)%npy;
    int ipx = i/npz/npy;
    int sx = ipx*nodesx/npx;
    int ex = (ipx+1)*nodesx/npx - 1;
    int sy = ipy*nodesy/npy;
    int ey = (ipy+1)*nodesy/npy - 1;
    int sz = ipz*nodesz/npz;
    int ez = (ipz+1)*nodesz/npz - 1;
    Vector diag(upper-lower);
    Point l(lower+diag*Vector(sx-1, sy-1, sz-1)/Vector(nx, ny, nz));
    Point u(lower+diag*Vector(ex+1, ey+1, ez+1)/Vector(nx, ny, nz));
    l=Max(l, lower); // For "ghost cell"
    u=Min(u, upper);
    return SubRegion(l, u, sx, sy, sz, ex, ey, ez);
}
