/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Region.h>
#include <SCICore/Math/MiscMath.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/NodeSubIterator.h>
#include <Uintah/Grid/SubRegion.h>
#include <Uintah/Math/Primes.h>
#include <values.h>

using namespace Uintah;
using namespace SCICore::Geometry;
using namespace std;
using SCICore::Math::Floor;

Region::Region(const Point& lower, const Point& upper,
	       const IntVector& res)
    : d_box(lower, upper), d_res(res)
{
}

Region::~Region()
{
}

#if 0
void Region::findCell(const Vector& pos, int& ix, int& iy, int& iz) const
{
    Vector cellpos = (pos-d_lower.asVector()) * 
                      Vector(d_nx, d_ny, d_nz) / (d_upper-d_lower);
    ix = Floor(cellpos.x());
    iy = Floor(cellpos.y());
    iz = Floor(cellpos.z());
}
#endif

bool Region::findCellAndWeights(const Point& pos,
				IntVector ni[8], double S[8]) const
{
   Vector cellpos = (pos-d_box.lower())*d_res/(d_box.upper()-d_box.lower());
   int ix = Floor(cellpos.x());
   int iy = Floor(cellpos.y());
   int iz = Floor(cellpos.z());
   ni[0] = IntVector(ix, iy, iz);
   ni[1] = IntVector(ix, iy, iz+1);
   ni[2] = IntVector(ix, iy+1, iz);
   ni[3] = IntVector(ix, iy+1, iz+1);
   ni[4] = IntVector(ix+1, iy, iz);
   ni[5] = IntVector(ix+1, iy, iz+1);
   ni[6] = IntVector(ix+1, iy+1, iz);
   ni[7] = IntVector(ix+1, iy+1, iz+1);
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
   return ix>= 0 && iy>=0 && iz>=0 && ix<d_res.x() && iy<d_res.y() && iz<d_res.z();
}

#if 0
bool Region::findCellAndShapeDerivatives(const Vector& pos,
					 Array3Index ni[8],
					 Vector d_S[8]) const
{
    Vector cellpos = (pos-d_lower.asVector())*
                      Vector(d_nx, d_ny, d_nz)/(d_upper-d_lower);
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
    return ix>= 0 && iy>=0 && iz>=0 && ix<d_nx && iy<d_ny && iz<d_nz;
}
#endif

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

#if 0
void Region::subregionIteratorPair(int i, int n,
				   NodeSubIterator& iter,
				   NodeSubIterator& end) const
{
    int npx, npy, npz;
    int nodesx = d_nx+1;
    int nodesy = d_ny+1;
    int nodesz = d_nz+1;
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
    int nodesx = d_nx+1;
    int nodesy = d_ny+1;
    int nodesz = d_nz+1;
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
    Vector diag(d_upper-d_lower);
    Point l(d_lower+diag*Vector(sx-1, sy-1, sz-1)/Vector(d_nx, d_ny, d_nz));
    Point u(d_lower+diag*Vector(ex+1, ey+1, ez+1)/Vector(d_nx, d_ny, d_nz));
    l=Max(l, d_lower); // For "ghost cell"
    u=Min(u, d_upper);
    return SubRegion(l, u, sx, sy, sz, ex, ey, ez);
}
#endif

ostream& operator<<(ostream& out, const Region* r)
{
  out << "(Region: box=" << r->getBox() << ", res=" << r->getNCells() << ")";
  return out;
}

long Region::totalCells() const
{
  return d_res.x()*d_res.y()*d_res.z();
}

void Region::performConsistencyCheck() const
{
  if(d_res.x() < 1 || d_res.y() < 1 || d_res.z() < 1)
    throw InvalidGrid("Degenerate region");
}

CellIterator Region::getCellIterator(const Box& b) const
{
   Vector diag = d_box.upper()-d_box.lower();
   Vector l = (b.lower() - d_box.lower())*d_res/diag;
   Vector u = (b.upper() - d_box.lower())*d_res/diag;
   return CellIterator((int)l.x(), (int)l.y(), (int)l.z(),
		       RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
}
      
NodeIterator Region::getNodeIterator() const
{
   return NodeIterator(0, 0, 0,
		       d_res.x()+1, d_res.y()+1, d_res.z()+1);
}
      

//
// $Log$
// Revision 1.11  2000/05/02 20:13:05  sparker
// Implemented findCellAndWeights
//
// Revision 1.10  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.9  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.8  2000/04/28 03:58:20  sparker
// Fixed countParticles
// Implemented createParticles, which doesn't quite work yet because the
//   data warehouse isn't there yet.
// Reduced the number of particles in the bar problem so that it will run
//   quickly during development cycles
//
// Revision 1.7  2000/04/27 23:18:50  sparker
// Added problem initialization for MPM
//
// Revision 1.6  2000/04/26 06:48:54  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/12 23:00:49  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//
