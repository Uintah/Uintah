/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Region.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/NodeSubIterator.h>
#include <Uintah/Grid/SubRegion.h>
#include <Uintah/Math/Primes.h>

#include <SCICore/Exceptions/InternalError.h>
#include <SCICore/Thread/AtomicCounter.h>

#include <values.h>
#include <iostream>

using namespace Uintah;
using namespace SCICore::Geometry;
using namespace std;
using SCICore::Exceptions::InternalError;
using SCICore::Math::Floor;
static SCICore::Thread::AtomicCounter ids("Region ID counter");

Region::Region(const Point& lower, const Point& upper,
	       const IntVector& lowIndex, const IntVector& highIndex,
	       int id)
    : d_box(lower, upper), d_lowIndex(lowIndex), d_highIndex(highIndex),
      d_id( id )
{
   d_res = highIndex - lowIndex;

   if(d_id == -1)
      d_id = ids++;

   for(int i=0;i<27;i++)
      d_neighbors[i]=0;
   d_neighbors[1*9+1*3+1]=this;

   int gc = 2; // Number of ghostcells for this region...

   determineGhostRegions( gc );
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
   ni[0] = IntVector(ix, iy, iz)+d_lowIndex;
   ni[1] = IntVector(ix, iy, iz+1)+d_lowIndex;
   ni[2] = IntVector(ix, iy+1, iz)+d_lowIndex;
   ni[3] = IntVector(ix, iy+1, iz+1)+d_lowIndex;
   ni[4] = IntVector(ix+1, iy, iz)+d_lowIndex;
   ni[5] = IntVector(ix+1, iy, iz+1)+d_lowIndex;
   ni[6] = IntVector(ix+1, iy+1, iz)+d_lowIndex;
   ni[7] = IntVector(ix+1, iy+1, iz+1)+d_lowIndex;
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


bool Region::findCellAndShapeDerivatives(const Point& pos,
					 IntVector ni[8],
					 Vector d_S[8]) const
{
    Vector cellpos = (pos-d_box.lower())*d_res/(d_box.upper()-d_box.lower());
    int ix = Floor(cellpos.x());
    int iy = Floor(cellpos.y());
    int iz = Floor(cellpos.z());
    ni[0] = IntVector(ix, iy, iz)+d_lowIndex;
    ni[1] = IntVector(ix, iy, iz+1)+d_lowIndex;
    ni[2] = IntVector(ix, iy+1, iz)+d_lowIndex;
    ni[3] = IntVector(ix, iy+1, iz+1)+d_lowIndex;
    ni[4] = IntVector(ix+1, iy, iz)+d_lowIndex;
    ni[5] = IntVector(ix+1, iy, iz+1)+d_lowIndex;
    ni[6] = IntVector(ix+1, iy+1, iz)+d_lowIndex;
    ni[7] = IntVector(ix+1, iy+1, iz+1)+d_lowIndex;
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
    return ix>= 0 && iy>=0 && iz>=0 && ix<d_res.x() && iy<d_res.y() && iz<d_res.z();
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

Region::BCType 
Region::getBCType(Region::FaceType face) const
{
  // Put in code to return whether the face is a symmetry plane,
  // a fixed boundary, borders a neighboring region or none
  // The value of face ranges from 0-5.

  return None;


}

string
Region::toString() const
{
  char str[ 1024 ];

  sprintf( str, "[ [%2.2lf, %2.2lf, %2.2lf] [%2.2lf, %2.2lf, %2.2lf] ]",
	   d_box.lower().x(), d_box.lower().y(), d_box.lower().z(),
	   d_box.upper().x(), d_box.upper().y(), d_box.upper().z() );

  return string( str );
}

CellIterator
Region::getCellIterator(const Box& b) const
{
   Vector diag = d_box.upper()-d_box.lower();
   Vector l = (b.lower() - d_box.lower())*d_res/diag;
   Vector u = (b.upper() - d_box.lower())*d_res/diag;
   return CellIterator((int)l.x(), (int)l.y(), (int)l.z(),
		       RoundUp(u.x()), RoundUp(u.y()), RoundUp(u.z()));
}
      
NodeIterator Region::getNodeIterator() const
{
   return NodeIterator(getNodeLowIndex(), getNodeHighIndex());
}

const Region* Region::getNeighbor(const IntVector& n) const
{
   if(n.x() == 0 && n.y() == 0 && n.z() == 0)
      return this;
   if(n.x() < -1 || n.y() < -1 || n.z() < -1
      || n.x() > 1 || n.y() > 1 || n.z() > 1)
      throw InternalError("Region::getNeighbor not implemented for distant neighbors");
   int ix=n.x()+1;
   int iy=n.y()+1;
   int iz=n.z()+1;
   int idx = ix*9+iy*3+iz;
   return d_neighbors[idx];
}

IntVector Region::getNodeHighIndex() const
{
   IntVector h(d_highIndex+
	       IntVector(getNeighbor(IntVector(1,0,0))?0:1,
			 getNeighbor(IntVector(0,1,0))?0:1,
			 getNeighbor(IntVector(0,0,1))?0:1));
   return h;
}

void Region::setNeighbor(const IntVector& n, const Region* neighbor)
{
   if(n.x() == 0 && n.y() == 0 && n.z() == 0)
      throw InternalError("Cannot set neighbor 0,0,0");
   if(n.x() < -1 || n.y() < -1 || n.z() < -1
      || n.x() > 1 || n.y() > 1 || n.z() > 1)
      throw InternalError("Region::getNeighbor not implemented for distant neighbors");
   int ix=n.x()+1;
   int iy=n.y()+1;
   int iz=n.z()+1;
   int idx = ix*9+iy*3+iz;
   cerr << "Region " << getID() << " neighbor " << n << " is now " << neighbor->getID() << '\n';
   d_neighbors[idx]=neighbor;
}

void
Region::determineGhostRegions( int numGhostCells )
{
   int gc = numGhostCells;

   // Determine the coordinates of all the sub-ghostRegions around
   // this region.

   int minX = Min( d_box.lower().x(), d_box.upper().x() );
   int minY = Min( d_box.lower().y(), d_box.upper().y() );
   int minZ = Min( d_box.lower().z(), d_box.upper().z() );

   int maxX = Max( d_box.lower().x(), d_box.upper().x() );
   int maxY = Max( d_box.lower().y(), d_box.upper().y() );
   int maxZ = Max( d_box.lower().z(), d_box.upper().z() );

   d_top.set( minX, minY, maxZ, maxX, maxY, maxZ + gc );
   d_topRight.set( maxX, minY, maxZ, maxX + gc, maxY, maxZ + gc );
   d_topLeft.set( minX - gc, minY, maxZ ,minX, maxY, maxZ + gc );
   d_topBack.set( minX, maxY, maxZ ,maxX, maxY + gc, maxZ + gc );
   d_topFront.set( minX, minY - gc, maxZ , maxX, minY, maxZ + gc );
   d_topRightBack.set( maxX, maxY, maxZ, maxX + gc, maxY + gc, maxZ + gc );
   d_topRightFront.set( maxX, minY, maxZ, maxX + gc, minY - gc, maxZ + gc );
   d_topLeftBack.set( minX, maxY, maxZ, minX - gc, maxY + gc, maxZ + gc );
   d_topLeftFront.set( minX - gc, minY - gc, maxX, minX, minY, maxZ + gc );
   d_bottom.set( minX, minY, minZ - gc, maxX, maxY, minZ );
   d_bottomRight.set( maxX, minY, minZ - gc, maxX + gc, maxY, minZ );
   d_bottomLeft.set( minX - gc, minY, minZ - gc, minX, maxY, minZ );
   d_bottomBack.set( minX, maxY, minZ - gc, maxX, maxY + gc, minZ );
   d_bottomFront.set( minX, minY - gc, minZ - gc, maxX, minY, minZ );
   d_bottomRightBack.set( maxX, maxY, minZ, maxX + gc, maxY + gc, minZ - gc );
   d_bottomRightFront.set( maxX, minY, minZ, maxX + gc, minY - gc, minZ - gc );
   d_bottomLeftBack .set( minX, maxY, minZ, minX - gc, maxY + gc, minZ - gc );
   d_bottomLeftFront.set( minX, minY, minZ, minX - gc, minY - gc, minZ - gc );
   d_right.set( maxX, minY, minZ, maxX + gc, maxY, maxZ );
   d_left.set( minX, minY, minZ, minX - gc, maxY, maxZ );
   d_back.set( minX, maxY, minZ, maxX, maxY + gc, maxZ );
   d_front.set( minX, minY, minZ, maxX, minY - gc, maxZ );
   d_rightBack.set( maxX, maxY, minZ, maxX + gc, maxY + gc, maxZ );
   d_rightFront.set( maxX, minY, minZ, maxX + gc, maxY, minZ - gc );
   d_leftBack.set( minX, maxY, minZ, minX - gc, maxY - gc, maxZ );
   d_leftFront.set( minX, minY, minZ, minX - gc, minY - gc, maxZ );
}
      

//
// $Log$
// Revision 1.20  2000/05/28 17:25:06  dav
// adding mpi stuff
//
// Revision 1.19  2000/05/20 08:09:26  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.18  2000/05/15 19:39:49  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.17  2000/05/10 20:03:02  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.16  2000/05/09 03:24:39  jas
// Added some enums for grid boundary conditions.
//
// Revision 1.15  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.14  2000/05/05 06:42:45  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.13  2000/05/04 19:06:48  guilkey
// Added the beginnings of grid boundary conditions.  Functions still
// need to be filled in.
//
// Revision 1.12  2000/05/02 20:30:59  jas
// Fixed the findCellAndShapeDerivatives.
//
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
