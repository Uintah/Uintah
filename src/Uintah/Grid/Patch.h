#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Uintah/Grid/SubPatch.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/Box.h>

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/MinMax.h>

#include <string>
#include <iosfwd>
#include <stdio.h>

using std::string;
using namespace Uintah;

namespace Uintah {
    
   using SCICore::Geometry::Point;
   using SCICore::Geometry::Vector;
   using SCICore::Geometry::IntVector;
   using SCICore::Math::RoundUp;
   using SCICore::Math::Min;
   using SCICore::Math::Max;
   
   class NodeSubIterator;
   class NodeIterator;
   class CellIterator;

   
/**************************************
      
CLASS
   Patch
      
   Short Description...
      
GENERAL INFORMATION
      
   Patch.h
      
   Steven G. Parker
   Department of Computer Science
   University of Utah
      
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
   Copyright (C) 2000 SCI Group
      
KEYWORDS
   Patch
      
DESCRIPTION
   Long description...
      
WARNING
     
****************************************/
    
   class Patch {
   public:

     enum BCType {
       None,
       Fixed,
       Symmetry,
       Neighbor
     };
     
     enum FaceType {
       xplus,
       xminus,
       yplus,
       yminus,
       zplus,
       zminus
     };

      //////////
      // Insert Documentation Here:
      Vector dCell() const {
	 return (d_box.upper()-d_box.lower())/d_res;
      }
      
      //////////
      // Insert Documentation Here:
      void findCell(const Point& pos, int& ix, int& iy, int& iz) const;
      
      //////////
      // Insert Documentation Here:
      bool findCellAndWeights(const SCICore::Geometry::Point& pos,
			      IntVector ni[8], double S[8]) const;

      //////////
      // Insert Documentation Here:
      bool findCellAndShapeDerivatives
			(const SCICore::Geometry::Point& pos,
		         IntVector ni[8],
			 SCICore::Geometry::Vector S[8]) const;

      //////////
      // Insert Documentation Here:
      CellIterator getCellIterator(const Box& b) const;

      //////////
      // Insert Documentation Here:
      NodeIterator getNodeIterator() const;

      //////////
      // Insert Documentation Here:
      void subpatchIteratorPair(int i, int n,
				 NodeSubIterator& iter,
				 NodeSubIterator& end) const;
      //////////
      // Insert Documentation Here:
      SubPatch subpatch(int i, int n) const;
      
      IntVector getNodeLowIndex() const {
	 return d_lowIndex;
      }
      IntVector getNodeHighIndex() const;
      IntVector getCellLowIndex() const {
	 return d_lowIndex;
      }
      IntVector getCellHighIndex() const {
	 return d_highIndex;
      }
      
      inline Box getBox() const {
	 return d_box;
      }
      
      inline IntVector getNCells() const {
	 return d_res;
      }
      inline IntVector getNNodes() const {
	 return d_res+IntVector(1,1,1);
      }
      
      long totalCells() const;
      
      void performConsistencyCheck() const;

      BCType getBCType(FaceType face) const;
      
      //////////
      // Insert Documentation Here:
      inline bool containsNode(const IntVector& idx) const {
	 IntVector l(getNodeLowIndex());
	 IntVector h(getNodeHighIndex());
	 return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.x()
	    && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      //////////
      // Determines if "patch" is within (or the same as) this
      // patch.
      inline bool contains(const Patch& patch) const {
	    int myMinX = Min( d_box.lower().x(), d_box.upper().x() );
	    int myMinY = Min( d_box.lower().y(), d_box.upper().y() );
	    int myMinZ = Min( d_box.lower().z(), d_box.upper().z() );

	    int myMaxX = Max( d_box.lower().x(), d_box.upper().x() );
	    int myMaxY = Max( d_box.lower().y(), d_box.upper().y() );
	    int myMaxZ = Max( d_box.lower().z(), d_box.upper().z() );

	    int regMinX = Min( patch.d_box.lower().x(), 
			       patch.d_box.upper().x() );
	    int regMinY = Min( patch.d_box.lower().y(), 
			       patch.d_box.upper().y() );
	    int regMinZ = Min( patch.d_box.lower().z(),
			       patch.d_box.upper().z() );

	    int regMaxX = Max( patch.d_box.lower().x(),
			       patch.d_box.upper().x() );
	    int regMaxY = Max( patch.d_box.lower().y(),
			       patch.d_box.upper().y() );
	    int regMaxZ = Max( patch.d_box.lower().z(),
			       patch.d_box.upper().z() );

	 return( myMinX >= regMinX && myMaxX <= regMaxX && 
		 myMinY >= regMinY && myMaxY <= regMaxY && 
		 myMinZ >= regMinZ && myMaxZ <= regMaxZ );
      }

      //////////
      // Insert Documentation Here:
      Point nodePosition(const IntVector& idx) const {
	 return d_box.lower() + dCell()*idx;
      }

      Box getGhostBox(int l, int h) const {
	 return Box(nodePosition(getCellLowIndex()+IntVector(l,l,l)),
		    nodePosition(getCellHighIndex()+IntVector(h,h,h)));
      }
				 
      string toString() const;

      int getID() const {
	 return d_id;
      }
      const Patch* getNeighbor(const IntVector&) const;
      void setNeighbor(const IntVector&, const Patch*);
   protected:
      friend class Level;
      
      //////////
      // Insert Documentation Here:
      Patch(const SCICore::Geometry::Point& min,
	     const SCICore::Geometry::Point& max,
	     const SCICore::Geometry::IntVector& d_lowIndex,
	     const SCICore::Geometry::IntVector& d_highIndex,
	     int id=-1);
      ~Patch();

   private:
      Patch(const Patch&);
      Patch& operator=(const Patch&);
      
      //////////
      // Insert Documentation Here:
      Box d_box;

      //////////
      // These are just coordinates of the ghostcell boxes around
      // this patch.  There are 26 of them.
      Box d_top, d_topRight, d_topLeft, d_topBack, d_topFront,
	  d_topRightBack, d_topRightFront, d_topLeftBack, 
	  d_topLeftFront;
      Box d_bottom, d_bottomRight, d_bottomLeft, d_bottomBack,
	  d_bottomFront, d_bottomRightBack, d_bottomRightFront,
	  d_bottomLeftBack, d_bottomLeftFront;
      Box d_right, d_left, d_back, d_front, d_rightBack,
	  d_rightFront, d_leftBack, d_leftFront;
      
      ////////// 
      // Pre-calculates the upper/lower points of the 26 adjacent
      // patches based on the number of ghost cells.
      void determineGhostPatches( int numGhostCells );

      //////////
      // Insert Documentation Here:
      IntVector d_lowIndex;
      IntVector d_highIndex;
      IntVector d_res;

      const Patch* d_neighbors[27];
      
      int d_id;
      friend class NodeIterator;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Patch* r);

//
// $Log$
// Revision 1.1  2000/05/30 20:19:32  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.21  2000/05/28 17:25:06  dav
// adding mpi stuff
//
// Revision 1.20  2000/05/20 08:09:27  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.19  2000/05/10 20:03:02  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.18  2000/05/09 03:24:40  jas
// Added some enums for grid boundary conditions.
//
// Revision 1.17  2000/05/07 06:02:12  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.16  2000/05/05 06:42:45  dav
// Added some _hopefully_ good code mods as I work to get the MPI stuff to work.
//
// Revision 1.15  2000/05/04 19:06:48  guilkey
// Added the beginnings of grid boundary conditions.  Functions still
// need to be filled in.
//
// Revision 1.14  2000/05/02 20:30:59  jas
// Fixed the findCellAndShapeDerivatives.
//
// Revision 1.13  2000/05/02 20:13:05  sparker
// Implemented findCellAndWeights
//
// Revision 1.12  2000/05/02 06:07:23  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.11  2000/05/01 16:18:18  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.10  2000/04/28 20:24:44  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.9  2000/04/27 23:18:50  sparker
// Added problem initialization for MPM
//
// Revision 1.8  2000/04/26 06:48:54  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/25 00:41:21  dav
// more changes to fix compilations
//
// Revision 1.6  2000/04/13 06:51:02  sparker
// More implementation to get this to work
//
// Revision 1.5  2000/04/12 23:00:50  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.4  2000/03/22 00:32:13  sparker
// Added Face-centered variable class
// Added Per-patch data class
// Added new task constructor for procedures with arguments
// Use Array3Index more often
//
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
