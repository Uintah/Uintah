#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Uintah/Grid/SubPatch.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/Box.h>
#include <Uintah/Grid/Level.h>

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
   
   class NodeIterator;
   class CellIterator;
   class BoundCond;
   
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
       xminus,
       xplus,
       yminus,
       yplus,
       zminus,
       zplus,
       startFace = xminus,
       endFace = zplus,
       numFaces // 6
     };
     
     //////////
     // Insert Documentation Here:
     Vector dCell() const {
       // This will need to change for stretched grids
       return d_level->dCell();
     }
     
     //////////
     // Find the index of a cell contaning the given Point. 
     bool findCell(const Point& pos, IntVector& ci) const;
     
     //////////
     // Find the 8 neighboring cell indexes according to a 
     // given node index.
     //    --tan
     void findCellsFromNode( const IntVector& nodeIndex,
			     IntVector cellIndex[8]) const;
     
     //////////
     // Find the 8 neighboring node indexes according to a 
     // given cell index.
     //    --tan
     void findNodesFromCell( const IntVector& cellIndex,
			     IntVector nodeIndex[8]) const;
     
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
     
     NodeIterator getNodeIterator(const Box& b) const;

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
       return Box(d_level->getNodePosition(d_lowIndex),
		  d_level->getNodePosition(d_highIndex));
     }
     
     inline IntVector getNFaces() const {
       // This is wrong for now
       return getNodeHighIndex()-getNodeLowIndex();
     }
     
     inline IntVector getNNodes() const {
       return getNodeHighIndex()-getNodeLowIndex();
     }
     
     long totalCells() const;
     
     void performConsistencyCheck() const;
     
     BCType getBCType(FaceType face) const;
     void setBCType(FaceType face, BCType newbc);
     void setBCValues(FaceType face, vector<BoundCond* >& bc);
     vector<BoundCond* > getBCValues(FaceType face) const;

     bool atEdge(FaceType face) const;
     static FaceType nextFace(FaceType face) {
       return (FaceType)((int)face+1);
     }
     
     //////////
     // Insert Documentation Here:
     inline bool containsNode(const IntVector& idx) const {
       IntVector l(getNodeLowIndex());
       IntVector h(getNodeHighIndex());
       return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
	 && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
     }
     
     //////////
     // Insert Documentation Here:
     inline bool containsCell(const IntVector& idx) const {
       IntVector l(getCellLowIndex());
       IntVector h(getCellHighIndex());
       return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
	 && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
     }
     
     //////////
     // Insert Documentation Here:
     Point nodePosition(const IntVector& idx) const {
       return d_level->getNodePosition(idx);
     }
     
     Box getGhostBox(const IntVector& lowOffset,
		     const IntVector& highOffset) const;
     
     string toString() const;
     
     int getID() const {
       return d_id;
     }
     const Level* getLevel() const {
       return d_level;
     }
     void getFace(FaceType face, int offset, IntVector& l, IntVector& h) const;
   protected:
     friend class Level;
     
     //////////
     // Insert Documentation Here:
     Patch(const Level*,
	   const SCICore::Geometry::IntVector& d_lowIndex,
	   const SCICore::Geometry::IntVector& d_highIndex,
	   int id=-1);
     ~Patch();
     
   private:
     Patch(const Patch&);
     Patch& operator=(const Patch&);
     
     const Level* d_level;
     
     //////////
     // Insert Documentation Here:
     IntVector d_lowIndex;
     IntVector d_highIndex;
     
     int d_id;
     BCType d_bctypes[numFaces];
     vector<vector<BoundCond* > > d_bcs;
     friend class NodeIterator;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Patch* r);

//
// $Log$
// Revision 1.12  2000/06/27 22:49:04  jas
// Added grid boundary condition support.
//
// Revision 1.11  2000/06/26 17:09:01  bigler
// Added getNodeIterator which takes a Box and returns the iterator
// that will loop over the nodes that lie withing the Box.
//
// Revision 1.10  2000/06/16 05:19:21  sparker
// Changed arrays to fortran order
//
// Revision 1.9  2000/06/15 21:57:19  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.8  2000/06/14 21:59:36  jas
// Copied CCVariable stuff to make FCVariables.  Implementation is not
// correct for the actual data storage and iteration scheme.
//
// Revision 1.7  2000/06/14 19:58:03  guilkey
// Added a different version of findCell.
//
// Revision 1.6  2000/06/07 18:30:50  tan
// Requirement for getHighGhostCellIndex() and getLowGhostCellIndex()
// cancelled.
//
// Revision 1.5  2000/06/05 19:25:02  tan
// I need the following two functions,
// (1) IntVector getHighGhostCellIndex() const;
// (2) IntVector getLowGhostCellIndex() const;
// The temporary empty functions are created.
//
// Revision 1.4  2000/06/04 04:35:52  tan
// Added function findNodesFromCell() to find the 8 neighboring node indexes
// according to a given cell index.
//
// Revision 1.3  2000/06/02 19:57:50  tan
// Added function findCellsFromNode() to find the 8 neighboring cell
// indexes according to a given node index.
//
// Revision 1.2  2000/06/01 22:13:54  tan
// Added findCell(const Point& pos).
//
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
