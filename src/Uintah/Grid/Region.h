#ifndef UINTAH_HOMEBREW_Region_H
#define UINTAH_HOMEBREW_Region_H

#include <Uintah/Grid/SubRegion.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/Box.h>

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Math/MiscMath.h>

#include <string>
#include <iosfwd>
#include <stdio.h>

using std::string;

namespace Uintah {
    
   using SCICore::Geometry::Point;
   using SCICore::Geometry::Vector;
   using SCICore::Geometry::IntVector;
   using SCICore::Math::RoundUp;
   
   class NodeSubIterator;
   class NodeIterator;
   class CellIterator;
   
/**************************************
      
CLASS
   Region
      
   Short Description...
      
GENERAL INFORMATION
      
   Region.h
      
   Steven G. Parker
   Department of Computer Science
   University of Utah
      
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
   Copyright (C) 2000 SCI Group
      
KEYWORDS
   Region
      
DESCRIPTION
   Long description...
      
WARNING
     
****************************************/
    
   class Region {
   public:
      
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
      bool findCellAndWeights(const SCICore::Geometry::Point& /*pos*/,
			      IntVector /*ni*/[8], double /*S*/[8]) const;

      //////////
      // Insert Documentation Here:
      bool findCellAndShapeDerivatives
			(const SCICore::Geometry::Point&/* pos*/,
		         IntVector /*ni*/[8],
			 SCICore::Geometry::Vector /*S*/[8]) const;

      //////////
      // Insert Documentation Here:
      CellIterator getCellIterator(const Box& b) const;

      //////////
      // Insert Documentation Here:
      NodeIterator getNodeIterator() const;

      //////////
      // Insert Documentation Here:
      void subregionIteratorPair(int i, int n,
				 NodeSubIterator& iter,
				 NodeSubIterator& end) const;
      //////////
      // Insert Documentation Here:
      SubRegion subregion(int i, int n) const;
      
      IntVector getLowIndex() const;
      IntVector getHighIndex() const;
      
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

      int getBCType(int face) const;
      
      //////////
      // Insert Documentation Here:
      inline bool contains(const IntVector& idx) const {
	 return idx.x() >= 0 && idx.y() >= 0 && idx.z() >= 0 &&
	   idx.x() <= d_res.x() && idx.y() <= d_res.y() && 
	   idx.z() <= d_res.z();
      }

      //////////
      // Determines if "region" is within (or the same as) this
      // region.
      inline bool contains(const Region& region) const {
	 return ( ( ( ( region.d_box.lower().x() >= d_box.lower().x() &&
			region.d_box.lower().x() <= d_box.upper().x() ) || 
		      ( region.d_box.lower().x() <= d_box.lower().x() &&
			region.d_box.lower().x() >= d_box.upper().x() ) ) &&

		    ( ( region.d_box.lower().y() >= d_box.lower().y() &&
			region.d_box.lower().y() <= d_box.upper().y() ) || 
		      ( region.d_box.lower().y() <= d_box.lower().y() &&
			region.d_box.lower().y() >= d_box.upper().y() ) ) &&

		    ( ( region.d_box.lower().z() >= d_box.lower().z() &&
			region.d_box.lower().z() <= d_box.upper().z() ) || 
		      ( region.d_box.lower().z() <= d_box.lower().z() &&
			region.d_box.lower().z() >= d_box.upper().z() ) ) ) &&

		  ( ( ( region.d_box.upper().x() >= d_box.lower().x() &&
			region.d_box.upper().x() <= d_box.upper().x() ) || 
		      ( region.d_box.upper().x() <= d_box.lower().x() &&
			region.d_box.upper().x() >= d_box.upper().x() ) ) &&

		    ( ( region.d_box.upper().y() >= d_box.lower().y() &&
			region.d_box.upper().y() <= d_box.upper().y() ) || 
		      ( region.d_box.upper().y() <= d_box.lower().y() &&
			region.d_box.upper().y() >= d_box.upper().y() ) ) &&

		    ( ( region.d_box.upper().z() >= d_box.lower().z() &&
			region.d_box.upper().z() <= d_box.upper().z() ) || 
		      ( region.d_box.upper().z() <= d_box.lower().z() &&
			region.d_box.upper().z() >= d_box.upper().z() ) ) ) );
      }

      //////////
      // Insert Documentation Here:
      Point nodePosition(const IntVector& idx) const {
	 return d_box.lower() + dCell()*idx;
      }

      string toString() const;

   protected:
      friend class Level;
      
      //////////
      // Insert Documentation Here:
      Region(const SCICore::Geometry::Point& min,
	     const SCICore::Geometry::Point& max,
	     const SCICore::Geometry::IntVector& res);
      ~Region();

   private:
      Region(const Region&);
      Region& operator=(const Region&);
      
      //////////
      // Insert Documentation Here:
      Box d_box;
      
      //////////
      // Insert Documentation Here:
      IntVector d_res;
      
      friend class NodeIterator;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Region* r);

//
// $Log$
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
// Added Per-region data class
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
