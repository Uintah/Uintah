#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>

#include <string>
#include <iosfwd>

namespace Uintah {

using namespace SCIRun;
  using std::string;
   
  class NodeIterator;
  class CellIterator;
  class BCData;
   
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
       numFaces, // 6
       invalidFace
     };
     
     //////////
     // Insert Documentation Here:
     Vector dCell() const {
       // This will need to change for stretched grids
       return d_level->dCell();
     }

     //////////
     // Find the closest node index to a point
     int findClosestNode(const Point& pos, IntVector& idx) const;
     
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
     void findCellAndWeights(const Point& pos,
			     IntVector ni[8], double S[8]) const;
     
     //////////
     // Insert Documentation Here:
     void findCellAndShapeDerivatives( const Point& pos,
				       IntVector ni[8],
				       Vector S[8]) const;

     void findCellAndWeightsAndShapeDerivatives(const Point& pos,
						IntVector ni[8], 
						double S[8],
						Vector d_S[8]) const;
     //////////
     // Insert Documentation Here:
     CellIterator getCellIterator(const IntVector gc = IntVector(0,0,0)) const;
     CellIterator getExtraCellIterator(const IntVector gc = 
				       IntVector(0,0,0)) const;
     
     CellIterator getCellIterator(const Box& b) const;
     CellIterator getExtraCellIterator(const Box& b) const;
     CellIterator getFaceCellIterator(const FaceType& face, 
                                const string& domain="minusEdgeCells") const;

     CellIterator getSFCXIterator(const int offset = 0) const;
     CellIterator getSFCYIterator(const int offset = 0) const;
     CellIterator getSFCZIterator(const int offset = 0) const;
     
     //////////
     // Insert Documentation Here:
     NodeIterator getNodeIterator() const;
     
     NodeIterator getNodeIterator(const Box& b) const;

     IntVector getNodeLowIndex() const {
       return d_lowIndex;
     }
 
     IntVector getInteriorNodeLowIndex()const;
     IntVector getInteriorNodeHighIndex()const;     
     IntVector getNodeHighIndex() const {
	 return d_nodeHighIndex;
      }

     IntVector getSFCXLowIndex() const {
       return d_lowIndex;
     }
     IntVector getSFCXHighIndex() const;

     IntVector getSFCYLowIndex() const {
       return d_lowIndex;
     }
     IntVector getSFCYHighIndex() const;

     IntVector getSFCZLowIndex() const {
       return d_lowIndex;
     }
     IntVector getSFCZHighIndex() const;

     IntVector getCellLowIndex() const {
       return d_lowIndex;
     }
     IntVector getCellHighIndex() const {
       return d_highIndex;
     }
     IntVector getInteriorCellLowIndex() const {
       return d_inLowIndex;
     }
     IntVector getInteriorCellHighIndex() const {
       return d_inHighIndex;
     }

     void setExtraIndices(const IntVector& l, const IntVector& h);

     // required for fortran interface
     IntVector getSFCXFORTLowIndex() const;
     IntVector getSFCXFORTHighIndex() const;

     IntVector getSFCYFORTLowIndex() const;
     IntVector getSFCYFORTHighIndex() const;

     IntVector getSFCZFORTLowIndex() const;
     IntVector getSFCZFORTHighIndex() const;

     IntVector getCellFORTLowIndex() const;
     IntVector getCellFORTHighIndex() const;

     // returns ghost cell index
     IntVector getGhostCellLowIndex(const int numGC) const;
     IntVector getGhostCellHighIndex(const int numGC) const;
     IntVector getGhostSFCXLowIndex(const int numGC) const;
     IntVector getGhostSFCXHighIndex(const int numGC) const;
     IntVector getGhostSFCYLowIndex(const int numGC) const;
     IntVector getGhostSFCYHighIndex(const int numGC) const;
     IntVector getGhostSFCZLowIndex(const int numGC) const;
     IntVector getGhostSFCZHighIndex(const int numGC) const;
     
     Box getBox() const;
     
     inline IntVector getNNodes() const {
       return getNodeHighIndex()-getNodeLowIndex();
     }
     
     long totalCells() const;
     
     void performConsistencyCheck() const;
     
     BCType getBCType(FaceType face) const;
     void setBCType(FaceType face, BCType newbc);
     void setBCValues(FaceType face, BCData& bc);
     BoundCondBase* getBCValues(int mat_id,string type,FaceType face) const;

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

     Point cellPosition(const IntVector& idx) const {
       return d_level->getCellPosition(idx);
     }

     Box getGhostBox(const IntVector& lowOffset,
		     const IntVector& highOffset) const;
     
     string toString() const;
     
     inline int getID() const {
       return d_id;
     }
     static const Patch* getByID(int);
     inline const Level* getLevel() const {
       return d_level;
     }
     void getFace(FaceType face, int offset, IntVector& l, IntVector& h) const;

     enum VariableBasis {
	CellBased,
	NodeBased,
	CellFaceBased,
	XFaceBased,
	YFaceBased,
	ZFaceBased,
	AllFaceBased
     };

     void computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				 int numGhostCells,
				 Level::selectType& neighbors,
				 IntVector& low, IntVector& high) const;
     void computeVariableExtents(TypeDescription::Type basis,
				 Ghost::GhostType gtype, int numGhostCells,
				 Level::selectType& neighbors,
				 IntVector& low, IntVector& high) const;

      class Compare {
      public:
	 inline bool operator()(const Patch* p1, const Patch* p2) const {
	    return p1->getID() < p2->getID();
	 }
      private:
      };

      // get the index into the Level::d_patches array
      int getLevelIndex() const { return d_level_index; }

       void setLayoutHint(const IntVector& pos);
       bool getLayoutHint(IntVector& pos) const;
   protected:
     friend class Level;
     
     //////////
     // Insert Documentation Here:
     Patch(const Level*,
	   const IntVector& d_lowIndex,
	   const IntVector& d_highIndex,
	   const IntVector& d_inLowIndex,
	   const IntVector& d_inHighIndex,
	   int id=-1);
     ~Patch();
     
   private:
     Patch(const Patch&);
     Patch& operator=(const Patch&);
     
     const Level* d_level; // I live in this grid level;
     int d_level_index;  // I'm at this index in the Level vector;
     
     // used only by friend class Level
     inline void setLevelIndex( int idx ){ d_level_index = idx;}
     
     //////////
     // Insert Documentation Here:
     IntVector d_lowIndex;
     IntVector d_highIndex;

     IntVector d_inLowIndex;
     IntVector d_inHighIndex;
     IntVector d_nodeHighIndex;
     
     int d_id;
     // Added an extra vector<> for each material
     BCType d_bctypes[numFaces];
     vector<BCData> d_bcs;
     friend class NodeIterator;
     bool in_database;
       bool have_layout;
       IntVector layouthint;
   };

} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Patch & r);

#endif
