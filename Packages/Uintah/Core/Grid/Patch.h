#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>

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
       Periodic,
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
     
     enum VariableBasis {
	NodeBased = Ghost::AroundNodes,
	CellBased = Ghost::AroundCells,
	XFaceBased = Ghost::AroundFacesX,
	YFaceBased = Ghost::AroundFacesY,
	ZFaceBased = Ghost::AroundFacesZ,
	AllFaceBased = Ghost::AroundFaces
     };

     static VariableBasis translateTypeToBasis(TypeDescription::Type type,
					       bool mustExist);
     
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
     CellIterator getSFCIterator( const int dir, const int offset = 0) const; 

     //////////
     // Insert Documentation Here:
     NodeIterator getNodeIterator() const;
     
     NodeIterator getNodeIterator(const Box& b) const;

     // If 'basis' is a constant (as in OnDemandDataWarehouse::getGridVar())
     // then these inlined functions will simplfy to one-liners.
     inline IntVector getLowIndex(VariableBasis basis) const;
     inline IntVector getHighIndex(VariableBasis basis) const;
     
     IntVector getLowIndex() const
     { return d_lowIndex; }
     
     IntVector getHighIndex() const
     { return d_highIndex; }
     
     IntVector getNodeLowIndex() const {
       return d_lowIndex;
     }

     IntVector getNodeHighIndex() const {
       return d_nodeHighIndex;
     }
 
     IntVector getInteriorNodeLowIndex()const;
     IntVector getInteriorNodeHighIndex()const;     

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
     IntVector getGhostCellLowIndex(int numGC) const;
     IntVector getGhostCellHighIndex(int numGC) const;
     /*
     IntVector getGhostSFCXLowIndex(const int numGC) const
     {  return d_lowIndex-getGhostSFCXLowOffset(numGC, d_bctypes); }
     IntVector getGhostSFCXHighIndex(const int numGC) const
     {  return d_highIndex+getGhostSFCXHighOffset(numGC, d_bctypes); }
     IntVector getGhostSFCYLowIndex(const int numGC) const
     {  return d_lowIndex-getGhostSFCYLowOffset(numGC, d_bctypes); }
     IntVector getGhostSFCYHighIndex(const int numGC) const
     {  return d_highIndex+getGhostSFCYHighOffset(numGC, d_bctypes); }
     IntVector getGhostSFCZLowIndex(const int numGC) const
     {  return d_lowIndex-getGhostSFCZLowOffset(numGC, d_bctypes); }
     IntVector getGhostSFCZHighIndex(const int numGC) const
     {  return d_highIndex+getGhostSFCZHighOffset(numGC, d_bctypes); }
     */
     
     // Passes back the low and high offsets for the given ghost cell
     // schenario.  Note: you should subtract the lowOffset (the offsets
     // should be >= 0 in each dimension).
     static void getGhostOffsets(VariableBasis basis, Ghost::GhostType gtype,
				 int numGhostCells,
				 IntVector& lowOffset, IntVector& highOffset);
     static void getGhostOffsets(TypeDescription::Type basis,
				 Ghost::GhostType gtype, int numGhostCells,
				 IntVector& l, IntVector& h)
     {
       bool basisMustExist = (gtype != Ghost::None);
       getGhostOffsets(translateTypeToBasis(basis, basisMustExist),
		       gtype, numGhostCells, l, h);
     }
     
     Box getBox() const;
     
     inline IntVector getNFaces() const {
       // NOT CORRECT
       return IntVector(0,0,0);
     }
     
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

     void computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				 int numGhostCells,
				 Level::selectType& neighbors,
				 IntVector& low, IntVector& high) const;
     /*
     void computeVariableExtents2(VariableBasis basis, Ghost::GhostType gtype,
				 int numGhostCells,
				 Level::selectType& neighbors,
				 IntVector& low, IntVector& high) const;
     */
     void computeVariableExtents(TypeDescription::Type basis,
				 Ghost::GhostType gtype, int numGhostCells,
				 Level::selectType& neighbors,
				 IntVector& low, IntVector& high) const
     {
       bool basisMustExist = (gtype != Ghost::None);
       computeVariableExtents(translateTypeToBasis(basis, basisMustExist),
			      gtype, numGhostCells, neighbors, low, high);

     }

     void computeVariableExtents(VariableBasis basis, Ghost::GhostType gtype,
				 int numGhostCells,
				 IntVector& low, IntVector& high) const;
     
     void computeVariableExtents(TypeDescription::Type basis,
				 Ghost::GhostType gtype, int numGhostCells,
				 IntVector& low, IntVector& high) const
     {
       bool basisMustExist = (gtype != Ghost::None);
       computeVariableExtents(translateTypeToBasis(basis, basisMustExist),
			      gtype, numGhostCells, low, high);

     }

     // helper for computeVariableExtents but also used externally
     // (in GhostOffsetVarMap)
     void computeExtents(VariableBasis basis, const IntVector& lowOffset,
			 const IntVector& highOffset,
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

     // true for wrap around patches (periodic boundary conditions) that
     // represent other real patches.
     bool isVirtual() const
     { return d_realPatch != 0; }

     const Patch* getRealPatch() const
     { return isVirtual() ? d_realPatch : this; }

     IntVector getVirtualOffset() const
     { return d_lowIndex - getRealPatch()->d_lowIndex; }

     Vector getVirtualOffsetVector() const
     { return cellPosition(d_lowIndex) -
	 cellPosition(getRealPatch()->d_lowIndex); }     
     
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

     Patch* createVirtualPatch(const IntVector& offset) const
     { return scinew Patch(this, offset); }
   private:
     Patch(const Patch&);
     Patch(const Patch* realPatch, const IntVector& virtualOffset);
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

     // NULL, unless this patch is a virtual patch (wrap-around
     // from periodic boundary conditions).
     const Patch* d_realPatch;
     
     // Added an extra vector<> for each material
     BCType d_bctypes[numFaces];
     vector<BCData> d_bcs;
     friend class NodeIterator;
     bool in_database;
     bool have_layout;
     IntVector layouthint;
   };

inline IntVector Patch::getLowIndex(VariableBasis basis) const
{
  switch (basis) {
  case CellBased:
    return getCellLowIndex();
  case NodeBased:
    return getNodeLowIndex();
  case XFaceBased:
    return getSFCXLowIndex();
  case YFaceBased:
    return getSFCYLowIndex();
  case ZFaceBased:
    return getSFCZLowIndex();
  case AllFaceBased:
    throw InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)");
  default:
    throw InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)");
  }
}

inline IntVector Patch::getHighIndex(VariableBasis basis) const
{
  switch (basis) {
  case CellBased:
    return getCellHighIndex();
  case NodeBased:
    return getNodeHighIndex();
  case XFaceBased:
    return getSFCXHighIndex();
  case YFaceBased:
    return getSFCYHighIndex();
  case ZFaceBased:
    return getSFCZHighIndex();
  case AllFaceBased:
    throw InternalError("AllFaceBased not implemented in Patch::getLowIndex(basis)");
  default:
    throw InternalError("Illegal VariableBasis in Patch::getLowIndex(basis)");
  }
}

} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Patch & r);

#endif
