#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Packages/Uintah/Core/Grid/Ghost.h>
//#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundCondData.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>

#undef None

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using std::string;
  using std::map;

  using SCIRun::Vector;
  using SCIRun::Point;
  using SCIRun::IntVector;

  class NodeIterator;
  class CellIterator;
  class BCData;
  class Level;
  class Box;
   
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
       Symmetry,
       Coarse,
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

     //Below for Fracture *************************************************
     void findCellNodes(const Point& pos,IntVector ni[8]) const;
     void findCellNodes27(const Point& pos,IntVector ni[27]) const;
 
     //determine if a point is in the patch
     inline bool containsPoint(const Point& p) const {
       IntVector l(getNodeLowIndex());
       IntVector h(getNodeHighIndex());
       Point lp = nodePosition(l);
       Point hp = nodePosition(h);
       return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
         && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
     }
     //determine if a point is in the patch's real cells
     inline bool containsPointInRealCells(const Point& p) const {
       IntVector l(getInteriorNodeLowIndex());
       IntVector h(d_inHighIndex);
       Point lp = nodePosition(l);
       Point hp = nodePosition(h);
       return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
         && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
     }
     //Above for Fracture *************************************************

     static VariableBasis translateTypeToBasis(TypeDescription::Type type,
					       bool mustExist);
     

     //////////
     // Insert Documentation Here:  
     Vector dCell() const;

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
     // The first three of these are for linear interpolation
     void findCellAndWeights(const Point& pos,
			     IntVector ni[8],
                             double S[8]) const;
     
     void findCellAndShapeDerivatives(const Point& pos,
			              IntVector ni[8],
				      Vector S[8]) const;

     void findCellAndWeightsAndShapeDerivatives(const Point& pos,
						IntVector ni[8], 
						double S[8],
						Vector d_S[8]) const;
     //////////
     // These are for higher order (27 node) interpolation
     void findCellAndWeights27(const Point& pos,
                                     IntVector ni[27],
                                     double S[27],const Vector& size) const;
     
     void findCellAndShapeDerivatives27(const Point& pos,
			                IntVector ni[27],
				        Vector S[27], const Vector& size) const;

     void findCellAndWeightsAndShapeDerivatives27(const Point& pos,
                                                  IntVector ni[27], 
                                                  double S[27], Vector d_S[27],
                                                  const Vector& size) const;
     //////////
     //////////
     // Insert Documentation Here:  
     CellIterator getCellIterator(const IntVector gc = IntVector(0,0,0)) const;
     CellIterator getExtraCellIterator(const IntVector gc = 
				       IntVector(0,0,0)) const;
     
     // This function will return all cells that are intersected by
     // the box.  This is based on the fact that boundaries of cells
     // are closed on the bottom and open on the top.
     CellIterator getCellIterator(const Box& b) const;
     // This function works on the assumption that we want all the cells
     // whose centers lie on or within the box.
     CellIterator getCellCenterIterator(const Box& b) const;
     // Insert Documentation Here:  
     CellIterator getExtraCellIterator(const Box& b) const;
     
     //__________________________________
     //   I C E - M P M I C E   I T E R A T O R S
     CellIterator getFaceCellIterator(const FaceType& face, 
                                const string& domain="minusEdgeCells") const;

     CellIterator getSFCXIterator(const int offset = 0) const;
     CellIterator getSFCYIterator(const int offset = 0) const;
     CellIterator getSFCZIterator(const int offset = 0) const;
     CellIterator getSFCIterator( const int dir, const int offset = 0) const;
     CellIterator addGhostCell_Iter(CellIterator hi_lo, const int nCells) const;
      
     //__________________________________
     //////////
     // Insert Documentation Here:
     NodeIterator getNodeIterator() const;
     
     // This will return an iterator which will include all the nodes
     // contained by the bounding box.  If a dimension of the widget
     // is degenerate (has a thickness of 0) the nearest node in that
     // dimension is used.
     NodeIterator getNodeIterator(const Box& b) const;

     IntVector getLowIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;
     IntVector getHighIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;
     
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
     void setBCValues(FaceType face, BoundCondData& bc);
     void setArrayBCValues(FaceType face, BCDataArray& bc);
     const BoundCondBase* getBCValues(int mat_id,string type,
				      FaceType face) const;

     BCDataArray* getBCDataArray(Patch::FaceType face) const;

     const BoundCondBase* getArrayBCValues(FaceType face,int mat_id,
					   string type,
					   vector<IntVector>& b,
					   vector<IntVector>& i,
					   vector<IntVector>& sfx,
					   vector<IntVector>& sfy,
					   vector<IntVector>& sfz,
					   vector<IntVector>& nb,
					   int child) const;
     

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
     inline bool containsSFCX(const IntVector& idx) const {
       IntVector l(getSFCXLowIndex());
       IntVector h(getSFCXHighIndex());
       return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
	 && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
     }
     
     //////////
     // Insert Documentation Here:
     inline bool containsSFCY(const IntVector& idx) const {
       IntVector l(getSFCYLowIndex());
       IntVector h(getSFCYHighIndex());
       return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
	 && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
     }
     
     //////////
     // Insert Documentation Here:
     inline bool containsSFCZ(const IntVector& idx) const {
       IntVector l(getSFCZLowIndex());
       IntVector h(getSFCZHighIndex());
       return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
	 && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
     }
     
     //////////
     // Insert Documentation Here:
     Point nodePosition(const IntVector& idx) const;

     Point cellPosition(const IntVector& idx) const;


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
     void getFace(FaceType face, const IntVector& insideOffset,
		  const IntVector& outsideOffset,
		  IntVector& l, IntVector& h) const;
     IntVector faceDirection(FaceType face) const;
     void getFaceNodes(FaceType face, int offset, IntVector& l,
		       IntVector& h) const;

     void getFaceCells(FaceType face, int offset, IntVector& l,
		       IntVector& h) const;

     static const int MAX_PATCH_SELECT = 32;
     typedef fixedvector<const Patch*, MAX_PATCH_SELECT> selectType;


     void computeVariableExtents(VariableBasis basis,
				 const IntVector& boundaryLayer,
				 Ghost::GhostType gtype, int numGhostCells,
				 selectType& neighbors,
				 IntVector& low, IntVector& high) const;
     void computeVariableExtents(TypeDescription::Type basis,
				 const IntVector& boundaryLayer,
				 Ghost::GhostType gtype, int numGhostCells,
				 selectType& neighbors,
				 IntVector& low, IntVector& high) const;

     void computeVariableExtents(VariableBasis basis,
				 const IntVector& boundaryLayer,
				 Ghost::GhostType gtype, int numGhostCells,
				 IntVector& low, IntVector& high) const;
     
     void computeVariableExtents(TypeDescription::Type basis,
				 const IntVector& boundaryLayer,
				 Ghost::GhostType gtype, int numGhostCells,
				 IntVector& low, IntVector& high) const;

     // helper for computeVariableExtents but also used externally
     // (in GhostOffsetVarMap)
     void computeExtents(VariableBasis basis,
			 const IntVector& boundaryLayer,
			 const IntVector& lowOffset,
			 const IntVector& highOffset,
			 IntVector& low, IntVector& high) const;

     /* Get overlapping patches on other levels. */
     
     void getFineLevelPatches(selectType& finePatches) const
     { getOtherLevelPatches(1, finePatches); }
     
     void getCoarseLevelPatches(selectType& coarsePatches) const
     { getOtherLevelPatches(-1, coarsePatches); }

     void getOtherLevelPatches(int levelOffset, selectType& patches)
       const;
     
     class Compare {
     public:
       inline bool operator()(const Patch* p1, const Patch* p2) const {
	 return (p1 != 0 && p2 != 0) ? (p1->getID() < p2->getID()) :
	   ((p2 != 0) ? true : false);
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

   public:
     
    /********************
      The following are needed in order to use Patch as a Box in
      Core/Container/SuperBox.h (see
      Packages/Uintah/CCA/Components/Schedulers/LocallyComputedPatchVarMap.cc)
    *********************/
    
    IntVector getLow() const
    { return getLowIndex(); }
    
    IntVector getHigh() const
    { return getHighIndex(); }
    
    int getVolume() const
    { return getVolume(getLow(), getHigh()); }
    
    int getArea(int side) const
    {
      int area = 1;
      for (int i = 0; i < 3; i++)
	if (i != side)
	  area *= getHigh()[i] - getLow()[i];
      return area;
    }
    
    static int getVolume(const IntVector& low, const IntVector& high)
    { return (high.x() - low.x()) * (high.y() - low.y()) *
	(high.z() - low.z()); }    
   protected:
     friend class Level;
     friend class NodeIterator;     
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

     // d_realPatch is NULL, unless this patch is a virtual patch
     // (wrap-around from periodic boundary conditions).
     const Patch* d_realPatch;
     
     const Level* d_level; // I live in this grid level;
     int d_level_index;  // I'm at this index in the Level vector;
     
     // used only by friend class Level
     inline void setLevelIndex( int idx ){ d_level_index = idx;}

     IntVector neighborsLow() const;
     IntVector neighborsHigh() const;
     
     //////////
     // Locations in space of opposite box corners.
     // These are in terms of cells positioned from the level's anchor,
     // and they include extra cells
     IntVector d_lowIndex;
     IntVector d_highIndex;

     //////////
     // Locations in space of opposite box corners.
     // There are in terms of cells positioned from the level's anchor,
     // and represent the interior cells (no extra cells)
     IntVector d_inLowIndex;
     IntVector d_inHighIndex;
     IntVector d_nodeHighIndex;
     
     int d_id; // Patch ID
     
     // Added an extra vector<> for each material
     BCType d_bctypes[numFaces];  // specifies bc type for each face
     vector<BoundCondData> d_bcs;
     map<Patch::FaceType,BCDataArray > array_bcs;
     bool in_database;
     bool have_layout;
     IntVector layouthint;
   };

} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Patch & r);

#endif
