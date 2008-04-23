#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>

#undef None

#include <sgi_stl_warnings_off.h>
#include   <string>
#include   <map>
#include   <iosfwd>
#include   <vector>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>

namespace Uintah {

  using std::string;
  using std::map;
  using std::vector;

  using SCIRun::Vector;
  using SCIRun::Point;
  using SCIRun::IntVector;

  class NodeIterator;
  class CellIterator;
  class BCData;
  class Level;
  class Box;
  class BCDataArray;
  class BoundCondBase;
   
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
     
    
   class UINTAHSHARE Patch {
   public:

     UINTAHSHARE friend std::ostream& operator<<(std::ostream& out, const Uintah::Patch & r);

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
 
     enum FaceIteratorType {
      ExtraMinusEdgeCells,            //Excludes the edge/corner cells.
      ExtraPlusEdgeCells,             //Includes the edge/corner cells.
      FaceNodes,                      //Includes all nodes on the face of the patch (without extra cells)
      SFCVars,                        //Includes SFC vars on the face           (without extra cells)
      InteriorFaceCells               //Includes cells on the interior of the face
     };
     
     class Compare {
      public:
       inline bool operator()(const Patch* p1, const Patch* p2) const {
         return (p1 != 0 && p2 != 0) ? (p1->getID() < p2->getID()) :
           ((p2 != 0) ? true : false);
       }
      private:
     };

    /**************New Public Interaface*******************
     *
     * This block will be used to store the new public interface as it goes live
     *
     */

    /**
     * Returns the cell low index not including extra cells or ghost cells.
     * This version of this function included to avoid unnecessary branching.
     */
     inline IntVector getCellLowIndex__New() const
     {
        return d_lowIndex__New;
     }
     
    /**
     * Returns the cell low index not including extra cells.
     * ngc specifies the number of ghost cells.
     */
     inline IntVector  getCellLowIndex__New(int ngc) const
     {  
        //if we have a neighbor subtract the number of ghost cells from the index
        return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngc:0,
                                getBCType(yminus)==Neighbor?ngc:0,
                                getBCType(zminus)==Neighbor?ngc:0); 
     }

    /**
     * Returns the cell high index not including extra cells or ghost cells.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline IntVector getCellHighIndex__New() const
    {
       return d_highIndex__New;
    }
    
     /**
     * Returns the cell high index not including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getCellHighIndex__New(int ngc) const
    {
      //if we have a neighbor add the number of ghost cells to the index
      return d_highIndex__New+IntVector(
                                getBCType(xplus)==Neighbor?ngc:0,
                                getBCType(yplus)==Neighbor?ngc:0,
                                getBCType(zplus)==Neighbor?ngc:0); 
    }
    
    /**
     * Returns the cell low index including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getExtraCellLowIndex__New(int ngc=0) const
    {
      //if have a neighbor subtract the number of ghost cells from the index
      //otherwise subtract the number of extra cells from the index
      return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngc:d_extraCells[0],
                                getBCType(yminus)==Neighbor?ngc:d_extraCells[1],
                                getBCType(zminus)==Neighbor?ngc:d_extraCells[2]); 
    }
    
    /**
     * Returns the cell high index including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getExtraCellHighIndex__New(int ngc=0) const
    {
      //if have a neighbor add the number of ghost cells to the index
      //otherwise add the number of extra cells to the index
      return d_highIndex__New+IntVector(
                                getBCType(xplus)==Neighbor?ngc:d_extraCells[0],
                                getBCType(yplus)==Neighbor?ngc:d_extraCells[1],
                                getBCType(zplus)==Neighbor?ngc:d_extraCells[2]); 
    }
    
    /**
     * Returns the node low index not including extra nodes or ghost nodes.
     * This version of this function is included to avoid unnecessary branching.
     */
     inline IntVector getNodeLowIndex__New() const
     {
        return d_lowIndex__New;
     }
    
    /**
     * Returns the node low index not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getNodeLowIndex__New(int ngn) const
    {
        //if we have a neighbor subtract the number of ghost nodes from the index
        return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngn:0,
                                getBCType(yminus)==Neighbor?ngn:0,
                                getBCType(zminus)==Neighbor?ngn:0); 
    }
    
    /**
     * Returns the node high index not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getNodeHighIndex__New(int ngn=0) const
    {
      //if we have a neighbor add the number of ghost nodes to the index
      //otherwise add 1 because we own the plus face node
      return d_highIndex__New+IntVector(
                                getBCType(xplus)==Neighbor?ngn:1,
                                getBCType(yplus)==Neighbor?ngn:1,
                                getBCType(zplus)==Neighbor?ngn:1); 
    } 
    
    /**
     * Returns the node low index including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getExtraNodeLowIndex__New(int ngn=0) const
    {
      //if have a neighbor subtract the number of ghost nodes from the index
      //otherwise subtract the number of extra nodes from the index
      return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngn:d_extraCells[0],
                                getBCType(yminus)==Neighbor?ngn:d_extraCells[1],
                                getBCType(zminus)==Neighbor?ngn:d_extraCells[2]); 
    }
    
    /**
     * Returns the node high index including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getExtraNodeHighIndex__New(int ngn=0) const
    {
      //if have a neighbor add the number of ghost nodes to the index
      //otherwise the number of extra nodes to the index and 1 for the plus face node
      return d_highIndex__New+IntVector(
                                getBCType(xplus)==Neighbor?ngn:d_extraCells[0]+1,
                                getBCType(yplus)==Neighbor?ngn:d_extraCells[1]+1,
                                getBCType(zplus)==Neighbor?ngn:d_extraCells[2]+1); 

    }
    
    /**
     * Returns the staggared face centered on X cell low index excluding extra cells
     */
    inline IntVector getSFCXLowIndex__New() const 
    {
       return getCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on X cell high index excluding extra cells
     */
    inline IntVector getSFCXHighIndex__New() const
    {
      return getCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
    }
    
    /**
     * Returns the staggared face centered on Y cell low index excluding extra cells
     */
    inline IntVector getSFCYLowIndex__New() const 
    {
       return getCellLowIndex__New();
    }

    /**
     * Returns the staggared face centered on Y cell high index excluding extra cells
     */
    inline IntVector getSFCYHighIndex__New() const
    {
      return getCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
    }
    
    /**
     * Returns the staggared face centered on Z cell low index excluding extra cells
     */
    IntVector getSFCZLowIndex__New() const 
    {
       return getCellLowIndex__New();
    }

    /**
     * Returns the staggared face centered on Z cell high index excluding extra cells
     */
    IntVector getSFCZHighIndex__New() const
    {
      return getCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
    }
    
    /**
     * Returns the staggared face centered on X cell low index including extra cells
     */
    inline IntVector getExtraSFCXLowIndex__New() const 
    {
       return getExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on X cell high index including extra cells
     */
    inline IntVector getExtraSFCXHighIndex__New() const
    {
      return getExtraCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
    }
    
    /**
     * Returns the staggared face centered on Y cell low index including extra cells
     */
    inline IntVector getExtraSFCYLowIndex__New() const 
    {
       return getExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on Y cell high index including extra cells
     */
    inline IntVector getExtraSFCYHighIndex__New() const
    {
      return getExtraCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
    }

    /**
     * Returns the staggared face centered on Z cell low index including extra cells
     */
    inline IntVector getExtraSFCZLowIndex__New() const 
    {
       return getExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on Z cell high index including extra cells
     */
    inline IntVector getExtraSFCZHighIndex__New() const
    {
      return getExtraCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
    }
     
    
    /**
     * Returns a cell iterator not including extra cells or ghost cells.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline CellIterator getCellIterator__New() const
    {
      return CellIterator(getCellLowIndex__New(),getCellHighIndex__New());
    }
    
    /**
     * Returns a cell iterator not including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline CellIterator getCellIterator__New(int ngc) const
    {
      return CellIterator(getCellLowIndex__New(ngc),getCellHighIndex__New(ngc));
    }
    
    /**
     * Returns a cell iterator including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline CellIterator getExtraCellIterator__New(int ngc=0) const
    {
      return CellIterator(getExtraCellLowIndex__New(ngc),getExtraCellHighIndex__New(ngc));
    }
    
    /**
     * Returns a node iterator not including extra nodes or ghost nodes.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline NodeIterator getNodeIterator__New() const
    {
      return NodeIterator(getNodeLowIndex__New(),getNodeHighIndex__New());
    }
    
    /**
     * Returns a node iterator not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline NodeIterator getNodeIterator__New(int ngn) const
    {
      return NodeIterator(getNodeLowIndex__New(ngn),getNodeHighIndex__New(ngn));
    }
    
    /**
     * Returns a node iterator including extra cells.
     * ngn specifies the number of ghost nodes.
     */
    NodeIterator getExtraNodeIterator__New(int ngn=0) const
    {
      return NodeIterator(getExtraNodeLowIndex__New(ngn),getExtraNodeHighIndex__New(ngn));
    }
    
    /**
     * Returns a staggared face centered on X cell iterator excluding extra cells
     */
    inline CellIterator getSFCXIterator__New() const
    {
      return CellIterator(getSFCXLowIndex__New(),getSFCXHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Y cell iterator excluding extra cells
     */
    inline CellIterator getSFCYIterator__New() const
    {
      return CellIterator(getSFCYLowIndex__New(),getSFCYHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Z cell iterator excluding extra cells
     */
    inline CellIterator getSFCZIterator__New() const
    {
      return CellIterator(getSFCZLowIndex__New(),getSFCZHighIndex__New());
    }

    /**
     * Returns a staggared face centered on X cell iterator including extra cells
     */
    inline CellIterator getExtraSFCXIterator__New() const
    {
      return CellIterator(getExtraSFCXLowIndex__New(),getExtraSFCXHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Y cell iterator including extra cells
     */
    inline CellIterator getExtraSFCYIterator__New() const
    {
      return CellIterator(getExtraSFCYLowIndex__New(),getExtraSFCYHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Z cell iterator including extra cells
     */
    inline CellIterator getExtraSFCZIterator__New() const
    {
      return CellIterator(getExtraSFCZLowIndex__New(),getExtraSFCZHighIndex__New());
    }

    /*****************************************************
     * Returns a face cell iterator
     *  face specifies which face will be returned
     *  domain specifies which type of iterator will be returned
     *  and can be one of the following:
     *     ExtraMinusEdgeCells:    All extra cells beyond the face excluding the edge cells.
     *     ExtraPlusEdgeCells:     All extra cells beyond the face including the edge cells.
     *     FaceNodes:              All nodes on the face.
     *     SFCVars:                All face centered nodes on the face.
     *     InteriorFaceCells:      All cells on the interior of the face.                                                             
     */
    CellIterator getFaceIterator__New(const FaceType& face, const FaceIteratorType& domain) const;

    /**
     *Returns an iterator to the edge of two intersecting faces.
     *if minusCornerCells is true the edge will exclude corner cells.
     */
    CellIterator getEdgeCellIterator__New(const FaceType& face0,const FaceType& face1,bool minusCornerCells=true) const;

    /*************************************************************
     *The following queries are for fortran.  Fortran indexing
     *is inclusiving meaning we have to subtract 1 from the high
     *indices.
     *************************************************************/
    
    /**
     * Returns the cell low index not including extra cells or ghost cells.
     * This version of this function included to avoid unnecessary branching.
     */
     inline IntVector getFortranCellLowIndex__New() const
     {
        return d_lowIndex__New;
     }
     
    /**
     * Returns the cell low index not including extra cells.
     * ngc specifies the number of ghost cells.
     */
     inline IntVector  getFortranCellLowIndex__New(int ngc) const
     {  
        //if we have a neighbor subtract the number of ghost cells from the index
        return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngc:0,
                                getBCType(yminus)==Neighbor?ngc:0,
                                getBCType(zminus)==Neighbor?ngc:0); 
     }

    /**
     * Returns the cell high index not including extra cells or ghost cells.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline IntVector getFortranCellHighIndex__New() const
    {
       return d_highIndex__New-IntVector(1,1,1);
    }
    
     /**
     * Returns the cell high index not including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getFortranCellHighIndex__New(int ngc) const
    {
      //if we have a neighbor add the number of ghost cells to the index
      return d_highIndex__New-IntVector(1,1,1)+IntVector(
                                getBCType(xplus)==Neighbor?ngc:0,
                                getBCType(yplus)==Neighbor?ngc:0,
                                getBCType(zplus)==Neighbor?ngc:0); 
    }
    
    /**
     * Returns the cell low index including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getFortranExtraCellLowIndex__New(int ngc=0) const
    {
      //if have a neighbor subtract the number of ghost cells from the index
      //otherwise subtract the number of extra cells from the index
      return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngc:d_extraCells[0],
                                getBCType(yminus)==Neighbor?ngc:d_extraCells[1],
                                getBCType(zminus)==Neighbor?ngc:d_extraCells[2]); 
    }
    
    /**
     * Returns the cell high index including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline IntVector getFortranExtraCellHighIndex__New(int ngc=0) const
    {
      //if have a neighbor add the number of ghost cells to the index
      //otherwise add the number of extra cells to the index
      return d_highIndex__New-IntVector(1,1,1)+IntVector(
                                getBCType(xplus)==Neighbor?ngc:d_extraCells[0],
                                getBCType(yplus)==Neighbor?ngc:d_extraCells[1],
                                getBCType(zplus)==Neighbor?ngc:d_extraCells[2]); 
    }
    
    /**
     * Returns the node low index not including extra nodes or ghost nodes.
     * This version of this function is included to avoid unnecessary branching.
     */
     inline IntVector getFortranNodeLowIndex__New() const
     {
        return d_lowIndex__New;
     }
    
    /**
     * Returns the node low index not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getFortranNodeLowIndex__New(int ngn) const
    {
        //if we have a neighbor subtract the number of ghost nodes from the index
        return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngn:0,
                                getBCType(yminus)==Neighbor?ngn:0,
                                getBCType(zminus)==Neighbor?ngn:0); 
    }
    
    /**
     * Returns the node high index not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getFortranNodeHighIndex__New(int ngn=0) const
    {
      //if we have a neighbor add the number of ghost nodes to the index
      //otherwise add 1 because we own the plus face node
      return d_highIndex__New-IntVector(1,1,1)+IntVector(
                                getBCType(xplus)==Neighbor?ngn:1,
                                getBCType(yplus)==Neighbor?ngn:1,
                                getBCType(zplus)==Neighbor?ngn:1); 
    } 
    
    /**
     * Returns the node low index including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getFortranExtraNodeLowIndex__New(int ngn=0) const
    {
      //if have a neighbor subtract the number of ghost nodes from the index
      //otherwise subtract the number of extra nodes from the index
      return d_lowIndex__New-IntVector(
                                getBCType(xminus)==Neighbor?ngn:d_extraCells[0],
                                getBCType(yminus)==Neighbor?ngn:d_extraCells[1],
                                getBCType(zminus)==Neighbor?ngn:d_extraCells[2]); 
    }
    
    /**
     * Returns the node high index including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline IntVector getFortranExtraNodeHighIndex__New(int ngn=0) const
    {
      //if have a neighbor add the number of ghost nodes to the index
      //otherwise the number of extra nodes to the index and 1 for the plus face node
      return d_highIndex__New-IntVector(1,1,1)+IntVector(
                                getBCType(xplus)==Neighbor?ngn:d_extraCells[0]+1,
                                getBCType(yplus)==Neighbor?ngn:d_extraCells[1]+1,
                                getBCType(zplus)==Neighbor?ngn:d_extraCells[2]+1); 

    }
    
    /**
     * Returns the staggared face centered on X cell low index excluding extra cells
     */
    inline IntVector getFortranSFCXLowIndex__New() const 
    {
       return getFortranCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on X cell high index excluding extra cells
     */
    inline IntVector getFortranSFCXHighIndex__New() const
    {
      return getFortranCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
    }
    
    /**
     * Returns the staggared face centered on Y cell low index excluding extra cells
     */
    inline IntVector getFortranSFCYLowIndex__New() const 
    {
       return getFortranCellLowIndex__New();
    }

    /**
     * Returns the staggared face centered on Y cell high index excluding extra cells
     */
    inline IntVector getFortranSFCYHighIndex__New() const
    {
      return getFortranCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
    }
    
    /**
     * Returns the staggared face centered on Z cell low index excluding extra cells
     */
    IntVector getFortranSFCZLowIndex__New() const 
    {
       return getFortranCellLowIndex__New();
    }

    /**
     * Returns the staggared face centered on Z cell high index excluding extra cells
     */
    IntVector getFortranSFCZHighIndex__New() const
    {
      return getFortranCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
    }
    
    /**
     * Returns the staggared face centered on X cell low index including extra cells
     */
    inline IntVector getFortranExtraSFCXLowIndex__New() const 
    {
       return getFortranExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on X cell high index including extra cells
     */
    inline IntVector getFortranExtraSFCXHighIndex__New() const
    {
      return getFortranExtraCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
    }
    
    /**
     * Returns the staggared face centered on Y cell low index including extra cells
     */
    inline IntVector getFortranExtraSFCYLowIndex__New() const 
    {
       return getFortranExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on Y cell high index including extra cells
     */
    inline IntVector getFortranExtraSFCYHighIndex__New() const
    {
      return getFortranExtraCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
    }

    /**
     * Returns the staggared face centered on Z cell low index including extra cells
     */
    inline IntVector getFortranExtraSFCZLowIndex__New() const 
    {
       return getFortranExtraCellLowIndex__New();
    }
    
    /**
     * Returns the staggared face centered on Z cell high index including extra cells
     */
    inline IntVector getFortranExtraSFCZHighIndex__New() const
    {
      return getFortranExtraCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
    }
     
    
    /**
     * Returns a cell iterator not including extra cells or ghost cells.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline CellIterator getFortranCellIterator__New() const
    {
      return CellIterator(getFortranCellLowIndex__New(),getFortranCellHighIndex__New());
    }
    
    /**
     * Returns a cell iterator not including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline CellIterator getFortranCellIterator__New(int ngc) const
    {
      return CellIterator(getFortranCellLowIndex__New(ngc),getFortranCellHighIndex__New(ngc));
    }
    
    /**
     * Returns a cell iterator including extra cells.
     * ngc specifies the number of ghost cells.
     */
    inline CellIterator getFortranExtraCellIterator__New(int ngc=0) const
    {
      return CellIterator(getFortranExtraCellLowIndex__New(ngc),getFortranExtraCellHighIndex__New(ngc));
    }
    
    /**
     * Returns a node iterator not including extra nodes or ghost nodes.
     * This version of this function is included to avoid unnecessary branching.
     */
    inline NodeIterator getFortranNodeIterator__New() const
    {
      return NodeIterator(getFortranNodeLowIndex__New(),getFortranNodeHighIndex__New());
    }
    
    /**
     * Returns a node iterator not including extra nodes.
     * ngn specifies the number of ghost nodes.
     */
    inline NodeIterator getFortranNodeIterator__New(int ngn) const
    {
      return NodeIterator(getFortranNodeLowIndex__New(ngn),getFortranNodeHighIndex__New(ngn));
    }
    
    /**
     * Returns a node iterator including extra cells.
     * ngn specifies the number of ghost nodes.
     */
    NodeIterator getFortranExtraNodeIterator__New(int ngn=0) const
    {
      return NodeIterator(getFortranExtraNodeLowIndex__New(ngn),getFortranExtraNodeHighIndex__New(ngn));
    }
    
    /**
     * Returns a staggared face centered on X cell iterator excluding extra cells
     */
    inline CellIterator getFortranSFCXIterator__New() const
    {
      return CellIterator(getFortranSFCXLowIndex__New(),getFortranSFCXHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Y cell iterator excluding extra cells
     */
    inline CellIterator getFortranSFCYIterator__New() const
    {
      return CellIterator(getFortranSFCYLowIndex__New(),getFortranSFCYHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Z cell iterator excluding extra cells
     */
    inline CellIterator getFortranSFCZIterator__New() const
    {
      return CellIterator(getFortranSFCZLowIndex__New(),getFortranSFCZHighIndex__New());
    }

    /**
     * Returns a staggared face centered on X cell iterator including extra cells
     */
    inline CellIterator getFortranExtraSFCXIterator__New() const
    {
      return CellIterator(getFortranExtraSFCXLowIndex__New(),getFortranExtraSFCXHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Y cell iterator including extra cells
     */
    inline CellIterator getFortranExtraSFCYIterator__New() const
    {
      return CellIterator(getFortranExtraSFCYLowIndex__New(),getFortranExtraSFCYHighIndex__New());
    }
    
    /**
     * Returns a staggared face centered on Z cell iterator including extra cells
     */
    inline CellIterator getFortranExtraSFCZIterator__New() const
    {
      return CellIterator(getFortranExtraSFCZLowIndex__New(),getFortranExtraSFCZHighIndex__New());
    }

    /**
     * Sets the number of extra cells for the patch class.
     */
    static inline void setExtraCells(IntVector extraCells) 
    {
      d_extraCells=extraCells;
    }
    
    /**
     * Gets the number of extra cells for the patch class.
     */
    static inline IntVector getExtraCells() 
    {
      return d_extraCells;
    }

    /**
     * Returns the number of cells excluding extra cells
     */
    inline int getNumCells() const
    {
      return getVolume(getCellLowIndex__New(),getCellHighIndex__New());
    }

    /**
     * Returns the number of cells including extra cells
     */
    inline int getNumExtraCells() const
    {
      return getVolume(getExtraCellLowIndex__New(),getExtraCellHighIndex__New());
    }

    /**
     * Returns the number of cells excluding extra cells
     */
    inline int getNumNodes() const
    {
      return getVolume(getNodeLowIndex__New(),getNodeHighIndex__New());
    }

    /**
     * Returns the number of cells excluding extra cells
     */
    inline int getNumExtraNodes() const
    {
      return getVolume(getExtraNodeLowIndex__New(),getExtraNodeHighIndex__New());
    }

    /**
     * Returns the dimension of the face
     * x=0, y=1, z=2
     */
    static inline int getFaceDimension(const FaceType& face)
    {
      return face/2;
    }

    /**
     * Returns the normal to the patch face
     */
    static inline IntVector getFaceDirection(const FaceType& face)
    {
      switch(face) 
      {
        case xminus:
          return IntVector(-1,0,0);
        case xplus:
          return IntVector(1,0,0);
        case yminus:
          return IntVector(0,-1,0);
        case yplus:
          return IntVector(0,1,0);
        case zminus:
          return IntVector(0,0,-1);
        case zplus:
          return IntVector(0,0,1);
        default:
          throw SCIRun::InternalError("Invalid FaceIteratorType Specified", __FILE__, __LINE__);
          return IntVector(0,0,0);
      }
    }
    
    /**
     * Returns the BC type for the specified face
     */
    inline BCType getBCType(FaceType face) const
    {
      switch(face)
      {
        case xminus:
          return d_patchState.xminus;
        case yminus:
          return d_patchState.yminus;
        case zminus:
          return d_patchState.zminus;
        case xplus:
          return d_patchState.xplus;
        case yplus:
          return d_patchState.yplus;
        case zplus:
          return d_patchState.zplus;
        default:
          throw SCIRun::InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
          return None;
      }
    }
    
    /**
     * sets the vector faces equal to the list of faces that have neighbors
     */
    inline void getNeighborFaces(vector<FaceType>& faces) const
    { 
      faces.clear();

      //for each face 
        //if we don't have a neigbor add that face to the boundary vector
      if(getBCType(xminus)==Neighbor) faces.push_back(xminus);
      if(getBCType(xplus)==Neighbor) faces.push_back(xplus);
      if(getBCType(yminus)==Neighbor) faces.push_back(yminus);
      if(getBCType(yplus)==Neighbor) faces.push_back(yplus);
      if(getBCType(zminus)==Neighbor) faces.push_back(zminus);
      if(getBCType(zplus)==Neighbor) faces.push_back(zplus);
    }
    
    /**
     * Returns true if a neighbor face exists on this patch
     */
    bool inline hasNeighborFaces() const
    { 
      return getBCType(xminus)==Neighbor || 
             getBCType(xplus)==Neighbor ||
             getBCType(yminus)==Neighbor ||
             getBCType(yplus)==Neighbor ||
             getBCType(zminus)==Neighbor ||
             getBCType(zplus)==Neighbor;
    }

    /**
     * sets the vector faces equal to the list of faces that are on the boundary
     */
    inline void getBoundaryFaces(vector<FaceType>& faces) const
    { 
      faces.clear();

      //for each face 
        //if the face condition is none add that face to the boundary vector
      if(getBCType(xminus)==None) faces.push_back(xminus);
      if(getBCType(xplus)==None) faces.push_back(xplus);
      if(getBCType(yminus)==None) faces.push_back(yminus);
      if(getBCType(yplus)==None) faces.push_back(yplus);
      if(getBCType(zminus)==None) faces.push_back(zminus);
      if(getBCType(zplus)==None) faces.push_back(zplus);
    }
    
    /**
     * Returns true if a boundary face exists on this patch
     */
    bool inline hasBoundaryFaces() const
    { 
      return getBCType(xminus)==None || 
             getBCType(xplus)==None ||
             getBCType(yminus)==None ||
             getBCType(yplus)==None ||
             getBCType(zminus)==None ||
             getBCType(zplus)==None;
    }
     
    
    /**
     * sets the vector faces equal to the list of faces that are coarse.
     * A face is considered coarse when neighboring cells only exist on
     * a coarser level.  Thus the face does not have a neighbor on the same 
     * level and is not on the boundary.
     */
    inline void getCoarseFaces(vector<FaceType>& faces) const
    { 
      faces.clear();

      //for each face 
        //if we don't have a neigbor add that face to the boundary vector
      if(getBCType(xminus)==Coarse) faces.push_back(xminus);
      if(getBCType(xplus)==Coarse) faces.push_back(xplus);
      if(getBCType(yminus)==Coarse) faces.push_back(yminus);
      if(getBCType(yplus)==Coarse) faces.push_back(yplus);
      if(getBCType(zminus)==Coarse) faces.push_back(zminus);
      if(getBCType(zplus)==Coarse) faces.push_back(zplus);
    }
     
    /**
     * Returns true if a coarse face exists on this patch
     */
    bool inline hasCoarseFaces() const
    { 
      return getBCType(xminus)==Coarse || 
             getBCType(xplus)==Coarse ||
             getBCType(yminus)==Coarse ||
             getBCType(yplus)==Coarse ||
             getBCType(zminus)==Coarse ||
             getBCType(zplus)==Coarse;
    }
     

  
    /**
     * sets the vector cells equal to the list of cells that are in the corners
     */
    void getCornerCells(vector<IntVector> & cells, const FaceType &face) const;

    /**
     * Returns the principal axis along a face and
     *the orthognonal axes to that face (right hand rule).
     */
    static inline IntVector faceAxes(const FaceType& face) 
    {
      switch(face)
      {
        case xminus: case xplus:
          return IntVector(0,1,2);
        case yminus: case yplus:
          return IntVector(1,2,0);
        case zminus: case zplus:
          return IntVector(2,0,1);
        default:
          throw SCIRun::InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
          return IntVector(0,0,0);
      };
    }
     
     /**
      * Returns a string equivalent of the face name (eg: "xminus")
      */
     static string getFaceName(FaceType face);
      
     /**
      * Sets the static pointer to the new grid
      */
     static inline void setNextGrid(GridP grid)
     {
       //update the index
       d_newGridIndex=(d_newGridIndex+1)%2;

       //set the grid pointer 
       //all patches that call finalize patch after
       //this point will point to this grid
       d_grid[d_newGridIndex]=grid;
     }
     

    /*
    void setBCType(FaceType face, BCType newbc)
    {
      switch(face)
      {
        case xminus:
          d_patchState.xminus=newbc;
          break;
        case yminus:
          d_patchState.yminus=newbc;
          break;
        case zminus:
          d_patchState.zminus=newbc;
          break;
        case xplus:
          d_patchState.xplus=newbc;
          break;
        case yplus:
          d_patchState.yplus=newbc;
          break;
        case zplus:
          d_patchState.zplus=newbc;
          break;
        default:
          throw SCIRun::InternalError("Invalid FaceType Specified", __FILE__, __LINE__);
      }
    }
    */
     
    /**
     * Returns the cell spacing Vector(dx,dy,dz)
     */
    inline Vector dCell() const
    {
      // This will need to change for stretched grids
      return getLevel()->dCell();
    }
     
    /**
     * Returns the level that the patch lives on
     */
    inline const Level* getLevel__New() const 
    {
      return d_grid[d_patchState.gridIndex]->getLevel(d_patchState.levelIndex).get_rep();
    }

    /**
     * Returns the domain coordinates of the node idx
     */
    inline Point getNodePosition(const IntVector& idx) const 
    {
      return getLevel()->getNodePosition(idx);
    }

    /**
     * Returns the domain coordinates of the cell idx
     */
    inline Point getCellPosition(const IntVector& idx) const 
    {
      return getLevel()->getCellPosition(idx);
    }
    
    //Below for Fracture *************************************************
    
    /**
     * Returns the 8 nodes found around the point pos
     */
    void findCellNodes(const Point& pos, IntVector ni[8]) const;
    
    /**
     * Returns the 27 nodes found around the point pos
     */
    void findCellNodes27(const Point& pos, IntVector ni[27]) const;
 
    /**
     * Returns true if the point p is contained within the patch
     * including extra cells
     */
    inline bool containsPointInExtraCells(const Point& p) const {
      IntVector l(getNodeLowIndex());
      IntVector h(getNodeHighIndex());
      Point lp = getNodePosition(l);
      Point hp = getNodePosition(h);
      return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
        && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
    }
    /**
     * Returns true if the point p is contained within the patch
     * excluding extra cells
     */
    inline bool containsPoint__New(const Point& p) const {
      IntVector l(getInteriorNodeLowIndex());
      IntVector h(d_inHighIndex);
      Point lp = getNodePosition(l);
      Point hp = getNodePosition(h);
      return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
       && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
    }
    //Above for Fracture *************************************************
     
    /**
     * Returns the cell that contains the point pos
     */
    inline bool findCell(const Point& pos, IntVector& ci) const
    {
      ci=getLevel()->getCellIndex(pos);
      return containsCell(ci);
    }
     
    /**
     * Must be called prior to using certain patch features.
     * Currently this sets the gridIndex.
     */
    void finalizePatch();
     
     
    /**
    * sets the array cellIndex equal to the 8 cells
    * that contribute to the node nodeIndex.
    */
    static void findCellsFromNode( const IntVector& nodeIndex,
                             IntVector cellIndex[8]);
     
    /**
     * sets the array nodeIndex equal to the 8 nodes that this
     * cell contributes to
     */
    static void findNodesFromCell( const IntVector& cellIndex,
                             IntVector nodeIndex[8]);

     
    /**************End New Public Interace****************/
   
    /*
    //Get refrences to the indices
    inline IntVector& low()
    { return d_lowIndex;  }

    inline IntVector& high()
    { return d_highIndex;  }
    */


    /********************
      The following are needed in order to use Patch as a Box in
      Core/Container/SuperBox.h (see
      Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.cc)
    *********************/

    void setBCType(FaceType face, BCType newbc);

    inline int getVolume() const
    { return getVolume(getLow(), getHigh()); }

    inline int getArea(int side) const
    {
      int area = 1;
      for (int i = 0; i < 3; i++)
        if (i != side)
          area *= getHigh()[i] - getLow()[i];
      return area;
    }

    static inline int getVolume(const IntVector& low, const IntVector& high)
    {
      return (high.x() -  low.x()) * (high.y() - low.y()) * (high.z() - low.z());
    } 


     static VariableBasis translateTypeToBasis(TypeDescription::Type type,
                                               bool mustExist);
     
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
     //   I C E - M P M I C E   S T U F F 
     CellIterator getFaceCellIterator(const FaceType& face, 
                                const string& domain="minusEdgeCells") const;
     CellIterator getEdgeCellIterator(const FaceType& face0, 
                                const FaceType& face1,
                                const string& domain="minusCornerCells") const;
     CellIterator getSFCIterator( const int dir, const int offset = 0) const;
     CellIterator addGhostCell_Iter(CellIterator hi_lo, const int nCells) const;
     
     // This will return an iterator which will include all the nodes
     // contained by the bounding box.  If a dimension of the widget
     // is degenerate (has a thickness of 0) the nearest node in that
     // dimension is used.
     NodeIterator getNodeIterator(const Box& b) const;

     // Get a larger set of nodes if using GIMP
     NodeIterator getNodeIterator(const string& interp_type) const;

     IntVector getLowIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;
     IntVector getHighIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;
     IntVector getInteriorLowIndex(VariableBasis basis) const;
     IntVector getInteriorHighIndex(VariableBasis basis) const;

     // do not get coarse-fine extra cells that are not the domain boundary
     IntVector getInteriorLowIndexWithBoundary(VariableBasis basis) const;
     IntVector getInteriorHighIndexWithBoundary(VariableBasis basis) const;
    
     /*
     int getInteriorVolume() const {
       return getVolume(d_inLowIndex,d_inHighIndex); 
     }
     */
     
     /*
     void setExtraIndices(const IntVector& l, const IntVector& h);
    */

     // For AMR.  When there are weird patch configurations, sometimes patches can overlap.
     // Find the intersection betwen the patch and the desired dependency, and then remove the intersection.
     // If the overlap IS the intersection, set the low to be equal to the high.
     void cullIntersection(VariableBasis basis, IntVector bl, const Patch* neighbor,
                           IntVector& region_low, IntVector& region_high) const;
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
     Box getInteriorBox() const;
     
     inline IntVector getNNodes() const {
       return getNodeHighIndex()-getNodeLowIndex();
     }
     
     inline IntVector getNInteriorNodes() const {
       return getInteriorNodeHighIndex()-getInteriorNodeLowIndex();
     }

     long totalCells() const;
     
     void performConsistencyCheck() const;
     
     void printPatchBCs(std::ostream& out) const;
     void setArrayBCValues(FaceType face, BCDataArray* bc);

     const BCDataArray* getBCDataArray(Patch::FaceType face) const;

     const BoundCondBase* getArrayBCValues(FaceType face,int mat_id,
                                           const string& type,
                                           vector<IntVector>*& b_ptr,
                                           vector<IntVector>*& nb_ptr,
                                           int child) const ;
     
     bool haveBC(FaceType face,int mat_id,const string& bc_type,
                 const string& bc_variable) const;

     bool atEdge(FaceType face) const;
     static FaceType nextFace(FaceType face) {
       return (FaceType)((int)face+1);
     }
     
     //__________________________________
     // returns and IntVector with 0 or 1 depending on
     // if there are neighboring patches.  Low examines
     // the minus faces and High the plus faces
     
     IntVector neighborsLow() const;
     IntVector neighborsHigh() const;
     
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
     
     void getFace(FaceType face, const IntVector& insideOffset,
                  const IntVector& outsideOffset,
                  IntVector& l, IntVector& h) const;
                
     IntVector faceDirection(FaceType face) const;
     
     void getFaceNodes(FaceType face, int offset, IntVector& l,
                       IntVector& h) const;

     void getFaceExtraNodes(FaceType face, int offset, IntVector& l,
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

     void getOtherLevelPatches(int levelOffset, selectType& patches, int numGhostCells = 0)
       const;
     
     // get the index into the Level::d_patches array
     int getLevelIndex() const { return d_level_index; }

     // get the index this patch would be if all levels were taken into account
     int getGridIndex() const;
     
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

     void setFaceMark(int markType, FaceType face, int mark) const { d_faceMarks[markType*numFaces + face] = mark; }
     int getFaceMark(int markType, FaceType face) const { return d_faceMarks[markType*numFaces + face]; }

     /***********************Old Interface that is been implemented in new interface*******************************/
    
     inline IntVector getLow() const
    { return d_lowIndex; }

    inline IntVector getHigh() const
    { return d_highIndex; }
     
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

     // returns ghost cell index
     IntVector getGhostCellLowIndex(int numGC) const;
     IntVector getGhostCellHighIndex(int numGC) const;
     
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
     
     
     
     CellIterator getCellIterator(const IntVector gc = IntVector(0,0,0)) const;
     CellIterator getExtraCellIterator(const IntVector gc = 
                                       IntVector(0,0,0)) const;

     NodeIterator getNodeIterator() const;

#if 1
     CellIterator getSFCXIterator(const int offset = 0) const;
     CellIterator getSFCYIterator(const int offset = 0) const;
     CellIterator getSFCZIterator(const int offset = 0) const;
#endif
     
     // required for fortran interface
     IntVector getSFCXFORTLowIndex() const;
     IntVector getSFCXFORTHighIndex() const;

     IntVector getSFCYFORTLowIndex() const;
     IntVector getSFCYFORTHighIndex() const;

     IntVector getSFCZFORTLowIndex() const;
     IntVector getSFCZFORTHighIndex() const;

     IntVector getCellFORTLowIndex() const;
     IntVector getCellFORTHighIndex() const;
     
     inline const Level* getLevel() const {
       return d_level;
     }
     
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


     
     /***********************End old interface********************************************************************/

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
           unsigned int levelIndex,
           int id=-1);
     ~Patch();

     Patch* createVirtualPatch(const IntVector& offset) const
     { return scinew Patch(this, offset); }
   private:

     
     /**
      * This struct will tightly store data that the patch needs
      * elements will only use the number of bits necessary for its
      * storage
      */
     struct PatchState
     {
        //The boundary conditions for each face
        BCType xminus : 2;
        BCType xplus : 2;
        BCType yminus : 2;
        BCType yplus : 2;
        BCType zminus : 2;
        BCType zplus : 2;
        unsigned int gridIndex : 1; //The grid index for this patch
        unsigned int levelIndex : 3; //The level index for this patch (max of 8 levels)
     };


    /****************New Private Interface*********************
     *Functions will slowly be moved into this block as it is verified
     ***********************************************************/
     
    
     /**
      * The low index of the patch without extra or ghost cells
      */
     IntVector d_lowIndex__New;
     /**
      * The high index of the patch without extra or ghost cells
      */
     IntVector d_highIndex__New;

     /**
      * The number of extra cells on the edge of the domain
      */
     static IntVector d_extraCells;

     /**
      * The patch state.  This stores much of the internal state
      * of the patch in a compact datastructure.
      */
     PatchState d_patchState;
      

      /**
       * This array will store pointers to grids.
       * Patches store an index into this array to specify
       * their grid. 
       */
      static GridP d_grid[2];

      /**
       *  The index of the new grid.  All new patches
       *  will point to this grid until setNextGrid is called
       */
      static int d_newGridIndex;


    //****************End of new private Interace**************/

     
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
     
     map<Patch::FaceType,BCDataArray* > array_bcs;

     //////////
     // add a method for the user to mark a patch face (for his own purposes)
     mutable int d_faceMarks[4*numFaces];
     
   }; // end class Patch
   
   inline Patch::FaceType operator++(Patch::FaceType &face,int)
   {
    return face=static_cast<Patch::FaceType>(face+1);
   }


} // End namespace Uintah

#endif
