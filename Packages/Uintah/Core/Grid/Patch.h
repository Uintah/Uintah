#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/fixedvector.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/SuperBox.h>

#undef None

#include <sgi_stl_warnings_off.h>
#include   <string>
#include   <map>
#include   <iosfwd>
#include   <vector>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>

//#define DELETE_OLD_INTERFACE

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
        None=0,
        Symmetry=1,
        Coarse=2,
        Neighbor=3
      };

      enum FaceType {
        xminus=0,
        xplus=1,
        yminus=2,
        yplus=3,
        zminus=4,
        zplus=5,
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
      
      static const int MAX_PATCH_SELECT = 32; 
      typedef fixedvector<const Patch*, MAX_PATCH_SELECT> selectType;


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
       * Returns the staggared face centered on X cell low index excluding extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCXLowIndex__New(int ngc) const 
      {
        return getCellLowIndex__New(ngc);
      }

      /**
       * Returns the staggared face centered on X cell high index excluding extra cells
       */
      inline IntVector getSFCXHighIndex__New() const
      {
        return getCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }
      
      /**
       * Returns the staggared face centered on X cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCXHighIndex__New(int ngc) const
      {
        return getCellHighIndex__New(ngc)+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }


      /**
       * Returns the staggared face centered on Y cell low index excluding extra cells
       */
      inline IntVector getSFCYLowIndex__New() const 
      {
        return getCellLowIndex__New();
      }
      
      /**
       * Returns the staggared face centered on Y cell low index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCYLowIndex__New(int ngc) const 
      {
        return getCellLowIndex__New(ngc);
      }


      /**
       * Returns the staggared face centered on Y cell high index excluding extra cells
       */
      inline IntVector getSFCYHighIndex__New() const
      {
        return getCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Y cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCYHighIndex__New(int ngc) const
      {
        return getCellHighIndex__New(ngc)+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index excluding extra cells
       */
      IntVector getSFCZLowIndex__New() const 
      {
        return getCellLowIndex__New();
      }

      /**
       * Returns the staggared face centered on Z cell low index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      IntVector getSFCZLowIndex__New(int ngc) const 
      {
        return getCellLowIndex__New(ngc);
      }

      /**
       * Returns the staggared face centered on Z cell high index excluding extra cells
       */
      IntVector getSFCZHighIndex__New() const
      {
        return getCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on Z cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      IntVector getSFCZHighIndex__New(int ngc) const
      {
        return getCellHighIndex__New(ngc)+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on X cell low index including extra cells
       */
      inline IntVector getExtraSFCXLowIndex__New() const 
      {
        return getExtraCellLowIndex__New();
      }

      /**
       * Returns the staggared face centered on X cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCXLowIndex__New(int ngc) const 
      {
        return getExtraCellLowIndex__New(ngc);
      }

      /**
       * Returns the staggared face centered on X cell high index including extra cells
       */
      inline IntVector getExtraSFCXHighIndex__New() const
      {
        return getExtraCellHighIndex__New()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }
      
      /**
       * Returns the staggared face centered on X cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCXHighIndex__New(int ngc) const
      {
        return getExtraCellHighIndex__New(ngc)+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }

      /**
       * Returns the staggared face centered on Y cell low index including extra cells
       */
      inline IntVector getExtraSFCYLowIndex__New() const 
      {
        return getExtraCellLowIndex__New();
      }
      
      /**
       * Returns the staggared face centered on Y cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCYLowIndex__New(int ngc) const 
      {
        return getExtraCellLowIndex__New(ngc);
      }


      /**
       * Returns the staggared face centered on Y cell high index including extra cells
       */
      inline IntVector getExtraSFCYHighIndex__New() const
      {
        return getExtraCellHighIndex__New()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Y cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCYHighIndex__New(int ngc) const
      {
        return getExtraCellHighIndex__New(ngc)+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index including extra cells
       */
      inline IntVector getExtraSFCZLowIndex__New() const 
      {
        return getExtraCellLowIndex__New();
      }

      /**
       * Returns the staggared face centered on Z cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCZLowIndex__New(int ngc) const 
      {
        return getExtraCellLowIndex__New(ngc);
      }

      /**
       * Returns the staggared face centered on Z cell high index including extra cells
       */
      inline IntVector getExtraSFCZHighIndex__New() const
      {
        return getExtraCellHighIndex__New()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on Z cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCZHighIndex__New(int ngc) const
      {
        return getExtraCellHighIndex__New(ngc)+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
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
       * Returns the number of cells in the range of low to high.
       * This should only be used by SuperBox code
       */
      static inline int getVolume(const IntVector& low, const IntVector& high)
      {
        return (high.x() -  low.x()) * (high.y() - low.y()) * (high.z() - low.z());
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
            return static_cast<BCType>(d_patchState.xminus);
          case yminus:
            return static_cast<BCType>(d_patchState.yminus);
          case zminus:
            return static_cast<BCType>(d_patchState.zminus);
          case xplus:
            return static_cast<BCType>(d_patchState.xplus);
          case yplus:
            return static_cast<BCType>(d_patchState.yplus);
          case zplus:
            return static_cast<BCType>(d_patchState.zplus);
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
      inline void setGrid(Grid* grid)
      {
        //set the grid pointer 
        d_grid=grid;
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
      inline const Level* getLevel() const 
      {
        return d_grid->getLevel(d_patchState.levelIndex).get_rep();
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
        IntVector l(getExtraNodeLowIndex__New());
        IntVector h(getExtraNodeHighIndex__New());
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
        IntVector l(getNodeLowIndex__New());
        IntVector h(getNodeHighIndex__New());
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

      /**
       * Returns true if the node idx is owned by this patch
       * including extra cells
       */
      inline bool containsNode(const IntVector& idx) const {
        IntVector l(getExtraNodeLowIndex__New());
        IntVector h(getExtraNodeHighIndex__New());
        return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
          && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      /**
       * Returns true if the cell idx is owned by this patch
       * including extra cells
       */
      inline bool containsCell(const IntVector& idx) const {
        IntVector l(getExtraCellLowIndex__New());
        IntVector h(getExtraCellHighIndex__New());
        return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
          && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      /**
       * Returns true if the SFCX idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCX(const IntVector& idx) const {
        IntVector l(getExtraSFCXLowIndex__New());
        IntVector h(getExtraSFCXHighIndex__New());
        return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
          && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      /**
       * Returns true if the SFCY idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCY(const IntVector& idx) const {
        IntVector l(getExtraSFCYLowIndex__New());
        IntVector h(getExtraSFCYHighIndex__New());
        return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
          && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      /**
       * Returns true if the SFCZ idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCZ(const IntVector& idx) const {
        IntVector l(getExtraSFCZLowIndex__New());
        IntVector h(getExtraSFCZHighIndex__New());
        return idx.x() >= l.x() && idx.y() >= l.y() && idx.z() >= l.z()
          && idx.x() < h.x() && idx.y() < h.y() && idx.z() < h.z();
      }

      /**
       * Returns the closest node to the point pos.  This node
       * is not guarenteed to be owned by this patch.
       */
      inline IntVector findClosestNode(const Point& pos) const
      {
        IntVector idx = getLevel()->getCellIndex(pos);
        Point cellP = getLevel()->getCellPosition(idx);
        for(int i=0;i<3;++i) {
          if( pos(i)>cellP(i) ) {
            idx[i]++;
          }
        }
        return idx;
      }

      /**
       *  This function will return an iterator that touches all cells
       *  that are are partially or fully within the region formed
       *  by intersecting the box b and this patch (including extra cells).
       *  The region is inclusive on the + faces.
       *
       *  The patch region includes extra cells
       *
       *  \todo rename to getExtraCellIterator
       */
      CellIterator getCellIterator(const Box& b) const;

      /**
       *  This function will return an iterator that touches all cells
       *  whose centers are within the region formed
       *  by intersecting the box b and this patch.
       *  The region is inclusive on the + faces (including extra cells).
       *
       *  The patch region includes extra cells
       *
       *  \todo rename to getExtraCellCenterIterator
       */
      CellIterator getCellCenterIterator(const Box& b) const;

      /**
       * This will return an iterator which will include all the nodes
       * contained by the bounding box which also intersect the patch 
       * (including extra cells).  If a dimension of the widget is 
       * degenerate (has a thickness of 0) the nearest node in that 
       * dimension is used.
       *
       * The patch region includes extra nodes
       *
       * \todo rename to getExtraNodeIterator
       */
      NodeIterator getNodeIterator(const Box& b) const;

      /**
       * Returns the VariableBasis for the TypeDescription::type specified
       * in type.  If mustExist is true this function will throw an exception
       * if the VariableBasis does not exist for the given type.
       */
      static VariableBasis translateTypeToBasis(TypeDescription::Type type,
          bool mustExist);

      /**
       * Returns the low index for a variable of type basis with extraCells or 
       * the boundaryLayer specified in boundaryLayer.  Having both extraCells
       * and a boundaryLayer is an error.
       */
      IntVector getExtraLowIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;

      /**
       * Returns the high index for a variable of type basis with extraCells or 
       * the boundaryLayer specified in boundaryLayer.  Having both extraCells
       * and a boundaryLayer is an error.
       */
      IntVector getExtraHighIndex(VariableBasis basis, const IntVector& boundaryLayer /*= IntVector(0,0,0)*/) const;

      /**
       * Returns the low index for a variable of type basis without a 
       * boundary layer and without extraCells.
       */
      IntVector getLowIndex(VariableBasis basis) const;

      /**
       * Returns the high index for a variable of type basis without a 
       * boundary layer and without extraCells.
       */
      IntVector getHighIndex(VariableBasis basis) const;

      /**
       * Returns the low index for a variable of type basis without extraCells
       * except on the boundary of the domain.
       */
      IntVector getLowIndexWithDomainLayer(VariableBasis basis) const;

      /**
       * Returns the high index for a variable of type basis without extraCells
       * except on the boundary of the domain.
       */
      IntVector getHighIndexWithDomainLayer(VariableBasis basis) const;
      
      /**
       * For AMR.  When there are weird patch configurations, sometimes patches can overlap.
       * Find the intersection betwen the patch and the desired dependency, and then remove the intersection.
       * If the overlap IS the intersection, set the low to be equal to the high.
       */
      void cullIntersection(VariableBasis basis, IntVector bl, const Patch* neighbor,
          IntVector& region_low, IntVector& region_high) const;

      /**
       * Returns a Box in domain coordinates of the patch including extra cells
       */
      Box getExtraBox() const;
      
      /**
       * Returns a Box in domain coordinates of the patch excluding extra cells
       */
      Box getBox() const;

      /**
       * Returns an IntVector with 0 or 1 depending on
       * if there are neighboring patches on the low faces.  
       */
      IntVector noNeighborsLow() const;
      
      /**
       * Returns an IntVector with 0 or 1 depending on
       * if there are neighboring patches on high low faces.  
       */
      IntVector noNeighborsHigh() const;
      
      /**
       * Returns the position of the node idx in domain coordinates.
       */
      Point nodePosition(const IntVector& idx) const;

      /**
       * Returns the position of the cell idx in domain coordinates.
       */
      Point cellPosition(const IntVector& idx) const;

      /**
       * Returns the next face.  Alternativly the ++ operator also 
       * works on the face.
       */
      static inline FaceType nextFace(FaceType face) {
        return (FaceType)((int)face+1);
      }

      /**
       * prints the patch boundary conditions to to ostream out.
       */
      void printPatchBCs(std::ostream& out) const;

      /**
       * returns a string with patch information
       */
      string toString() const;
     
      /**
       * returns a unique patch id
       */
      inline int getID() const {
        return d_id;
      }
      
      /**
       * Returns the patches on the level offset by levelOffset of this level
       * that overlap with this patch.
       */
      void getOtherLevelPatches(int levelOffset, selectType& patches, int numGhostCells = 0)
        const;
      
      /**
       * Returns the patches on the finer level that overlap
       * this patch
       **/
      void inline getFineLevelPatches(selectType& finePatches) const
      { getOtherLevelPatches(1, finePatches); }

      /**
       * Returns the patches on the coarser level that overlap
       * this patch
       **/
      void inline getCoarseLevelPatches(selectType& coarsePatches) const
      { getOtherLevelPatches(-1, coarsePatches); }
      
      /**
       * Returns an IntVector which has a -1 or +1 on the component of 
       * the dimension that the face is on.  a -1 indicates a minus face 
       * and a +1 indicates a plus face.
       */
      IntVector faceDirection(FaceType face) const;


      /**
       * Returns the index that this patch would be
       * if all of the levels were taken into account.
       * This query is O(L) where L is the number of levels.
       */
      int getGridIndex() const;
     

      /** The following functions should be looked at and verified.  Right now they are obscure. **/
      
      /**
       * Passes back the low and high offsets for the given ghost cell
       * scenario.  Note: you should subtract the lowOffset (the offsets
       * should be >= 0 in each dimension).
       */
      static void getGhostOffsets(VariableBasis basis, Ghost::GhostType gtype,
          int numGhostCells,
          IntVector& lowOffset, IntVector& highOffset);
      
      /**
       * Passes back the low and high offsets for the given ghost cell
       * scenario.  Note: you should subtract the lowOffset (the offsets
       * should be >= 0 in each dimension).
       */
      static void getGhostOffsets(TypeDescription::Type basis,
          Ghost::GhostType gtype, int numGhostCells,
          IntVector& l, IntVector& h)
      {
        bool basisMustExist = (gtype != Ghost::None);
        getGhostOffsets(translateTypeToBasis(basis, basisMustExist),
            gtype, numGhostCells, l, h);
      }
      
      /**
       * Computes the variables extents (high and low points)
       * for the given basis, boundary layer, and ghost cells.
       * Neighboring patches are returned in neighbors.
       */
      void computeVariableExtents(VariableBasis basis,
          const IntVector& boundaryLayer,
          Ghost::GhostType gtype, int numGhostCells,
          selectType& neighbors,
          IntVector& low, IntVector& high) const;
      
      /**
       * Computes the variables extents (high and low points)
       * for the given basis, boundary layer, and ghost cells
       * Neighboring patches are returned in neighbors.
       */
      void computeVariableExtents(TypeDescription::Type basis,
          const IntVector& boundaryLayer,
          Ghost::GhostType gtype, int numGhostCells,
          selectType& neighbors,
          IntVector& low, IntVector& high) const;

      /**
       * Computes the variables extents (high and low points)
       * for the given basis, boundary layer, and ghost cells
       */
      void computeVariableExtents(VariableBasis basis,
          const IntVector& boundaryLayer,
          Ghost::GhostType gtype, int numGhostCells,
          IntVector& low, IntVector& high) const;

      /**
       * Computes the variables extents (high and low points)
       * for the given basis, boundary layer, and ghost cells
       */
      void computeVariableExtents(TypeDescription::Type basis,
          const IntVector& boundaryLayer,
          Ghost::GhostType gtype, int numGhostCells,
          IntVector& low, IntVector& high) const;

      /**
       * helper for computeVariableExtents but also used externally
       */
      void computeExtents(VariableBasis basis,
          const IntVector& boundaryLayer,
          const IntVector& lowOffset,
          const IntVector& highOffset,
          IntVector& low, IntVector& high) const;

      /**
       * Verifies that the patch is valid
       */
      void performConsistencyCheck() const;
     
      /**
       * Returns the box including the ghostCells specified
       * in lowOffset and highOffset
       */
      Box getGhostBox(const IntVector& lowOffset,
          const IntVector& highOffset) const;

      /**
       * Returns true if a patch is virtual.
       * Virtual patches exist beyond periiodic boundaries and 
       * point to real patches on the other side of the domain
       */
      bool isVirtual() const
      { return d_realPatch != 0; }

      /**
       * Returns a pointer to the real patch.
       * If the patch is virtual it will return a pointer
       * to a patch on the opposite side of the domain.  If the 
       * patch is real it will return a pointer to itself.
       */
      const Patch* getRealPatch() const
      { return isVirtual() ? d_realPatch : this; }

      /**
       * Returns the offset between the virtual patch and the real patch 
       * in index coordinates.
       */ 
      IntVector getVirtualOffset() const
      { return getCellLowIndex__New() - getRealPatch()->getCellLowIndex__New(); }

      /**
       * Returns the offset between the virtual patch and the real patch 
       * in domain coordinates.
       */
      Vector getVirtualOffsetVector() const
      { return cellPosition(getCellLowIndex__New()) -
        cellPosition(getRealPatch()->getCellLowIndex__New()); 
      }     

      /**************End New Public Interace****************/


      void setBCType(FaceType face, BCType newbc);


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


      /*****Boundary condition code to be worked on by John******/
      void setArrayBCValues(FaceType face, BCDataArray* bc);

      const BCDataArray* getBCDataArray(Patch::FaceType face) const;

      const BoundCondBase* getArrayBCValues(FaceType face,int mat_id,
                                            const string& type,
                                            Iterator& b_ptr,
                                            Iterator& e_ptr,
                                            int child) const ;


      bool haveBC(FaceType face,int mat_id,const string& bc_type,
          const string& bc_variable) const;

      void initializeBoundaryConditions();

      /*****end boundary condition ****/


      /***This section is functions that have yet to be migrated to the new interface.
       * It also includes functions that may be removed from patch in the future*/

      void setFaceMark(int markType, FaceType face, int mark) const { d_faceMarks[markType*numFaces + face] = mark; }
      
      void getFace(FaceType face, const IntVector& insideOffset,
          const IntVector& outsideOffset,
          IntVector& l, IntVector& h) const;

      void getFaceNodes(FaceType face, int offset, IntVector& l,
          IntVector& h) const;

      void getFaceExtraNodes(FaceType face, int offset, IntVector& l,
          IntVector& h) const;

      void getFaceCells(FaceType face, int offset, IntVector& l,
          IntVector& h) const;


      // get the index into the Level::d_patches array
      int getLevelIndex() const { return d_level_index; }

      int getFaceMark(int markType, FaceType face) const { return d_faceMarks[markType*numFaces + face]; }

      /*  End the section of unupdated functions */
      
      /***********************Old Interface that is been implemented in new interface*******************************/
#ifndef DELETE_OLD_INTERFACE

      /**
       * Replace with getExtraCellLowIndex__New()
       */
      IntVector getCellLowIndex() const {
        return d_lowIndex;
      }

      /**
       * Replace with getExtraCellHighIndex__New()
       */
      IntVector getCellHighIndex() const {
        return d_highIndex;
      }

      /**
       * Replace with getCellLowIndex__New()
       */
      IntVector getInteriorCellLowIndex() const {
        return d_inLowIndex;
      }

      /**
       * Replace with getCellHighIndex__New()
       */
      IntVector getInteriorCellHighIndex() const {
        return d_inHighIndex;
      }

      /**
       * Replace with getExtraCellLowIndex__New(numGC)
       */
      IntVector getGhostCellLowIndex(int numGC) const;

      /**
       * Replace with getExtraCellHighIndex__New(numGC)
       */
      IntVector getGhostCellHighIndex(int numGC) const;

      /**
       * Replace with getExtraCellLowIndex__New()
       */
      IntVector getLowIndex() const
      { return d_lowIndex; }

      /**
       * Replace with getExtraCellHighIndex__New()
       */
      IntVector getHighIndex() const
      { return d_highIndex; }

      /**
       * Replace with getExtraNodeLowIndex__New(numGC)
       */
      IntVector getNodeLowIndex() const {
        return d_lowIndex;
      }

      /**
       * Replace with getExtraNodeHighIndex__New(numGC)
       */
      IntVector getNodeHighIndex() const {
        return d_nodeHighIndex;
      }

      /**
       * Replace with getNodeLowIndex__New(numGC)
       */
      IntVector getInteriorNodeLowIndex()const;

      /**
       * Replace with getNodeHighIndex__New(numGC)
       */
      IntVector getInteriorNodeHighIndex()const;     

      /**
       * Replace with getExtraSFCXLowIndex__New()
       */
      IntVector getSFCXLowIndex() const {
        return d_lowIndex;
      }

      /**
       * Replace with getExtraSFCXHighIndex__New()
       */
      IntVector getSFCXHighIndex() const;

      /**
       * Replace with getExtraSFCYLowIndex__New()
       */
      IntVector getSFCYLowIndex() const {
        return d_lowIndex;
      }

      /**
       * Replace with getExtraSFCYHighIndex__New()
       */
      IntVector getSFCYHighIndex() const;

      /**
       * Replace with getExtraSFCZLowIndex__New()
       */
      IntVector getSFCZLowIndex() const {
        return d_lowIndex;
      }

      /**
       * Replace with getExtraSFCZHighIndex__New()
       */
      IntVector getSFCZHighIndex() const;


      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      CellIterator getCellIterator(const IntVector gc = IntVector(0,0,0)) const;

      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      CellIterator getExtraCellIterator(const IntVector gc = 
          IntVector(0,0,0)) const;

      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      NodeIterator getNodeIterator() const;

      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      CellIterator getSFCXIterator(const int offset = 0) const;

      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      CellIterator getSFCYIterator(const int offset = 0) const;

      /**
       * No need to replace old name and new name will be the same thus I can change it over
       * in the header file alone
       */
      CellIterator getSFCZIterator(const int offset = 0) const;

      /**
       * Replace with getFortranSFCXLowIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCXFORTLowIndex() const;

      /**
       * Replace with getFortranSFCXHighIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCXFORTHighIndex() const;

      /**
       * Replace with getFortranSFCYLowIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCYFORTLowIndex() const;

      /**
       * Replace with getFortranSFCYHighIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCYFORTHighIndex() const;

      /**
       * Replace with getFortranSFCZLowIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCZFORTLowIndex() const;

      /**
       * Replace with getFortranSFCZHighIndex__New()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCZFORTHighIndex() const;

      /**
       * Replace with getFortranCellLowIndex__New()
       */
      IntVector getCellFORTLowIndex() const;
      /**
       * Replace with getFortranCellHighIndex__New()
       */
      IntVector getCellFORTHighIndex() const;

      /**
       * determine if a point is in the patch
       * Replace with containsPointInExtraCells()
       */
      inline bool containsPoint(const Point& p) const {
        IntVector l(getNodeLowIndex());
        IntVector h(getNodeHighIndex());
        Point lp = nodePosition(l);
        Point hp = nodePosition(h);
        return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
          && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
      }

      /**
       * determine if a point is in the patch
       * Replace with containsPoint__New()
       */
      inline bool containsPointInRealCells(const Point& p) const {
        IntVector l(getInteriorNodeLowIndex());
        IntVector h(d_inHighIndex);
        Point lp = nodePosition(l);
        Point hp = nodePosition(h);
        return p.x() >= lp.x() && p.y() >= lp.y() && p.z() >= lp.z()
          && p.x() < hp.x() && p.y() < hp.y() && p.z() < hp.z();
      }

      /**
       * Replace this with getExtraNodeIterator__New()
       *  this assumes when using gimp or 3rdorderBS the extra cells = IntVector(1,1,1)
       *  when not using gimp or 3rdorderBS the extracells=IntVector(0,0,0)
       */
      NodeIterator getNodeIterator(const string& interp_type) const;

      /**
       * Replace with getFaceIterator__New(FaceType face,FaceIteratorType type)
       */
      CellIterator getFaceCellIterator(const FaceType& face, 
          const string& domain="minusEdgeCells") const;
      /**
       * Replace with getEdgeCellIterator__New(FaceType face0,FaceType face1, bool minusCornerCells)
       */
      CellIterator getEdgeCellIterator(const FaceType& face0, 
          const FaceType& face1,
          const string& domain="minusCornerCells") const;
      /**
       * Adding ghhostcells to an iterator is now obsolete.  You should be able to get ghostcells by providing the number
       * you want to the get*Iterator(numGC) queries
       */
      CellIterator addGhostCell_Iter(CellIterator hi_lo, const int nCells) const;
#endif
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
        unsigned int xminus : 2;
        unsigned int xplus : 2;
        unsigned int yminus : 2;
        unsigned int yplus : 2;
        unsigned int zminus : 2;
        unsigned int zplus : 2;
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
       * This stores a pointer to the grid grids.
       */
      Grid* d_grid;

      /**
       * A unique patch id.
       * If more storage is needed then the id could be moved into 
       * the patch state with less bits.
       */
      int d_id;

      /**This section includes members that may be phased out in the future**/
      
      
      /** 
       * A pointer to the real patch 
       * This pointer is null for non-virtual patches.
       * Virtual patches exist for wrap-around on periodic boundary 
       * conditions.
       */
      const Patch* d_realPatch;

      //****************End of new private Interace**************/


      Patch(const Patch&);
      Patch(const Patch* realPatch, const IntVector& virtualOffset);
      Patch& operator=(const Patch&);


      int d_level_index;  // I'm at this index in the Level vector;

      // used only by friend class Level
      inline void setLevelIndex( int idx ){ d_level_index = idx;}
      
      vector<BCDataArray*>* d_arrayBCS;

      //////////
      // add a method for the user to mark a patch face (for his own purposes)
      mutable int d_faceMarks[4*numFaces];

      /*****************Members to be phased out*****************/
#ifndef DELETE_OLD_INTERFACE      
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
#endif      
      /******************End of members to be phased out***********/


      /********************
        The following are needed in order to use Patch as a Box in
        Core/Container/SuperBox.h (see
        Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.cc)

        These are private so that other people don't try to use them.  Please
        use the other more descriptive queries.
       *********************/

      friend class SCIRun::InternalAreaSuperBoxEvaluator<const Uintah::Patch*, int>;
      friend class SCIRun::SuperBox<const Patch*, IntVector, int, int,
             SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> >;
      friend class SCIRun::BasicBox<const Patch*, IntVector, int, int, 
             SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> >;
      
      /**
       * Returns the low index including extra cells
       */
      inline IntVector getLow() const
      { return getExtraCellLowIndex__New(); }

      /**
       * Returns the high index including extra cells
       */
      inline IntVector getHigh() const
      { return getExtraCellHighIndex__New(); }

      /** 
       * Returns the number of cells in a patch including extraCells.
       * This should only be used by SuperBox code
       */
      inline int getVolume() const
      { return getVolume(getExtraCellLowIndex__New(), getExtraCellHighIndex__New()); }

      /** 
       * Returns the number of cells on a patches face including extraCells.
       * This should only be used by SuperBox code.  side specifies the dimension.
       */
      inline int getArea(int side) const
      {
        int area = 1;
        for (int i = 0; i < 3; i++)
          if (i != side)
            area *= getExtraCellHighIndex__New()[i] - getExtraCellLowIndex__New()[i];
        return area;
      }

     
   }; // end class Patch
   
   inline Patch::FaceType operator++(Patch::FaceType &face,int)
   {
    return face=static_cast<Patch::FaceType>(face+1);
   }


} // End namespace Uintah

#endif
