/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_Patch_H
#define UINTAH_HOMEBREW_Patch_H

#include <Core/Grid/Grid.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Level.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/fixedvector.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/Iterator.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Containers/SuperBox.h>

#undef None

#include   <string>
#include   <map>
#include   <iosfwd>
#include   <vector>



#if defined( __PGI )
#  define WARNS_ABOUT_UNREACHABLE_STATEMENTS 1
#endif

namespace Uintah {

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
      
      
KEYWORDS
   Patch
      
DESCRIPTION
   Long description...
      
WARNING
     
****************************************/
     
    
  class Patch {
    public:

    friend std::ostream& operator<<(std::ostream& out, const Uintah::Patch & r);

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

      enum EdgeIteratorType {
        ExtraCells,                     //Extra cells on the edge
        ExtraCellsMinusCorner,          //Extra cells on the edge without the corners
        ExtraSFC,                       //SFC variables on the edge
        ExtraSFCMinusCorner,            //SFC variables on the edge without the corners
        SFC,                            //SFC variables on the edge between normal and extra cells
        SFCMinusCorner                  //Same as SFC without the corners
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
      inline IntVector getCellLowIndex() const
      {
        return d_lowIndex;
      }

      /**
       * Returns the cell low index not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector  getCellLowIndex(int ngc) const
      {  
        //if we have a neighbor subtract the number of ghost cells from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngc:0,
            getBCType(yminus)==Neighbor?ngc:0,
            getBCType(zminus)==Neighbor?ngc:0); 
      }

      /**
       * Returns the cell high index not including extra cells or ghost cells.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline IntVector getCellHighIndex() const
      {
        return d_highIndex;
      }

      /**
       * Returns the cell high index not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getCellHighIndex(int ngc) const
      {
        //if we have a neighbor add the number of ghost cells to the index
        return d_highIndex+IntVector(
            getBCType(xplus)==Neighbor?ngc:0,
            getBCType(yplus)==Neighbor?ngc:0,
            getBCType(zplus)==Neighbor?ngc:0); 
      }

      /**
       * Returns the cell low index including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraCellLowIndex(int ngc=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor subtract the number of ghost cells from the index
        //otherwise subtract the number of extra cells from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngc:ec[0],
            getBCType(yminus)==Neighbor?ngc:ec[1],
            getBCType(zminus)==Neighbor?ngc:ec[2]); 
      }

      /**
       * Returns the cell high index including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraCellHighIndex(int ngc=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor add the number of ghost cells to the index
        //otherwise add the number of extra cells to the index
        return d_highIndex+IntVector(
            getBCType(xplus)==Neighbor?ngc:ec[0],
            getBCType(yplus)==Neighbor?ngc:ec[1],
            getBCType(zplus)==Neighbor?ngc:ec[2]); 
      }

      /**
       * Returns the node low index not including extra nodes or ghost nodes.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline IntVector getNodeLowIndex() const
      {
        return d_lowIndex;
      }

      /**
       * Returns the node low index not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getNodeLowIndex(int ngn) const
      {
        //if we have a neighbor subtract the number of ghost nodes from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngn:0,
            getBCType(yminus)==Neighbor?ngn:0,
            getBCType(zminus)==Neighbor?ngn:0); 
      }

      /**
       * Returns the node high index not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getNodeHighIndex(int ngn=0) const
      {
        //if we have a neighbor add the number of ghost nodes to the index
        //otherwise add 1 because we own the plus face node
        return d_highIndex+IntVector(
            getBCType(xplus)==Neighbor?ngn:1,
            getBCType(yplus)==Neighbor?ngn:1,
            getBCType(zplus)==Neighbor?ngn:1); 
      } 

      /**
       * Returns the node low index including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getExtraNodeLowIndex(int ngn=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor subtract the number of ghost nodes from the index
        //otherwise subtract the number of extra nodes from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngn:ec[0],
            getBCType(yminus)==Neighbor?ngn:ec[1],
            getBCType(zminus)==Neighbor?ngn:ec[2]); 
      }

      /**
       * Returns the node high index including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getExtraNodeHighIndex(int ngn=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor add the number of ghost nodes to the index
        //otherwise the number of extra nodes to the index and 1 for the plus face node
        return d_highIndex+IntVector(
            getBCType(xplus)==Neighbor?ngn:ec[0]+1,
            getBCType(yplus)==Neighbor?ngn:ec[1]+1,
            getBCType(zplus)==Neighbor?ngn:ec[2]+1); 

      }

      /**
       * Returns the staggared face centered on X cell low index excluding extra cells
       */
      inline IntVector getSFCXLowIndex() const 
      {
        return getCellLowIndex();
      }
      
      /**
       * Returns the staggared face centered on X cell low index excluding extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCXLowIndex(int ngc) const 
      {
        return getCellLowIndex(ngc);
      }

      /**
       * Returns the staggared face centered on X cell high index excluding extra cells
       */
      inline IntVector getSFCXHighIndex() const
      {
        return getCellHighIndex()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }
      
      /**
       * Returns the staggared face centered on X cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCXHighIndex(int ngc) const
      {
        return getCellHighIndex(ngc)+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }


      /**
       * Returns the staggared face centered on Y cell low index excluding extra cells
       */
      inline IntVector getSFCYLowIndex() const 
      {
        return getCellLowIndex();
      }
      
      /**
       * Returns the staggared face centered on Y cell low index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCYLowIndex(int ngc) const 
      {
        return getCellLowIndex(ngc);
      }


      /**
       * Returns the staggared face centered on Y cell high index excluding extra cells
       */
      inline IntVector getSFCYHighIndex() const
      {
        return getCellHighIndex()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Y cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getSFCYHighIndex(int ngc) const
      {
        return getCellHighIndex(ngc)+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index excluding extra cells
       */
      IntVector getSFCZLowIndex() const 
      {
        return getCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Z cell low index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      IntVector getSFCZLowIndex(int ngc) const 
      {
        return getCellLowIndex(ngc);
      }

      /**
       * Returns the staggared face centered on Z cell high index excluding extra cells
       */
      IntVector getSFCZHighIndex() const
      {
        return getCellHighIndex()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on Z cell high index excluding extra cells
       * ngc specifies the number of ghost cells.
       */
      IntVector getSFCZHighIndex(int ngc) const
      {
        return getCellHighIndex(ngc)+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on X cell low index including extra cells
       */
      inline IntVector getExtraSFCXLowIndex() const 
      {
        return getExtraCellLowIndex();
      }

      /**
       * Returns the staggared face centered on X cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCXLowIndex(int ngc) const 
      {
        return getExtraCellLowIndex(ngc);
      }

      /**
       * Returns the staggared face centered on X cell high index including extra cells
       */
      inline IntVector getExtraSFCXHighIndex() const
      {
        return getExtraCellHighIndex()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }
      
      /**
       * Returns the staggared face centered on X cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCXHighIndex(int ngc) const
      {
        return getExtraCellHighIndex(ngc)+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }

      /**
       * Returns the staggared face centered on Y cell low index including extra cells
       */
      inline IntVector getExtraSFCYLowIndex() const 
      {
        return getExtraCellLowIndex();
      }
      
      /**
       * Returns the staggared face centered on Y cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCYLowIndex(int ngc) const 
      {
        return getExtraCellLowIndex(ngc);
      }


      /**
       * Returns the staggared face centered on Y cell high index including extra cells
       */
      inline IntVector getExtraSFCYHighIndex() const
      {
        return getExtraCellHighIndex()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Y cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCYHighIndex(int ngc) const
      {
        return getExtraCellHighIndex(ngc)+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index including extra cells
       */
      inline IntVector getExtraSFCZLowIndex() const 
      {
        return getExtraCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Z cell low index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCZLowIndex(int ngc) const 
      {
        return getExtraCellLowIndex(ngc);
      }

      /**
       * Returns the staggared face centered on Z cell high index including extra cells
       */
      inline IntVector getExtraSFCZHighIndex() const
      {
        return getExtraCellHighIndex()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on Z cell high index including extra cells
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getExtraSFCZHighIndex(int ngc) const
      {
        return getExtraCellHighIndex(ngc)+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }


      /**
       * Returns a cell iterator not including extra cells or ghost cells.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline CellIterator getCellIterator() const
      {
        return CellIterator(getCellLowIndex(),getCellHighIndex());
      }

      /**
       * Returns a cell iterator not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline CellIterator getCellIterator(int ngc) const
      {
        return CellIterator(getCellLowIndex(ngc),getCellHighIndex(ngc));
      }

      /**
       * Returns a cell iterator including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline CellIterator getExtraCellIterator(int ngc=0) const
      {
        return CellIterator(getExtraCellLowIndex(ngc),getExtraCellHighIndex(ngc));
      }

      /**
       * Returns a node iterator not including extra nodes or ghost nodes.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline NodeIterator getNodeIterator() const
      {
        return NodeIterator(getNodeLowIndex(),getNodeHighIndex());
      }

      /**
       * Returns a node iterator not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline NodeIterator getNodeIterator(int ngn) const
      {
        return NodeIterator(getNodeLowIndex(ngn),getNodeHighIndex(ngn));
      }

      /**
       * Returns a node iterator including extra cells.
       * ngn specifies the number of ghost nodes.
       */
      NodeIterator getExtraNodeIterator(int ngn=0) const
      {
        return NodeIterator(getExtraNodeLowIndex(ngn),getExtraNodeHighIndex(ngn));
      }

      /**
       * Returns a staggared face centered on X cell iterator excluding extra cells
       */
      inline CellIterator getSFCXIterator() const
      {
        return CellIterator(getSFCXLowIndex(),getSFCXHighIndex());
      }

      /**
       * Returns a staggared face centered on Y cell iterator excluding extra cells
       */
      inline CellIterator getSFCYIterator() const
      {
        return CellIterator(getSFCYLowIndex(),getSFCYHighIndex());
      }

      /**
       * Returns a staggared face centered on Z cell iterator excluding extra cells
       */
      inline CellIterator getSFCZIterator() const
      {
        return CellIterator(getSFCZLowIndex(),getSFCZHighIndex());
      }

      /**
       * Returns a staggared face centered on X cell iterator including extra cells
       */
      inline CellIterator getExtraSFCXIterator() const
      {
        return CellIterator(getExtraSFCXLowIndex(),getExtraSFCXHighIndex());
      }

      /**
       * Returns a staggared face centered on Y cell iterator including extra cells
       */
      inline CellIterator getExtraSFCYIterator() const
      {
        return CellIterator(getExtraSFCYLowIndex(),getExtraSFCYHighIndex());
      }

      /**
       * Returns a staggared face centered on Z cell iterator including extra cells
       */
      inline CellIterator getExtraSFCZIterator() const
      {
        return CellIterator(getExtraSFCZLowIndex(),getExtraSFCZHighIndex());
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
      CellIterator getFaceIterator(const FaceType& face, const FaceIteratorType& domain) const;

      /**
       *Returns an iterator to the edge of two intersecting faces.
       *if minusCornerCells is true the edge will exclude corner cells.
       */
      CellIterator getEdgeCellIterator(const FaceType& face0,const FaceType& face1,const EdgeIteratorType &type) const;

      /*************************************************************
       *The following queries are for fortran.  Fortran indexing
       *is inclusiving meaning we have to subtract 1 from the high
       *indices.
       *************************************************************/

      /**
       * Returns the cell low index not including extra cells or ghost cells.
       * This version of this function included to avoid unnecessary branching.
       */
      inline IntVector getFortranCellLowIndex() const
      {
        return d_lowIndex;
      }

      /**
       * Returns the cell low index not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector  getFortranCellLowIndex(int ngc) const
      {  
        //if we have a neighbor subtract the number of ghost cells from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngc:0,
            getBCType(yminus)==Neighbor?ngc:0,
            getBCType(zminus)==Neighbor?ngc:0); 
      }

      /**
       * Returns the cell high index not including extra cells or ghost cells.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline IntVector getFortranCellHighIndex() const
      {
        return d_highIndex-IntVector(1,1,1);
      }

      /**
       * Returns the cell high index not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getFortranCellHighIndex(int ngc) const
      {
        //if we have a neighbor add the number of ghost cells to the index
        return d_highIndex-IntVector(1,1,1)+IntVector(
            getBCType(xplus)==Neighbor?ngc:0,
            getBCType(yplus)==Neighbor?ngc:0,
            getBCType(zplus)==Neighbor?ngc:0); 
      }

      /**
       * Returns the cell low index including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getFortranExtraCellLowIndex(int ngc=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor subtract the number of ghost cells from the index
        //otherwise subtract the number of extra cells from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngc:ec[0],
            getBCType(yminus)==Neighbor?ngc:ec[1],
            getBCType(zminus)==Neighbor?ngc:ec[2]); 
      }

      /**
       * Returns the cell high index including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline IntVector getFortranExtraCellHighIndex(int ngc=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor add the number of ghost cells to the index
        //otherwise add the number of extra cells to the index
        return d_highIndex-IntVector(1,1,1)+IntVector(
            getBCType(xplus)==Neighbor?ngc:ec[0],
            getBCType(yplus)==Neighbor?ngc:ec[1],
            getBCType(zplus)==Neighbor?ngc:ec[2]); 
      }

      /**
       * Returns the node low index not including extra nodes or ghost nodes.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline IntVector getFortranNodeLowIndex() const
      {
        return d_lowIndex;
      }

      /**
       * Returns the node low index not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getFortranNodeLowIndex(int ngn) const
      {
        //if we have a neighbor subtract the number of ghost nodes from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngn:0,
            getBCType(yminus)==Neighbor?ngn:0,
            getBCType(zminus)==Neighbor?ngn:0); 
      }

      /**
       * Returns the node high index not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getFortranNodeHighIndex(int ngn=0) const
      {
        //if we have a neighbor add the number of ghost nodes to the index
        //otherwise add 1 because we own the plus face node
        return d_highIndex-IntVector(1,1,1)+IntVector(
            getBCType(xplus)==Neighbor?ngn:1,
            getBCType(yplus)==Neighbor?ngn:1,
            getBCType(zplus)==Neighbor?ngn:1); 
      } 

      /**
       * Returns the node low index including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getFortranExtraNodeLowIndex(int ngn=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor subtract the number of ghost nodes from the index
        //otherwise subtract the number of extra nodes from the index
        return d_lowIndex-IntVector(
            getBCType(xminus)==Neighbor?ngn:ec[0],
            getBCType(yminus)==Neighbor?ngn:ec[1],
            getBCType(zminus)==Neighbor?ngn:ec[2]); 
      }

      /**
       * Returns the node high index including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline IntVector getFortranExtraNodeHighIndex(int ngn=0) const
      {
        IntVector ec=getExtraCells();
        //if have a neighbor add the number of ghost nodes to the index
        //otherwise the number of extra nodes to the index and 1 for the plus face node
        return d_highIndex-IntVector(1,1,1)+IntVector(
            getBCType(xplus)==Neighbor?ngn:ec[0]+1,
            getBCType(yplus)==Neighbor?ngn:ec[1]+1,
            getBCType(zplus)==Neighbor?ngn:ec[2]+1); 

      }

      /**
       * Returns the staggared face centered on X cell low index excluding extra cells
       */
      inline IntVector getFortranSFCXLowIndex() const 
      {
        return getFortranCellLowIndex();
      }

      /**
       * Returns the staggared face centered on X cell high index excluding extra cells
       */
      inline IntVector getFortranSFCXHighIndex() const
      {
        return getFortranCellHighIndex()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }

      /**
       * Returns the staggared face centered on Y cell low index excluding extra cells
       */
      inline IntVector getFortranSFCYLowIndex() const 
      {
        return getFortranCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Y cell high index excluding extra cells
       */
      inline IntVector getFortranSFCYHighIndex() const
      {
        return getFortranCellHighIndex()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index excluding extra cells
       */
      IntVector getFortranSFCZLowIndex() const 
      {
        return getFortranCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Z cell high index excluding extra cells
       */
      IntVector getFortranSFCZHighIndex() const
      {
        return getFortranCellHighIndex()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }

      /**
       * Returns the staggared face centered on X cell low index including extra cells
       */
      inline IntVector getFortranExtraSFCXLowIndex() const 
      {
        return getFortranExtraCellLowIndex();
      }

      /**
       * Returns the staggared face centered on X cell high index including extra cells
       */
      inline IntVector getFortranExtraSFCXHighIndex() const
      {
        return getFortranExtraCellHighIndex()+IntVector(getBCType(xplus) == Neighbor?0:1, 0, 0);
      }

      /**
       * Returns the staggared face centered on Y cell low index including extra cells
       */
      inline IntVector getFortranExtraSFCYLowIndex() const 
      {
        return getFortranExtraCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Y cell high index including extra cells
       */
      inline IntVector getFortranExtraSFCYHighIndex() const
      {
        return getFortranExtraCellHighIndex()+IntVector(0, getBCType(yplus) == Neighbor?0:1, 0);
      }

      /**
       * Returns the staggared face centered on Z cell low index including extra cells
       */
      inline IntVector getFortranExtraSFCZLowIndex() const 
      {
        return getFortranExtraCellLowIndex();
      }

      /**
       * Returns the staggared face centered on Z cell high index including extra cells
       */
      inline IntVector getFortranExtraSFCZHighIndex() const
      {
        return getFortranExtraCellHighIndex()+IntVector(0,0, getBCType(zplus) == Neighbor?0:1);
      }


      /**
       * Returns a cell iterator not including extra cells or ghost cells.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline CellIterator getFortranCellIterator() const
      {
        return CellIterator(getFortranCellLowIndex(),getFortranCellHighIndex());
      }

      /**
       * Returns a cell iterator not including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline CellIterator getFortranCellIterator(int ngc) const
      {
        return CellIterator(getFortranCellLowIndex(ngc),getFortranCellHighIndex(ngc));
      }

      /**
       * Returns a cell iterator including extra cells.
       * ngc specifies the number of ghost cells.
       */
      inline CellIterator getFortranExtraCellIterator(int ngc=0) const
      {
        return CellIterator(getFortranExtraCellLowIndex(ngc),getFortranExtraCellHighIndex(ngc));
      }

      /**
       * Returns a node iterator not including extra nodes or ghost nodes.
       * This version of this function is included to avoid unnecessary branching.
       */
      inline NodeIterator getFortranNodeIterator() const
      {
        return NodeIterator(getFortranNodeLowIndex(),getFortranNodeHighIndex());
      }

      /**
       * Returns a node iterator not including extra nodes.
       * ngn specifies the number of ghost nodes.
       */
      inline NodeIterator getFortranNodeIterator(int ngn) const
      {
        return NodeIterator(getFortranNodeLowIndex(ngn),getFortranNodeHighIndex(ngn));
      }

      /**
       * Returns a node iterator including extra cells.
       * ngn specifies the number of ghost nodes.
       */
      NodeIterator getFortranExtraNodeIterator(int ngn=0) const
      {
        return NodeIterator(getFortranExtraNodeLowIndex(ngn),getFortranExtraNodeHighIndex(ngn));
      }

      /**
       * Returns a staggared face centered on X cell iterator excluding extra cells
       */
      inline CellIterator getFortranSFCXIterator() const
      {
        return CellIterator(getFortranSFCXLowIndex(),getFortranSFCXHighIndex());
      }

      /**
       * Returns a staggared face centered on Y cell iterator excluding extra cells
       */
      inline CellIterator getFortranSFCYIterator() const
      {
        return CellIterator(getFortranSFCYLowIndex(),getFortranSFCYHighIndex());
      }

      /**
       * Returns a staggared face centered on Z cell iterator excluding extra cells
       */
      inline CellIterator getFortranSFCZIterator() const
      {
        return CellIterator(getFortranSFCZLowIndex(),getFortranSFCZHighIndex());
      }

      /**
       * Returns a staggared face centered on X cell iterator including extra cells
       */
      inline CellIterator getFortranExtraSFCXIterator() const
      {
        return CellIterator(getFortranExtraSFCXLowIndex(),getFortranExtraSFCXHighIndex());
      }

      /**
       * Returns a staggared face centered on Y cell iterator including extra cells
       */
      inline CellIterator getFortranExtraSFCYIterator() const
      {
        return CellIterator(getFortranExtraSFCYLowIndex(),getFortranExtraSFCYHighIndex());
      }

      /**
       * Returns a staggared face centered on Z cell iterator including extra cells
       */
      inline CellIterator getFortranExtraSFCZIterator() const
      {
        return CellIterator(getFortranExtraSFCZLowIndex(),getFortranExtraSFCZHighIndex());
      }

      /**
       * Gets the number of extra cells for the patch class.
       */
      inline IntVector getExtraCells() const 
      {
        return getLevel()->getExtraCells();
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
       * Returns the number of cells excluding extra cells and ghost cells
       */
      inline int getNumCells() const
      {
        return getVolume(getCellLowIndex(),getCellHighIndex());
      }

      /**
       * Returns the number of cells including extra cells
       */
      inline int getNumExtraCells() const
      {
        return getVolume(getExtraCellLowIndex(),getExtraCellHighIndex());
      }

      /**
       * Returns the number of cells excluding extra cells
       */
      inline int getNumNodes() const
      {
        return getVolume(getNodeLowIndex(),getNodeHighIndex());
      }

      /**
       * Returns the number of cells excluding extra cells
       */
      inline int getNumExtraNodes() const
      {
        return getVolume(getExtraNodeLowIndex(),getExtraNodeHighIndex());
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
#if !WARNS_ABOUT_UNREACHABLE_STATEMENTS
            return IntVector(0,0,0);
#endif
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
#if !WARNS_ABOUT_UNREACHABLE_STATEMENTS
            return None;
#endif
        }
      }

      /**
       * sets the vector faces equal to the list of faces that have neighbors
       */
      inline void getNeighborFaces(std::vector<FaceType>& faces) const
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
      inline void getBoundaryFaces(std::vector<FaceType>& faces) const
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
      inline void getCoarseFaces(std::vector<FaceType>& faces) const
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
      void getCornerCells(std::vector<IntVector> & cells, const FaceType &face) const;

      /**
       * Returns the principal axis along a face and
       *the orthognonal axes to that face (right hand rule).
       */
      static inline IntVector getFaceAxes(const FaceType& face) 
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
#if !WARNS_ABOUT_UNREACHABLE_STATEMENTS
            return IntVector(0,0,0);
#endif
        };
      }

      /**
       * Returns a string equivalent of the face name (eg: "xminus")
       */
      static std::string getFaceName(FaceType face);

      /**
       * Sets a pointer to the new grid
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

      inline const LevelP& getLevelP() const 
      {
        return d_grid->getLevel(d_patchState.levelIndex);
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
      
      /**
       * Returns the cell index of the coordinate pos
       */
      inline IntVector getCellIndex(const Point& pos) const 
      {
        return getLevel()->getCellIndex(pos);
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
        IntVector l(getExtraCellLowIndex());
        IntVector h(getExtraCellHighIndex());
        IntVector c=getLevel()->getCellIndex(p);          
        return containsIndex(l,h,c);
      }
      /**
       * Returns true if the point p is contained within the patch
       * excluding extra cells
       */
      inline bool containsPoint(const Point& p) const {
        IntVector l(getCellLowIndex());
        IntVector h(getCellHighIndex());
        IntVector c=getLevel()->getCellIndex(p);          
        return containsIndex(l,h,c);
      }
      //Above for Fracture *************************************************

      static inline bool containsIndex(const IntVector &low, const IntVector &high, const IntVector &cell)
      {
        return  low.x() <= cell.x() && 
                low.y() <= cell.y() &&
                low.z() <= cell.z() &&
                high.x() > cell.x() && 
                high.y() > cell.y() &&
                high.z() > cell.z();

      }
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
        IntVector l(getExtraNodeLowIndex());
        IntVector h(getExtraNodeHighIndex());
        return containsIndex(l,h,idx);
      }

      /**
       * Returns true if the cell idx is owned by this patch
       * including extra cells
       */
      inline bool containsCell(const IntVector& idx) const {
        IntVector l(getExtraCellLowIndex());
        IntVector h(getExtraCellHighIndex());
        return containsIndex(l,h,idx);
      }

      /**
       * Returns true if the SFCX idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCX(const IntVector& idx) const {
        IntVector l(getExtraSFCXLowIndex());
        IntVector h(getExtraSFCXHighIndex());
        return containsIndex(l,h,idx);
      }

      /**
       * Returns true if the SFCY idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCY(const IntVector& idx) const {
        IntVector l(getExtraSFCYLowIndex());
        IntVector h(getExtraSFCYHighIndex());
        return containsIndex(l,h,idx);
      }

      /**
       * Returns true if the SFCZ idx is owned by this patch
       * including extra cells
       */
      inline bool containsSFCZ(const IntVector& idx) const {
        IntVector l(getExtraSFCZLowIndex());
        IntVector h(getExtraSFCZHighIndex());
        return containsIndex(l,h,idx);
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
       * Returns an IntVector with 0 or 1 depending on
       * if there are neighboring patches on the low faces.  
       */
      IntVector neighborsLow() const;
      
      /**
       * Returns an IntVector with 0 or 1 depending on
       * if there are neighboring patches on high low faces.  
       */
      IntVector neighborsHigh() const;
      
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
      std::string toString() const;
     
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
      void computeExtents( VariableBasis basis,
                           const IntVector& boundaryLayer,
                           const IntVector& lowOffset,
                           const IntVector& highOffset,
                           IntVector& low, IntVector& high ) const;

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
      { return getCellLowIndex() - getRealPatch()->getCellLowIndex(); }

      /**
       * Returns the offset between the virtual patch and the real patch 
       * in domain coordinates.
       */
      Vector getVirtualOffsetVector() const
      { return cellPosition(getCellLowIndex()) -
        cellPosition(getRealPatch()->getCellLowIndex()); 
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
                                            const std::string& type,
                                            Iterator& b_ptr,
                                            Iterator& e_ptr,
                                            int child) const ;


      bool haveBC(FaceType face,int mat_id,const std::string& bc_type,
          const std::string& bc_variable) const;

      void initializeBoundaryConditions();

      /*****end boundary condition ****/

      /**
      * Returns a vector of Regions that
      * do not have any overlapping finer level patches.
      * Use the following to iterate over the cells
      * that have no finer level overlapping cells.
      */
     void  getFinestRegionsOnPatch(std::vector<Region>& difference) const;



      /***This section is functions that have yet to be migrated to the new interface.
       * It also includes functions that may be removed from patch in the future*/
      
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

      /*  End the section of unupdated functions */
      /*  Bugged functions that are being kept around until fixed */
      /**
       * Replace with getFortranSFCXLowIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCXFORTLowIndex__Old() const;

      /**
       * Replace with getFortranSFCXHighIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCXFORTHighIndex__Old() const;

      /**
       * Replace with getFortranSFCYLowIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCYFORTLowIndex__Old() const;

      /**
       * Replace with getFortranSFCYHighIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCYFORTHighIndex__Old() const;

      /**
       * Replace with getFortranSFCZLowIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCZFORTLowIndex__Old() const;

      /**
       * Replace with getFortranSFCZHighIndex()
       * This does not line up with the old call and transferring must be donw
       * with care.  The old versions of this function was bugged and faces on 
       * the boundary of two patches were owned by the patch on the plus side of 
       * the face where it should have belonged to the patch on the minus side of 
       * the face.
       */
      IntVector getSFCZFORTHighIndex__Old() const;
      
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
      IntVector d_lowIndex;
      /**
       * The high index of the patch without extra or ghost cells
       */
      IntVector d_highIndex;

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
      
      std::vector<BCDataArray*>* d_arrayBCS;

      /********************
        The following are needed in order to use Patch as a Box in
        Core/Container/SuperBox.h (see
        Core/Grid/Variables/LocallyComputedPatchVarMap.cc)

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
      { return getExtraCellLowIndex(); }

      /**
       * Returns the high index including extra cells
       */
      inline IntVector getHigh() const
      { return getExtraCellHighIndex(); }

      /** 
       * Returns the number of cells in a patch including extraCells.
       * This should only be used by SuperBox code
       */
      inline int getVolume() const
      { return getVolume(getExtraCellLowIndex(), getExtraCellHighIndex()); }

      /** 
       * Returns the number of cells on a patches face including extraCells.
       * This should only be used by SuperBox code.  side specifies the dimension.
       */
      inline int getArea(int side) const
      {
        int area = 1;
        for (int i = 0; i < 3; i++)
          if (i != side)
            area *= getExtraCellHighIndex()[i] - getExtraCellLowIndex()[i];
        return area;
      }

     
   }; // end class Patch
   
   inline Patch::FaceType operator++(Patch::FaceType &face,int)
   {
    return face=static_cast<Patch::FaceType>(face+1);
   }

} // End namespace Uintah

#endif
