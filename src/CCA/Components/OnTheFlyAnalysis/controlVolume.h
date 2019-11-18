/*
 * The MIT License
 *
 * Copyright (c) 1997-2019 The University of Utah
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


#ifndef CCA_Components_ontheflyAnalysis_controlVolume_h
#define CCA_Components_ontheflyAnalysis_controlVolume_h

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Patch.h>
#include <string>

namespace Uintah {

  class CellIterator;
  class Box;

  
  class controlVolume {
  public:
  
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


    enum FaceIteratorType {
      MinusEdgeCells,                 //Excludes the edge/corner cells.
      InteriorFaceCells               //Includes cells on the interior of the face
    };
    
    //______________________________________________________________________
    // Returns a cell iterator over control volume cells on this patch
    inline CellIterator getCellIterator(const Patch* patch) const ;
    
    
    //______________________________________________________________________
    // Returns a face cell iterator
    //  domain specifies which type of iterator will be returned
    //     MinusEdgeCells:         All cells on the face excluding the edge cells.
    //     InteriorFaceCells:      All cells on the interior of the face.
    
    CellIterator getFaceIterator(const controlVolume::FaceType& face, 
                                 const FaceIteratorType& domain,
                                 const Patch* patch) const;

    //______________________________________________________________________
    // Returns the normal to the patch face
    IntVector getFaceDirection( const controlVolume::FaceType & face ) const;
    
    //______________________________________________________________________
    // Sets the vector faces equal to the list of faces that are on the boundary
    inline void getBoundaryFaces(std::vector<FaceType>& faces,
                                 const Patch* patch) const;

    //______________________________________________________________________
    // Returns true if a control volume boundary face exists on this patch
    bool inline hasBoundaryFaces( const Patch* patch) const {

      bool test = doesIntersect( d_lowIndex, d_highIndex, patch->getCellLowIndex(), patch->getCellHighIndex() );
      return test;
    }
    
    //______________________________________________________________________
    //
    std::string toString(const Patch* patch) const;

    //______________________________________________________________________
    // Returns the principal axis along a face and
    // the orthognonal axes to that face (right hand rule).
    IntVector getFaceAxes( const controlVolume::FaceType & face ) const;
    
    //______________________________________________________________________
    // Returns a string equivalent of the face name (eg: "xminus")
    std::string getFaceName( controlVolume::FaceType face ) const;;
    
    //______________________________________________________________________
    // Returns the cell area dx*dy.
    double cellArea( const controlVolume::FaceType face,
                     const Patch* patch ) const;
  

  protected:
    controlVolume( const IntVector & lowIndex,
                   const IntVector & highIndex);
    ~controlVolume();

    IntVector d_lowIndex;
    IntVector d_highIndex;
};
}  // end namespace Uintah
#endif
