/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef UINTAH_GRID_BoundCondReader_H
#define UINTAH_GRID_BoundCondReader_H

#include <map>
#include <string>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

/*!
        
  \class BoundCondReader
        
  \brief Reads in the boundary conditions and stores them for later processing
  by Patch.cc.
        
  Reads in the boundary conditions for a given face and a given material.
  Multiple material bc specifications may be combined within a single 
  \<Face\> \</Face\>.

  The boundary condition specification consist of two components, a 
  geometrical component and a type-value pair component.  The geometrical 
  component describes where on a patch's face the boundary condition is
  applied.  The side specification will apply the bc's type-value pair to
  the entire face.  The circle specification will apply the bc to a circular
  region on a face.  The rectangle specification will apply the bc to a
  rectangular region on a face.  There may be multiple specifications of 
  circles and rectangles on a give face.  There must be at least one side
  specification for each face.  The type-value pair component are described 
  in the BoundCondBase class.

  A significant component of the BoundCondReader class is to take the 
  geometrical specification and arrange the various instances of side,
  circle, and rectangle in such a manner as to ensure that the entire face
  has an bc value specified.  To accomplish this, auxillary classes 
  UnionBCData, and DifferenceBCData are used to manage the union of multiple
  instances of rectangle and circle bcs for a given side and the difference
  of that collection with the base side bc.  Unions and difference bcs are 
  not available to the user.

  Currently, this class supports the old way of specifying the geometric
  component and the new way.  The old way only allowed a side specification.
  The new way allows for circles, rectangles and a side specification 
  for a given face.  At some point, the old way infrastructure will be go 
  away.


  \author John A. Schmidt \n
  Department of Mechanical Engineering \n
  University of Utah \n
  Center for the Simulation of Accidental Fires and Explosions (C-SAFE) \n\n

*/

  class BoundCondReader  {
  public:

    /// Constructor
    BoundCondReader();

    /// Destructor
    ~BoundCondReader();

    void whichPatchFace(const std::string fc, Patch::FaceType& face_side,
                        int& plusMinusFaces, int& p_dir);

    /// Read in the boundary conditions given a problem specification, ps.
    /// Each face is read in and processed using the function 
    /// createBoundaryConditionFace() which indicates whether the boundary
    /// condition should be applied on the side, circle, or rectangle region. 
    /// Then the individual boundary conditions such as Pressure, Density,
    /// etc. are processed.  The individual boundary conditions for a given
    /// face may be for several different materials.   Boundary conditions
    /// are then separated out by material ids.  Following separation, the
    /// boundary conditions are then combined (combineBCS()) so that any 
    /// circles and rectangles specified on a face are combined into a union.  
    /// The union is the subtracted out from the side case using a difference
    /// boundary condition.
    /// 
    /// 

    void read(ProblemSpecP& ps,const ProblemSpecP& grid_ps );

    /// Read in the geometric tags: side, circle, and rectangle.  Performs
    /// error checking if the tag is not present or if the circle and rectangle
    /// tags are not specified correctly.
    BCGeomBase* createBoundaryConditionFace(ProblemSpecP& ps,
                                            const ProblemSpecP& grid_ps,
                                            Patch::FaceType& face_side);


    /// Combine the boundary conditions for a given face into union and 
    /// difference operations for the face.  Multiple circles and rectangles
    /// are stored in a union.  The resultant union is then subtracted from
    /// the side and stored as a difference bc.  This operation only happens
    /// if there are more than one bc specified for a given face.  
    void combineBCS();

    void combineBCS_NEW();
    
    void bulletProofing();

    ///
    bool compareBCData(BCGeomBase* b1, BCGeomBase* b2);

    /// not used
    const BCDataArray getBCDataArray(Patch::FaceType& face) const;

   private:
    friend class Level;
    friend class Patch;
    std::map<Patch::FaceType,BCDataArray > d_BCReaderData;
  };

  void print(BCGeomBase* p);

} // End namespace Uintah

#endif


