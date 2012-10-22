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


#ifndef __INTERSECTION_GEOMETRY_OBJECT_H__
#define __INTERSECTION_GEOMETRY_OBJECT_H__      

#include <Core/GeometryPiece/GeometryPiece.h>
#include <vector>

namespace Uintah {


/**************************************
        
CLASS
   IntersectionGeometryPiece
        
   Creates the intersection of geometry pieces from the xml input 
   file description. 

        
GENERAL INFORMATION
        
   IntersectionGeometryPiece.h
        
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
        
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
        
 
        
KEYWORDS
   IntersectionGeometryPiece
        
DESCRIPTION
   Creates a intersection of different geometry pieces from the xml input 
   file description.
   Requires multiple inputs: specify multiple geometry pieces.  
   There are methods for checking if a point is inside the intersection of 
   pieces and also for determining the bounding box for the collection.
   The input form looks like this:
       <intersection>
         <box>
           <min>[0.,0.,0.]</min>
           <max>[1.,1.,1.]</max>
         </box>
         <sphere>
           <origin>[.5,.5,.5]</origin>
           <radius>1.5</radius>
         </sphere>
       </intersection>
        
WARNING
        
****************************************/

      class IntersectionGeometryPiece : public GeometryPiece {
         
      public:
         //////////
         // Constructor that takes a ProblemSpecP argument.   It reads the xml 
         // input specification and builds the intersection of geometry pieces.
         IntersectionGeometryPiece(ProblemSpecP &);
         
         /// Copy constructor
         IntersectionGeometryPiece(const IntersectionGeometryPiece&);

         /// Assignment operator
         IntersectionGeometryPiece& operator=(const IntersectionGeometryPiece& );
         //////////
         // Destructor
         virtual ~IntersectionGeometryPiece();

         static const string TYPE_NAME;
         virtual std::string getType() const { return TYPE_NAME; }

         /// Make a clone
         GeometryPieceP clone() const;
         
         //////////
         // Determines whether a point is inside the intersection piece.  
         virtual bool inside(const Point &p) const;
         
         //////////
         // Returns the bounding box surrounding the intersection piece.
         virtual Box getBoundingBox() const;
         
      private:
         virtual void outputHelper( ProblemSpecP & ps) const;

         std::vector<GeometryPieceP> child_;
      };

} // End namespace Uintah
      
#endif // __INTERSECTION_GEOMETRY_PIECE_H__
