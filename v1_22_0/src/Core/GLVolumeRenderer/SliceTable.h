/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef SLICETABLE_H
#define SLICETABLE_H


#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/GLVolumeRenderer/Brick.h>

namespace SCIRun {


/**************************************

CLASS
   SliceTable
   
   SliceTable Class for 3D Texturing 

GENERAL INFORMATION

   SliceTable.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SliceTable

DESCRIPTION
   SliceTable class for 3D Texturing.  A slice table is a global virtual
   table that determines slice distances when volume rendering in a multi-
   resolution volume renderer.  Given the size of the entire volume, a 
   view point and the number of slices per brick calculate the min and 
   ray parameters and the ray step dt.

  
WARNING
  
****************************************/


class SliceTable 
{
public:

  // GROUP: Constructors:
  //////////
  // Consructor
  SliceTable( const Point& min,  const Point& max, const  Ray& view,
	      int slices, int treedepth);

  // GROUP: Destructors
  //////////
  // Destructor
  ~SliceTable();

  // GROUP: Computation
  //////////
  void getParameters(const Brick&,double& tmin,
		     double& tmax, double& dt) const;

private:
  Ray view;
  int slices;
  int order[8];
  int treedepth;
  double minT, maxT, DT;
};


} // End namespace SCIRun
#endif
