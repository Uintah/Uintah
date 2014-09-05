/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
