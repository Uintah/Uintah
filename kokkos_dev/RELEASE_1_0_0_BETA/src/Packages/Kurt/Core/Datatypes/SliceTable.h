#ifndef SLICETABLE_H
#define SLICETABLE_H


#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include "Brick.h"

namespace Kurt {

using namespace SCIRun;
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
	      int slices, int treedepth );

  // GROUP: Destructors
  //////////
  // Destructor
  ~SliceTable();

  // GROUP: Computation
  //////////
  //
  void getParameters(const Brick&,double& tmin,
		     double& tmax, double& dt) const;

private:
  Ray view;
  int slices;
  int treedepth;
  int order[8];
  double minT, maxT, DT;
};

} // End namespace Kurt

#endif
