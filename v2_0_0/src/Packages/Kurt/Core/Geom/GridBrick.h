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

#ifndef My_GRIDBRICK_H
#define My_GRIDBRICK_H


#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Containers/Array3.h>
#include <Core/GLVolumeRenderer/Brick.h>
#include <Core/GLVolumeRenderer/VolRenState.h>

namespace Kurt {

using SCIRun::Point;
using SCIRun::Ray;
using SCIRun::IntVector;
using SCIRun::Array3;
using SCIRun::Brick;

/**************************************

CLASS
   GridBrick
   
   GridBrick Class for 3D Texturing 

GENERAL INFORMATION

   GridBrick.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2001 SCI Group

KEYWORDS
   GridBrick

DESCRIPTION
   GridBrick class for 3D Texturing.  Stores the texture associated with
   the GridBrick and the bricks location is space.  For a given view ray,
   min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/

class GridBrick : public Brick
 {
public:
   friend class VolRen;
  // GROUP: Constructors:
  //////////
  // Constructor
  GridBrick(const Point& min, const Point& max,
	    const IntVector& imin, const IntVector& imax,
	    int padx, int pady, int padz,int level,
	    Array3<unsigned char>* tex);
  GridBrick();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~GridBrick(){}

  IntVector& get_min_index(){ return i_min_; }
  IntVector& get_max_index(){ return i_max_; }
  void get_index_range(IntVector& min, IntVector& max)
    { min = i_min_; max = i_max_; }

protected:

  IntVector i_min_;
  IntVector i_max_;

};

} //end namespace Kurt


#endif
