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
#include <Packages/Kurt/Core/Geom/GridBrick.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <stdlib.h>
#include <iostream>
#include <string>
using std::cerr;
using std::endl;
using std::string;
using std::cin;
using std::vector;


namespace Kurt {

using SCIRun::Brick;
using SCIRun::intersectParam;
using SCIRun::Dot;



GridBrick::GridBrick() :
  i_min_(IntVector(0,0,0)), i_max_(IntVector(0,0,0))
{
}

GridBrick::GridBrick(const Point& min, const Point& max,
		     const IntVector& imin, const IntVector& imax,
		     int padx, int pady, int padz,
		     int level,
		     Array3<unsigned char>* tex) :
  Brick(min,max,padx,pady,padz,level,tex),
  i_min_(imin), i_max_(imax)
{
}

}
