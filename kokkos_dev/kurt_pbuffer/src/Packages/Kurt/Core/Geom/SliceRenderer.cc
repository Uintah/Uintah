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

#include <Packages/Kurt/Core/Geom/SliceRenderer.h>
#include <Core/Util/NotFinished.h>
// #include <Core/Geometry/Point.h>
// #include <Core/Datatypes/Field.h>

namespace Kurt {

// using SCIRun::FieldHandle;
// using SCIRun::Point;



SliceRenderer::SliceRenderer() 
{
  NOT_FINISHED("SliceRenderer::SliceRenderer(int id)");
}


SliceRenderer::SliceRenderer( GridVolRen* gvr,
			       FieldHandle tex,
			       ColorMapHandle map,
			       bool fixed,
			       double min, double max)
  : VolumeRenderer(gvr, tex, map, fixed, min, max)
{ lighting_ = 1;}

SliceRenderer::SliceRenderer(const SliceRenderer& copy)
  : VolumeRenderer( copy )
{
} 

SliceRenderer::~SliceRenderer()
{
}

} // End namespace Uintah
