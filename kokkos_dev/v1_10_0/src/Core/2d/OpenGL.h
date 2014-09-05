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

  
/*
 *  OpenGL.h: Displayable Geometry
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef GeomOpenGL_h
#define GeomOpenGL_h 

#ifdef _WIN32
#define WINGDIAPI __declspec(dllimport)
#define APIENTRY __stdcall
#define CALLBACK APIENTRY
#endif

#include <stddef.h>
#include <stdlib.h>
#include <sci_glu.h>
#include <GL/glx.h>

#include <sci_config.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {


} // End namespace SCIRun


#endif /* OpenGL_h  */

