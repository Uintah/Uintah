/* glprintf.h */

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

#ifndef SCIRun_Core_2d_glprintf_h
#define SCIRun_Core_2d_glprintf_h 1

#ifdef __cplusplus
extern "C" {
#endif

  //! The upper left (or upper right) corner of 
  // the first (or last) character of the text 
  // to be printed when aligned GL_LEFT (or GL_RIGHT).
  // pos must be a 3D point, i.e. an array with
  // 3 locations starting at zero.
  void glTextAnchor(double* pos);

  //! The normal to the plane in which the text will
  // be printed
  // norm must be a 3D vector, i.e. an array with
  // 3 locations starting at zero.
  void glTextNormal(double* norm);

  //! The up orientation for the text to be printed
  // pos must be a 3D vector, i.e. an array with
  // 3 locations starting at zero.
  // Note that the rendered text may appear skewed
  // if the ratio of the projection dimensions is 
  // not identical to the ratio of the viewport
  // dimensions.  Compensate by making up non-unit 
  // length normal (multiply by ratio difference)
  void glTextUp(double* up);

  //! The width and height of a single character of 
  // the text to be printed.  The size is used for 
  // all characters in the text
  // width and height should be proportional to the
  // width and height of the projection (ortho or 
  // or perspective) at the time glprintf() is called
  void glTextSize(double width, double height);

  //! describes whether text is anchored at the
  // upper left or upper right and rendered
  // left-to-right or right-to-left (respectively)
  // must be GL_LEFT or GL_RIGHT
  void glTextAlign(int align);

  //! identical to the standard printf()
  int  glprintf(const char* format, ...);

  //! must be called (at least once) for each render
  // context before it intends to use glprintf()  
  void init_glprintf();
  
#ifdef __cplusplus
}
#endif

#endif

