/* glprintf.h */

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

