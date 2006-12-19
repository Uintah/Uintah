//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : GeomColorMappedNrrdTextureObj.h
//    Author : McKay Davis
//    Date   : Tue Oct  3 15:03:56 2006

#ifndef SCI_GEOMCOLORMAPPEDNRRDTEXTUREOBJ_H
#define SCI_GEOMCOLORMAPPEDNRRDTEXTUREOBJ_H

#include <Core/Geom/GeomObj.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <Core/Geom/share.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>

namespace SCIRun {
class SCISHARE GeomColorMappedNrrdTextureObj : public GeomObj {
  ColorMappedNrrdTextureObjHandle cmnto_;
  double alpha_cutoff_;
  double offset_;
public:
  GeomColorMappedNrrdTextureObj() {};
  GeomColorMappedNrrdTextureObj(ColorMappedNrrdTextureObjHandle &);
  GeomColorMappedNrrdTextureObj(const GeomColorMappedNrrdTextureObj&);
  virtual ~GeomColorMappedNrrdTextureObj();
  void set_alpha_cutoff(double alpha);
  void set_offset(double offset);
  virtual GeomObj* clone();
  virtual void get_bounds(BBox&);
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

  
#endif
