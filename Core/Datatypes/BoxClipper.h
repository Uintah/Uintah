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

//    File   : BoxClipper.h
//    Author : Michael Callahan
//    Date   : September 2001

#if !defined(BoxClipper_h)
#define BoxClipper_h

#include <Core/Datatypes/Clipper.h>
#include <Core/Geometry/Transform.h>

namespace SCIRun {


class BoxClipper : public Clipper
{
  Transform trans_;

public:

  //BoxClipper(ScaledBoxWidget *box);
  BoxClipper(Transform &t);
  BoxClipper(const BoxClipper &bc);
  virtual ~BoxClipper();

  virtual bool inside_p(const Point &p);

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
};


} // end namespace SCIRun

#endif // BoxClipper_h
