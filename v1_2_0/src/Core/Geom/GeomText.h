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
 *  GeomText.h:  Texts of GeomObj's
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 1998
 *
 *  Copyright (C) 1998 SCI Text
 */

#ifndef SCI_Geom_Text_h
#define SCI_Geom_Text_h 1

#include <Core/Geom/GeomObj.h>
#ifdef _WIN32
#define WINGDIAPI __declspec(dllimport)
#define APIENTRY __stdcall
#define CALLBACK APIENTRY
#endif
#include <GL/gl.h>

namespace SCIRun {

class SCICORESHARE GeomText : public GeomObj {
  static int init;
  static GLuint fontbase;
public:
  string text;
  Point at;
  Color c;
public:
  GeomText();
  GeomText( const string &, const Point &, const Color &c = Color(1,1,1));
  GeomText(const GeomText&);
  virtual ~GeomText();
  virtual GeomObj* clone();

  virtual void reset_bbox();
  virtual void get_bounds(BBox&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
    virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};

} // End namespace SCIRun


#endif /* SCI_Geom_Text_h */

