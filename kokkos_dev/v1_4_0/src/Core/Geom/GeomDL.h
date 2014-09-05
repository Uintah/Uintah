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
 *  GeomDL.h: ?
 *
 *  Written by:
 *   Author Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Date July 2000
 *
 *  Copyright (C) 2000  SCI Group
 */

#ifndef SCI_Geom_GeomDL_h 
#define SCI_Geom_GeomDL_h 1

#ifdef SCI_OPENGL
#include <Core/Geom/GeomOpenGL.h>
#endif
#include <Core/Geom/GeomObj.h>


namespace SCIRun {
    
    class SCICORESHARE GeomDL : public GeomObj {
      GeomObj* child;
      int polygons;
      unsigned int dl;
      bool have_dl;
      int type;
      int lighting;
      
    public:
      GeomDL(GeomObj*);
      virtual ~GeomDL();
      
      virtual GeomObj* clone();
      virtual void get_bounds(BBox&);
      
#ifdef SCI_OPENGL
      virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
      virtual void get_triangles(Array1<float> &);
      virtual void io(Piostream&);
      static PersistentTypeID type_id;	
      virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
    };
    
} // End namespace SCIRun

#endif
