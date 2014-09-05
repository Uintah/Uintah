
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
#include <SCICore/Geom/GeomOpenGL.h>
#endif
#include <SCICore/Geom/GeomObj.h>


namespace SCICore {
  namespace GeomSpace {
    
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
      virtual bool saveobj(std::ostream&, const clString& format, GeomSave*);
    };
    
  } // End namespace GeomSpace
} // End namespace SCICore

#endif
