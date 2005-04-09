
/*
 *  HeadLight.h:  A Point light source
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_HeadLight_h
#define SCI_Geom_HeadLight_h 1

#include <Geom/Light.h>
#include <Geom/Color.h>
#include <Geometry/Point.h>

class HeadLight : public Light {
    Color c;
public:
    HeadLight(const clString& name, const Color&);
    virtual ~HeadLight();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&);
    virtual GeomObj* geom();
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx);
#endif
    virtual void lintens(const OcclusionData& od, const Point& hit_position,
			 Color& light, Vector& light_dir);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Geom_HeadLight_h */

