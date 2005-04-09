
/*
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <config.h>
#include <Classlib/Persistent.h>
#include <Classlib/String.h>

class Color;
class DrawInfoOpenGL;
class GeomObj;
class OcclusionData;
class Point;
class Vector;
class View;

class Light : public Persistent {
protected:
    Light(const clString& name);
public:
    clString name;
    virtual ~Light();
    virtual void compute_lighting(const View& view, const Point& at,
				  Color&, Vector&)=0;
    virtual GeomObj* geom()=0;
#ifdef SCI_OPENGL
    virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
    virtual void lintens(const OcclusionData& od, const Point& hit_position,
			 Color& light, Vector& light_dir)=0;
    virtual void io(Piostream&);
    friend void Pio(Piostream&, Light*&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Geom_Light_h */
