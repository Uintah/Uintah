
#ifndef GeomSave_h
#define GeomSave_h 1

#include <Geometry/Vector.h>

class ostream;

struct GeomSave {
    int nindent;
    void indent();
    void unindent();
    void indent(ostream&);

    // For VRML.
    void start_sep(ostream&);
    void end_sep(ostream&);
    void start_tsep(ostream&);
    void end_tsep(ostream&);
    void orient(ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    // For RIB.
    void start_attr(ostream&);
    void end_attr(ostream&);
    void start_trn(ostream&);
    void end_trn(ostream&);
    void rib_orient(ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));

    void translate(ostream&, const Point& p);
    void rotateup(ostream&, const Vector& up, const Vector& new_up);
    void start_node(ostream&, char*);
    void end_node(ostream&);
};

#endif
