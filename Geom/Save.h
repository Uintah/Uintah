
#ifndef GeomSave_h
#define GeomSave_h 1

class ostream;

struct GeomSave {
    int nindent;
    void indent();
    void unindent();
    void indent(ostream&);
    void start_sep(ostream&);
    void end_sep(ostream&);
    void start_tsep(ostream&);
    void end_tsep(ostream&);
    void translate(ostream&, const Point& p);
    void rotateup(ostream&, const Vector& up, const Vector& new_up);
    void start_node(ostream&, char*);
    void end_node(ostream&);

    void orient(ostream&, const Point& center, const Vector& up,
		const Vector& new_up=Vector(0,1,0));
};

#endif
