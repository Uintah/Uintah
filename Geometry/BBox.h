
#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <Geometry/Point.h>
class Vector;
class Piostream;
class BBox {
protected:
friend void Pio(Piostream &, BBox& );
    int have_some;
    Point cmin;
    Point cmax;
public:
    BBox();
    ~BBox();
    BBox(const BBox&);
    inline int valid(){return have_some;}
    void reset();
    void extend(const Point& p);
    void extend(const Point& p, double radius);
    void extend(BBox b);
    Point center();
    double longest_edge();
    void translate(const Vector &v);
    void scale(double s, const Vector &o);
    Point min();
    Point max();
    Vector diagonal();
    Point corner(int);
};

#endif
