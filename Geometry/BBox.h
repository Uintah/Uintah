
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
    void extend(const BBox& b);
    void extend_cyl(const Point& cen, const Vector& normal, double r);
    Point center();
    double longest_edge();
    void translate(const Vector &v);
    void scale(double s, const Vector &o);
    Point min() const;
    Point max() const;
    Vector diagonal() const;
    Point corner(int);
};

#endif
