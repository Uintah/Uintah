
#ifndef Geometry_Transform_h
#define Geometry_Transform_h 1

class Vector;
class Point;

class Transform {
    double mat[4][4];
    double imat[4][4];
    int inverse_valid;
    void install_mat(double[4][4]);
    void compute_imat();
    void mulmat(double[4][4]);
public:
    Transform();
    Transform(const Transform&);
    Transform& operator=(const Transform&);
    ~Transform();

    void scale(const Vector&);
    void rotate(double, const Vector& axis);
    void translate(const Vector&);
    Point unproject(const Point& p);
    Point project(const Point& p);
    Vector project(const Vector& p);
    void get(double*);
    void set(double*);
    void load_identity();
    void lookat(const Point&, const Point&, const Vector&);
};

class PTransform : public Transform {
public:
    PTransform();
    PTransform(const PTransform&);
    PTransform& operator=(const PTransform&);
    ~PTransform();
    void perspective(double, double, double, double);
    void ortho(const Point&, const Point&);
};

#endif

