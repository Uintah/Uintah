
#ifndef Geometry_Transform_h
#define Geometry_Transform_h 1

namespace rtrt {

class Vector;
class Point;
class Ray;

class Transform {
    double mat[4][4];
    double imat[4][4];
    int inverse_valid;
    void install_mat(double[4][4]);
    void compute_imat();
    void build_rotate(double m[4][4], double, const Vector&);
    void build_scale(double m[4][4], const Vector&);
    void build_translate(double m[4][4], const Vector&);
    void pre_mulmat(double[4][4]);
    void post_mulmat(double[4][4]);
    void invmat(double[4][4]);
    void switch_rows(double m[4][4], int row1, int row2) const;
    void sub_rows(double m[4][4], int row1, int row2, double mul) const;
    void load_identity(double[4][4]);
public:
    Transform();
    Transform(const Transform&);
    Transform& operator=(const Transform&);
    ~Transform();
    Transform(const Point&, const Vector&, const Vector&, const Vector&);

    void load_frame(const Point&,const Vector&, const Vector&, const Vector&);
    void change_basis(Transform&);

    void post_trans(Transform&);
    void print(void);
    void printi(void);

    void pre_scale(const Vector&);
    void post_scale(const Vector&);
    void pre_rotate(double, const Vector& axis);
    void post_rotate(double, const Vector& axis);
    void rotate(const Vector& from, const Vector& to);
    void pre_translate(const Vector&);
    void post_translate(const Vector&);
    Point unproject(const Point& p);
    Point project(const Point& p);
    Vector project(const Vector& p);
    void get(double*);
    void get_trans(double*);
    void set(double*);
    void load_identity();
    void perspective(const Point& eyep, const Point& lookat,
		     const Vector& up, double fov,
		     double znear, double zfar,
		     int xres, int yres);
    void invert();

    Ray xray(const Ray& ray, double& dist_scale);
    Vector project_normal(const Vector&);
};

} // end namespace rtrt

#endif

