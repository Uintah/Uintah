
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
    void pre_mulmat(double[4][4]);
    void post_mulmat(double[4][4]);
    void invmat(double[4][4]);
    void switch_rows(double m[4][4], int row1, int row2) const;
    void sub_rows(double m[4][4], int row1, int row2, double mul) const;
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
    void pre_translate(const Vector&);
    void post_translate(const Vector&);
    Point unproject(const Point& p);
    Point project(const Point& p);
    Vector project(const Vector& p);
    void get(double*);
    void set(double*);
    void load_identity();
    void perspective(const Point& eyep, const Point& lookat,
		     const Vector& up, double fov,
		     double znear, double zfar,
		     int xres, int yres);
    void invert();
};

#endif

