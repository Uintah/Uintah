
#include "Transform.h"
#include "Point.h"
#include "Vector.h"
#include "Ray.h"
#include <iostream>

#include <stdio.h>

using namespace rtrt;

Transform::Transform()
{
    load_identity();
    inverse_valid=0;
}

Transform::Transform(const Transform& copy)
{
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    mat[i][j]=copy.mat[i][j];
	    imat[i][j]=copy.imat[i][j];
	}
    }
    inverse_valid=copy.inverse_valid;
}

Transform::~Transform()
{
}

void Transform::load_frame(const Point&,
			   const Vector& x, 
			   const Vector& y, 
			   const Vector& z)
{
    mat[3][3] = imat[3][3] = 1.0;
    mat[0][3] = mat[1][3] = mat[2][3] = 0.0; // no perspective
    imat[0][3] = imat[1][3] = imat[2][3] = 0.0; // no perspective

    mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
    imat[3][0] = imat[3][1] = imat[3][2] = 0.0;

    mat[0][0] = x.x();
    mat[1][0] = x.y();
    mat[2][0] = x.z();

    mat[0][1] = y.x();
    mat[1][1] = y.y();
    mat[2][1] = y.z();

    mat[0][2] = z.x();
    mat[1][2] = z.y();
    mat[2][2] = z.z();

    imat[0][0] = x.x();
    imat[0][1] = x.y();
    imat[0][2] = x.z();

    imat[1][0] = y.x();
    imat[1][1] = y.y();
    imat[1][2] = y.z();

    imat[2][0] = z.x();
    imat[2][1] = z.y();
    imat[2][2] = z.z();

    inverse_valid = 1;
}

void Transform::change_basis(Transform& T)
{
    pre_mulmat(T.imat);
    post_mulmat(T.mat);
}

void Transform::post_trans(Transform& T)
{
    post_mulmat(T.mat);
}

void Transform::print(void)
{
    for(int i=0;i<4;i++) {
	for(int j=0;j<4;j++)
	    printf("%f ",mat[i][j]); 
	printf("\n");
    }
    printf("\n");
	
}

void Transform::printi(void)
{
    for(int i=0;i<4;i++) {
	for(int j=0;j<4;j++)
	    printf("%f ",imat[i][j]); 
	printf("\n");
    }
    printf("\n");
	
}

void Transform::build_scale(double m[4][4], const Vector& v)
{
    load_identity(m);
    m[0][0]=v.x();
    m[1][1]=v.y();
    m[2][2]=v.z();
}
    
void Transform::pre_scale(const Vector& v)
{
    double m[4][4];
    build_scale(m,v);
    pre_mulmat(m);
    inverse_valid=0;
}

void Transform::post_scale(const Vector& v)
{
    double m[4][4];
    build_scale(m,v);
    post_mulmat(m);
    inverse_valid=0;
}

void Transform::build_translate(double m[4][4], const Vector& v)
{
    load_identity(m);
    m[0][3]=v.x();
    m[1][3]=v.y();
    m[2][3]=v.z();
}

void Transform::pre_translate(const Vector& v)
{
    double m[4][4];
    build_translate(m,v);
    pre_mulmat(m);
    inverse_valid=0;
}

void Transform::post_translate(const Vector& v)
{
    double m[4][4];
    build_translate(m,v);    
    post_mulmat(m);
    inverse_valid=0;
}

void Transform::build_rotate(double m[4][4], double angle, const Vector& axis)
{
    // From Foley and Van Dam, Pg 227
    // NOTE: Element 0,1 is wrong in the text!
    double sintheta=sin(angle);
    double costheta=cos(angle);
    double ux=axis.x();
    double uy=axis.y();
    double uz=axis.z();
    m[0][0]=ux*ux+costheta*(1-ux*ux);
    m[0][1]=ux*uy*(1-costheta)-uz*sintheta;
    m[0][2]=uz*ux*(1-costheta)+uy*sintheta;
    m[0][3]=0;

    m[1][0]=ux*uy*(1-costheta)+uz*sintheta;
    m[1][1]=uy*uy+costheta*(1-uy*uy);
    m[1][2]=uy*uz*(1-costheta)-ux*sintheta;
    m[1][3]=0;

    m[2][0]=uz*ux*(1-costheta)-uy*sintheta;
    m[2][1]=uy*uz*(1-costheta)+ux*sintheta;
    m[2][2]=uz*uz+costheta*(1-uz*uz);
    m[2][3]=0;

    m[3][0]=0;
    m[3][1]=0;
    m[3][2]=0;
    m[3][3]=1;
}

void Transform::pre_rotate(double angle, const Vector& axis)
{
    double m[4][4];
    build_rotate(m, angle, axis);
    pre_mulmat(m);
    inverse_valid=0;
}	

void Transform::post_rotate(double angle, const Vector& axis)
{
    double m[4][4];
    build_rotate(m, angle, axis);
    post_mulmat(m);
    inverse_valid=0;
}	

void Transform::rotate(const Vector& from, const Vector& to)
{
        Vector axis(from.cross(to));
        if(axis.length2() < 0.00001)return; // Don't bother
        double sintheta=axis.normalize();
        if(sintheta >= 1.0){
            pre_rotate(M_PI/2, axis);
        } else if(sintheta <= -1.0){
            pre_rotate(-M_PI/2, axis);
        } else {
            double theta=asin(sintheta);
            pre_rotate(theta, axis);
        }
}
void Transform::get(double* gmat)
{
    double* p=gmat;
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    *p++=mat[i][j];
	}
    }
}

// GL stores its matrices column-major.  Need to take the transpose...
void Transform::get_trans(double* gmat)
{
    double* p=gmat;
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    *p++=mat[j][i];
	}
    }
}

void Transform::set(double* pmat)
{
    double* p=pmat;
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    mat[i][j]= *p++;
	}
    }
    inverse_valid=0;
}

void Transform::load_identity()
{
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    mat[i][j]=0;
	}
	mat[i][i]=1.0;
    }
    inverse_valid=0;
}

void Transform::install_mat(double m[4][4])
{
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    mat[i][j]=m[i][j];
	}
    }
}

void Transform::load_identity(double m[4][4]) 
{
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    m[i][j]=0;
	}
	m[i][i]=1.0;
    }
}
    
void Transform::compute_imat()
{
    cerr << "Transform::compute_imat not finished\n";
}

void Transform::post_mulmat(double mmat[4][4])
{
    double newmat[4][4];
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    newmat[i][j]=0.0;
	    for(int k=0;k<4;k++){
		newmat[i][j]+=mat[i][k]*mmat[k][j];
	    }
	}
    }
    install_mat(newmat);
}

void Transform::pre_mulmat(double mmat[4][4])
{
    double newmat[4][4];
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    newmat[i][j]=0.0;
	    for(int k=0;k<4;k++){
		newmat[i][j]+=mmat[i][k]*mat[k][j];
	    }
	}
    }
    install_mat(newmat);
}

Transform& Transform::operator=(const Transform& copy)
{
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
            mat[i][j]=copy.mat[i][j];
    inverse_valid=0;
    return *this;
}

Vector Transform::project(const Vector& p)
{
    double x=mat[0][0]*p.x()+mat[0][1]*p.y()+mat[0][2]*p.z();
    double y=mat[1][0]*p.x()+mat[1][1]*p.y()+mat[1][2]*p.z();
    double z=mat[2][0]*p.x()+mat[2][1]*p.y()+mat[2][2]*p.z();
    return Vector(x, y, z);
}

Point Transform::project(const Point& p)
{
    double x=mat[0][0]*p.x()+mat[0][1]*p.y()+mat[0][2]*p.z()+mat[0][3];
    double y=mat[1][0]*p.x()+mat[1][1]*p.y()+mat[1][2]*p.z()+mat[1][3];
    double z=mat[2][0]*p.x()+mat[2][1]*p.y()+mat[2][2]*p.z()+mat[2][3];
    double w=mat[3][0]*p.x()+mat[3][1]*p.y()+mat[3][2]*p.z()+mat[3][3];
    return Point(x/w, y/w, z/w);
}

Ray Transform::xray(const Ray& r, double& dist_scale)
{
    Vector v(project(r.direction()));
    dist_scale=v.normalize();
    return Ray(project(r.origin()), v);
}

Vector Transform::project_normal(const Vector& p)
{
    double x=mat[0][0]*p.x()+mat[0][1]*p.x()+mat[0][2]*p.x()+mat[0][3];
    double y=mat[1][0]*p.y()+mat[1][1]*p.y()+mat[1][2]*p.y()+mat[1][3];
    double z=mat[2][0]*p.z()+mat[2][1]*p.z()+mat[2][2]*p.z()+mat[2][3];
    return Vector(x, y, z);
}
