
#include <Geometry/Transform.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>
#include <iostream.h>

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


void Transform::scale(const Vector& v)
{
    for(int i=0;i<4;i++){
	mat[0][i]*=v.x();
	mat[1][i]*=v.y();
	mat[2][i]*=v.z();
    }
    inverse_valid=0;
}

void Transform::translate(const Vector& v)
{
    double dmat[4][4];
    dmat[0][0]=mat[0][0]+v.x()*mat[3][0];
    dmat[0][1]=mat[0][1]+v.x()*mat[3][1];
    dmat[0][2]=mat[0][2]+v.x()*mat[3][2];
    dmat[0][3]=mat[0][3]+v.x()*mat[3][3];
    dmat[1][0]=mat[1][0]+v.y()*mat[3][0];
    dmat[1][1]=mat[1][1]+v.y()*mat[3][1];
    dmat[1][2]=mat[1][2]+v.y()*mat[3][2];
    dmat[1][3]=mat[1][3]+v.y()*mat[3][3];
    dmat[2][0]=mat[2][0]+v.z()*mat[3][0];
    dmat[2][1]=mat[2][1]+v.z()*mat[3][1];
    dmat[2][2]=mat[2][2]+v.z()*mat[3][2];
    dmat[2][3]=mat[2][3]+v.z()*mat[3][3];
    dmat[3][0]=mat[3][0];
    dmat[3][1]=mat[3][1];
    dmat[3][2]=mat[3][2];
    dmat[3][3]=mat[3][3];
    install_mat(dmat);
    inverse_valid=0;
}

void Transform::rotate(double angle, const Vector& axis)
{
    // From Foley and Van Dam, Pg 227
    // NOTE: Element 0,1 is wrong in the text!
    double sintheta=Sin(angle);
    double costheta=Cos(angle);
    double ux=axis.x();
    double uy=axis.y();
    double uz=axis.z();
    double newmat[4][4];
    newmat[0][0]=ux*ux+costheta*(1-ux*ux);
    newmat[0][1]=ux*uy*(1-costheta)-uz*sintheta;
    newmat[0][2]=uz*ux*(1-costheta)+uy*sintheta;
    newmat[0][3]=0;

    newmat[1][0]=ux*uy*(1-costheta)+uz*sintheta;
    newmat[1][1]=uy*uy+costheta*(1-uy*uy);
    newmat[1][2]=uy*uz*(1-costheta)-ux*sintheta;
    newmat[1][3]=0;

    newmat[2][0]=uz*ux*(1-costheta)-uy*sintheta;
    newmat[2][1]=uy*uz*(1-costheta)+ux*sintheta;
    newmat[2][2]=uz*uz+costheta*(1-uz*uz);
    newmat[2][3]=0;

    newmat[3][0]=0;
    newmat[3][1]=0;
    newmat[3][2]=0;
    newmat[3][3]=1;

    mulmat(newmat);
    inverse_valid=0;
}	

Point Transform::project(const Point& p)
{
    return Point(mat[0][0]*p.x()+mat[0][1]*p.y()+mat[0][2]*p.z()+mat[0][3],
		 mat[1][0]*p.x()+mat[1][1]*p.y()+mat[1][2]*p.z()+mat[1][3],
		 mat[2][0]*p.x()+mat[2][1]*p.y()+mat[2][2]*p.z()+mat[2][3],
		 mat[3][0]*p.x()+mat[3][1]*p.y()+mat[3][2]*p.z()+mat[3][3]);
}

Vector Transform::project(const Vector& p)
{
    return Vector(mat[0][0]*p.x()+mat[0][1]*p.y()+mat[0][2]*p.z(),
		 mat[1][0]*p.x()+mat[1][1]*p.y()+mat[1][2]*p.z(),
		 mat[2][0]*p.x()+mat[2][1]*p.y()+mat[2][2]*p.z());
}

Point Transform::unproject(const Point& p)
{
    if(!inverse_valid)compute_imat();
    return Point(imat[0][0]*p.x()+imat[0][1]*p.y()+imat[0][2]*p.z()+imat[0][3],
		 imat[1][0]*p.x()+imat[1][1]*p.y()+imat[1][2]*p.z()+imat[1][3],
		 imat[2][0]*p.x()+imat[2][1]*p.y()+imat[2][2]*p.z()+imat[2][3],
		 imat[3][0]*p.x()+imat[3][1]*p.y()+imat[3][2]*p.z()+imat[3][3]);
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

void Transform::lookat(const Point&, const Point&, const Vector&)
{
    cerr << "Transform::lookat not finished\n";
}

void Transform::install_mat(double m[4][4])
{
    for(int i=0;i<4;i++){
	for(int j=0;j<4;j++){
	    mat[i][j]=m[i][j];
	}
    }
}

void Transform::compute_imat()
{
    cerr << "Transform::compute_imat not finished\n";
}

void Transform::mulmat(double mmat[4][4])
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

PTransform::PTransform()
{
}

PTransform::~PTransform()
{
}

void PTransform::perspective(double, double, double, double)
{
    cerr << "PTransform::perspective not finished\n";
}

void PTransform::ortho(const Point& min, const Point& max)
{
    Vector d(max-min);
    d.x(1/d.x());
    d.y(1/d.y());
    d.z(1/d.z());
    scale(d);
    translate(Point(0,0,0)-min);
}

PTransform::PTransform(const PTransform& copy)
: Transform(copy)
{
}
