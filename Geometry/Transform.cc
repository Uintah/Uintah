
#include <Geometry/Transform.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Trig.h>
#include <iostream.h>

#include <stdio.h>

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
void Transform::pre_scale(const Vector& v)
{
    for(int i=0;i<4;i++){
	mat[0][i]*=v.x();
	mat[1][i]*=v.y();
	mat[2][i]*=v.z();
    }
    inverse_valid=0;
}

void Transform::post_scale(const Vector& v)
{
    for(int i=0;i<4;i++){
	mat[i][0]*=v.x();
	mat[i][1]*=v.y();
	mat[i][2]*=v.z();
    }
    inverse_valid=0;
}

void Transform::pre_translate(const Vector& v)
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

void Transform::post_translate(const Vector& v)
{
    mat[0][3]+=mat[0][0]*v.x()+mat[0][1]*v.y()+mat[0][2]*v.z()+mat[0][3];
    mat[1][3]+=mat[1][0]*v.x()+mat[1][1]*v.y()+mat[1][2]*v.z()+mat[1][3];
    mat[2][3]+=mat[2][0]*v.x()+mat[2][1]*v.y()+mat[2][2]*v.z()+mat[2][3];
    mat[3][3]+=mat[3][0]*v.x()+mat[3][1]*v.y()+mat[3][2]*v.z()+mat[3][3];
    inverse_valid=0;
}

void Transform::pre_rotate(double angle, const Vector& axis)
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

    pre_mulmat(newmat);
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

void Transform::perspective(const Point& eyep, const Point& lookat,
			    const Vector& up, double fov,
			    double znear, double zfar,
			    int xres, int yres)
{
    Vector lookdir(lookat-eyep);
    Vector z(lookdir);
    z.normalize();
    Vector x(Cross(z, up));
    x.normalize();
    Vector y(Cross(x, z));
    double xviewsize=Tan(DtoR(fov/2.))*2.;
    double yviewsize=xviewsize*yres/xres;
    double zscale=-znear;
    double xscale=xviewsize*0.5;
    double yscale=yviewsize*0.5;
    x*=xscale;
    y*=yscale;
    z*=zscale;
//    pre_translate(Point(0,0,0)-eyep);
    double m[4][4];
    // Viewing...
    m[0][0]=x.x(); m[0][1]=y.x(); m[0][2]=z.x(); m[0][3]=eyep.x();
    m[1][0]=x.y(); m[1][1]=y.y(); m[1][2]=z.y(); m[1][3]=eyep.y();
    m[2][0]=x.z(); m[2][1]=y.z(); m[2][2]=z.z(); m[2][3]=eyep.z();
    m[3][0]=0;     m[3][1]=0; m[3][2]=0.0;   m[3][3]=1.0;
    invmat(m);
    pre_mulmat(m);
    
    // Perspective...
    m[0][0]=1.0; m[0][1]=0.0; m[0][2]=0.0; m[0][3]=0.0;
    m[1][0]=0.0; m[1][1]=1.0; m[1][2]=0.0; m[1][3]=0.0;
    m[2][0]=0.0; m[2][1]=0.0; m[2][2]=-(zfar-1)/(1+zfar); m[2][3]=-2*zfar/(1+zfar);
    m[3][0]=0.0; m[3][1]=0.0; m[3][2]=-1.0; m[3][3]=0.0;
    pre_mulmat(m);

    pre_scale(Vector(1,-1,1)); // X starts at the top...
    pre_translate(Vector(1,1,0));
    pre_scale(Vector(xres/2., yres/2., 1.0));	
    m[3][3]+=1.0; // hack
}

void Transform::invmat(double m[4][4])
{
    double imat[4][4];
    int i;
    for(i=0;i<4;i++){
        for(int j=0;j<4;j++){
            imat[i][j]=0.0;
        }
        imat[i][i]=1.0;
    }

    // Gauss-Jordan with partial pivoting
    for(i=0;i<4;i++){
        double max=Abs(m[i][i]);
        int row=i;
	int j;
        for(j=i+i;j<4;j++){
            if(Abs(m[j][i]) > max){
                max=Abs(m[j][i]);
                row=j;
            }
        }
        ASSERT(max!=0);
        if(row!=i){
            switch_rows(m, i, row);
            switch_rows(imat, i, row);
        }
        double denom=1./m[i][i];
        for(j=i+1;j<4;j++){
            double factor=m[j][i]*denom;
            sub_rows(m, j, i, factor);
            sub_rows(imat, j, i, factor);
        }
    }

    // Jordan
    for(i=1;i<4;i++){
        ASSERT(m[i][i]!=0);
        double denom=1./m[i][i];
        for(int j=0;j<i;j++){
            double factor=m[j][i]*denom;
            sub_rows(m, j, i, factor);
            sub_rows(imat, j, i, factor);
        }
    }

    // Normalize
    for(i=0;i<4;i++){
        ASSERT(m[i][i]!=0);
        double factor=1./m[i][i];
        for(int j=0;j<4;j++){
            imat[i][j] *= factor;
	    m[i][j]=imat[i][j];
	}
    }
}

void Transform::switch_rows(double m[4][4], int r1, int r2) const
{
    for(int i=0;i<4;i++){
        double tmp=m[r1][i];
        m[r1][i]=m[r2][i];
        m[r2][i]=tmp;
    }
}


void Transform::sub_rows(double m[4][4], int r1, int r2, double mul) const
{
    for(int i=0;i<4;i++)
        m[r1][i] -= m[r2][i]*mul;
}

Transform& Transform::operator=(const Transform& copy)
{
    for(int i=0;i<4;i++)
        for(int j=0;j<4;j++)
            mat[i][j]=copy.mat[i][j];
    inverse_valid=0;
    return *this;
}

