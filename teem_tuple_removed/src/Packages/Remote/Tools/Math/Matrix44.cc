#include <Packages/Remote/Tools/Math/Matrix44.h>
#include <Packages/Remote/Tools/Math/Vector.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

namespace Remote {
//////////////////////////////////////////////////////////
// Private member functions.

void Matrix44::build_identity(double m[4][4]) const
{
	m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0;
	m[0][1] = m[0][2] = m[0][3] = 0.0;
	m[1][0] = m[1][2] = m[1][3] = 0.0;
	m[2][0] = m[2][1] = m[2][3] = 0.0;
	m[3][0] = m[3][1] = m[3][2] = 0.0;
}

void Matrix44::build_scale(double m[4][4], const Vector& v) const
{
	build_identity(m);
	m[0][0]=v.x;
	m[1][1]=v.y;
	m[2][2]=v.z;
}

void Matrix44::build_translate(double m[4][4], const Vector& v) const
{
	build_identity(m);
	m[0][3]=v.x;
	m[1][3]=v.y;
	m[2][3]=v.z;
}

void Matrix44::build_rotate(double m[4][4], double angle, const Vector& axis) const
{
	// NOTE: Element 0,1 is wrong in Foley and Van Dam, Pg 227!
	double sintheta=sin(angle);
	double costheta=cos(angle);
	double ux=axis.x;
	double uy=axis.y;
	double uz=axis.z;
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

// Returns true if it worked, false if non-invertible.
bool Matrix44::build_inverse(double m[4][4]) const
{
  double p[4][4];
  build_identity(p);
	
  // Make it upper triangular using Gauss-Jordan with partial pivoting.
  int i;
  for(i=0; i<4; i++)
    {
      // Find largest row.
      double max=Abs(m[i][i]);
      int row=i;
      int j;
      for(j=i+1; j<4; j++)
	{
	  if(Abs(m[j][i]) > max)
	    {
	      max=Abs(m[j][i]);
	      row=j;
	    }
	}
		
      // Pivot around largest row.
      if(max <= 0)
	return false;

      if(row!=i)
	{
	  switch_rows(m, i, row);
	  switch_rows(p, i, row);
	}
		
      // Subtract scaled rows to eliminate column i.
      ASSERT(m[i][i]!=0);
      double denom = 1./m[i][i];
      for(j=i+1; j<4; j++)
	{
	  double factor = m[j][i] * denom;
	  sub_rows(m, j, i, factor);
	  sub_rows(p, j, i, factor);
	}
    }
	
  // Diagonalize m using Jordan.
  for(i=1; i<4; i++)
    {
      ASSERT(m[i][i]!=0);
      double denom = 1./m[i][i];
      for(int j=0; j<i; j++)
	{
	  double factor = m[j][i] * denom;
	  sub_rows(m, j, i, factor);
	  sub_rows(p, j, i, factor);
	}
    }
	
  // Normalize m to the identity and copy p over m.
  for(i=0; i<4; i++)
    {
      ASSERT(m[i][i]!=0);
      double factor = 1./m[i][i];
      for(int j=0; j<4; j++)
	{
	  // As if we were doing m[i][j] *= factor
	  p[i][j] *= factor;
	  m[i][j] = p[i][j];
	}
    }

  return true;
}

void Matrix44::build_transpose(double m[4][4]) const
{
  double t;
  t = m[0][1]; m[0][1] = m[1][0]; m[1][0] = t;
  t = m[0][2]; m[0][2] = m[2][0]; m[2][0] = t;
  t = m[0][3]; m[0][3] = m[3][0]; m[3][0] = t;
  t = m[1][2]; m[1][2] = m[2][1]; m[2][1] = t;
  t = m[1][3]; m[1][3] = m[3][1]; m[3][1] = t;
  t = m[2][3]; m[2][3] = m[3][2]; m[3][2] = t;
}

void Matrix44::post_mulmat(double to[4][4], const double from[4][4])
{
  double newmat[4][4];
  for(int i=0; i<4; i++)
    {
      for(int j=0; j<4; j++)
	{
	  newmat[i][j] = 0.0;
	  for(int k=0; k<4; k++)
	    {
	      newmat[i][j] += to[i][k] * from[k][j];
	    }
	}
    }
	
  copy_mat((double *)to, (double *)newmat);
}

void Matrix44::pre_mulmat(double to[4][4], const double from[4][4])
{
  double newmat[4][4];
  for(int i=0; i<4; i++)
    {
      for(int j=0; j<4; j++)
	{
	  newmat[i][j]=0.0;
	  for(int k=0; k<4; k++)
	    {
	      newmat[i][j] += from[i][k] * to[k][j];
	    }
	}
    }
	
  copy_mat((double *)to, (double *)newmat);
}

//////////////////////////////////////////////////////////
// Public member functions.

string Matrix44::print() const
{
	char xx[40];

	string st;
	for(int i=0; i<4; i++)
	{
		st += string("[");
		st += gcvt(mat[i][0], PRDIG, xx) + string(", ");
		st += gcvt(mat[i][1], PRDIG, xx) + string(", ");
		st += gcvt(mat[i][2], PRDIG, xx) + string(", ");
		st += gcvt(mat[i][3], PRDIG, xx) + string("]\n");
	}

	return st;
}

string Matrix44::printInv() const
{
	char xx[40];

	string st;
	for(int i=0; i<4; i++)
	{
		st += string("[");
		for(int j=0; j<4; j++)
			st += gcvt(imat[i][j], PRDIG, xx) + string(", ");
		st += string("]\n");
	}

	return st;
}

void Matrix44::PostTrans(const Matrix44& T)
{
	if(T.is_identity)
		return;
	if(is_identity)
	{
		// Copying is faster than matrix multiply.
		copy_mat((double *)mat, (double *)T.mat);
		copy_mat((double *)imat, (double *)T.imat);
		is_identity = false;
		inverse_valid = T.inverse_valid;
		return;
	}
	
	post_mulmat(mat, T.mat);
	pre_mulmat(imat, T.imat);
	inverse_valid = inverse_valid && T.inverse_valid;
}

void Matrix44::PreTrans(const Matrix44& T)
{
	if(T.is_identity)
		return;
	if(is_identity)
	{
		// Copying is faster than matrix multiply.
		copy_mat((double *)mat, (double *)T.mat);
		copy_mat((double *)imat, (double *)T.imat);
		is_identity = false;
		inverse_valid = T.inverse_valid;
		return;
	}
	
	pre_mulmat(mat, T.mat);
	post_mulmat(imat, T.imat);
	inverse_valid = inverse_valid && T.inverse_valid;
}

void Matrix44::LoadFrame(const Vector& x, const Vector& y, const Vector& z)
{
	mat[3][3] = imat[3][3] = 1.0;
	mat[0][3] = mat[1][3] = mat[2][3] = 0.0;
	imat[0][3] = imat[1][3] = imat[2][3] = 0.0;
	
	mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
	imat[3][0] = imat[3][1] = imat[3][2] = 0.0;
	
	mat[0][0] = x.x;
	mat[1][0] = x.y;
	mat[2][0] = x.z;
	
	mat[0][1] = y.x;
	mat[1][1] = y.y;
	mat[2][1] = y.z;
	
	mat[0][2] = z.x;
	mat[1][2] = z.y;
	mat[2][2] = z.z;
	
	imat[0][0] = x.x;
	imat[0][1] = x.y;
	imat[0][2] = x.z;
	
	imat[1][0] = y.x;
	imat[1][1] = y.y;
	imat[1][2] = y.z;
	
	imat[2][0] = z.x;
	imat[2][1] = z.y;
	imat[2][2] = z.z;
	
	inverse_valid = true;
	is_identity = false;
}

// Loads a rotation frame and an offset.
void Matrix44::LoadFrame(const Vector& x, const Vector& y, 
						 const Vector& z, const Vector& t)
{
	LoadFrame(x, y, z);
	Translate(t);
}

void Matrix44::GetFrame(Vector &c0, Vector &c1, Vector &c2)
{
  c0.x = mat[0][0]; c1.x = mat[0][1]; c2.x = mat[0][2];
  c0.y = mat[1][0]; c1.y = mat[1][1]; c2.y = mat[1][2];
  c0.z = mat[2][0]; c1.z = mat[2][1]; c2.z = mat[2][2];
}

void Matrix44::GetFrame(Vector &c0, Vector &c1, Vector &c2, Vector &c3)
{
  c0.x = mat[0][0]; c1.x = mat[0][1]; c2.x = mat[0][2]; c3.x = mat[0][3];
  c0.y = mat[1][0]; c1.y = mat[1][1]; c2.y = mat[1][2]; c3.y = mat[1][3];
  c0.z = mat[2][0]; c1.z = mat[2][1]; c2.z = mat[2][2]; c3.z = mat[2][3];
}

void Matrix44::ChangeBasis(const Matrix44& T)
{
	// If T.imat is invalid it will not only make
	// imat invalid, but mat will be invalid.
	if(!inverse_valid)
		compute_inverse();
	
	// XXX Not yet optimized for the identity.
	pre_mulmat(mat, T.imat);
	post_mulmat(mat, T.mat);
	
	// XXX I need to check this.
	pre_mulmat(imat, T.imat);
	post_mulmat(imat, T.mat);
	is_identity = false;
}

void Matrix44::Scale(const Vector& v)
{
	double m[4][4];
	build_scale(m, v);
	if(is_identity)
		copy_mat((double *)mat, (double *)m);
	else
		post_mulmat(mat, m);
	
	m[0][0] = 1. / m[0][0];
	m[1][1] = 1. / m[1][1];
	m[2][2] = 1. / m[2][2];
	if(is_identity)
		copy_mat((double *)imat, (double *)m);
	else
		pre_mulmat(imat, m);
	is_identity = false;
}

void Matrix44::Rotate(double angle, const Vector& axis)
{
	double m[4][4];
	build_rotate(m, angle, axis);
	if(is_identity)
		copy_mat((double *)mat, (double *)m);
	else
		post_mulmat(mat, m);
	
	build_transpose(m);
	if(is_identity)
		copy_mat((double *)imat, (double *)m);
	else
		pre_mulmat(imat, m);
	is_identity = false;
} 

void Matrix44::Translate(const Vector& v)
{
	double m[4][4];
	build_translate(m, v);
	if(is_identity)
		copy_mat((double *)mat, (double *)m);
	else
		post_mulmat(mat, m);
	
	m[0][3] = -m[0][3];
	m[1][3] = -m[1][3];
	m[2][3] = -m[2][3];
	
	if(is_identity)
		copy_mat((double *)imat, (double *)m);
	else
		pre_mulmat(imat, m);
	is_identity = false;
}

bool Matrix44::Invert()
{
	if(inverse_valid)
	{
		if(is_identity) return true;
		
		// Just swap it with its inverse.
		double temp[4][4];
		copy_mat((double *)temp, (double *)mat);
		copy_mat((double *)mat, (double *)imat);
		copy_mat((double *)imat, (double *)temp);
		return true;
	}
	else
	{
		// Copy mat to imat, then invert old mat.
		copy_mat((double *)imat, (double *)mat);
		inverse_valid = true;
		if(is_identity) return true;
		
		return inverse_valid = build_inverse(mat);
	}
}

void Matrix44::Transpose()
{
	if(is_identity) return;

	build_transpose(mat);

	// XXX I don't know what to do to the inverse, so toss it.
	inverse_valid = false;
}

void Matrix44::Frustum(double l, double r, double b, double t, double n, double f)
{
	ASSERT(n>0 && f>0);
	double m[4][4];
	
	m[0][0] = (n+n)/(r-l);
	m[0][1] = 0;
	m[0][2] = (r+l)/(r-l);
	m[0][3] = 0;
	
	m[1][0] = 0;
	m[1][1] = (n+n)/(t-b);
	m[1][2] = (t+b)/(t-b);
	m[1][3] = 0;
	
	m[2][0] = 0;
	m[2][1] = 0;
	m[2][2] = -(f+n)/(f-n);
	m[2][3] = -2.0*f*n/(f-n);
	
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = -1;
	m[3][3] = 0;
	
	if(is_identity)
		copy_mat((double *)mat, (double *)m);
	else
		post_mulmat(mat, m);
	
	m[0][0] = (r-l)/(n+n);
	m[0][1] = 0;
	m[0][2] = 0;
	m[0][3] = (r+l)/(n+n);
	
	m[1][0] = 0;
	m[1][1] = (t-b)/(n+n);
	m[1][2] = 0;
	m[1][3] = (t+b)/(n+n);
	
	m[2][0] = 0;
	m[2][1] = 0;
	m[2][2] = 0;
	m[2][3] = -1;
	
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = -(f-n)/(2.0*f*n);
	m[3][3] = (f+n)/(2.0*f*n);
	
	if(is_identity)
		copy_mat((double *)imat, (double *)m);
	else
		pre_mulmat(imat, m);
	
	is_identity = false;
}

void Matrix44::Perspective(double fovy, double aspect, double znear, double zfar)
{
	ASSERT(znear>0 && zfar>0);
	double top = znear * tan(fovy * 0.5);
	double bottom = -top;
	double left = bottom * aspect;
	double right = top * aspect;
	Frustum(left, right, bottom, top, znear, zfar);
}

void Matrix44::LookAt(const Vector& eye, const Vector& lookat, const Vector& up)
{
	Vector f(lookat - eye);
	f.normalize();
	
	Vector upn(up);
	upn.normalize();
	
	Vector s(Cross(f, upn));
	s.normalize();
	
	Vector u(Cross(s, f));
	// u.normalize(); // This normalize shouldn't be necessary.
	
	double m[4][4];
	
	m[0][0] = s.x;
	m[0][1] = s.y;
	m[0][2] = s.z;
	m[0][3] = 0;
	
	m[1][0] = u.x;
	m[1][1] = u.y;
	m[1][2] = u.z;
	m[1][3] = 0;
	
	m[2][0] = -f.x;
	m[2][1] = -f.y;
	m[2][2] = -f.z;
	m[2][3] = 0;
	
	m[3][0] = 0;
	m[3][1] = 0;
	m[3][2] = 0;
	m[3][3] = 1;
	
	if(is_identity)
		copy_mat((double *)mat, (double *)m);
	else
		post_mulmat(mat, m);
	
	build_transpose(m);
	
	if(is_identity)
		copy_mat((double *)imat, (double *)m);
	else
		pre_mulmat(imat, m);
	
	is_identity = false;
	
	Translate(-eye);
}

Vector Matrix44::Project(const Vector& p) const
{
	// XXX Should I put an optimization here for is_identity?
	
	double w1 = mat[3][0]*p.x + mat[3][1]*p.y + mat[3][2]*p.z + mat[3][3];
	w1 = 1. / w1;
	
	double xw = mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3];
	double yw = mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3];
	double zw = mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3];
	
	return Vector(xw * w1, yw * w1, zw * w1);
}

Vector Matrix44::Project(const Vector& p, const double w) const
{
	// XXX Should I put an optimization here for is_identity?
	
	double w1 = mat[3][0]*p.x + mat[3][1]*p.y + mat[3][2]*p.z + mat[3][3]*w;
	w1 = 1. / w1;
	
	double xw = mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3]*w;
	double yw = mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3]*w;
	double zw = mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3]*w;
	
	return Vector(xw * w1, yw * w1, zw * w1);
}

void Matrix44::Project(double &x, double &y, double &z, double &w) const
{
	// XXX Should I put an optimization here for is_identity?
	
	double x1 = mat[0][0]*x + mat[0][1]*y + mat[0][2]*z + mat[0][3]*w;
	double y1 = mat[1][0]*x + mat[1][1]*y + mat[1][2]*z + mat[1][3]*w;
	double z1 = mat[2][0]*x + mat[2][1]*y + mat[2][2]*z + mat[2][3]*w;
	double w1 = mat[3][0]*x + mat[3][1]*y + mat[3][2]*z + mat[3][3]*w;
	x=x1; y=y1; z=z1; w=w1;
}

Vector Matrix44::ProjectDirection(const Vector& p) const
{
	// XXX Should I put an optimization here for is_identity?
	
	return Vector(mat[0][0]*p.x+mat[0][1]*p.y+mat[0][2]*p.z,
		mat[1][0]*p.x+mat[1][1]*p.y+mat[1][2]*p.z,
		mat[2][0]*p.x+mat[2][1]*p.y+mat[2][2]*p.z);
}

Vector Matrix44::UnProject(const Vector& p)
{
	// XXX Should I put an optimization here for is_identity?
	
	if(!inverse_valid)
		compute_inverse();
	
	double w1 = imat[3][0]*p.x+imat[3][1]*p.y+imat[3][2]*p.z+imat[3][3];
	w1 = 1. / w1;
	
	double xw = imat[0][0]*p.x+imat[0][1]*p.y+imat[0][2]*p.z+imat[0][3];
	double yw = imat[1][0]*p.x+imat[1][1]*p.y+imat[1][2]*p.z+imat[1][3];
	double zw = imat[2][0]*p.x+imat[2][1]*p.y+imat[2][2]*p.z+imat[2][3];
	
	return Vector(xw * w1, yw * w1, zw * w1);
}

bool Matrix44::CheckNaN() const
{
  double *m = (double *)mat;
  for(int i=0; i<16; i++)
    ASSERTERR(!isnan(m[i]), "Matrix has a NaN");

  return true;
}
} // End namespace Remote


