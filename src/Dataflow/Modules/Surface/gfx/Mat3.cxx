#include <gfx/std.h>
#include <gfx/math/Mat3.h>

Mat3 Mat3::identity(Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1));
Mat3 Mat3::zero(Vec3(0,0,0), Vec3(0,0,0), Vec3(0,0,0));
Mat3 Mat3::unit(Vec3(1,1,1), Vec3(1,1,1), Vec3(1,1,1));

Mat3 Mat3::diag(const Vec3& v)
{
    Mat3 M(zero);

    M(0,0) = v[X];
    M(1,1) = v[Y];
    M(2,2) = v[Z];

    return M;
}

Mat3 Mat3::extend(const Vec3& v)
{
    return Mat3(v,v,v);
}



Mat3 Mat3::operator*(const Mat3& m) const
{
    Mat3 A;
    int i,j;

    for(i=0;i<3;i++)
	for(j=0;j<3;j++)
	    A(i,j) = row[i]*m.col(j);

    return A;
}

real Mat3::det()
{
    return row[0] * (row[1] ^ row[2]);
}

Mat3 Mat3::transpose()
{
    return Mat3(col(0), col(1), col(2));
}

Mat3 Mat3::adjoint()
{
    return Mat3(row[1]^row[2],
		row[2]^row[0],
		row[0]^row[1]);
}

real Mat3::inverse(Mat3& inv)
{
    Mat3 A = adjoint();
    real d = A.row[0] * row[0];

    if( d==0.0 )
	return 0.0;

    inv = A.transpose() / d;
    return d;
}
