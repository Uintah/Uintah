#include <gfx/std.h>
#include <gfx/math/Mat2.h>

Mat2 Mat2::identity(Vec2(1,0), Vec2(0,1));
Mat2 Mat2::zero(Vec2(0,0), Vec2(0,0));
Mat2 Mat2::unit(Vec2(1,1), Vec2(1,1));

Mat2 Mat2::operator*(const Mat2& m) const
{
    Mat2 A;
    int i,j;

    for(i=0;i<2;i++)
	for(j=0;j<2;j++)
	    A(i,j) = row[i]*m.col(j);

    return A;
}

real Mat2::det()
{
    return (row[0][0]*row[1][1] - row[0][1]*row[1][0]);
}

Mat2 Mat2::transpose()
{
    return Mat2(col(0), col(1));
}

real Mat2::inverse(Mat2& inv)
{
    real d = det();

    if( d==0.0 )
	return 0.0;

    inv.row[0][0] = row[1][1]/d;
    inv.row[0][1] = -row[1][0]/d;
    inv.row[1][0] = -row[0][1]/d;
    inv.row[1][1] = row[0][0]/d;

    return d;
}
