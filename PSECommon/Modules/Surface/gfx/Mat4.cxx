#include <gfx/std.h>
#include <gfx/math/Mat4.h>

Mat4 Mat4::identity(Vec4(1,0,0,0),Vec4(0,1,0,0),Vec4(0,0,1,0),Vec4(0,0,0,1));
Mat4 Mat4::zero(Vec4(0,0,0,0),Vec4(0,0,0,0),Vec4(0,0,0,0),Vec4(0,0,0,0));
Mat4 Mat4::unit(Vec4(1,1,1,1),Vec4(1,1,1,1),Vec4(1,1,1,1),Vec4(1,1,1,1));

Mat4 Mat4::trans(real x, real y, real z)
{
    return Mat4(Vec4(1,0,0,x),
		Vec4(0,1,0,y),
		Vec4(0,0,1,z),
		Vec4(0,0,0,1));
}

Mat4 Mat4::scale(real x, real y, real z)
{
    return Mat4(Vec4(x,0,0,0),
		Vec4(0,y,0,0),
		Vec4(0,0,z,0),
		Vec4(0,0,0,1));
}

Mat4 Mat4::xrot(real a)
{
    real c = cos(a);
    real s = sin(a);

    return Mat4(Vec4(1, 0, 0, 0),
		Vec4(0, c,-s, 0),
		Vec4(0, s, c, 0),
		Vec4(0, 0, 0, 1));
}

Mat4 Mat4::yrot(real a)
{
    real c = cos(a);
    real s = sin(a);

    return Mat4(Vec4(c, 0, s, 0),
		Vec4(0, 1, 0, 0),
		Vec4(-s,0, c, 0),
		Vec4(0, 0, 0, 1));
}

Mat4 Mat4::zrot(real a)
{
    real c = cos(a);
    real s = sin(a);

    return Mat4(Vec4(c,-s, 0, 0),
		Vec4(s, c, 0, 0),
		Vec4(0, 0, 1, 0),
		Vec4(0, 0, 0, 1));
}

Mat4 Mat4::operator*(const Mat4& m) const
{
    Mat4 A;
    int i,j;

    for(i=0;i<4;i++)
	for(j=0;j<4;j++)
	    A(i,j) = row[i]*m.col(j);

    return A;
}

real Mat4::det() const
{
    return row[0] * cross(row[1], row[2], row[3]);
}

Mat4 Mat4::transpose() const
{
    return Mat4(col(0), col(1), col(2), col(3));
}

Mat4 Mat4::adjoint() const
{
    Mat4 A;

    A.row[0] = cross( row[1], row[2], row[3]);
    A.row[1] = cross(-row[0], row[2], row[3]);
    A.row[2] = cross( row[0], row[1], row[3]);
    A.row[3] = cross(-row[0], row[1], row[2]);
        
    return A;
}

real Mat4::cramerInverse(Mat4& inv) const
{
    Mat4 A = adjoint();
    real d = A.row[0] * row[0];

    if( d==0.0 )
	return 0.0;

    inv = A.transpose() / d;
    return d;
}



// Matrix inversion code for 4x4 matrices.
// Originally ripped off and degeneralized from Paul's matrix library
// for the view synthesis (Chen) software.
//
// Returns determinant of a, and b=a inverse.
// If matrix is singular, returns 0 and leaves trash in b.
//
// Uses Gaussian elimination with partial pivoting.

#define SWAP(a, b, t)   {t = a; a = b; b = t;}
real Mat4::inverse(Mat4& B) const
{
    Mat4 A(*this);

    int i, j, k;
    real max, t, det, pivot;

    /*---------- forward elimination ----------*/

    for (i=0; i<4; i++)                 /* put identity matrix in B */
        for (j=0; j<4; j++)
            B(i, j) = (real)(i==j);

    det = 1.0;
    for (i=0; i<4; i++) {               /* eliminate in column i, below diag */
        max = -1.;
        for (k=i; k<4; k++)             /* find pivot for column i */
            if (fabs(A(k, i)) > max) {
                max = fabs(A(k, i));
                j = k;
            }
        if (max<=0.) return 0.;         /* if no nonzero pivot, PUNT */
        if (j!=i) {                     /* swap rows i and j */
            for (k=i; k<4; k++)
                SWAP(A(i, k), A(j, k), t);
            for (k=0; k<4; k++)
                SWAP(B(i, k), B(j, k), t);
            det = -det;
        }
        pivot = A(i, i);
        det *= pivot;
        for (k=i+1; k<4; k++)           /* only do elems to right of pivot */
            A(i, k) /= pivot;
        for (k=0; k<4; k++)
            B(i, k) /= pivot;
        /* we know that A(i, i) will be set to 1, so don't bother to do it */

        for (j=i+1; j<4; j++) {         /* eliminate in rows below i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=i+1; k<4; k++)       /* subtract scaled row i from row j */
                A(j, k) -= A(i, k)*t;   /* (ignore k<=i, we know they're 0) */
            for (k=0; k<4; k++)
                B(j, k) -= B(i, k)*t;
        }
    }

    /*---------- backward elimination ----------*/

    for (i=4-1; i>0; i--) {             /* eliminate in column i, above diag */
        for (j=0; j<i; j++) {           /* eliminate in rows above i */
            t = A(j, i);                /* we're gonna zero this guy */
            for (k=0; k<4; k++)         /* subtract scaled row i from row j */
                B(j, k) -= B(i, k)*t;
        }
    }

    return det;
}
