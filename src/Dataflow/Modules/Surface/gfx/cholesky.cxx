#include <gfx/std.h>
#include <gfx/math/Mat4.h>

// Adapted directly from Numerical Recipes in C
//
// Takes a symmetric positive definite matrix a
// (only the upper triangle is actually necessary)
// and computes the Cholesky decomposition of a.
// A return value of False indicates that the decomposition does not exist.
// On return, the lower triangle of a contains the factor and the
// vector p contains the diagonal elements of the factor.
//
bool cholesky(Mat4& a, Vec4& p)
{
    int n = 4;
    int i,j,k;
    real sum;

    for(i=0; i<n; i++)
    {
        for(j=i; j<n; j++)
        {
            for(sum=a(i,j), k=i-1; k>=0; k--) sum -= a(i,k)*a(j,k);
            if( i==j )
            {
                if( sum<=0.0 )
                    return False;
                p[i] = sqrt(sum);
            }
            else
                a(j,i) = sum/p[i];
        }
    }

    return True;
}
