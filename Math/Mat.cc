

/*
 *  Mat.cc:  Simple matrix calculations
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Math/Mat.h>

void matsolve3by3(double mat[3][3], double rhs[3])
{
    double B[3];
    double t7 = mat[2][1]*mat[0][2];
    double t20 = mat[2][1]*mat[0][0];
    double t23 = mat[0][1]*mat[2][0];
    double t26 = 1/(mat[2][2]*mat[0][0]*mat[1][1]-mat[2][2]*mat[1][0]*mat[0][1]-mat[2][0]*mat[0][2]*mat[1][1]-t20*mat[1][2]+t7*mat[1][0]+t23*mat[1][2]);
    double t34 = rhs[0]*mat[2][0];
    B[0] = (mat[0][1]*mat[1][2]*rhs[2]-mat[0][1]*rhs[1]*mat[2][2]-mat[0][2]*rhs[2]*mat[1][1]+t7*rhs[1]+rhs[0]*mat[2][2]*mat[1][1]-rhs[0]*mat[2][1]*mat[1][2])*t26;
    B[1] = -(mat[0][0]*mat[1][2]*rhs[2]-mat[0][0]*rhs[1]*mat[2][2]+rhs[1]*mat[2][0]*mat[0][2]-t34*mat[1][2]-mat[0][2]*mat[1][0]*rhs[2]+mat[1][0]*rhs[0]*mat[2][2])*t26;
    B[2] = (rhs[2]*mat[0][0]*mat[1][1]-rhs[2]*mat[1][0]*mat[0][1]-t34*mat[1][1]-t20*rhs[1]+mat[2][1]*mat[1][0]*rhs[0]+t23*rhs[1])*t26;
    rhs[0]=B[0];
    rhs[1]=B[1];
    rhs[2]=B[2];
}
