

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

#include <SCICore/Math/Mat.h>

#define Abs(x) ((x)<0?-(x):(x))
#define Max(x,y) ((x)<(y)?(y):(x))
#define SWITCH_ROWS(m, r1, r2) \
    for(k=0;k<3;k++){ \
        double tmp=m[r1][k]; \
        m[r1][k]=m[r2][k]; \
        m[r2][k]=tmp; \
    }

#define SUB_ROWS(m, r1, r2, mul) \
    for(k=0;k<3;k++){ \
        m[r1][k] -= m[r2][k]*mul; \
    }

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

void matsolve3by3_cond(double mat[3][3], double rhs[3], double* rcond)
{
    double imat[3][3];
    double norm=0;
    int i, j, k;
    double denom;
    double inorm;
    double lhs[3];
    double factor;

    for(i=0;i<3;i++){
	double sum=0;
	for(j=0;j<3;j++)
	    sum+=Abs(mat[i][j]);
	norm=Max(norm, sum);
    }

    for(i=0;i<3;i++){
        for(j=0;j<3;j++){
            imat[i][j]=0.0;
        }
        imat[i][i]=1.0;
    }

    /* Gauss-Jordan with partial pivoting */
    for(i=0;i<3;i++){
        double max=Abs(mat[i][i]);
        int row=i;
        for(j=i+i;j<3;j++){
            if(Abs(mat[j][i]) > max){
                max=Abs(mat[j][i]);
                row=j;
            }
        }
	if(max==0){
	    rcond=0;
	    return;
	}
        if(row!=i){
	    SWITCH_ROWS(mat, i, row);
	    SWITCH_ROWS(imat, i, row);
        }
        denom=1./mat[i][i];
        for(j=i+1;j<3;j++){
            double factor=mat[j][i]*denom;
            SUB_ROWS(mat, j, i, factor);
            SUB_ROWS(imat, j, i, factor);
        }
    }

    /* Jordan */
    for(i=1;i<3;i++){
	if(mat[i][i]==0){
	    rcond=0;
	    return;
	}
        denom=1./mat[i][i];
        for(j=0;j<i;j++){
            double factor=mat[j][i]*denom;
            SUB_ROWS(mat, j, i, factor);
            SUB_ROWS(imat, j, i, factor);
        }
    }

    /* Normalize */
    for(i=0;i<3;i++){
	if(mat[i][i]==0){
	    rcond=0;
	    return;
	}
        factor=1./mat[i][i];
        for(j=0;j<3;j++){
            imat[i][j] *= factor;
	}
    }

    inorm=0;
    for(i=0;i<3;i++){
	double sum=0;
	for(j=0;j<3;j++)
	    sum+=Abs(imat[i][j]);
	inorm=Max(inorm, sum);
    }
    *rcond=1./(norm*inorm);

    /* Compute the solution... */
    for(i=0;i<3;i++){
	lhs[i]=0;
	for(j=0;j<3;j++){
	    lhs[i]+=imat[i][j]*rhs[j];
	}
    }
    for(i=0;i<3;i++){
	rhs[i]=lhs[i];
    }
}
