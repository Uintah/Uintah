#include "Equation.h"

#include <iostream>
#include <math.h>

namespace Uintah {

using namespace std;

Equation::
Equation()
{
}

void
Equation::
solve()
{
  int i,j,k,n,io,jo,i1;
  n = 4;
  
  int m[4];

  for(i=0;i<n;++i) m[i] = i;

  for(k=0;k<n;++k){
    double p = 0;
    for(i=k;i<n;++i)
    for(j=k;j<n;++j) {
      double tmp = fabs( mat[i][j] );
      if( tmp > p ) {
        p = tmp;
        io = i;
        jo = j;
      }
    }
    if(p <= 1.e-13) {
      throw "Sigular Matrix";
    }
    p = mat[io][jo];

    if(jo != k) {
      for(i=0;i<n;++i) {
        swap( mat[i][jo], mat[i][k] );
      }
      swap( m[jo], m[k] );
    }

    if(io != k) {
      for(j=k;j<n;++j) swap( mat[io][j], mat[k][j] );
      swap( vec[io], vec[k] );
    }

    if(k != n-1) for(j=k;j<n-1;++j) mat[k][j+1] /= p;
    vec[k] /= p;

    if(k != n-1) {
      for(i=k;i<n-1;++i) {
        for(j=k;j<n-1;++j)
          mat[i+1][j+1] -= mat[i+1][k] * mat[k][j+1];
          vec[i+1] -= mat[i+1][k] * vec[k];
      }
    }
  }

  for(i1=1;i1<n;++i1) {
    i = n - 1 - i1;
    for(j=i;j<n-1;++j)
    vec[i] -= mat[i][j+1] * vec[j+1];
  }
  for(k=0;k<n;++k) mat[0][m[k]] = vec[k];
  for(k=0;k<n;++k) vec[k] = mat[0][k];
}

template<class T>
void swap(T& a, T& b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

template<class T>
T SQR(const T& a)
{
  return a*a;
}

double pythag(double a, double b)
{
  double absa,absb;
  absa=fabs(a);
  absb=fabs(b);
  if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
  else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}


void HouseholderReduction(Matrix3& a, Vector& d, Vector& e)
{
  int l,k,j,i;
  double scale,hh,h,g,f;
  int n=3;
  
  for (i=n;i>=2;i--) {
    l=i-1;
    h=scale=0.0;
    if(l > 1) {
      for (k=1;k<=l;k++)
        scale += fabs(a(i,k));
      if (scale == 0.0)  //Skip transformation.
        e[i]=a(i,l);
      else {
        for (k=1;k<=l;k++) {
          a(i,k) /= scale;       //Use scaled a's for transformation.
          h += a(i,k)*a(i,k);   //Form Ù in h.
        }
        f=a(i,l);
        g=(f >= 0.0 ? -sqrt(h) : sqrt(h));
        e[i]=scale*g;
        h -= f*g;
        a(i,l)=f-g;              //Store u in the ith row of a.
        f=0.0;
        for (j=1;j<=l;j++) {
        /* Next statement can be omitted if eigenvectors not wanted */
          a(j,i)=a(i,j)/h;   //Store u=H in ith column of a.
          g=0.0;   //Form an element of A  u in g.
          for (k=1;k<=j;k++)
            g += a(j,k)*a(i,k);
          for (k=j+1;k<=l;k++)
            g += a(k,j)*a(i,k);
          e[j]=g/h;   //Form element of p in temporarily unused element of e.
          f += e[j]*a(i,j);
        }
        hh=f/(h+h);   //Form K, equation (11.2.11).
        for (j=1;j<=l;j++) {   //Form q and store in e overwriting p.
          f=a(i,j);
          e[j]=g=e[j]-hh*f;
          for (k=1;k<=j;k++)   //Reduce a, equation (11.2.13).
            a(j,k) -= (f*e[k]+g*a(i,k));
        }
      }
    } else
      e[i]=a(i,l);
    d[i]=h;
  }
  /* Next statement can be omitted if eigenvectors not wanted */
  d[1]=0.0;
  e[1]=0.0;
  /* Contents of this loop can be omitted if eigenvectors not
     wanted except for statement d[i]=a(i,i); */
  for (i=1;i<=n;i++) { //Begin accumulation of transformation matrices.
    l=i-1;
    if (d[i]) {   //This block skipped when i=1.
      for (j=1;j<=l;j++) {
        g=0.0;
        for (k=1;k<=l;k++) //Use u and u=H stored in a to form P  Q.
          g += a(i,k)*a(k,j);
        for (k=1;k<=l;k++)
          a(k,j) -= g*a(k,i);
      }
    }
    d[i]=a(i,i); //This statement remains.
    a(i,i)=1.0; //Reset row and column of a to identity matrix for next iteration.
    for (j=1;j<=l;j++) a(j,i)=a(i,j)=0.0;
  }
}

void QLAlgorithm(Matrix3& z, Vector& d)
{
  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;
  Vector e;
  int n=3;
  
  HouseholderReduction(z,d,e);

  for (i=2;i<=n;i++) e[i-1]=e[i]; //Convenient to renumber the elements of e. 
  e[n]=0.0;
  for (l=1;l<=n;l++) {
    iter=0;
    do {
      for (m=l;m<=n-1;m++) { //Look for a single small subdiagonal element to split the matrix.
        dd=fabs(d[m])+fabs(d[m+1]);
        if ( fabs(e[m]) < dd*1.e-6) break;
      }
      if (m != l) {
        iter++;
        assert(iter<30);
	//cout<<"iter: "<<iter<<endl;
        g=(d[l+1]-d[l])/(2.0*e[l]); //Form shift.
        r=pythag(g,1.0);
        g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
        s=c=1.0;
        p=0.0;
        for (i=m-1;i>=l;i--) { //A plane rotation as in the original QL, followed by Givens rotations to restore tridiagonal form.
          f=s*e[i];
          b=c*e[i];
          e[i+1]=(r=pythag(f,g));
          if (r == 0.0) { //Recover from underflow.
            d[i+1] -= p;
            e[m]=0.0;
            break;
          }
          s=f/r;
          c=g/r;
          g=d[i+1]-p;
          r=(d[i]-g)*s+2.0*c*b;
          d[i+1]=g+(p=s*r);
          g=c*r-b;
          /* Next loop can be omitted if eigenvectors not wanted*/
          for (k=1;k<=n;k++) { //Form eigenvectors.
            f=z(k,i+1);
            z(k,i+1)=s*z(k,i)+c*f;
            z(k,i)=c*z(k,i)-s*f;
          }
        }
        if (r == 0.0 && i >= l) continue;
        d[l] -= p;
        e[l]=g;
        e[m]=0.0;
      }
    } while (m != l);
  }
}

double getMaxEigenvalue(const Matrix3& mat, Vector& eigenVector)
{
  Matrix3 a = mat;
  Vector d;
  int index;

  QLAlgorithm(a, d);
       if(d[1]>=d[2] && d[1]>=d[3]) index = 1;
  else if(d[2]>=d[1] && d[2]>=d[3]) index = 2;
  else index = 3;

  eigenVector[1] = a(1,index);
  eigenVector[2] = a(2,index);
  eigenVector[3] = a(3,index);
  eigenVector.normalize();
  return d[index];
}

void getFirstAndSecondEigenvalue(const Matrix3& mat, 
                                 Vector& eigenVector1, double& eigenValue1,
				 Vector& eigenVector2, double& eigenValue2)
{
  Matrix3 a = mat;
  Vector d;
  int index1,index2;

  QLAlgorithm(a, d);

       if(d[1]>=d[2] && d[1]>=d[3]) index1 = 1;
  else if(d[2]>=d[1] && d[2]>=d[3]) index1 = 2;
  else index1 = 3;

       if( (d[1]-d[2]) *(d[1]-d[3]) <= 0 ) index2 = 1;
  else if( (d[2]-d[1]) *(d[2]-d[3]) <= 0 ) index2 = 2;
  else index2 = 3;

  ASSERT(index1 != index2);

  eigenVector1[1] = a(1,index1);
  eigenVector1[2] = a(2,index1);
  eigenVector1[3] = a(3,index1);
  eigenVector1.normalize();
  eigenValue1 = d[index1];
  
  eigenVector2[1] = a(1,index2);
  eigenVector2[2] = a(2,index2);
  eigenVector2[3] = a(3,index2);
  eigenVector2.normalize();
  eigenValue2 = d[index2];
}

void getEigenInfo(const Matrix3& mat, 
                  Vector& eigenVector1,double& eigenValue1,
		  Vector& eigenVector2,double& eigenValue2,
		  Vector& eigenVector3,double& eigenValue3)
{
  Matrix3 a = mat;
  Vector d;

  QLAlgorithm(a, d);

  eigenVector1[1] = a(1,1);
  eigenVector1[2] = a(2,1);
  eigenVector1[3] = a(3,1);
  eigenVector1.normalize();
  eigenValue1 = d[1];

  eigenVector2[1] = a(1,2);
  eigenVector2[2] = a(2,2);
  eigenVector2[3] = a(3,2);
  eigenVector2.normalize();
  eigenValue2 = d[2];

  eigenVector3[1] = a(1,3);
  eigenVector3[2] = a(2,3);
  eigenVector3[3] = a(3,3);
  eigenVector3.normalize();
  eigenValue3 = d[3];
}


} // End namespace Uintah
