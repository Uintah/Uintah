/****************************************************************
 * Protozoa module (improved simplex search)for the SCIRun      *
 *                                                              *
 *  Written by:                                                 *
 *   Kris Zyp                                                   *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   Sept 2000                                                  *
 *                                                              *
 *  Copyright (C) 2000 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/
#include <math.h>
#include <iostream>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#define NRANSI
using namespace std;
using namespace SCICore::Containers;
#define GET_PSUM \
for (j=1;j<=ndim;j++) {\
   for (sum=0.0,k=1;k<=mpts;k++) sum += guesses(j,k);\
      psum[j]=sum;}
#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}
const CREATE_EARLY_GUESSES = 1;
const USE_CROSS_TERMS = 1;
const MIN_LEAST_SQUARES_EXTRAS = 1;
const MAX_LEAST_SQUARES_EXTRAS = 10;
double errCorrect(double initialError)
    // we are squaring the error so that it is linear with the sum of sqares.  We are inverting the square because we want hlower values to be of more importance
{
    return 1/initialError*initialError;
}
void sort(Array2<double>* guesses, Array1<double>* errors, int cols, int rows, int fullsort)
{
    double swap;
    int i,j,k;
    if (fullsort == 0) fullsort = rows;
    for (i=rows;i>=fullsort;i--)
	for (j=1;j<i;j++)
	    if ((*errors)[j] > (*errors)[i]) {
		SWAP((*errors)[j],(*errors)[i])
		    for (k=1;k<=cols;k++) SWAP((*guesses)(k,j),(*guesses)(k,i))    
							}
}
void solveMatrix(Array2<double>* A, Array1<double>* x, int cols, int rows)
{
    int i, j,k,maxrow;
    double tmp;
    Array2<double> newA(cols+2,cols+1);
    



    // if it rectangular (doing least squares method) then make A = (A'*A) and b = A'*b
    if (cols != rows) { 
	// first do a weighting scheme that gives emphasis to points closer to the solution
	for (j=1;j<=rows;j++)
	    for (i=1;i<=cols+1;i++)
		(*A)(i,j) = (*A)(i,j) / (*A)(cols+1,j);

	// now multiply everything by A' (A' means transpose of A)

	for (i=1;i<=cols;i++)
	    for (j=1;j<=cols;j++){
		newA (i,j) = 0;
		for (k=1;k<=rows;k++)
		    newA(i,j) += newA(i,j) + (*A)(i,k) * (*A)(j,k);
	    }
	for (i=1;i<=cols;i++) {
	    newA(cols+1,i) = 0;
	    for (j=1;j<=rows;j++)
		newA(cols+1,i) += newA(cols+1,i) + (*A)(i,j)* (*A)(cols+1,j);
	}
	A = &newA;
    }
    


    // do the gaussian elimination
   for (i=1;i<=cols;i++) {

      /* Find the rows with the largest first value */
      maxrow = i;
      for (j=i+1;j<=cols;j++) {
         if (abs((*A)(i,j)) > abs((*A)(i,maxrow)))
            maxrow = j;
      }

      /* Swap the maxrow and ith row */
      for (k=i;k<=cols+1;k++) {
         tmp = (*A)(k,i);
         (*A)(k,i) = (*A)(k,maxrow);
         (*A)(k,maxrow) = tmp;
      }

      /* Singular matrix? */
      //    if (abs((*A)(i,i)) < 0.00000000001)
	  //  cerr << "urghh... This matrix isn't looking very healthy" << endl;

      /* Eliminate the ith element of the jth row */
      for (j=i+1;j<=cols;j++) {
         for (k=cols+1;k>=i;k--) {
            (*A)(k,j) -= (*A)(k,i) * (*A)(i,j) / (*A)(i,i);
         }
      }
   }

   /* Do the back substitution */
   for (j=cols;j>=1;j--) {
      tmp = 0;
      for (k=j+1;k<=cols;k++)
         tmp += (*A)(k,j) * (*x)[k];
      (*x)[j] = ((*A)(cols+1,j) - tmp) / (*A)(j,j);
   }

}


void protozoa(double **p, double y[], int ndim, double ftol,
	    double *(*funk)(int), int *nfunk, int extra)
{
    int squareTermSize = 2*ndim+1;  // size of the matrix with x, x^2, etc...
    int crossTermSize = squareTermSize+ndim*(ndim-1)/2; // size of the matrix with x, x^2, xy, etc...
    int i,k,ihi,ilo,inhi,j,iteration,mpts=crossTermSize;
    double rtol,sum,swap,errSum,errRef;
    Array1<double> x(mpts+1);
    Array2<double> A(mpts+2,mpts+1+MAX_LEAST_SQUARES_EXTRAS);
    Array2<double> guesses(ndim+extra+1,mpts+1+MAX_LEAST_SQUARES_EXTRAS);
    Array1<double> errors(mpts+1+MAX_LEAST_SQUARES_EXTRAS);
    
    Array1<double> psum(mpts+1);
    Array1<int> indx(mpts+1);
    
    int NMAX=*nfunk;
    *nfunk=0;
    cerr << "Starting Protozoa\n";

    // fill the internal array of guesses
    for (i=1;i<=(ndim+2);i++) {
	errors[i] = y[i];
	for (j=1;j<=(ndim+extra);j++)
	    guesses(j,i) = p[i][j];
    }
    sort(&guesses,&errors,ndim+extra,ndim+1,1);

    cerr << "starting on guessing" << endl;
    if (CREATE_EARLY_GUESSES) {
	
	// the first guess is a weighted average guess
	iteration = ndim+2;
	errRef = 1/y[ndim]/y[ndim]/2 + 1/y[ndim-1]/y[ndim-1] / 2;  // rather abritrary, but it is kind of mean/median value
	errSum = 0;
	for (i=1;i<=iteration-1;i++) 
	    errSum += 1/errors[i]/errors[i] -errRef;
	
	for (j=1;j<=ndim;j++) {
	    guesses(j,iteration) = 0;	
	    for(i=1;i<=iteration-1;i++)
		guesses(j,iteration) += guesses(j,i) * (1/errors[i]/errors[i] - errRef) / errSum;
	    cerr << " " << guesses(j,iteration);
	}
   // put the values back in the outside matrix
   for(i=1;i<=ndim+extra;i++)
       p[ndim+2][i] = guesses(i,iteration);

   // calculate error
   errors[iteration] = *((*funk)(ndim+2));
   cerr << " " << errors[iteration] << endl;
   
   sort(&guesses,&errors,ndim+extra,iteration,0);
   
   // check to see if we are done
   rtol=2.0*fabs(errors[iteration]-errors[iteration-ndim-1])/(fabs(errors[iteration])+fabs(errors[iteration-ndim-1]));

   if (rtol < ftol) {
       y[1] = errors[iteration];
       for(i=1;i<=ndim+extra;i++)
	   p[1][i] = guesses(i,iteration);
       return;	
   }
   
   *nfunk += 1;

    
    }
    iteration = ndim+3;

    // circular parabolic fit (no crossterms
    cerr << "circular parabolic fitting\n";
    for (;iteration<=squareTermSize+MIN_LEAST_SQUARES_EXTRAS;iteration++) {

	    //    cerr << "assign the matrix\n";

	    // This parts creates the matrix of of ax^2 + bx + cy^2 + dy +... + 1 = error
	    mpts = iteration-1;
	    for (i=1;i<=mpts;i++) {
		A(ndim+2,i)=1;
		A(ndim+3,i)= errors[i]*errors[i];
		A(1+ndim,i) = 0;
		for (j=1;j<=ndim;j++){ 
		    A(j,i)= guesses(j,i);
		    A(1+ndim,i) += guesses(j,i)*guesses(j,i);
		}
	    }
	    //    cerr << "invert the matrix\n";
	
	    solveMatrix(&A,&x,ndim+2,mpts);
	    
   // so you have the matrix solution.  Now to find the minimum value for the quadratic equation you take the derivative of ax^2 + bx (for each dimension x,y,z) and set it to zero so x = -b/2a
   for (i=1;i<=ndim;i++){
       if (x[1+ndim] <=  0) {  // it is convex instead of concave
	   cerr << "convexity detected on axis " << i << endl;
	   GET_PSUM
	   guesses(i,iteration) = psum[i]/mpts;
       }
       else
	   guesses(i,iteration) = -x[i]/(2*x[1+ndim]);	
       cerr << " " << guesses(i,iteration);
   }	

   // put the values back in the outside matrix
   for(i=1;i<=ndim+extra;i++)
       p[ndim+2][i] = guesses(i,iteration);

   // calculate error
   errors[iteration] = *((*funk)(ndim+2));
   cerr << " " << errors[iteration] << endl;
   
   sort(&guesses,&errors,ndim+extra,iteration,0);
   
   // check to see if we are done
   rtol=2.0*fabs(errors[iteration]-errors[iteration-ndim-1])/(fabs(errors[iteration])+fabs(errors[iteration-ndim-1]));

   if (rtol < ftol) {
       y[1] = errors[iteration];
       for(i=1;i<=ndim+extra;i++)
	   p[1][i] = guesses(i,iteration);
       return;	
   }
   

   *nfunk += 1;

	}



    // Do the quadratic 7x7 matrix guesses (no cross terms)
    cerr << "simple quadratic fitting\n";

    for (;iteration<=crossTermSize+MIN_LEAST_SQUARES_EXTRAS;iteration++) {

	    //    cerr << "assign the matrix\n";

	    // This parts creates the matrix of of ax^2 + bx + cy^2 + dy +... + 1 = error
	    mpts = iteration-1;
	    for (i=1;i<=mpts;i++) {
		A(squareTermSize,i)=1;
		A(squareTermSize+1,i)= errors[i]*errors[i];
		for (j=1;j<=ndim;j++) {
		    A(j,i)= guesses(j,i);
		    A(j+ndim,i) = guesses(j,i)*guesses(j,i);
		}
	    }
	    //    cerr << "invert the matrix\n";
	
	    solveMatrix(&A,&x,squareTermSize,mpts);

	    
   // so you have the matrix solution.  Now to find the minimum value for the quadratic equation you take the derivative of ax^2 + bx (for each dimension x,y,z) and set it to zero so x = -b/2a
   for (i=1;i<=ndim;i++){
       if (x[i+ndim] <=  0) {  // it is convex instead of concave
	   cerr << "convexity detected on axis " << i << endl;
	   GET_PSUM
	   guesses(i,iteration) = psum[i]/mpts;
       }
       else
	   guesses(i,iteration) = -x[i]/(2*x[i+ndim]);	
       cerr << " " << guesses(i,iteration);
   }	

   // put the values back in the outside matrix
   for(i=1;i<=ndim+extra;i++)
       p[ndim+2][i] = guesses(i,iteration);

   // calculate error
   errors[iteration] = *((*funk)(ndim+2));
   cerr << " " << errors[iteration] << endl;
   
   sort(&guesses,&errors,ndim+extra,iteration,0);
   
   // check to see if we are done
   rtol=2.0*fabs(errors[iteration]-errors[iteration-ndim-1])/(fabs(errors[iteration])+fabs(errors[iteration-ndim-1]));

   if (rtol < ftol) {
       y[1] = errors[iteration];
       for(i=1;i<=ndim+extra;i++)
	   p[1][i] = guesses(i,iteration);
       return;	
   }
   
   *nfunk += 1;

	}


	// do the quadratic fit 10x10 matrix guesses (cross terms included)
    cerr << "cross term fitting\n";
	for (;;) {
	    //    cerr << "assign the matrix\n";

	    // This parts creates the matrix of of ax^2 + bx + cy^2 + dy +... + 1 = error
	    int crossTermCounter = squareTermSize;
	    mpts = iteration-1;
	    for (i=1;i<=iteration-1;i++) {
		A(crossTermSize,i)=1;
		A(crossTermSize+1,i)= errors[i]*errors[i];
		crossTermCounter = squareTermSize;
		for (j=1;j<=ndim;j++) {
		    A(j,i)= guesses(j,i);
		    A(j+ndim,i) = guesses(j,i)*guesses(j,i);
		    for (k=j+1;k<=ndim;k++) {
			A(crossTermCounter++,i) = guesses(j,i)*guesses(k,i);
		    }
		}
	    }
	    //    cerr << "invert the matrix\n";
	
	    //do some gaussian elimination:
   solveMatrix(&A,&x,crossTermSize,iteration-1);
   // so you have the matrix solution.  Now to find the minimum value for the quadratic equation you take the derivatives of Dx^2 + Ax + Gxy + Ixz and so on.. Then you use the following equation to solve:
   A(1,1) = 2*x[4];
   A(2,1) = x[7];
   A(3,1) = x[8];
   A(1,2) = x[7];
   A(2,2) = 2*x[5];
   A(3,2) = x[9];
   A(1,3) = x[8];
   A(2,3) = x[9];
   A(3,3) = 2*x[6];
   A(4,1) = -x[1];
   A(4,2) = -x[2];
   A(4,3) = -x[3];
   solveMatrix(&A,&x,3,3);
   

   for (i=1;i<=ndim;i++){
       if (x[i+ndim] <=  0) {  // it is convex instead of concave
	   cerr << " convexity detected on axis " << i << " ";
	   GET_PSUM
	   guesses(i,iteration) = psum[i]/mpts;
       }
       else
	   guesses(i,iteration) = x[i];	
       cerr << " " << guesses(i,iteration);
   }
	    //	    cerr << "ihi " << ihi << endl;

   // put the values back in the outside matrix
   for(i=1;i<=ndim+extra;i++)
       p[ndim+2][i] = guesses(i,iteration);

   // calculate error
   errors[iteration] = *((*funk)(ndim+2));
   cerr << " " << errors[iteration] << endl;
   
   sort(&guesses,&errors,ndim+extra,iteration,0);
   
   // check to see if we are done
   rtol=2.0*fabs(errors[iteration]-errors[iteration-ndim-1])/(fabs(errors[iteration])+fabs(errors[iteration-ndim-1]));

   if (rtol < ftol) {
       y[1] = errors[iteration];
       for(i=1;i<=ndim+extra;i++)
	   p[1][i] = guesses(i,iteration);
       break;	
   }
   
   if (*nfunk >= NMAX) {
       cerr << "NMAX exceeded" << endl;
       break;
   }

   *nfunk += 1;
   if (iteration < crossTermSize+MAX_LEAST_SQUARES_EXTRAS) iteration++;
}
}


#undef NRANSI

#undef SWAP
#undef GET_PSUM
