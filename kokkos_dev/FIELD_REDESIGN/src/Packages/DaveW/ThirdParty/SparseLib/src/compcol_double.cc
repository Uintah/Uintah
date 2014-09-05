/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*             ********   ***                                 SparseLib++    */
/*          *******  **  ***       ***      ***               v. 1.5c        */
/*           *****      ***     ******** ********                            */
/*            *****    ***     ******** ********              R. Pozo        */
/*       **  *******  ***   **   ***      ***                 K. Remington   */
/*        ********   ********                                 A. Lumsdaine   */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                                           */
/*                                                                           */
/*                     SparseLib++ : Sparse Matrix Library                   */
/*                                                                           */
/*               National Institute of Standards and Technology              */
/*                        University of Notre Dame                           */
/*              Authors: R. Pozo, K. Remington, A. Lumsdaine                 */
/*                                                                           */
/*                                 NOTICE                                    */
/*                                                                           */
/* Permission to use, copy, modify, and distribute this software and         */
/* its documentation for any purpose and without fee is hereby granted       */
/* provided that the above notice appear in all copies and supporting        */
/* documentation.                                                            */
/*                                                                           */
/* Neither the Institutions (National Institute of Standards and Technology, */
/* University of Notre Dame) nor the Authors make any representations about  */
/* the suitability of this software for any purpose.  This software is       */
/* provided ``as is'' without expressed or implied warranty.                 */
/*                                                                           */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*          Compressed column sparse matrix (0-based)                    */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#include <iostream>
#include <stdlib.h>

using namespace std;

#define _NON_TEMPLATE_COMPLEX

#include "compcol_double.h"
#include "comprow_double.h"
#include "coord_double.h"

#include "spblas.h"

/*****************************/
/*  Constructor(s)           */
/*****************************/

CompCol_Mat_double::CompCol_Mat_double(void)
        : val_(0), rowind_(0), colptr_(0), base_(0), nz_(0)
{
        dim_[0] = 0;
        dim_[1] = 0;
}

/*****************************/
/*  Copy constructor         */
/*****************************/

CompCol_Mat_double::CompCol_Mat_double(const CompCol_Mat_double &S) :
        val_(S.val_), rowind_(S.rowind_), colptr_(S.colptr_), 
        base_(S.base_), nz_(S.nz_)
{
        dim_[0] = S.dim_[0];
        dim_[1] = S.dim_[1];
}

/***********************************/
/*  Construct from storage vectors */
/***********************************/

CompCol_Mat_double::CompCol_Mat_double(int M, int N, int nz, double *val,       
                                     int *r, int *c, int base) :
        val_(val, nz), rowind_(r, nz), colptr_(c, N+1), base_(base), nz_(nz)
{
        dim_[0] = M;
        dim_[1] = N;
}

CompCol_Mat_double::CompCol_Mat_double(int M, int N, int nz, 
                    const VECTOR_double &val, const VECTOR_int &r, 
                    const VECTOR_int  &c, int base) :
        val_(val), rowind_(r), colptr_(c), base_(base), nz_(nz)
{
        dim_[0] = M;
        dim_[1] = N;
}
/**********************************************************************/ 
/*  Construct a CompCol_Mat_double from a CompRow_Mat_double            */ 
/*  (i.e. convert compressed row storage to compressed column storage)*/ 
/**********************************************************************/

CompCol_Mat_double::CompCol_Mat_double(const CompRow_Mat_double &R) :
        val_(R.NumNonzeros()), rowind_(R.NumNonzeros()), colptr_(R.dim(1) +1),
        base_(R.base()), nz_(R.NumNonzeros())
{

        dim_[0] = R.dim(0);
        dim_[1] = R.dim(1);

        int i,j;
        VECTOR_int tally(R.dim(1)+1, 0);
//      First pass through nonzeros.  Tally entries in each column.
//      And calculate colptr array.
        for (i=0;i<nz_;i++) tally(R.col_ind(i))++;
        colptr_(0) = 0;
        for (j=0;j<dim_[1];j++) colptr_(j+1) = colptr_(j)+tally(j);
//      Make copy of colptr for use in second pass.
        tally = colptr_;
//      Second pass through nonzeros.  Fill in index and value entries.
        int count = 0;
        for (i=1;i<=dim_[0];i++)
        {
           for (j=count;j<R.row_ptr(i);j++)
           {
              val_(tally(R.col_ind(j))) = R.val(j);
              rowind_(tally(R.col_ind(j))) = i-1;
              tally(R.col_ind(count))++;
              count++;
           }
        }
}

/**********************************************************************/
/*  Construct a CompCol_Mat_double from a Coord_Mat_double              */
/*  (i.e. convert coordinate storage to compressed column storage)    */
/**********************************************************************/

CompCol_Mat_double::CompCol_Mat_double(const Coord_Mat_double &CO) :
        val_(CO.NumNonzeros()), rowind_(CO.NumNonzeros()), 
        colptr_(CO.dim(1) +1), base_(CO.base()), nz_(CO.NumNonzeros()) 
{

        dim_[0] = CO.dim(0);
        dim_[1] = CO.dim(1);

        int i,j;
        VECTOR_int tally(CO.dim(1)+1, 0);
//      First pass through nonzeros.  Tally entries in each column.
//      And calculate colptr array.
        for (i=0;i<nz_;i++) tally(CO.col_ind(i))++;
        colptr_(0) = 0;
        for (j=0;j<dim_[1];j++) colptr_(j+1) = colptr_(j)+tally(j);
//      Make copy of colptr for use in second pass.
        tally = colptr_;
//      Second pass through nonzeros.   Fill in index and value entries.
        for (i=0;i<nz_;i++)
        {
           val_(tally(CO.col_ind(i))) = CO.val(i);
           rowind_(tally(CO.col_ind(i))) = CO.row_ind(i);
           tally(CO.col_ind(i))++;
        }
}

/***************************/
/* Assignment operator...  */
/***************************/

CompCol_Mat_double& CompCol_Mat_double::operator=(const CompCol_Mat_double &C)  
{
        dim_[0] = C.dim_[0];
        dim_[1] = C.dim_[1];
        base_   = C.base_;
        nz_     = C.nz_;
        val_    = C.val_; 
        rowind_ = C.rowind_; 
        colptr_ = C.colptr_; 
        return *this;
}

/***************************/
/* newsize()               */
/***************************/

CompCol_Mat_double& CompCol_Mat_double::newsize(int M, int N, int nz)
{
        dim_[0] = M;
        dim_[1] = N;

        nz_ = nz;
        val_.newsize(nz);
        rowind_.newsize(nz);
        colptr_.newsize(N+1);
        return *this;
}

/*********************/
/*   Array access    */
/*********************/

double CompCol_Mat_double::operator()(int i, int j)  const
{
        for (int t=colptr_(j); t<colptr_(j+1); t++)
           if (rowind_(t) == i) return val_(t);
        if (i < dim_[0] && j < dim_[1]) return 0.0;
        else 
        {  
            cerr << "Array accessing exception -- out of bounds." << endl;
            exit(1);
            return (0);   // return to suppress compiler warning message
        }
}

double& CompCol_Mat_double::set(int i, int j)
{        
        for (int t=colptr_(j); t<colptr_(j+1); t++)
           if (rowind_(t) == i) return val_(t);
        cerr << "Array element (" << i << "," << j ;
        cerr << ") not in sparse structure -- cannot assign." << endl;
        exit(1);
    return val_(0);   // return to suppress compiler warning message
}


/*************/
/*   I/O     */
/*************/

ostream& operator << (ostream & os, const CompCol_Mat_double & mat)
{
        int M = mat.dim(0);
        int N = mat.dim(1);
        int rowp1, colp1; 
        int flag = 0;
        long olda = os.setf(ios::right,ios::adjustfield);
        long oldf = os.setf(ios::scientific,ios::floatfield);
        int oldp = os.precision(12);

//      Loop through columns
        for (int j = 0; j < N ; j++) 
           for (int i=mat.col_ptr(j);i<mat.col_ptr(j+1);i++)
           {
              rowp1 = mat.row_ind(i)+1;
              colp1 = j + 1;
              if ( rowp1 == M && colp1 == N ) flag = 1;
              os.width(14);
              os <<  rowp1 ; os << "    " ;
              os.width(14);
              os <<  colp1 ; os << "    " ;
              os.width(20);
              os <<  mat.val(i) << "\n";
           }

        if (flag == 0)
        {
           os.width(14);
           os <<  M ; os << "    " ;
           os.width(14);
           os <<  N ; os << "    " ;
           os.width(20);
           os <<  mat(M-1,N-1) << "\n";
        }
        os.setf(olda,ios::adjustfield);
        os.setf(oldf,ios::floatfield);
        os.precision(oldp);

        return os;
}


/***************************************/
/* Matrix-Vector multiplication...  */
/***************************************/

VECTOR_double CompCol_Mat_double::operator*(const VECTOR_double &x) 
        const
{
        int M = dim_[0];
        int N = dim_[1];

//      Check for compatible dimensions:
        if (x.size() != N) 
        {
           cerr << "Error in CompCol Matvec -- incompatible dimensions." 
                << endl;
           exit(1);
           return x;
        }

        VECTOR_double result(M, 0.0);
        VECTOR_double work(M);
  
        int descra[9];
        descra[0] = 0;
        descra[1] = 0;
        descra[2] = 0;

        F77NAME(dcscmm) (0, M, 1, N, 1.0,
                 descra, &val_(0), &rowind_(0), &colptr_(0),
                 &x(0), N, 1.0, &result(1), M,
                 &work(1), M);

        return result;
}

/**********************************************/
/* Matrix-Transpose-Vector multiplication...  */
/**********************************************/

VECTOR_double CompCol_Mat_double::trans_mult(const VECTOR_double &x)  
        const
{
        int M = dim_[0];
        int N = dim_[1];

//      Check for compatible dimensions:
        if (x.size() != M) 
        {
          cerr << "Error in CompCol TransMatvec -- incompatible dimensions." 
               << endl;
          exit(1);
          return x;
        }

        VECTOR_double result(N, 0.0);
        VECTOR_double work(N);
  
        int descra[9];
        descra[0] = 0;
        descra[1] = 0;
        descra[2] = 0;

        F77NAME(dcscmm) (1, N, 1, M, 1.0,
                     descra, &val_(0), &rowind_(0), &colptr_(0),
                 &x(1), M, 1.0, &result(0), N,
                 &work(0), N);

        return result;
}
       
