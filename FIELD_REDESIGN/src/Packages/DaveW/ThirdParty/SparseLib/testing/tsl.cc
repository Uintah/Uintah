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

#include <stdlib.h>

#include "compcol_double.h"
#include "comprow_double.h"
#include "coord_double.h"
#include "iohb_double.h"

main(int argc, char * argv[])
{

        if (argc < 2)
        {
          cerr << "Usage: " << argv[0] << " HBfile [-v]" << endl;
          exit(-1);
        }
        int verbose = 0;
        if (argc > 2 ) 
        {
          if (argv[2][1] == 'v') verbose = 1;
          else
          {
            cerr << "Usage: " << argv[0] << " HBfile [-v]" << endl;
            exit(-1);
          }
        }

/******************************************************************/
//      Testing readHB functions...                               //
/******************************************************************/
        if (verbose) cout << "Testing readHB functions:" << endl << endl;
        CompCol_Mat_double A1;
        int M, N, nonzeros, nrhs; 
        readHB_info(argv[1], &M, &N, &nonzeros, &nrhs);
        if (verbose) 
        {
          cout << "Return values from readHB_info:" << endl;
          cout << "M = " << M << " N = " << N ;
          cout << " Nonzeros = " << nonzeros << " Nrhs = " << nrhs << endl;
          cout << "Output from readHB_info in verbose mode: " << endl;
          cout << "......................................................"<<endl;
          readHB_info(argv[1],  &M, &N, &nonzeros, &nrhs);
          cout << "......................................................"<<endl
               << endl;
        
          cout << "Reading the matrix from " << argv[1] << "..." << endl;
        }
        readHB_mat(argv[1], &A1);
        if ( nrhs > 0)
        {
        if (verbose) 
          cout << "Reading a rhs from " << argv[1] << "..." << endl << endl;
          VECTOR_double b(N); 
          readHB_rhs(argv[1], &b);
        }

/******************************************************************/
//      Generate small test matrix for testing conversions...     //
//                                                                //
//                   [    1   2  0   0   3   ]                    //
//                                                                //
//                   |    4   5  6   0   0   |                    //
//                                                                //
//                   |    0   7  8   0   9   |                    //
//                                                                //
//                   |    0   0  0  10   0   |                    //
//                                                                //
//                   [   11   0  0   0  12   ]                    //
//                                                                //
/******************************************************************/

        double val[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 
                          7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        int colind[12] = {0, 1, 4, 0, 1, 2, 1, 2, 4, 3, 0, 4};
        int rowptr[6]  = { 0,      3,       6,    9,10,   12};
        double rowsum[5] = {6.0, 15.0, 24.0, 10.0, 23.0};
        double colsum[5] = {16.0, 14.0, 14.0, 10.0, 24.0};

        CompRow_Mat_double R(5,5,12,val,rowptr,colind);

        if (verbose) 
        {
          cout << "Generated row-oriented matrix from data." << endl;
          cout << "Converting: Row to Column... " << endl;
        }
        CompCol_Mat_double C(R);
        if (verbose) cout << "            Column to Coord... " << endl;
        Coord_Mat_double CO1(C);
        if (verbose) cout << "            Coord to Row... " << endl;
        CompRow_Mat_double R2(CO1);
  
        double err = 0.0;
        int i,j;
        for (i=0;i<5;i++)
          for (j=0;j<5;j++)
            if ( (R2(i,j) - R(i,j)) < 0) err -= R2(i,j) - R(i,j);
            else  err += R2(i,j) - R(i,j);

        if ( err > 1.e-8 )
        {
          cout << "Error in conversions too high. Halting execution. " << endl;
          exit(1);
        }
        if (verbose) 
        {
          cout << "Accumulated error = " << err << "          success. " << endl;
          cout << "Reverse direction..." << endl;
          cout << "Converting: Row to Coord... " << endl;
        }

        Coord_Mat_double CO2(R);
        if (verbose) cout << "            Coord to Column... " << endl;
        CompCol_Mat_double C2(CO2);
        if (verbose) cout << "            Column to Row... " << endl;
        CompRow_Mat_double R3(C2);

        err = 0.0;
        for (i=0;i<5;i++)
          for (j=0;j<5;j++)
            if ( (R3(i,j) - R(i,j)) < 0) err -= (R3(i,j) - R(i,j));
            else err += (R3(i,j) - R(i,j));
        if ( err > 1.e-8 )
        {
          cout << "Error in conversions too high. Halting execution. " << endl;
          exit(1);
        }
        if (verbose) 
        {
          cout << "Accumulated error = " << err << "          success. " << endl;
          cout << endl;
          cout << "Testing sparse matrix - dense vector multiplies... " << endl;
          cout << "Small test matrix:                                 " << endl;
        }

        VECTOR_double x(5,0.0);
        VECTOR_double v(5,1.0);

        if (verbose) cout << "Mat-vec (CompCol)...           ";
        x = C*v;
        for (i=0;i<5;i++) 
          if (x(i) != rowsum[i]) 
          {
            cout << "Mat-vec (Compcol) error." <<endl;
            exit(1);
          }
        if (verbose) cout << "success." << endl;

        x = 0.0;
        if (verbose) cout << "Mat-trans_mult-vec (CompCol)...";
        x = C.trans_mult(v);
        for (i=0;i<5;i++) 
          if (x(i) != colsum[i]) 
          {
             cout << "Mat-trans_mult-vec (CompCol) error." <<endl;
             exit(1);
           }
        if (verbose) cout << "success." << endl;

        x = 0.0;
        if (verbose) cout << "Mat-vec (CompRow)...           ";
        x = R*v;
        for (i=0;i<5;i++) 
          if (x(i) != rowsum[i]) 
          {
             cout << "Mat-vec (CompRow) error." <<endl;
             exit(1);
          }
        if (verbose) cout << "success." << endl;

        x = 0.0;
        if (verbose) cout << "Mat-trans_mult-vec (CompRow)...";
        x = R.trans_mult(v);
        for (i=0;i<5;i++) 
          if (x(i) != colsum[i]) 
          {
             cout << "Mat-trans_mult-vec (CompRow) error." <<endl;
             exit(1);
          }
        if (verbose) cout << "success." << endl;

        x = 0.0;
        if (verbose) cout << "Mat-vec (Coord)...             ";
        x = CO1*v;
        for (i=0;i<5;i++) 
          if (x(i) != rowsum[i]) 
          {
             cout << "Mat-vec (Coord) error." <<endl;
             exit(1);
          }
        if (verbose) cout << "success." << endl;

        x = 0.0;
        if (verbose) cout << "Mat-trans_mult-vec (Coord)...  ";
        x = CO1.trans_mult(v);
        for (i=0;i<5;i++) 
          if (x(i) != colsum[i]) 
          {
             cout << "Mat-trans_mult-vec (Coord) error." <<endl;
             exit(1);
          }
        if (verbose) 
        {
          cout << "success." << endl;
          cout << endl;
          cout << "Testing sparse matrix - dense vector multiplies... " << endl;
          cout << "Matrix from Harwell-Boeing file:                   " << endl;
        }

        VECTOR_double x1(M,0.0);
        VECTOR_double v1(N,1.0);

        if (verbose) cout << "Mat-vec (CompCol)...           "<< endl;
        x1= A1*v1;

        VECTOR_double x2(M,0.0);
        CompRow_Mat_double A2(A1);
        if (verbose) cout << "Mat-vec (CompRow)...           "<< endl;
        x2 = A2*v1;
       
        VECTOR_double x3(M,0.0);
        Coord_Mat_double A3(A2);
        if (verbose) cout << "Mat-vec (Coord)...             "<< endl;
        x3 = A3*v1;

        if (verbose) cout << "Comparing results...           ";

        err = 0.0;
        for (i=0;i<M;i++)
        {
          if ( (x1(i) - x2(i)) < 0 ) err -= (x1(i) - x2(i));
          else err += (x1(i) - x2(i));
        }
        if ( err > 1.e-5 ) 
        {
          cout << endl << "Error in Matvecs on HB generated matrix.  " << endl;
          exit(1);
        } 

        err = 0.0;
        for (i=0;i<M;i++)
        {
          if ( (x2(i) - x3(i)) < 0 ) err -= (x2(i) - x3(i));
          else err += (x2(i) - x3(i));
        }
        if ( err > 1.e-5 ) 
        {
          cout << endl << "Error in Matvecs on HB generated matrix.  " << endl;
          exit(1);
        }
        if (verbose) cout << "success. " << endl;

        v1 = 0.0;
        x1 = 1.0;
        if (verbose) cout << "Mat-trans-vec (CompCol)...           "<< endl;
        v1 = A1.trans_mult(x1);
        
        VECTOR_double v2(N,0.0);
        if (verbose) cout << "Mat-trans-vec (CompRow)...           "<< endl;
        v2 = A2.trans_mult(x1);

        VECTOR_double v3(N,0.0);
        if (verbose) cout << "Mat-trans-vec (Coord)...             "<< endl;
        v3 = A3.trans_mult(x1);

        if (verbose) cout << "Comparing results...           ";

        err = 0.0;
        for (i=0;i<N;i++)
        {
          if ( (v1(i) - v2(i)) < 0 ) err -= (v1(i) - v2(i));
          else err += (v1(i) - v2(i));
        }
        if ( err > 1.e-5 ) 
        {
          cout << endl << "Error in Mat-tran_mult-vecs on HB generated matrix.  " 
               << endl;
          exit(1);
        }

        err = 0.0;
        for (i=0;i<N;i++)
        {
          if ( (v2(i) - v3(i)) < 0 ) err -= (v2(i) - v3(i));
          else err += (v2(i) - v3(i));
        }
        if ( err > 1.e-5 ) 
        {
          cout << endl << "Error in Mat-tran_mult-vecs on HB generated matrix.  " 
               << endl;
          exit(1);
        }
        if (verbose) cout << "success. " << endl;

        cout << endl << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << "+    Successful completion of testing for SparseLib++   +" << endl;
        cout << "+   No errors detected in conversion or blas routines.  +" << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << endl << endl;
}
