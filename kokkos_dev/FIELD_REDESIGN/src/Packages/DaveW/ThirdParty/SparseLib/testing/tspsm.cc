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
#include "ilupre_double.h"
#include "icpre_double.h"
#include "iohb_double.h"
#include "spblas.h"

int
main()
{
/******************************************************************/
//                                                                //
//                   [    1   0  0   0   0   ]                    //
//                                                                //
//                   |    4   5  0   0   0   |                    //
//                                                                //
//                   |    0   7  8   0   0   |                    //
//                                                                //
//                   |    0   0  9  10   0   |                    //
//                                                                //
//                   [   11   0  0   0  12   ]                    //
//                                                                //
/******************************************************************/

    int verbose =0;

  if (verbose) cout << "Testing Sparse BLAS" << endl;

  int errcount = 0;

  double val[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  int colind[] = {0,   0,   1,   1,   2,   2,   3,    0,    4   };
  int rowptr[] = {0,   1,        3,        5,         7,           9};

  CompRow_Mat_double Ar(5,5,9,val,rowptr,colind);
  int m = Ar.dim(0), n = Ar.dim(1);
  
  VECTOR_double b(n), x(m), y(m);
  
  for (int i = 0; i < m; i++)
    x(i) = i;
  
  b = Ar * x;

  CompCol_Mat_double Ac;
  Ac = Ar;

  int descra[9];
  descra[0] = 0;

  // lower diag
  descra[1] = 1;
  descra[2] = 0;

  y = b;
  F77NAME(dcsrsm) (0, m, 1, 1, NULL, 1.0,
           descra, val, colind, rowptr,
           &y(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // upper diag
  descra[1] = 2;
  descra[2] = 0;

  y = b;
  F77NAME(dcscsm) (1, m, 1, 1, NULL, 1.0,
           descra, val, colind, rowptr,
           &y(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  y = b;
  F77NAME(dcsrsm) (1, m, 1, 1, NULL, 1.0,
           descra, &Ac.val(0), &Ac.row_ind(0), &Ac.col_ptr(0),
           &y(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // lower diag
  descra[1] = 1;
  descra[2] = 0;

  y = b;
  F77NAME(dcscsm) (0, m, 1, 1, NULL, 1.0,
           descra, &Ac.val(0), &Ac.row_ind(0), &Ac.col_ptr(0),
           &y(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }


/******************************************************************/
//                                                                //
//                   [    1   0  0   0   0   ]                    //
//                                                                //
//                   |    4   1  0   0   0   |                    //
//                                                                //
//                   |    0   7  1   0   0   |                    //
//                                                                //
//                   |    0   0  9   1   0   |                    //
//                                                                //
//                   [    0   0  0   0   1   ]                    //
//                                                                //
/******************************************************************/

  double uval[] = {1.0, 4.0, 1.0, 7.0, 1.0, 9.0, 1.0, 1.0};
  int ucolind[] = {0,   0,   1,   1,   2,   2,   3,    4};
  int urowptr[] = {0,   1,        3,        5,         7,    8};
  double xval[] = {4.0, 7.0, 9.0 };
  int xcolind[] = {0,   1,   2   };  
  int xrowptr[] = {0,   0,   1,  2, 3, 3 };

  CompRow_Mat_double uAr(5,5,8,uval,urowptr,ucolind);
  CompRow_Mat_double xAr(5,5,3,xval,xrowptr,xcolind);
  
  b = uAr * x;

  CompCol_Mat_double xAc;
  xAc = xAr;

  descra[0] = 0;

  // lower unit
  descra[1] = 1;
  descra[2] = 1;

  F77NAME(dcsrsm) (0, m, 1, 1, NULL, 1.0,
           descra, xval, xcolind, xrowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  F77NAME(dcscsm) (1, m, 1, 1, NULL, 1.0,
           descra, xval, xcolind, xrowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // upper unit
  descra[1] = 2;
  descra[2] = 1;

  F77NAME(dcsrsm) (1, m, 1, 1, NULL, 1.0,
           descra, &xAc.val(0), &xAc.row_ind(0), &xAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // lower unit
  descra[1] = 1;
  descra[2] = 1;

  F77NAME(dcscsm) (0, m, 1, 1, NULL, 1.0,
           descra, &xAc.val(0), &xAc.row_ind(0), &xAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }


/******************************************************************/
//                                                                //
//                   [    1   4   0   0   0   ]                    //
//                                                                //
//                   |    0   5   7   0   0   |                    //
//                                                                //
//                   |    0   0   8   9   0   |                    //
//                                                                //
//                   |    0   0   0  10   0   |                    //
//                                                                //
//                   [    0   0   0   0  12   ]                    //
//                                                                //
/******************************************************************/

  double yval[] = {1.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0};
  int ycolind[] = {0,   1,   1,   2,   2,   3,   3,    4};
  int yrowptr[] = {0,        2,        4,        6,    7,    8};


  CompRow_Mat_double yAr(5,5,8,yval,yrowptr,ycolind);
  b = yAr * x;

  CompCol_Mat_double yAc;
  yAc = yAr;

  // upper diag
  descra[1] = 2;
  descra[2] = 0;

  F77NAME(dcsrsm) (0, m, 1, 1, NULL, 1.0,
           descra, yval, ycolind, yrowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }


  // lower diag
  descra[1] = 1;
  descra[2] = 0;

  F77NAME(dcscsm) (1, m, 1, 1, NULL, 1.0,
           descra, yval, ycolind, yrowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  F77NAME(dcsrsm) (1, m, 1, 1, NULL, 1.0,
           descra, &yAc.val(0), &yAc.row_ind(0), &yAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }


  // lower diag
  descra[1] = 2;
  descra[2] = 0;

  F77NAME(dcscsm) (0, m, 1, 1, NULL, 1.0,
           descra, &yAc.val(0), &yAc.row_ind(0), &yAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }


/******************************************************************/
//                                                                //
//                   [    1   4   0   0   0   ]                    //
//                                                                //
//                   |    0   1   7   0   0   |                    //
//                                                                //
//                   |    0   0   1   9   0   |                    //
//                                                                //
//                   |    0   0   0   1   0   |                    //
//                                                                //
//                   [    0   0   0   0   1   ]                    //
//                                                                //
/******************************************************************/

  double zval[] = {1.0, 4.0, 1.0, 7.0, 1.0, 9.0, 1.0, 1.0};
  int zcolind[] = {0,   1,   1,   2,   2,   3,   3,    4};
  int zrowptr[] = {0,        2,        4,        6,    7,    8};
  double aval[] = {4.0, 7.0, 9.0 };
  int acolind[] = {1,   2,   3   };  
  int arowptr[] = {0,   1,  2, 3, 3, 3 };


  CompRow_Mat_double zAr(5,5,8,zval,zrowptr,zcolind);
  CompRow_Mat_double aAr(5,5,3,aval,arowptr,acolind);
  
  b = zAr * x;

  CompCol_Mat_double aAc;
  aAc = aAr;

  descra[0] = 0;

  // upper unit
  descra[1] = 2;
  descra[2] = 1;

  F77NAME(dcsrsm) (0, m, 1, 1, NULL, 1.0,
           descra, aval, acolind, arowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  F77NAME(dcscsm) (1, m, 1, 1, NULL, 1.0,
           descra, aval, acolind, arowptr,
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // lower unit
  descra[1] = 1;
  descra[2] = 1;

  F77NAME(dcsrsm) (1, m, 1, 1, NULL, 1.0,
           descra, &aAc.val(0), &aAc.row_ind(0), &aAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSRSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  // upper unit
  descra[1] = 2;
  descra[2] = 1;

  F77NAME(dcscsm) (0, m, 1, 1, NULL, 1.0,
           descra, &aAc.val(0), &aAc.row_ind(0), &aAc.col_ptr(0),
           &b(0), m, 0.0, &y(1), m,
           NULL, 0);

  if (verbose) cout << "CSCSM y-x " << norm(y-x) << endl;

  if (norm(y-x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "x" << endl << x << endl;
      cout << "y" << endl << y << endl;
    }
  }

  if (errcount > 0) {
    cout << "There were " << errcount << "errors" << endl;
    if (verbose == 0) {
      cout << "Run again with -v to find out where" << endl;      
    }
  } else {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "+   Successful completion of testing for SparseLib++   +" << endl;
    cout << "+      No errors detected in Sparse BLAS routines.     +" << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << endl << endl;
  }

  return errcount;
}
