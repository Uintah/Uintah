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

int
main(int argc, char * argv[])
{
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " HBfile [-v]" << endl;
    exit(-1);
  }

  int verbose = 0;
  int errcount = 0;

  if (argc > 2 )  {
    if (argv[2][1] == 'v')
      verbose = 1;
    else {
      cerr << "Usage: " << argv[0] << " HBfile [-v]" << endl;
      return -1;
    }
  }

  CompCol_Mat_double Ac;
  CompRow_Mat_double Ar;

  readHB_mat(argv[1], &Ac);
  Ar = Ac;
  
  int m = Ar.dim(0), n = Ar.dim(1);

  VECTOR_double b(n), x(m), c(m);
  
  for (int i = 0; i < m; i++)
    x(i) = i;

  b = Ar * x;

  CompCol_ILUPreconditioner_double Mc(Ac);
  CompRow_ILUPreconditioner_double Mr(Ar);
  ICPreconditioner_double cMc(Ac);
  ICPreconditioner_double cMr(Ar);

  if (verbose) cout << "Mc" << endl;
  VECTOR_double yc = Mc.solve(b);
  if (verbose) cout << "ILU yc-x " << norm(yc-x)/norm(x) << endl;
  if (norm(yc-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yc" << endl << yc << endl;
    }
  }

  if (verbose) cout << "Mr" << endl;
  VECTOR_double yr = Mr.solve(b);
  if (verbose) cout << "ILU yr-x " << norm(yr-x)/norm(x) << endl;
  if (norm(yr-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yr" << endl << yr << endl;
    }
  }
  
  if (verbose) cout << "cMc" << endl;
  yc = cMc.solve(b);
  if (verbose) cout << "IC yc-x " << norm(yc-x)/norm(x) << endl;
  if (norm(yc-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yc" << endl << yc << endl;
    }
  }
  
  if (verbose) cout << "cMr" << endl;
  yr = cMr.solve(b);
  if (verbose) cout << "IC yr-x " << norm(yr-x)/norm(x) << endl;
  if (norm(yr-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yr" << endl << yr << endl;
    }
  }


  c = Ar.trans_mult(x);

  if (norm(b-c)/norm(b) > 1.e-14) {
    errcount++;
    if (verbose) {
      cout << "b" << endl << b << endl;
      cout << "c" << endl << c << endl;
    }
  }

  if (verbose) cout << "Mc" << endl;
  yc = Mc.trans_solve(b);
  if (verbose) cout << "ILU yc-x " << norm(yc-x)/norm(x) << endl;
  if (norm(yc-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yc" << endl << yc << endl;
    }
  }

  if (verbose) cout << "Mr" << endl;
  yr = Mr.trans_solve(b);
  if (verbose) cout << "ILU yr-x " << norm(yr-x)/norm(x) << endl;
  if (norm(yr-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yr" << endl << yr << endl;
    }
  }

  if (verbose) cout << "cMc" << endl;
  yc = cMc.trans_solve(b);
  if (verbose) cout << "IC yc-x " << norm(yc-x)/norm(x) << endl;
  if (norm(yc-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yc" << endl << yc << endl;
    }
  }

  if (verbose) cout << "cMr" << endl;
  yr = cMr.trans_solve(b);
  if (verbose) cout << "IC yr-x " << norm(yr-x)/norm(x) << endl;
  if (norm(yr-x)/norm(x) > 1.e-12) {
    errcount++;
    if (verbose) {
      cout << "x" << endl << x << endl;
      cout << "yr" << endl << yr << endl;
    }
  }

  if (errcount > 0) {
    cout << "There were " << errcount << " errors" << endl;
    if (verbose == 0) {
      cout << "Run again with -v to find out where" << endl;      
    }
  } else {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "+   Successful completion of testing for SparseLib++   +" << endl;
    cout << "+    No errors detected in preconditioner routines.    +" << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << endl << endl;
  }

  return errcount;
}
