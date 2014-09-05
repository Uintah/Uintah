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
#include "iotext_double.h"

int
main(int argc, char * argv[])
{

  if (argc < 6) {
      cerr << "Usage: A  x  y  A*x   A'*y   (filenames)" << endl;
    exit(-1);
  }

  int verbose = 0;
  int errcount = 0;

  if (argc > 6 )  {
    if (argv[6][1] == 'v')
      verbose = 1;
    else {
      cerr << "Usage: A  x  y  A*x   A'*y   (filenames)" << endl;
      return -1;
    }
  }

  char *A_name = argv[1];
  char *x_name = argv[2];
  char *y_name = argv[3];
  char *Ax_name  =argv[4];
  char *Aty_name = argv[5];

  CompCol_Mat_double Acol;
  CompRow_Mat_double Arow;
  Coord_Mat_double Acoord;

  VECTOR_double x, y;
  VECTOR_double Ax, Aty;

  readtxtfile_mat(A_name, &Acoord);
  Acol = Acoord;
  Arow = Acol;

  readtxtfile_vec(x_name, &x);
  readtxtfile_vec(y_name, &y);
  readtxtfile_vec(Ax_name, &Ax);
  readtxtfile_vec(Aty_name, &Aty);

  if (verbose)
  {
    cout << "Dimensons: " << endl;
    cout << " A ("<< Acoord.dim(0) << "," << Acoord.dim(1) << ") ";
    cout << " x: (" << x.size() << ")  y: (" << y.size() << ")"  << endl;
    cout << " A*x: " << Ax.size()  <<  endl;
    cout << " A'*y:"  << Aty.size()  << endl; 
  } 
  if (norm(Ax - Acol*x) > 1e-8 )
  {
    errcount++;
    if (verbose) cout << "A*x failed. (col)\n";
  }

  if (norm(Aty - Acol.trans_mult(y)) > 1e-8)
  {
    errcount++;
    if (verbose) cout << "A'*y failed. (col) \n";
  }

  if (norm(Ax - Acoord*x) > 1e-8)
  {
    errcount++;
    if (verbose) cout << "A*x failed. (coord)\n";
  }
  
  if (norm(Aty - Acoord.trans_mult(y)) > 1e-8)
  {
    errcount++;
    if (verbose) cout << "A'*y failed. (coord)\n";
  }
  
  if (norm(Ax - Arow*x) > 1e-8)
  {
    errcount++;
    if (verbose) cout << "A*x failed (row).\n";
  }
    
  if (norm(Aty - Arow.trans_mult(y)) > 1e-8)
  {
    errcount++;
    if (verbose) cout << "A'*y failed (row).\n";
  }

  if (errcount > 0) {
    cout << "There were " << errcount << " errors" << endl;
    if (verbose == 0) {
      cout << "Run again with -v to find out where" << endl;      
    }
  } else {
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "+   Successful completion of SparseLib++ * and '*      +" << endl;
    cout << "+    No errors detected in computational routines.     +" << endl;
    cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << endl << endl;
  }

  return errcount;
}
