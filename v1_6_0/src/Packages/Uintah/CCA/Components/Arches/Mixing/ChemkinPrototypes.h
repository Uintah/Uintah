//----- ChemkinPrototypes.h -----------------------------------------------

#ifndef Uintah_Component_Arches_Mixing_ChemkinPrototypes_h
#define Uintah_Component_Arches_Mixing_ChemkinPrototypes_h

  /****************************************************************************
   CLASS
      None
      
   GENERAL INFORMATION
      ChemkinPrototypes.h - Header file to prototype Chemkin routines

      Author: Jennifer Spinti (spinti@crsim.utah.edu) & Rajesh Rawat

      Creation Date: 20 October 1999
 
      C-SAFE

      Copyright U of U 1999

   KEYWORDS
      Chemkin, Reaction_Model

   DESCRIPTION

   PATTERNS
      None

   WARNINGS
      None

   POSSIBLE REVISIONS:
      None

  ***************************************************************************/

#if !defined(_AIX)
#  define ckinterp ckinterp_
#  define mrhlen mrhlen_
#  define mrhinit mrhinit_
#  define mrhsyms mrhsyms_
#  define mrhsyme mrhsyme_
#  define ckindx ckindx_
#  define ckwt ckwt_
#  define ckawt ckawt_
#  define ckxty ckxty_
#  define ckytx ckytx_
#  define ckcpbs ckcpbs_
#  define ckhms ckhms_
#  define ckwyp ckwyp_
#  define ckhbms ckhbms_
#  define ckmmwy ckmmwy_
#  define ckrhoy ckrhoy_
#  define equil equil_
#  define eqsol eqsol_
#  define dvode dvode_
#  define dgelss dgelss_
#endif

// GROUP: Function Declarations:
////////////////////////////////////////////////////////////////////////
//

extern "C"
{
 
  // Fortran program for reading in thermodynamic and chemical kinetics files
  // and creating a binary output file
  void ckinterp();

  // Hacked interface to cklen
  void mrhlen(int *leniwk, int *lenrwk, int *lencwk,
	      int *linc, int *lout, char *cklinkfile,
	      int *namelength);

  // Hacked interface to ckinit
  void mrhinit(int *leniwk, int *lenrwk, int *lencwk,
	       int *linc, int *lout, int *ickwrk,
	       double *rckwrk, char *cckwrk, char *cklinkfile,
	       int *namelength);

  // Hacked interface to cksyms
  void mrhsyms(char *cckwrk, int *lout, char *kname, int *kerr);

  // Hacked interface to cksyme
  void mrhsyme(char *cckwrk, int *lout, char *ename, int *kerr);
   
  void ckindx(int *ickwrk, double *rckwrk,
	      int *mm, int *kk, int *ii, int *nfit);

  void ckwt(int *ickwrk, double *rckwrk, double *returnvalue);

  void ckawt(int *ickwrk, double *rckwrk, double *returnvalue);

  void ckxty(double *x, int *ickwrk, double *rckwrk, double *y);

  void ckytx(double *x, int *ickwrk, double *rckwrk, double *y);

  void ckcpbs(double *T,double *y,
	      int *ickwrk, double *rckwrk, double *returnvalue);

  void ckhms(double *T, int *ickwrk, double *rckwrk, double *ret);
   
  void ckwyp(double *pressure, double *T, double *y,
	     int *ickwrk, double *rckwrk, double *cdot);

  void ckhbms(double *T, double *y, int *ickwrk, 
	      double *rckwrk, double *h);

  void ckmmwy(double *y, int *ickwrk, double *rckwrk, double *wtm);

  void ckrhoy(double *pressure, double *T, double *y,
	      int *ickrwk, double *rckwrk, double *returnvalue);
   
  void equil(int *lout, int *lprnt, int *lsave, int *leqst, int *lcntue,
	     int *ickwrk, double* rckwrk, int *lenieq, int *ieqwrk, 
	     int *lenreq, double *reqwrk, int *nel, int *nsp, 
	     char *ename, char *sp_name, int *nop, int *kmon, double *xeq, 
	     double *tin, double *test_t, double *pin, double *Press, 
	     int *ncon, int *kcon, double *xcon, int *ierr);
	     //*iadiabatic, double *hin, int *ierr);

  void eqsol(int *nsp, double* reqwrk, double *xeq, double *yeq, 
	     double *tin, double *pin, double *hout, double *vout,
	     double *sout, double *wmout, double *cout_s, double *cdet_out);

  void dvode(int (*Fun)(int *n, double *t, double *phi,
                        double *ydot, double *rpar, int *ipar),
	     int *neq, double *phi,
	     double *t1, double *t2,
	     int *itol, double *rtol, double *atol, int *itask,
	     int *istate, int *iopt, double *rwork, int *lrw,
	     int *iwork, int *liw,
	     int (*jac)(int *n, double *t, double *y, int *ml, int *mu,
			double *pd, int *nrowpd, double *rpar, int *ipar),
	     int *MF, double *rpar, int *ipar);
  void dgelss(int *m, int *n, int *nrhs, double A[5][8], int *lda, double *b,
	      int *ldb, double *s, double *rcond, int *rank, double *work,
	      int *lwork, int *info);


}

#endif

