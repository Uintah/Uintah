/*______________________________________________________________________
*   To shutup the warning messages from -fullwarn (Steve Parker)
*_______________________________________________________________________*/
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1174
#endif


#ifndef _NR_UTILS_H_
#define _NR_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

static float sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)

static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

static double dmaxarg1,dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))

static double dminarg1,dminarg2;
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
        (dminarg1) : (dminarg2))

static float maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
        (maxarg1) : (maxarg2))

static float minarg1,minarg2;
#define FMIN(a,b) (minarg1=(a),minarg2=(b),(minarg1) < (minarg2) ?\
        (minarg1) : (minarg2))

static long lmaxarg1,lmaxarg2;
#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ?\
        (lmaxarg1) : (lmaxarg2))

static long lminarg1,lminarg2;
#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ?\
        (lminarg1) : (lminarg2))

static int imaxarg1,imaxarg2;
#define IMAX(a,b) (imaxarg1=(a),imaxarg2=(b),(imaxarg1) > (imaxarg2) ?\
        (imaxarg1) : (imaxarg2))

static int iminarg1,iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

void nrerror(char error_text[]);
float *vector_nr(long nl, long nh);
int *ivector_nr(long nl, long nh);
unsigned char *cvector_nr(long nl, long nh);
unsigned long *lvector_nr(long nl, long nh);
double *dvector_nr(long nl, long nh);
float **matrix(long n1dl, long n1dh, long n2dl, long n2dh);
double **dmatrix(long n1dl, long n1dh, long n2dl, long n2dh);
int **imatrix(long n1dl, long n1dh, long n2dl, long n2dh);
float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
	long newrl, long newcl);
float **convert_matrix(float *a, long n1dl, long n1dh, long n2dl, long n2dh);
/*______________________________________________________________________
* Modified by Todd Harman 2.26.99
*-----------------------------------------------------------------------*/
float   ***f3tensor(            long n1dl,   long n1dh,   long n2dl,   long n2dh,
                                long n3dl,   long n3dh);
                                                                
double  ***darray_3d(           long n1dl,   long n1dh,   long n2dl,   long n2dh,
                                long n3dl,   long n3dh);
                                
double  ****darray_4d(          long n1dl,   long n1dh,   long n2dl,   long n2dh,
                                long n3dl,   long n3dh,   long n4dl,  long n4dh);
                                
double  *****darray_5d(         long n1dl,   long n1dh,   long n2dl,   long n2dh, 
                                long n3dl,   long n3dh,   long n4dl,  long n4dh,
                                long n5dl,  long n5dh);
                                
double  ******darray_6d(        long n1dl,   long n1dh,   long n2dl,   long n2dh, 
                                long n3dl,   long n3dh,   long n4dl,  long n4dh,
                                long n5dl,  long n5dh,   long n6dl,  long n6dh);
                                
int     ***iarray_3d(           long n1dl,   long n1dh,   long n2dl,   long n2dh,
                                long n3dl,   long n3dh);
                                
float  *convert_darray_2d_to_vector(double **t,         int n1dl,   int n1dh, 
                                int n2dl,   int n2dh,   int *max_len);
                                                                 
float  *convert_darray_3d_to_vector(double ***t,        int n1dl,   int n1dh, 
                                 int n2dl,  int n2dh,   int n3dl,   int n3dh,
                                 int *max_len);
                                 
float  *convert_iarray_3d_to_vector(int ***t,           int n1dl,   int n1dh, 
                                 int n2dl,  int n2dh,   int n3dl,   int n3dh,
                                 int *max_len); 
                                                                 
float *convert_darray_4d_to_vector(double ****t,        int n1dl,   int n1dh, 
                                 int n2dl,  int n2dh,   int n3dl,   int n3dh,
                                 int n4l,   int n4h,
                                 int *max_len);
                                                                  
float *convert_darray_5d_to_vector(double *****t,       int n1dl,   int n1dh, 
                                 int n2dl,  int n2dh,   int n3dl,   int n3dh,
                                 int n4l,   int n4h,    int n5l,    int n5h,
                                 int *max_len,          int ptrflag);
                                 
float *convert_darray_6d_to_vector(double ******t,      int n1dl,   int n1dh, 
                                 int n2dl,  int n2dh,   int n3dl,   int n3dh,
                                 int n4l,   int n4h,    int n5l,    int n5h,
                                 int *max_len);
/*______________________________________________________________________
*
*_______________________________________________________________________*/
void free_vector_nr(float *v, long nl, long nh);
void free_ivector_nr(int *v, long nl, long nh);
void free_cvector_nr(unsigned char *v, long nl, long nh);
void free_lvector_nr(unsigned long *v, long nl, long nh);
void free_dvector_nr(double *v, long nl, long nh);
void free_matrix(float **m, long n1dl, long n1dh, long n2dl, long n2dh);
void free_dmatrix(double **m, long n1dl, long n1dh, long n2dl, long n2dh);
void free_imatrix(int **m, long n1dl, long n1dh, long n2dl, long n2dh);
void free_submatrix(float **b, long n1dl, long n1dh, long n2dl, long n2dh);
void free_convert_matrix(float **b, long n1dl, long n1dh, long n2dl, long n2dh);
void free_f3tensor(float ***t, long n1dl, long n1dh, long n2dl, long n2dh,
	long n3dl, long n3dh);
void free_darray_3d(double ***t, long n1dl, long n1dh, long n2dl, long n2dh,
	long n3dl, long n3dh);
       
void free_iarray_3d(int ***t, long n1dl, long n1dh, long n2dl, long n2dh,
	long n3dl, long n3dh);
       
void free_darray_4d(double ****t, long n1dl, long n1dh, long n2dl, long n2dh,
	long n3dl, long n3dh, long n4dl, long n4dh);

void free_darray_5d(double *****t,      long n1dl,   long n1dh,   long n2dl, long n2dh,
	long n3dl, long n3dh, long n4dl,   long n4dh,  long n5dl,  long n5dh);

void free_darray_6d(double ******t,     long n1dl,   long n1dh,   long n2dl, long n2dh,
	long n3dl, long n3dh, long n4dl, long n4dh,  long n5dl,  long n5dh,
       long n6dl,   long n6dh);

#ifdef __cplusplus
}
#endif
#endif /* _NR_UTILS_H_ */
/*______________________________________________________________________
*   To shutup the warning messages from -fullwarn (Steve Parker)
*_______________________________________________________________________*/
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1174
#endif
