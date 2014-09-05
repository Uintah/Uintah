#include <nrrd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
cc -I/home/gk/c/nrrd/include -L/home/gk/c/nrrd/lib -o st st.c -lnrrd -lm
*/

int debug = 0, verbose;
int i, sx, sy, sz, xi, yi, zi;
float parm;

void
normalize(double *v) {
  double len;

  len = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  v[0] /= len;
  v[1] /= len;
  v[2] /= len;
}

#define CROSS(v3, v1, v2) \
  (v3[0] = v1[1]*v2[2] - v1[2]*v2[1], \
   v3[1] = v1[2]*v2[0] - v1[0]*v2[2], \
   v3[2] = v1[0]*v2[1] - v1[1]*v2[0])

#define COPY(v2, v1) \
  (v2[0] = v1[0],     \
   v2[1] = v1[1],     \
   v2[2] = v1[2])

#define ADD(v3, v1, v2)  \
  (v3[0] = v1[0] + v2[0], \
   v3[1] = v1[1] + v2[1], \
   v3[2] = v1[2] + v2[2])

#define DOT(v1, v2) \
  (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])

#define SCALE(v, a) \
  (v[0] *= a,        \
   v[1] *= a,        \
   v[2] *= a)

double v1[3], v2[3], v3[3], m1[9], m2[9], m3[9], m4[9], t[9];

/************************************************************************/
/************************************************************************/
/************************************************************************/

char desc[1024];

double
eval(int i, double x, double y, double z) {
  double bound, ret, tmp, r, f;

  r = sqrt(x*x + y*y + z*z);
  
  bound = 0.5 - 0.5*erf(20*(r-0.8));  /* 1 on inside, 0 on outside */
  
  switch (i) {
  case 1:
    f = NRRD_AFFINE(0.0, parm, 2.0, 1.0, 0.0001);
    break;
  case 2:
    f = NRRD_AFFINE(0.0, parm, 2.0, 0.0001, 1.0);
    break;
  case 3:
    f = 0.0001;
    break;
  }
  ret = NRRD_AFFINE(1.0, bound, 0.0, f, 1.0/3.0);
  return ret;
}

void
evec1(double x, double y, double z) {
  double r;
  
  r = sqrt(x*x + y*y + z*z);

  /* looking towards positive Y, v1 points counter clockwise */
  v1[0] = -z;
  v1[1] = 0;
  v1[2] = x;
  normalize(v1);
}

void
evec2(double x, double y, double z) {
  double tmp[3];

  /* v2 points towards pole at positive Y */
  tmp[0] = -x;
  tmp[1] = -y;
  tmp[2] = -z;
  normalize(tmp);
  CROSS(v2, tmp, v1);
}

/************************************************************************/
/************************************************************************/
/************************************************************************/

void
matrixC(double *v1, double *v2, double *v3, double *m) {

  m[0] = v1[0]; m[1] = v2[0]; m[2] = v3[0];
  m[3] = v1[1]; m[4] = v2[1]; m[5] = v3[1];
  m[6] = v1[2]; m[7] = v2[2]; m[8] = v3[2];
}

void
matrixR(double *v1, double *v2, double *v3, double *m) {

  m[0] = v1[0]; m[1] = v1[1]; m[2] = v1[2];
  m[3] = v2[0]; m[4] = v2[1]; m[5] = v2[2];
  m[6] = v3[0]; m[7] = v3[1]; m[8] = v3[2];
}

void
diagonal(double a, double b, double c, double *m) {

  m[0] = a; m[1] = 0; m[2] = 0;
  m[3] = 0; m[4] = b; m[5] = 0;
  m[6] = 0; m[7] = 0; m[8] = c;
}

void
matmul(double *m3, double *m1, double *m2) {

  /* m1      m2      m3
     0 1 2   0 1 2   0 1 2
     3 4 5   3 4 5   3 4 5
     6 7 8   6 7 8   6 7 8 */
  m3[0] = m1[0]*m2[0] + m1[1]*m2[3] + m1[2]*m2[6];
  m3[1] = m1[0]*m2[1] + m1[1]*m2[4] + m1[2]*m2[7];
  m3[2] = m1[0]*m2[2] + m1[1]*m2[5] + m1[2]*m2[8];
  m3[3] = m1[3]*m2[0] + m1[4]*m2[3] + m1[5]*m2[6];
  m3[4] = m1[3]*m2[1] + m1[4]*m2[4] + m1[5]*m2[7];
  m3[5] = m1[3]*m2[2] + m1[4]*m2[5] + m1[5]*m2[8];
  m3[6] = m1[6]*m2[0] + m1[7]*m2[3] + m1[8]*m2[6];
  m3[7] = m1[6]*m2[1] + m1[7]*m2[4] + m1[8]*m2[7];
  m3[8] = m1[6]*m2[2] + m1[7]*m2[5] + m1[8]*m2[8];
}

void
tensor(double x, double y, double z, double *t) {
  
  evec1(x, y, z);    /* sets v1 */
  evec2(x, y, z);    /* sets v2 */
  CROSS(v3, v1, v2);
  normalize(v3);
  if (debug) {
    printf("xi = %d; yi = %d; zi = %d\n", xi, yi, zi);
    printf("x = %g, y = %g, z = %g\n", x, y, z);
    printf("evects: v1: % 21.10f % 21.10f % 21.10f\n", v1[0], v1[1], v1[2]);
    printf("        v2: % 21.10f % 21.10f % 21.10f\n", v2[0], v2[1], v2[2]);
    printf("        v3: % 21.10f % 21.10f % 21.10f\n", v3[0], v3[1], v3[2]);
  }
  matrixC(v1, v2, v3, m1);
  diagonal(eval(1, x, y, z),
	   eval(2, x, y, z),
	   eval(3, x, y, z), m2);
  if (debug) {
    printf("  eval1: % 21.10f\n", eval(1, x, y, z));
    printf("  eval2: % 21.10f\n", eval(2, x, y, z));
    printf("  eval3: % 21.10f\n", eval(3, x, y, z));
  }
  matrixR(v1, v2, v3, m3);
  matmul(m4, m2, m3);
  matmul(t, m1, m4);
  if (debug) {
    printf("tensor:  % 21.10f % 21.10f % 21.10f\n", t[0], t[1], t[2]);
    printf("         % 21.10f % 21.10f % 21.10f\n", t[3], t[4], t[5]);
    printf("         % 21.10f % 21.10f % 21.10f\n", t[6], t[7], t[8]);
  }
}

void
usage(char *me) {
  /*                      0    1    2    3    4       5 */
  fprintf(stderr, "usage: %s <sx> <sy> <sz> <parm> <outputname>\n", me);
  exit(1);
}
  /* parm: 
     0 for linear along v1, 
     1 for planar, 
     2 for linear along v2 */

int
main(int argc, char **argv) {
  char *me, *out;
  float x, y, z, *data;
  Nrrd *nrrd;
  FILE *fout;

  me = argv[0];
  if (!(6 == argc)) {
    usage(me);
  }
  if (1 != sscanf(argv[4], "%f", &parm)) {
    fprintf(stderr, "%s: couldn't parse %s as float\n", me, argv[4]);
    exit(1);
  }
  out = argv[5];
  if (1 != sscanf(argv[1], "%d", &sx) ||
      1 != sscanf(argv[2], "%d", &sy) ||
      1 != sscanf(argv[3], "%d", &sz)) {
    fprintf(stderr, "%s: couldn't parse \"%s\", \"%s\", \"%s\" as int sizes\n",
	    me, argv[1], argv[2], argv[3]);
    usage(me);
  }
  if (!(fout = fopen(out, "w"))) {
    fprintf(stderr, "%s: couldn't open \"%s\" for writing\n", me, out);
    exit(1);
  }

  if (!(nrrd = nrrdNewAlloc(sx*sy*sz*7, nrrdTypeFloat, 4))) {
    fprintf(stderr, "%s: can't allocate nrrd\n", me);
    exit(1);
  }
  nrrd->size[0] = 7;
  nrrd->size[1] = sx;
  nrrd->size[2] = sy;
  nrrd->size[3] = sz;
  nrrd->spacing[0] = 0.0;
  nrrd->spacing[1] = 1.0;
  nrrd->spacing[2] = 1.0;
  nrrd->spacing[3] = 1.0;
  sprintf(desc, "gridlock %g", parm);
  strcpy(nrrd->content, desc);
  nrrd->encoding = nrrdEncodingRaw;
  data = nrrd->data;

  for (zi=0; zi<=sz-1; zi++) {
    z = NRRD_AFFINE(0, zi, sz-1, -1.0, 1.0);
    for (yi=0; yi<=sy-1; yi++) {
      y = NRRD_AFFINE(0, yi, sy-1, -1.0, 1.0);
      for (xi=0; xi<=sx-1; xi++) {
	x = NRRD_AFFINE(0, xi, sx-1, -1.0, 1.0);

	tensor(x, y, z, t);
	data[0 + 7*(xi + sx*(yi + sy*zi))] = 1.0;
	data[1 + 7*(xi + sx*(yi + sy*zi))] = t[0];
	data[2 + 7*(xi + sx*(yi + sy*zi))] = t[1];
	data[3 + 7*(xi + sx*(yi + sy*zi))] = t[2];
	data[4 + 7*(xi + sx*(yi + sy*zi))] = t[4];
	data[5 + 7*(xi + sx*(yi + sy*zi))] = t[5];
	data[6 + 7*(xi + sx*(yi + sy*zi))] = t[8];
      }
    }
  }
  strcpy(nrrd->label[0], "tens");
  strcpy(nrrd->label[1], "x");
  strcpy(nrrd->label[2], "y");
  strcpy(nrrd->label[3], "z");
  if (nrrdWrite(fout, nrrd)) {
    fprintf(stderr, "%s: error writing nrrd:\n%s", me, nrrdStrdupErr());
    exit(1);
  }

  exit(0);
}
