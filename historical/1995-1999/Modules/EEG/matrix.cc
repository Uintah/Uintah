#include <stdio.h>
#include "main.h"
#include <stdlib.h>
#include <iostream.h>
#include "vector.h"

double norm(double **a, int nx, int ny) {
    double total=0;
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    total+=a[i][j]*a[i][j];
	}
    }
    return total;
}

double normI(double **a, int nx, int ny) {
    double total=0;
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    if (i==j) {
		total+=(a[i][j]-1)*(a[i][j]-1);
	    } else {
		total+=a[i][j]*a[i][j];
	    }
	}
    }
    return total;
}

double **makeMatrix(int nx, int ny) {
    double **m = (double **) malloc ((nx+1)*sizeof(double*));
    double *mm = (double *) malloc ((nx+1)*sizeof(double)*(ny+1));
    if (m==0 || mm==0) {
	cerr << "Unable to allocate "<<nx<<"x"<<ny<<" matrix.  Exiting.\n";
	exit(0);
    }
    for (int i=1; i<=nx; i++) {
	m[i]=&(mm[i*(ny+1)]);
    }
    return m;
}

void freeMatrix(double **m) {
//    free(*m);
//    free(m);
}

void getColumn(double **a, int u, int idx, double *c) {
    for (int i=1; i<=u; i++) {
	c[i]=a[i][idx];
    }
}

void putColumn(double **a, int u, int idx, double *c) {
    for (int i=1; i<=u; i++) {
	a[i][idx]=c[i];
    }
}

void printMatrix(double **a, int u, int v) {
    int c=0;
    int i, j;
    for (i=1; i<=u; i++) {
	for (j=1; j<=v; j++) {
	    if (a[i][j] != 0) c++;
	}
    }
    FILE *f=fopen("/tmp/A.mat", "wt");
    fprintf(f, "%d %d %d\n", u, v, c);
    for (i=1; i<=u; i++) {
	for (j=1; j<=v; j++) {
	    if (a[i][j] != 0) {
		fprintf(f, "%d %d %.6lf\n", i-1, j-1, a[i][j]);
	    }
	}
    }
    fclose(f);
}

void printMatrixToFile(char *name, double **a, int u, int v) {
    int i, j;
    FILE *f=fopen(name, "wt");
    fprintf(f, "%d %d\n", u, v);
    for (i=1; i<=u; i++) {
	for (j=1; j<=v; j++) {
	    fprintf(f, "%lf ", a[i][j]);
	}
	fprintf(f, "\n");
    }
    fclose(f);
}

void printMatrix2(double **a, int u, int v) {
    int c=0;
    int i, j;
    for (i=1; i<=u; i++) {
	for (j=1; j<=v; j++) {
	    if (a[i][j] != 0) c++;
	}
    }
    FILE *f=fopen("/tmp/U.mat", "wt");
    fprintf(f, "%d %d %d\n", u, v, c);
    for (i=1; i<=u; i++) {
	for (j=1; j<=v; j++) {
	    if (a[i][j] != 0) {
		fprintf(f, "%d %d %.6lf\n", i-1, j-1, a[i][j]);
	    }
	}
    }
    fclose(f);
}

void printSparseMatrix(double **a, int u) {
    for (int i=1; i<=u; i++) {
	printf("%d (%d):  ", i, (int)a[i][0]);
	for (int j=1; j<=(int)a[i][0]; j++) {
	    printf("[%d] %.6lf  ", (int)a[i][j*2-1], a[i][j*2]);
	}
	printf("\n");
    }
    printf("\n");
}

void printSparseMatrix2(char *name, double **a, int u) {
    int c=0;
    int i, j;
    for (i=1; i<=u; i++) c+=a[i][0];

    FILE *f=fopen(name, "wt");
    fprintf(f, "%d %d\n", u, c);
    for (i=1; i<=u; i++) {
	for (j=1; j<=(int)a[i][0]; j++) {
	    fprintf(f, "%d %d %.6lf\n", i-1,
		   (int)a[i][j*2-1]-1, a[i][j*2]);
	}
    }
    fclose(f);
}

double percentFull(double **a, int nx, int ny) {
    int full=0;
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    if (a[i][j] > .00001 || a[i][j] < -.00001) {
		full++;
	    }
	}
    }
    return full*100./(nx*ny);
}

void matMatMult(int x, int y, int z, double **a, double **b, double **c) {
    for (int i=1; i<=x; i++) {
	for (int k=1; k<=z; k++) {
	    c[i][k]=0;
	}
    }

    for (i=1; i<=x; i++) {
	for (int j=1; j<=y; j++) {
	    double fact=a[i][j];
	    double* dst=c[i]+1;
	    double* src=b[j]+1;
	    for (int k=1; k<=z; k++) {
		//c[i][k] += fact*b[j][k];
		*dst++ +=fact* *src++;
	    }
	}
    }
}

void newMatMatMult(int x, int y, int z, double **a, double **b, double **c) {
    for (int i=1; i<=x; i++) {
	for (int k=1; k<=z; k++) {
	    c[i][k]=0;
	}
    }

    for (i=1; i<=x; i++) {
	for (int j=1; j<=y; j++) {
	    double fact=a[i][j];
	    if (fact == 0) continue;
	    double* dst=c[i]+1;
	    double* src=b[j]+1;
	    for (int k=1; k<=z; k++) {
		//c[i][k] += fact*b[j][k];
		*dst++ +=fact* *src++;
	    }
	}
    }
}

void matSub(int nx, int ny, double **a, double **b, double **c) {
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    c[i][j]=a[i][j]-b[i][j];
	}
    }
}

void matVecMult(int x, int y, double **a, double *b, double *c) {
    for (int i=1; i<=x; i++) {
	c[i]=0;
    }

    for (i=1; i<=x; i++) {
	for (int j=1; j<=y; j++) {
	    c[i]+=a[i][j]*b[j];
	}
    }
}


void matTranspose(int nx, int ny, double **u, double **ut) {
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    ut[j][i]=u[i][j];
	}
    }
}

void sortem2(int *idx, double *data, int nn) {
    double tmpData;
    int tmpIdx;
    int swapped=1;
    while (swapped) {
	swapped=0;
	for (int j=0; j<nn-1; j++) {
	    if (idx[j]>idx[j+1]) {
		tmpIdx=idx[j];
		tmpData=data[j];
		idx[j]=idx[j+1];
		data[j]=data[j+1];
		idx[j+1]=tmpIdx;
		data[j+1]=tmpData;
		swapped=1;
	    }
	}
    }
}

// small to big
void sortem(double *w, double *n, int nn) {
    for (int j=1; j<=nn; j++) n[j]=w[j]; 
    double tmp;
    int swapped=1;
//    printf("W's:  ");
//    for (j=1; j<nn; j++) printf("%le ", n[j]);
//    printf("\n");
    while (swapped) {
	swapped=0;
	for (j=1; j<nn; j++) {
	    if (n[j]>n[j+1]) {
		tmp=n[j];
		n[j]=n[j+1];
		n[j+1]=tmp;
		swapped=1;
	    }
	}
    }
//    printf("W's:  ");
//    for (j=1; j<nn; j++) printf("%le ", n[j]);
//    printf("\n");
}

void svdTruncateW(double *w, int nn, double truncMin) {
    if (truncMin == 0) return;
    double *n=makeVector(nn);
    sortem(w, n, nn);
    printf("Min value: %le    Max value: %le   Condition Number: %le\n", 
	   n[1], n[nn], n[1]/n[nn]);

    int last=nn-(nn-1)*truncMin;

    double wmin=n[last];

    int counter=0;
    for (int j=1; j<=nn; j++) {
	if (w[j] <= wmin) {
	    w[j]=0.0;
	    counter++;
	}
    }
    printf("Hacked off %d terms from w vector in SVD\n", counter);
}

void svdCompose(double **u, int nx, int ny, double *w, double **v, 
		double **uTr, double **ui){
    for (int i=1; i<=ny; i++) {
	for (int j=1; j<=ny; j++) {
	    if (w[j] != 0.0) {
		v[i][j]/=w[j];
	    }
	}
    }
    matTranspose(nx, ny, u, uTr);
    matMatMult(ny, ny, nx, v, uTr, ui);
}

int isIdentity(int x, double **a) {
    for (int i=1; i<=x; i++) {
	for (int j=1; j<=x; j++) {
	    if (i==j) {
		if (a[i][j] > 1.0001 || a[i][j] < .9999) return 0;
	    } else {
		if (a[i][j] > 0.0001 || a[i][j] < -0.0001) return 0;
	    }
	}
    }
    return 1;
}

FILE *readSizeOfMatrix(int *ns, int *nv, int *nc, FILE *f, char *prob) {
    char fMatName[250];
    sprintf(fMatName, "%s.mat", prob);
    if ((f = fopen(fMatName, "rt")) == NULL) {
	printf("Coudln't open %s.  Exiting...\n", fMatName);
	exit(0);
    }	
    fscanf(f, "%d %d %d", ns, nv, nc);
    return f;
}

void readMatrixVV(double **Ass, double **Asv, double **Asc, double **Avs, 
		double **Avv, double **Avc, double **Azz, int ns, int nv, int nc, FILE *f) {
    for (int i=1; i<=ns; i++) {	
	for (int j=1; j<=ns; j++) {	// Initialize Ass
	    Ass[i][j]=0;
	}
	for (j=1; j<=nv; j++) {		// Initialize Asv and Avs
	    Asv[i][j]=Avs[j][i]=0;
	}
	for (j=1; j<=nc; j++) {		// Initialize Asc
	    Asc[i][j]=0;
	}
    }
    for (i=1; i<=nv; i++) {
	for (int j=1; j<=nc; j++) {		// Initialize Avc
	    Avc[i][j]=0;
	}
	for (j=1; j<=nv; j++) {		// need to figure out exact value here!
	    Avv[i][j]=0;
	}
    }
    int x, y;
    int lastS, lastV;
    lastS=ns;
    lastV=nv+lastS;
    double tt;
    while (!feof(f)) {
	if (fscanf(f, "%d %d %lf", &x, &y, &tt) == 3) {
	    if (x<=lastS) {
		if (y<=lastS) {
		    Ass[x][y]=tt;
		} else if (y<=lastV) {
		    Asv[x][y-lastS]=Avs[y-lastS][x]=tt;
		} else {
		    Asc[x][y-lastV]=tt;
		}
	    } else if (x<=lastV){
		if (y<=lastS) {
		} else if (y<=lastV) {
		    int last=(int)Avv[x-lastS][0];
		    Avv[x-lastS][last*2+1]=y-lastS;
		    Avv[x-lastS][last*2+2]=tt;
		    Avv[x-lastS][0]=last+1;
		    Azz[x-lastS][y-lastS] = tt;
		} else {
		    Avc[x-lastS][y-lastV]=tt;
		}
	    }
	}
    }
    fclose(f);
printf("Done reading file!\n");
}

void copyMatrix(double **a, double **b, int nx, int ny) {
    for (int i=1; i<=nx; i++) {
	for (int j=1; j<=ny; j++) {
	    b[i][j]=a[i][j];
	}
    }
}
