#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
#include <stdio.h>
#include <stdlib.h> 
#include <malloc.h>
#include <math.h>
#include <string.h>

/*
  int main(void){
  MESH * mesh;
  init();
  read_mesh(&mesh);
  read_conductivity(mesh);
  compute_volumes(mesh);
  compute_matrix(mesh);
  dump_vis(mesh);
  return(1);
  }
*/

/* INIT prints out the header */
void init(){
  printf("\n\n\n");
  printf("\t\t********************************************\n");
  printf("\t\t*   VULCAN Vertex centered finite volume   *\n");
  printf("\t\t*  discretization for cardiac propagation  *\n");
  printf("\t\t*                                          *\n");
  printf("\t\t*     Writen By Chris Penland 1997-99      *\n");
  printf("\t\t*  Converted to C by Joe Tranquillo 2001   *\n");
  printf("\t\t********************************************\n");
  printf("\n\n");
	
}

/*
  READ_MESH generates or reads from the file a 
  list of vertices and forms the volume elements
*/
void read_mesh(MESH** mesh){
  char filename[100],line[100];
  int type,xdim,ydim,zdim,i,j,k,tem,tem2,p[10];
  int nodenumber=0,numelem=0;
  double dx,dy,dz,temp11;
  char *c;
  FILE *fp;

  *mesh=(MESH*)malloc(sizeof(MESH));
  if(*mesh==NULL){
    printf("\n\nOut of memory allocating mesh.\n");
    exit(-1);
  }

  printf("Grid type: (1) Cartesian, (2) File, (3) TrueGrid(ALE3D) -> ");
  scanf("%d",&type);

  switch(type){
  case 1:	
    printf("Enter Dimensions (x y z) in cm -> ");
    scanf("%d %d %d",&xdim,&ydim,&zdim);

    printf("Enter Spacing (dx dy dz) in cm -> ");
    scanf("%lf %lf %lf",&dx,&dy,&dz);

    (*mesh)->numvtx=xdim*ydim*zdim;
    (*mesh)->vtx=(VERTEX*)malloc(sizeof(VERTEX)*xdim*ydim*zdim);
    if((*mesh)->vtx==NULL)
    {
      printf("\n\nError allocating vertex array. Insufficient Memory.\n");
      exit(-1);
    }

    for(k=0; k<=zdim-1; k++){
      for(j=0; j<=ydim-1; j++){
	for(i=0; i<=xdim-1; i++){
	  nodenumber = i+j*xdim+k*xdim*ydim;
	  (*mesh)->vtx[nodenumber].x = i*dx;
	  (*mesh)->vtx[nodenumber].y = j*dy;
	  (*mesh)->vtx[nodenumber].z = k*dz;			
	}
      }
    }

    (*mesh)->numelement=(xdim-1)*(ydim-1)*(zdim-1);
    (*mesh)->elements=(VOLUME_ELEMENT*)malloc(sizeof(VOLUME_ELEMENT)*(*mesh)->numelement);
    if((*mesh)->elements==NULL)
    {
      printf("\n\nError allocating elements array. Insufficient Memory.\n");
      exit(-1);
    }
    for(k=0; k<=zdim-2; k++){
      for(j=0; j<=ydim-2; j++){
	for(i=0; i<=xdim-2; i++){
	  (*mesh)->elements[numelem].vtx[0] = i+j*xdim+k*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[1] = (i+1)+j*xdim+k*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[2] = (i+1)+(j+1)*xdim+k*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[3] = i+(j+1)*xdim+k*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[4] = i+j*xdim+(k+1)*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[5] = (i+1)+j*xdim+(k+1)*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[6] = (i+1)+(j+1)*xdim+(k+1)*xdim*ydim;
	  (*mesh)->elements[numelem].vtx[7] = i+(j+1)*xdim+(k+1)*xdim*ydim;
	  numelem++;
	}
      }
    }
    break;

  case 2:
    /* Get Node Information */
    printf("Enter ASCII xyz coordinate file -> ");
    scanf("%s",filename);
    printf("Number of Nodes -> ");
    scanf("%d",&nodenumber);

    (*mesh)->numvtx=nodenumber;
    (*mesh)->vtx=(VERTEX*)malloc(sizeof(VERTEX)*(*mesh)->numvtx);
    if((*mesh)->vtx==NULL)
    {
      printf("\n\nError allocating vertex array. Insufficient Memory.\n");
      exit(-1);
    }			
    nodenumber = 0;
    fp = fopen(filename, "r" );
    do {
      c = fgets(line,150,fp);
      if (c!=NULL){ 
	sscanf(line,"%f %f %f\n",&(*mesh)->vtx[nodenumber].x,&(*mesh)->vtx[nodenumber].y,&(*mesh)->vtx[nodenumber].z);		       
      } 
      nodenumber++;
    } while (c!=NULL);
    fclose(fp);

    if((nodenumber-1)!=(*mesh)->numvtx){
      printf("\n\nLength of coordinate file not equal to Number of Nodes.\n");
      exit(-1);
    }

    /* Get Connectivity Information */
    printf("Enter ASCII connectivity file -> ");
    scanf("%s",filename);
    printf("Number of Elements -> ");
    scanf("%d",&numelem);

    (*mesh)->numelement=numelem;
    (*mesh)->elements=(VOLUME_ELEMENT*)malloc(sizeof(VOLUME_ELEMENT)*(*mesh)->numelement);
    if((*mesh)->elements==NULL)
    {
      printf("\n\nError allocating elements array. Insufficient Memory.\n");
      exit(-1);
    }

    numelem= 0;
    fp = fopen(filename, "r" );
    do {
      c = fgets(line,150,fp);
      if (c!=NULL){ 
	sscanf(line,"%d %d %d %d %d %d %d %d\n",
	       &(*mesh)->elements[numelem].vtx[0],
	       &(*mesh)->elements[numelem].vtx[1],
	       &(*mesh)->elements[numelem].vtx[2],
	       &(*mesh)->elements[numelem].vtx[3],
	       &(*mesh)->elements[numelem].vtx[4],
	       &(*mesh)->elements[numelem].vtx[5],
	       &(*mesh)->elements[numelem].vtx[6],
	       &(*mesh)->elements[numelem].vtx[7]);
      } 
      numelem++;
    } while (c!=NULL);
    fclose(fp);

    if((numelem-1)!=(*mesh)->numelement){
      printf("\n\nLength of connectivity file not equal to number of elements.\n");
      exit(-1);
    }	
    break;

  case 3:
    printf("Enter TrueGrid ALE3D file -> ");
    scanf("%s",filename);
    fp = fopen(filename, "r" );
    fgets(line,100,fp); 
    fgets(line,100,fp);		
    sscanf(line,"%d %d %d\n",&tem,&nodenumber,&numelem);
		

    (*mesh)->numvtx=nodenumber;
    (*mesh)->vtx=(VERTEX*)malloc(sizeof(VERTEX)*(*mesh)->numvtx);
    if((*mesh)->vtx==NULL)
    {
      printf("\n\nError allocating vertex array. Insufficient Memory.\n");
      exit(-1);
    }


    (*mesh)->numelement=numelem;
    (*mesh)->elements=(VOLUME_ELEMENT*)malloc(sizeof(VOLUME_ELEMENT)*(*mesh)->numelement);
    if((*mesh)->elements==NULL)
    {
      printf("\n\nError allocating elements array. Insufficient Memory.\n");
      exit(-1);
    }


    do {
      c = fgets(line,150,fp);
      if (c!=NULL){ 
	sscanf(line,"%d\n",&tem);		       
      } 
    } while (tem!=1);

    sscanf(line,"%d %lf %f %f %f\n",&tem,&temp11,&(*mesh)->vtx[0].x,
	   &(*mesh)->vtx[0].y,&(*mesh)->vtx[0].z);
		

    for(i=1; i<(*mesh)->numvtx; i++){
      fgets(line,100,fp);
      sscanf(line,"%d %lf %f %f %f\n",&tem,&temp11,&(*mesh)->vtx[i].x,
	     &(*mesh)->vtx[i].y,&(*mesh)->vtx[i].z);		       			
    }



    for(i=0; i<(*mesh)->numelement; i++){
      fgets(line,100,fp);
      sscanf(line,"%d %d %d %d %d %d %d %d %d\n",&tem,&tem2,
	     &p[0],&p[1],&p[2],&p[3],&p[4],&p[5],&p[6]);
      fgets(line,100,fp);
      sscanf(line,"%d\n",&p[7]);
      (*mesh)->elements[i].vtx[0]=p[0]-1;
      (*mesh)->elements[i].vtx[1]=p[1]-1;
      (*mesh)->elements[i].vtx[2]=p[2]-1;
      (*mesh)->elements[i].vtx[3]=p[3]-1;
      (*mesh)->elements[i].vtx[4]=p[4]-1;
      (*mesh)->elements[i].vtx[5]=p[5]-1;
      (*mesh)->elements[i].vtx[6]=p[6]-1;
      (*mesh)->elements[i].vtx[7]=p[7]-1;
    }
    fclose(fp);
    break;
  default:
    printf("\n\nError in Entry of Grid Type\n");
  }

}


/* READ_CONDUCTIVITY generates or reads from the file
   the conductivity tensor D at each vertex*/
void read_conductivity(MESH* mesh){
  char filename[100],line[100];
  float U[3][3],sigma[3][3],D[3][3],temp[3][3],UT[3][3];
  float dot_result;
  int type,i,j,k,counter=0;
  char *c;
  FILE *fp;
	
  printf("Grid type: (1) Basic, (2) File -> ");
  scanf("%d",&type);

  switch(type){
  case 1:
    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	U[i][j] = 0.0;
	UT[i][j] = 0.0;
	sigma[i][j] = 0.0;
	D[i][j] = 0.0;
	temp[i][j] = 0.0;	
      }
    }


    printf("Enter the relative conductivity scale [longitudinal transverse sheet] in mS/cm: ");
    scanf("%f %f %f", &sigma[0][0], &sigma[1][1], &sigma[2][2]);
    printf("Enter the longitudinal unit vector [x y z]: ");
    scanf("%f %f %f", &U[0][0], &U[0][1], &U[0][2]);
    printf("Enter the transverse unit vector [x y z]: ");
    scanf("%f %f %f", &U[1][0], &U[1][1], &U[1][2]);


    dot_result = U[0][0]*U[1][0]+U[0][1]*U[1][1]+U[0][2]*U[1][2];
    if(dot_result>0.0001){
      printf("\n\nUnit Vectors not orthogonal!\n");
      exit(-1);
    }
    /* Cross Product */
    U[2][0]=U[0][1]*U[1][2]-U[0][2]*U[1][1]; 
    U[2][1]=U[0][2]*U[1][0]-U[0][0]*U[1][2];
    U[2][2]=U[0][0]*U[1][1]-U[0][1]*U[1][0];
    printf("The sheet unit vector is [%2.2f %2.2f %2.2f].\n",U[2][0],U[2][1],U[2][2]);

    /* U Transpose */
    UT[0][0] = U[0][0];
    UT[0][1] = U[1][0];
    UT[0][2] = U[2][0];
    UT[1][0] = U[0][1];
    UT[1][1] = U[1][1];
    UT[1][2] = U[2][1];
    UT[2][0] = U[0][2];
    UT[2][1] = U[1][2];
    UT[2][2] = U[2][2];

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	for(k=0; k<3; k++){
	  temp[i][j]=temp[i][j]+U[i][k]*sigma[k][j]; 
	}
      }
    }

    for(i=0; i<3; i++){
      for(j=0; j<3; j++){
	for(k=0; k<3; k++){
	  D[i][j]=D[i][j]+temp[i][k]*UT[k][j]; 
	}
      }
    }

    for(i=0;i<mesh->numvtx;i++)
    {
      mesh->vtx[i].sxx=D[0][0];
      mesh->vtx[i].sxy=D[0][1];
      mesh->vtx[i].sxz=D[0][2];			
      mesh->vtx[i].syy=D[1][1];
      mesh->vtx[i].syz=D[1][2];
      mesh->vtx[i].szz=D[2][2];
    }
		
    break;
  case 2:
    printf("Enter Conductivity Filename -> ");
    scanf("%s",filename);
    fp = fopen(filename, "r" );
    do {
      c = fgets(line,150,fp);
      if (c!=NULL){ 
	sscanf(line,"%f %f %f %f %f %f\n",
	       &mesh->vtx[counter].sxx,&mesh->vtx[counter].sxy,
	       &mesh->vtx[counter].sxz,&mesh->vtx[counter].syy,
	       &mesh->vtx[counter].syz,&mesh->vtx[counter].szz); 
      } 
			
      counter++;
    } while (c!=NULL);
    fclose(fp);
			
    if((counter-1)!=mesh->numvtx){
      printf("\n\nError in number of Conductivity Values\n");			
      exit(-1);
    }		
    break;

  default:
    printf("\n\nError in Entry of Grid Type\n");
  }

}


/* Computes the volume associated with each vertex */
void compute_volumes(MESH *mesh, const char *fname){
  int i,j,x;
  float a=1.0/sqrt(3.0);
  float detJ;
  double totalvol;
  float xi[8],eta[8],zeta[8];
  float Jac[3][3];
  float Xbox[8],Ybox[8],Zbox[8];
  double *TempVol;
  FILE *fp=0;

  for (i=0; i<8; i++){
    Xbox[i] = 0.0;
    Ybox[i] = 0.0;
    Zbox[i] = 0.0;
    xi[i] = 0.0;
    eta[i] = 0.0;
    zeta[i] = 0.0;
  }

  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      Jac[i][j] = 0.0;
    }
  }
	

  TempVol=(double*)malloc(sizeof(double)*(mesh->numvtx));
  xi[0]=-a;xi[1]=a;xi[2]=a;xi[3]=-a;
  xi[4]=-a;xi[5]=a;xi[6]=a;xi[7]=-a;
  eta[0]=-a;eta[1]=-a;eta[2]=a;eta[3]=a;
  eta[4]=-a;eta[5]=-a;eta[6]=a;eta[7]=a;
  zeta[0]=-a;zeta[1]=-a;zeta[2]=-a;zeta[3]=-a;
  zeta[4]=a;zeta[5]=a;zeta[6]=a;zeta[7]=a;

  for (i=0; i<mesh->numelement; i++) {
    for(j=0; j<8; j++){
      Xbox[j] = mesh->vtx[mesh->elements[i].vtx[j]].x;
      Ybox[j] = mesh->vtx[mesh->elements[i].vtx[j]].y;
      Zbox[j] = mesh->vtx[mesh->elements[i].vtx[j]].z;
    }
	
    for (j=0; j<8; j++) {
      jacobian(xi[j],eta[j],zeta[j],Xbox,Ybox,Zbox,Jac); 
      detJ = Jac[0][0]*(Jac[1][1]*Jac[2][2]-Jac[1][2]*Jac[2][1]) -
	Jac[0][1]*(Jac[1][0]*Jac[2][2]-Jac[1][2]*Jac[2][0]) +
	Jac[0][2]*(Jac[1][0]*Jac[2][1]-Jac[1][1]*Jac[2][0]);  
      mesh->vtx[mesh->elements[i].vtx[j]].volume = mesh->vtx[mesh->elements[i].vtx[j]].volume + detJ;

    }
  }

  /* Dump Volume Vector */
  if (strlen(fname)) fp = fopen(fname,"w");
  /*	if (fp ==NULL){
	printf("\n\nProblems opening Volume.vec for writting\n");
	} */

  totalvol = 0.0;
  for(i=0; i<mesh->numvtx; i++){
    TempVol[i] = mesh->vtx[i].volume;
    totalvol += TempVol[i];
  }
  printf("Total Volume = %lf cm^3\n",totalvol);

  if (fp) {
    x = 1;
    if (*(char *)&x == 1) {
      fprintf(fp, "VL%i: %i\n",sizeof(double), mesh->numvtx);
    } else {
      fprintf(fp, "VB%i: %i\n",sizeof(double), mesh->numvtx);
    }

    fseek(fp, 128, SEEK_SET);
    fwrite(TempVol, sizeof(double), mesh->numvtx, fp);  
    fclose(fp);
  }
  free(TempVol);


}

/* Computes the jacobian at a point in an element */
void jacobian(float xitemp,float etatemp,float zetatemp,float *Xbox,float *Ybox, float *Zbox,float Jac[][3]){
  int i;
  float dN[8][3];
  char ch = 'd';

  trial(xitemp,etatemp,zetatemp,ch,dN);

  for(i=0; i<3; i++){
    Jac[0][i] = 0.0;
    Jac[1][i] = 0.0;
    Jac[2][i] = 0.0;
  }


  for(i=0; i<8; i++){
    Jac[0][0] = Jac[0][0] + Xbox[i]*dN[i][0]; 
    Jac[1][0] = Jac[1][0] + Xbox[i]*dN[i][1];
    Jac[2][0] = Jac[2][0] + Xbox[i]*dN[i][2];
    Jac[0][1] = Jac[0][1] + Ybox[i]*dN[i][0]; 
    Jac[1][1] = Jac[1][1] + Ybox[i]*dN[i][1];
    Jac[2][1] = Jac[2][1] + Ybox[i]*dN[i][2];
    Jac[0][2] = Jac[0][2] + Zbox[i]*dN[i][0]; 
    Jac[1][2] = Jac[1][2] + Zbox[i]*dN[i][1];
    Jac[2][2] = Jac[2][2] + Zbox[i]*dN[i][2];
  }
}


/* Evaulate trial functions and derivatives for trilinear brick
   Returns vector N with length 8 containing
   either the trial functions (MODE='f') or the derivatives of the 
   trial functions (MODE='xi','eta','zeta' depending on which 
   derivative is needed) evaluated at the point (XI,ETA,ZETA) in the 
   brick.
*/
void trial(float xitemp,float etatemp,float zetatemp,char ch,float N[][3]){
  int i,j;
  float signxi[8],signeta[8],signzeta[8];

  for(i=0; i<8; i++){
    signxi[i] = 0.0;
    signeta[i] = 0.0;
    signzeta[i] = 0.0;		
  }

  for(i=0; i<8; i++){
    for(j=0; j<3; j++){
      N[i][j] = 0.0;
    }
  }


  signxi[0]=-1.0;signxi[1]=1.0;signxi[2]=1.0;signxi[3]=-1.0;
  signxi[4]=-1.0;signxi[5]=1.0;signxi[6]=1.0;signxi[7]=-1.0;

  signeta[0]=-1.0;signeta[1]=-1.0;signeta[2]=1.0;signeta[3]=1.0;
  signeta[4]=-1.0;signeta[5]=-1.0;signeta[6]=1.0;signeta[7]=1.0;

  signzeta[0]=-1.0;signzeta[1]=-1.0;signzeta[2]=-1.0;signzeta[3]=-1.0;
  signzeta[4]=1.0;signzeta[5]=1.0;signzeta[6]=1.0;signzeta[7]=1.0;

  switch(ch) {
  case 'f':
    for(i=0; i<8; i++){
      N[i][0]=0.125*(1.0+xitemp*signxi[i])*(1.0+etatemp*signeta[i])*(1.0+zetatemp*signzeta[i]);
    }
    break;
  case 'd':	
    for(i=0; i<8; i++){
      N[i][0] = 0.125*signxi[i]*(1.0+etatemp*signeta[i])*(1.0+zetatemp*signzeta[i]);
      N[i][1] = 0.125*signeta[i]*(1.0+xitemp*signxi[i])*(1.0+zetatemp*signzeta[i]);
      N[i][2] = 0.125*signzeta[i]*(1.0+xitemp*signxi[i])*(1.0+etatemp*signeta[i]);
    }
    break;
  default:
    printf("Error in Trial Function\n");
  }

}


/*
  Computes the global coupling matrix for mesh of elements
  Computes a sparse containing the resistive coupling
  between vertices of a mesh defined in VERTEX and connected into brick
  elements by ELEMENT. VERTEX is a structure containing fields X, Y, Z
  which are arrays representing the real space coordinates of the
  vertices. D is a substructure of VERTEX and contains fields XX, XY,
  XZ, YY, YZ, ZZ which are also arrays containing the conductivity
  tensor components at the element vertices. ELEMENT is a structure
  containing field VTX which is an array of length 8 arrays that define
  the vertices of the brick elements in the mesh.
*/
void compute_matrix(MESH *mesh, const char *fname){
  int i,j,k,li,lj,gi,gj,loopvar,x;
  int counter,min,pmin,maxloopvar=0;
  int maxbandwidth=0,minbandwidth=0;
  int TEMPjcoef;
  float D[8][9];
  float Xbox[8],Ybox[8],Zbox[8];
  float lmatrix[8][8];
  double TEMPcoef;
  float *tempcoef;
  int *tempjcoef;
  int **jcoef,*count;
  float **coef;
  FILE *sp=0;

  for(i=0; i<8; i++){
    for(j=0; j<8; j++){
      lmatrix[i][j] = 0.0;
      D[i][j] = 0.0;
    }
    Xbox[i] = 0.0;
    Ybox[i] = 0.0;
    Zbox[i] = 0.0;
  }
  /* Initialize count */

  count=malloc(sizeof(int)*(mesh->numvtx+1));

  for(i=0; i<mesh->numvtx; i++){
    count[i] = -1;
  }

  /* Initialize jcoef */
  jcoef=malloc(sizeof(int *)*(mesh->numvtx+1));
  if(jcoef == NULL){
    printf("\n\nError allocating memory in jcoef.\n");
    exit(-1);
  }
  for(i=0; i<mesh->numvtx; i++){
    jcoef[i] = malloc(sizeof(int)*(27));
    if(jcoef[i] == NULL){
      printf("\n\nError allocating memory in jcoef.\n");
      exit(-1);
    }
  }

  for(i=0; i<mesh->numvtx; i++){
    for(j=0; j<27; j++){
      jcoef[i][j] = -1;
    }
  }

  /* Initialize coef */
  coef=malloc(sizeof(float *)*(mesh->numvtx+1));
  if(coef == NULL){
    printf("\n\nError allocating memory in coef.\n");
    exit(-1);
  }
  for(i=0; i<mesh->numvtx; i++){
    coef[i] = malloc(sizeof(float)*(27));
    if(coef[i] == NULL){
      printf("\n\nError allocating memory in coef.\n");
      exit(-1);
    }
  }

  for(i=0; i<mesh->numvtx; i++){
    for(j=0; j<27; j++){
      coef[i][j] = 0.0;
    }
  }

  for(i=0; i<mesh->numelement; i++){	
	
    for(j=0; j<8; j++){
      Xbox[j] = mesh->vtx[mesh->elements[i].vtx[j]].x;
      Ybox[j] = mesh->vtx[mesh->elements[i].vtx[j]].y;
      Zbox[j] = mesh->vtx[mesh->elements[i].vtx[j]].z;
      D[j][0] = mesh->vtx[mesh->elements[i].vtx[j]].sxx;
      D[j][1] = mesh->vtx[mesh->elements[i].vtx[j]].sxy;
      D[j][2] = mesh->vtx[mesh->elements[i].vtx[j]].sxz;
      D[j][3] = mesh->vtx[mesh->elements[i].vtx[j]].sxy;
      D[j][4] = mesh->vtx[mesh->elements[i].vtx[j]].syy;
      D[j][5] = mesh->vtx[mesh->elements[i].vtx[j]].syz;
      D[j][6] = mesh->vtx[mesh->elements[i].vtx[j]].sxz;
      D[j][7] = mesh->vtx[mesh->elements[i].vtx[j]].syz;
      D[j][8] = mesh->vtx[mesh->elements[i].vtx[j]].szz;
    }

    fvelement(Xbox,Ybox,Zbox,lmatrix,D);
		
    /* Generate unordered coef and jcoef */
    for(li=0; li<8; li++){
      for(lj=0; lj<8; lj++){
	gi = mesh->elements[i].vtx[li];
	gj = mesh->elements[i].vtx[lj];

	if(count[gi]==-1){	/* first entry on this row */
	  loopvar=0;
	}else{
	  loopvar=count[gi]+1;
	}

	if(loopvar>maxloopvar){
	  maxloopvar = loopvar;
	}

	for(j=0; j<=loopvar; j++){
	  if(jcoef[gi][j]==gj){	
	    coef[gi][j] += lmatrix[li][lj];
	    break;
	  } else if((jcoef[gi][j]==-1)&&(lmatrix[li][lj]!=0.0)){
	    count[gi] =count[gi]+1;
	    jcoef[gi][count[gi]]=gj;
	    coef[gi][count[gi]] = lmatrix[li][lj];	
	  } else{
				
	  }				
	}
      }
    }
  }

  tempjcoef=malloc(sizeof(int)*(maxloopvar+1));
  tempcoef=malloc(sizeof(float)*(maxloopvar+1));
  /* Sort coef and jcoef matrices */
 

  for(i=0; i<mesh->numvtx; i++){ 	
    for(j=0; j<maxloopvar; j++){
      tempjcoef[j] = 0;
      tempcoef[j] = 0.0;
    }
		
    for(j=0; j<=count[i]; j++){
      if(jcoef[i][j]==i){
	tempjcoef[0] = i;
	tempcoef[0] = coef[i][j];
      }	
    }
		
    counter = 1;
    pmin = -10;
    for(j=0; j<=count[i]; j++){
      min =mesh->numvtx+1;
      for(k=0; k<=count[i]; k++){
	if((jcoef[i][k]<min)&&(jcoef[i][k]!=i)&&(jcoef[i][k]>pmin)&&(jcoef[i][k]!=-1)){
	  min = jcoef[i][k];
	  tempjcoef[counter] = jcoef[i][k];
	  tempcoef[counter] = coef[i][k];
	}
      }
      counter++;
      pmin = min;
    }

    for(j=count[i]+1; j<maxloopvar; j++){
      tempjcoef[j] = i;
      tempcoef[j] = 0.0;
    }

    for(j=0; j<=maxloopvar; j++){
      jcoef[i][j] = tempjcoef[j];
      coef[i][j] = tempcoef[j];	
    }

    /* find min and max bandwidth */
    for(j=0; j<=count[i]; j++){
      if((jcoef[i][j]-i)>maxbandwidth){
	maxbandwidth=jcoef[i][j]-i;
      }

      if((jcoef[i][j]-i)<minbandwidth){
	minbandwidth= jcoef[i][j]-i;
      }
    }
  } 

  /* Dump SPR file */
  if (strlen(fname)) sp = fopen(fname,"w");
  /*	if (sp ==NULL){
	printf("\n\nProblems opening Vulcan.spr for writting\n");
	} */
  if (sp) {
    x = 1;
    if(*(char *)&x == 1){
      fprintf( sp, "SL%i%i: %i %i %i %i %i %i\n", sizeof(double), 
	       sizeof(int),mesh->numvtx,mesh->numvtx,maxbandwidth,
	       minbandwidth,1,maxloopvar);
    }else{
      fprintf( sp, "SB%i%i: %i %i %i %i %i %i\n", sizeof(double), 
	       sizeof(int),mesh->numvtx,mesh->numvtx,maxbandwidth,
	       minbandwidth,1,maxloopvar);
    }
    fseek( sp, 128, SEEK_SET );
	  
    for(j=0; j<maxloopvar; j++){
      for(i=0; i<mesh->numvtx; i++){	
	TEMPjcoef = jcoef[i][j]+1; 
	fwrite(&TEMPjcoef, sizeof(int),1, sp );
      }
    }
    for(j=0; j<maxloopvar; j++){
      for(i=0; i<mesh->numvtx; i++){
	TEMPcoef = (double)coef[i][j];
	fwrite(&TEMPcoef, sizeof(double),1, sp ); 
      }
    }	
    fclose(sp); 
  }

  /* Free jcoef and coef */
  for(i=0; i<mesh->numvtx; i++){
    free(jcoef[i]);
    free(coef[i]);
  }
  free(jcoef);
  free(coef);
}

/*
  Computes the coupling matrix for a trilinear brick element
  Returns 8x8 square "lmatrix" containing the local resistive
  coupling across the Voronoi faces internal to the element. LOCALVERTEX
  is a structure containing fields X, Y, Z which are arrays of length 8
  representing the real space coordinates of the vertices of the element
  D is a substructure of LOCALVERTEX and contains fields XX, XY, XZ, YY,
  YZ, ZZ which are also length 8 arrays containing the conductivity
  tensor components at the element vertices. 
*/

void fvelement(float *Xbox,float *Ybox,float *Zbox,float lmatrix[][8],float D[][9]){
  int i,j,k,start[12],end[12];
  float a=1.0,detJ;
  float xi[12],eta[12],zeta[12];
  char ch1='f',ch2='d';
  float dN[8][3],Ni[8][3];
  float CondD[9],Jac[3][3],JacInv[3][3];
  float normal[3][12],Temp[8],temp[3],temp2[3];


	

  for(i=0; i<12; i++){
    xi[i] = 0.0;
    eta[i] = 0.0;
    zeta[i] = 0.0;
    start[i] = 0.0;
    end[i] = 0.0;
    normal[0][i]=0.0;normal[1][i]=0.0;normal[2][i]=0.0;
  }

  for(i=0; i<8; i++){
    Temp[i] = 0.0;
    CondD[i] = 0.0;
    dN[i][0]=0.0;dN[i][1]=0.0;dN[i][2]=0.0;
    Ni[i][0]=0.0;Ni[i][1]=0.0;Ni[i][2]=0.0;
    lmatrix[i][0]=0.0;lmatrix[i][1]=0.0;lmatrix[i][2]=0.0;lmatrix[i][3]=0.0;
    lmatrix[i][4]=0.0;lmatrix[i][5]=0.0;lmatrix[i][6]=0.0;lmatrix[i][7]=0.0;		
  }
  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      Jac[i][j] = 0.0;
      JacInv[i][j] = 0.0;
    }
    temp[i] = 0.0;
    temp2[i] = 0.0;
  }


  xi[0]=0.0;xi[1]=a;xi[2]=0.0;xi[3]=-a;
  xi[4]=0.0;xi[5]=a;xi[6]=0.0;xi[7]=-a;
  xi[8]=-a;xi[9]=a;xi[10]=a;xi[11]=-a;

  eta[0]=-a;eta[1]=0.0;eta[2]=a;eta[3]=0.0;
  eta[4]=-a;eta[5]=0.0;eta[6]=a;eta[7]=0.0;
  eta[8]=-a;eta[9]=-a;eta[10]=a;eta[11]=a;

  zeta[0]=-a;zeta[1]=-a;zeta[2]=-a;zeta[3]=-a;
  zeta[4]=a;zeta[5]=a;zeta[6]=a;zeta[7]=a;
  zeta[8]=0.0;zeta[9]=0.0;zeta[10]=0.0;zeta[11]=0.0;

  start[0]=0;start[1]=1;start[2]=2;start[3]=3;
  start[4]=4;start[5]=5;start[6]=6;start[7]=7;
  start[8]=0;start[9]=1;start[10]=2;start[11]=3;

  end[0]=1;end[1]=2;end[2]=3;end[3]=0;
  end[4]=5;end[5]=6;end[6]=7;end[7]=4;
  end[8]=4;end[9]=5;end[10]=6;end[11]=7;
	
  normals(Xbox,Ybox,Zbox,D,normal);

  for(i=0; i<12; i++){
    trial(xi[i],eta[i],zeta[i],ch1,Ni);
    trial(xi[i],eta[i],zeta[i],ch2,dN);

    for(k=0; k<9; k++){
      CondD[k] = 0.0; 
      for(j=0; j<8; j++){
	CondD[k] = CondD[k] + Ni[j][0]*D[j][k];
      }
    }

    jacobian(xi[i],eta[i],zeta[i],Xbox,Ybox,Zbox,Jac); 
    /* Inverse of Jacobian */
    detJ = Jac[0][0]*(Jac[1][1]*Jac[2][2]-Jac[1][2]*Jac[2][1]) -
      Jac[0][1]*(Jac[1][0]*Jac[2][2]-Jac[1][2]*Jac[2][0]) +
      Jac[0][2]*(Jac[1][0]*Jac[2][1]-Jac[1][1]*Jac[2][0]);  

    JacInv[0][0] = (Jac[1][1]*Jac[2][2]-Jac[1][2]*Jac[2][1])/detJ;
    JacInv[0][1] = -(Jac[0][1]*Jac[2][2]-Jac[0][2]*Jac[2][1])/detJ;
    JacInv[0][2] = (Jac[0][1]*Jac[1][2]-Jac[0][2]*Jac[1][1])/detJ;
    JacInv[1][0] = -(Jac[1][0]*Jac[2][2]-Jac[1][2]*Jac[2][0])/detJ;
    JacInv[1][1] = (Jac[0][0]*Jac[2][2]-Jac[0][2]*Jac[2][0])/detJ;
    JacInv[1][2] = -(Jac[0][0]*Jac[1][2]-Jac[0][2]*Jac[1][0])/detJ;
    JacInv[2][0] = (Jac[1][0]*Jac[2][1]-Jac[1][1]*Jac[2][0])/detJ;
    JacInv[2][1] = -(Jac[0][0]*Jac[2][1]-Jac[0][1]*Jac[2][0])/detJ;
    JacInv[2][2] = (Jac[0][0]*Jac[1][1]-Jac[0][1]*Jac[1][0])/detJ;

    /* compute da'*D */
    temp[0]=normal[0][i]*CondD[0]+normal[1][i]*CondD[3]+normal[2][i]*CondD[6];
    temp[1]=normal[0][i]*CondD[1]+normal[1][i]*CondD[4]+normal[2][i]*CondD[7];
    temp[2]=normal[0][i]*CondD[2]+normal[1][i]*CondD[5]+normal[2][i]*CondD[8];

    /* compute (da'*D)*JacInv */
    temp2[0] = temp[0]*JacInv[0][0]+temp[1]*JacInv[1][0]+temp[2]*JacInv[2][0];
    temp2[1] = temp[0]*JacInv[0][1]+temp[1]*JacInv[1][1]+temp[2]*JacInv[2][1];
    temp2[2] = temp[0]*JacInv[0][2]+temp[1]*JacInv[1][2]+temp[2]*JacInv[2][2];

    /* compute (da'*D*JacInv)*dN' */		
    Temp[0] = temp2[0]*dN[0][0]+temp2[1]*dN[0][1]+temp2[2]*dN[0][2];
    Temp[1] = temp2[0]*dN[1][0]+temp2[1]*dN[1][1]+temp2[2]*dN[1][2];
    Temp[2] = temp2[0]*dN[2][0]+temp2[1]*dN[2][1]+temp2[2]*dN[2][2];
    Temp[3] = temp2[0]*dN[3][0]+temp2[1]*dN[3][1]+temp2[2]*dN[3][2];
    Temp[4] = temp2[0]*dN[4][0]+temp2[1]*dN[4][1]+temp2[2]*dN[4][2];
    Temp[5] = temp2[0]*dN[5][0]+temp2[1]*dN[5][1]+temp2[2]*dN[5][2];
    Temp[6] = temp2[0]*dN[6][0]+temp2[1]*dN[6][1]+temp2[2]*dN[6][2];
    Temp[7] = temp2[0]*dN[7][0]+temp2[1]*dN[7][1]+temp2[2]*dN[7][2];

    for(j=0; j<8; j++){
      lmatrix[start[i]][j] = lmatrix[start[i]][j] + Temp[j];
      lmatrix[end[i]][j] = lmatrix[end[i]][j] - Temp[j];	
    }
			
  }
}


/*
  Computes the normal vectors for each face in the element
  Returns in "normal" (3x12) 12 column vectors that are the area scaled
  normal vectors for the 12 internal Voronoi faces in the element
  defined by LOCALVERTEX. This is done by computing the locations of
  fiducial points on the element that are the centroids of the element, 
  the element faces and the element edges.
*/
void normals(float *Xbox,float *Ybox,float *Zbox,float D[][9],float normal[][12]){
  int i;
  float edgevec[3],vec1[3],vec2[3],da[3];
  float sign;
  typedef struct {
    float x,y,z;
  }Vert;

  Vert centroid,south,east,west,north,top,bottom;
  Vert e12,e23,e34,e41,e56,e67,e78,e85,e15,e26,e37,e48;

  for(i=0; i<3; i++){
    edgevec[i] = 0.0;
    vec1[i] = 0.0;
    vec2[i] = 0.0;
  }

  centroid.x = 0.125*(Xbox[0]+Xbox[1]+Xbox[2]+Xbox[3]+Xbox[4]+Xbox[5]+Xbox[6]+Xbox[7]);
  centroid.y = 0.125*(Ybox[0]+Ybox[1]+Ybox[2]+Ybox[3]+Ybox[4]+Ybox[5]+Ybox[6]+Ybox[7]);
  centroid.z = 0.125*(Zbox[0]+Zbox[1]+Zbox[2]+Zbox[3]+Zbox[4]+Zbox[5]+Zbox[6]+Zbox[7]);

  south.x = 0.25*(Xbox[0]+Xbox[1]+Xbox[4]+Xbox[5]);
  south.y = 0.25*(Ybox[0]+Ybox[1]+Ybox[4]+Ybox[5]);
  south.z = 0.25*(Zbox[0]+Zbox[1]+Zbox[4]+Zbox[5]);

  east.x = 0.25*(Xbox[1]+Xbox[2]+Xbox[5]+Xbox[6]);
  east.y = 0.25*(Ybox[1]+Ybox[2]+Ybox[5]+Ybox[6]);
  east.z = 0.25*(Zbox[1]+Zbox[2]+Zbox[5]+Zbox[6]);

  west.x = 0.25*(Xbox[0]+Xbox[3]+Xbox[4]+Xbox[7]);
  west.y = 0.25*(Ybox[0]+Ybox[3]+Ybox[4]+Ybox[7]);
  west.z = 0.25*(Zbox[0]+Zbox[3]+Zbox[4]+Zbox[7]);

  north.x = 0.25*(Xbox[2]+Xbox[3]+Xbox[6]+Xbox[7]);
  north.y = 0.25*(Ybox[2]+Ybox[3]+Ybox[6]+Ybox[7]);
  north.z = 0.25*(Zbox[2]+Zbox[3]+Zbox[6]+Zbox[7]);

  top.x = 0.25*(Xbox[4]+Xbox[5]+Xbox[6]+Xbox[7]);
  top.y = 0.25*(Ybox[4]+Ybox[5]+Ybox[6]+Ybox[7]);
  top.z = 0.25*(Zbox[4]+Zbox[5]+Zbox[6]+Zbox[7]);

  bottom.x = 0.25*(Xbox[0]+Xbox[1]+Xbox[2]+Xbox[3]);
  bottom.y = 0.25*(Ybox[0]+Ybox[1]+Ybox[2]+Ybox[3]);
  bottom.z = 0.25*(Zbox[0]+Zbox[1]+Zbox[2]+Zbox[3]);

  e12.x = 0.5*(Xbox[0]+Xbox[1]);
  e12.y = 0.5*(Ybox[0]+Ybox[1]);
  e12.z = 0.5*(Zbox[0]+Zbox[1]);

  e23.x = 0.5*(Xbox[1]+Xbox[2]);
  e23.y = 0.5*(Ybox[1]+Ybox[2]);
  e23.z = 0.5*(Zbox[1]+Zbox[2]);

  e34.x = 0.5*(Xbox[2]+Xbox[3]);
  e34.y = 0.5*(Ybox[2]+Ybox[3]);
  e34.z = 0.5*(Zbox[2]+Zbox[3]);

  e41.x = 0.5*(Xbox[0]+Xbox[3]);
  e41.y = 0.5*(Ybox[0]+Ybox[3]);
  e41.z = 0.5*(Zbox[0]+Zbox[3]);

  e56.x = 0.5*(Xbox[4]+Xbox[5]);
  e56.y = 0.5*(Ybox[4]+Ybox[5]);
  e56.z = 0.5*(Zbox[4]+Zbox[5]);

  e67.x = 0.5*(Xbox[5]+Xbox[6]);
  e67.y = 0.5*(Ybox[5]+Ybox[6]);
  e67.z = 0.5*(Zbox[5]+Zbox[6]);

  e78.x = 0.5*(Xbox[6]+Xbox[7]);
  e78.y = 0.5*(Ybox[6]+Ybox[7]);
  e78.z = 0.5*(Zbox[6]+Zbox[7]);

  e85.x = 0.5*(Xbox[7]+Xbox[4]);
  e85.y = 0.5*(Ybox[7]+Ybox[4]);
  e85.z = 0.5*(Zbox[7]+Zbox[4]);

  e15.x = 0.5*(Xbox[0]+Xbox[4]);
  e15.y = 0.5*(Ybox[0]+Ybox[4]);
  e15.z = 0.5*(Zbox[0]+Zbox[4]);

  e26.x = 0.5*(Xbox[1]+Xbox[5]);
  e26.y = 0.5*(Ybox[1]+Ybox[5]);
  e26.z = 0.5*(Zbox[1]+Zbox[5]);

  e37.x = 0.5*(Xbox[2]+Xbox[6]);
  e37.y = 0.5*(Ybox[2]+Ybox[6]);
  e37.z = 0.5*(Zbox[2]+Zbox[6]);

  e48.x = 0.5*(Xbox[3]+Xbox[7]);
  e48.y = 0.5*(Ybox[3]+Ybox[7]);
  e48.z = 0.5*(Zbox[3]+Zbox[7]);

  /* 1 -> 2 face */
  edgevec[0] = Xbox[1]-Xbox[0]; 
  edgevec[1] = Ybox[1]-Ybox[0];
  edgevec[2] = Zbox[1]-Zbox[0];	
  vec1[0] = centroid.x-e12.x;
  vec1[1] = centroid.y-e12.y;
  vec1[2] = centroid.z-e12.z;
  vec2[0] = south.x-bottom.x;
  vec2[1] = south.y-bottom.y;
  vec2[2] = south.z-bottom.z;
  /* cross product of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][0] = da[0]*sign;
  normal[1][0] = da[1]*sign;
  normal[2][0] = da[2]*sign;


  /* 2 -> 3 face */
  edgevec[0] = Xbox[2]-Xbox[1]; 
  edgevec[1] = Ybox[2]-Ybox[1];
  edgevec[2] = Zbox[2]-Zbox[1];	
  vec1[0] = centroid.x-e23.x;
  vec1[1] = centroid.y-e23.y;
  vec1[2] = centroid.z-e23.z;
  vec2[0] = east.x-bottom.x;
  vec2[1] = east.y-bottom.y;
  vec2[2] = east.z-bottom.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][1] = da[0]*sign;
  normal[1][1] = da[1]*sign;
  normal[2][1] = da[2]*sign;

  /* 3 -> 4 face */
  edgevec[0] = Xbox[3]-Xbox[2]; 
  edgevec[1] = Ybox[3]-Ybox[2];
  edgevec[2] = Zbox[3]-Zbox[2];	
  vec1[0] = centroid.x-e34.x;
  vec1[1] = centroid.y-e34.y;
  vec1[2] = centroid.z-e34.z;
  vec2[0] = north.x-bottom.x;
  vec2[1] = north.y-bottom.y;
  vec2[2] = north.z-bottom.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][2] = da[0]*sign;
  normal[1][2] = da[1]*sign;
  normal[2][2] = da[2]*sign;
	
  /* 4 -> 1 face */
  edgevec[0] = Xbox[0]-Xbox[3]; 
  edgevec[1] = Ybox[0]-Ybox[3];
  edgevec[2] = Zbox[0]-Zbox[3];	
  vec1[0] = centroid.x-e41.x;
  vec1[1] = centroid.y-e41.y;
  vec1[2] = centroid.z-e41.z;
  vec2[0] = west.x-bottom.x;
  vec2[1] = west.y-bottom.y;
  vec2[2] = west.z-bottom.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][3] = da[0]*sign;
  normal[1][3] = da[1]*sign;
  normal[2][3] = da[2]*sign;

  /* 5 -> 6 face */
  edgevec[0] = Xbox[5]-Xbox[4]; 
  edgevec[1] = Ybox[5]-Ybox[4];
  edgevec[2] = Zbox[5]-Zbox[4];	
  vec1[0] = centroid.x-e56.x;
  vec1[1] = centroid.y-e56.y;
  vec1[2] = centroid.z-e56.z;
  vec2[0] = top.x-south.x;
  vec2[1] = top.y-south.y;
  vec2[2] = top.z-south.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][4] = da[0]*sign;
  normal[1][4] = da[1]*sign;
  normal[2][4] = da[2]*sign;

  /* 6-> 7 face */
  edgevec[0] = Xbox[6]-Xbox[5]; 
  edgevec[1] = Ybox[6]-Ybox[5];
  edgevec[2] = Zbox[6]-Zbox[5];	
  vec1[0] = centroid.x-e67.x;
  vec1[1] = centroid.y-e67.y;
  vec1[2] = centroid.z-e67.z;
  vec2[0] = top.x-east.x;
  vec2[1] = top.y-east.y;
  vec2[2] = top.z-east.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][5] = da[0]*sign;
  normal[1][5] = da[1]*sign;
  normal[2][5] = da[2]*sign;
	
  /* 7 -> 8 face */
  edgevec[0] = Xbox[7]-Xbox[6]; 
  edgevec[1] = Ybox[7]-Ybox[6];
  edgevec[2] = Zbox[7]-Zbox[6];	
  vec1[0] = centroid.x-e78.x;
  vec1[1] = centroid.y-e78.y;
  vec1[2] = centroid.z-e78.z;
  vec2[0] = top.x-north.x;
  vec2[1] = top.y-north.y;
  vec2[2] = top.z-north.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][6] = da[0]*sign;
  normal[1][6] = da[1]*sign;
  normal[2][6] = da[2]*sign;

  /* 8 -> 5 face */
  edgevec[0] = Xbox[4]-Xbox[7]; 
  edgevec[1] = Ybox[4]-Ybox[7];
  edgevec[2] = Zbox[4]-Zbox[7];	
  vec1[0] = centroid.x-e85.x;
  vec1[1] = centroid.y-e85.y;
  vec1[2] = centroid.z-e85.z;
  vec2[0] = top.x-west.x;
  vec2[1] = top.y-west.y;
  vec2[2] = top.z-west.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][7] = da[0]*sign;
  normal[1][7] = da[1]*sign;
  normal[2][7] = da[2]*sign;

  /* 1 -> 5 face */
  edgevec[0] = Xbox[4]-Xbox[0]; 
  edgevec[1] = Ybox[4]-Ybox[0];
  edgevec[2] = Zbox[4]-Zbox[0];	
  vec1[0] = centroid.x-e15.x;
  vec1[1] = centroid.y-e15.y;
  vec1[2] = centroid.z-e15.z;
  vec2[0] = west.x-south.x;
  vec2[1] = west.y-south.y;
  vec2[2] = west.z-south.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][8] = da[0]*sign;
  normal[1][8] = da[1]*sign;
  normal[2][8] = da[2]*sign;

  /* 2 -> 6 face */
  edgevec[0] = Xbox[5]-Xbox[1]; 
  edgevec[1] = Ybox[5]-Ybox[1];
  edgevec[2] = Zbox[5]-Zbox[1];	
  vec1[0] = centroid.x-e26.x;
  vec1[1] = centroid.y-e26.y;
  vec1[2] = centroid.z-e26.z;
  vec2[0] = south.x-east.x;
  vec2[1] = south.y-east.y;
  vec2[2] = south.z-east.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][9] = da[0]*sign;
  normal[1][9] = da[1]*sign;
  normal[2][9] = da[2]*sign;

  /* 3 -> 7 face */
  edgevec[0] = Xbox[6]-Xbox[2]; 
  edgevec[1] = Ybox[6]-Ybox[2];
  edgevec[2] = Zbox[6]-Zbox[2];	
  vec1[0] = centroid.x-e37.x;
  vec1[1] = centroid.y-e37.y;
  vec1[2] = centroid.z-e37.z;
  vec2[0] = east.x-north.x;
  vec2[1] = east.y-north.y;
  vec2[2] = east.z-north.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][10] = da[0]*sign;
  normal[1][10] = da[1]*sign;
  normal[2][10] = da[2]*sign;

  /* 4 -> 8 face */
  edgevec[0] = Xbox[7]-Xbox[3]; 
  edgevec[1] = Ybox[7]-Ybox[3];
  edgevec[2] = Zbox[7]-Zbox[3];	
  vec1[0] = centroid.x-e48.x;
  vec1[1] = centroid.y-e48.y;
  vec1[2] = centroid.z-e48.z;
  vec2[0] = north.x-west.x;
  vec2[1] = north.y-west.y;
  vec2[2] = north.z-west.z;
  /* cross producte of vec1 and vec2 divided by 2 into vec1*/
  da[0] = (vec1[1]*vec2[2]-vec1[2]*vec2[1])/2.0; 
  da[1] = (vec1[2]*vec2[0]-vec1[0]*vec2[2])/2.0;
  da[2] = (vec1[0]*vec2[1]-vec1[1]*vec2[0])/2.0;
  if((da[0]*edgevec[0]+da[1]*edgevec[1]+da[2]*edgevec[2])>=0.0){
    sign=1.0;
  } else{
    sign=-1.0;
  }
  normal[0][11] = da[0]*sign;
  normal[1][11] = da[1]*sign;
  normal[2][11] = da[2]*sign;
}


void dump_vis(MESH *mesh, const char *fname){
  int i,x;
  double temp;
  FILE *nodesout=0;
  
  if (strlen(fname)) {
    nodesout = fopen(fname, "w" );
  }
  if (nodesout) { 
    fwrite(&mesh->numelement,sizeof(int),1,nodesout);
    fseek( nodesout, 128, SEEK_SET );
    fwrite(&mesh->numvtx,sizeof(int),1,nodesout);
    fseek( nodesout, 256, SEEK_SET );
    
    for(i=0; i<mesh->numvtx; i++){
      temp = mesh->vtx[i].x;
      fwrite(&temp,sizeof(double),1,nodesout);
      temp = mesh->vtx[i].y;
      fwrite(&temp,sizeof(double),1,nodesout);
      temp = mesh->vtx[i].z;
      fwrite(&temp,sizeof(double),1,nodesout);
    }
    
    for(i=0; i<mesh->numelement; i++){
      fwrite(&(mesh->elements[i].vtx[0]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[1]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[2]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[3]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[4]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[5]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[6]),sizeof(int),1,nodesout); 
      fwrite(&(mesh->elements[i].vtx[7]),sizeof(int),1,nodesout); 
    }
    
    x = 1;
    if(*(char *)&x == 1){
      printf("\n\nWarning: Vis.vtx is being dumped in Little Endian\n");
    }
    
    fclose(nodesout);
  }
}
