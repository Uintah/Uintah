//static char *id="@(#) $Id$";

/*
 *  Anneal.cc:  
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Mar. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Tester/RigorousTest.h>
#include <SCICore/Util/NotFinished.h>

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace SCICore::Containers;

/* constants */

class Anneal : public Module {
    #define pi (3.141592653589793)
    #define ra (8.0)
    #define rb (8.5)
    #define rc (9.2)

    #define nparameters (3)

    typedef double point[130][3];
    typedef double parameterset[nparameters];
    typedef double polynomial[1001];
    typedef double statevector[130];

    double potential(parameterset,double,int,int,int);
    double A_n(double,double,double,int);
    double S_n(double,double,double,int);
    double U_n(double,double,double,int);
    double T_n(double,double,double,int);
    double W_n(double,double,double,int);

    void LegendreP(int,double,polynomial);
    
    double bruteforce(parameterset);
    double primitive(parameterset);

    double errorfunction(parameterset);

    int nelectrodes,ielectrode_A,ielectrode_B;
    double I_inject;
    point xyz;
    statevector vdata;

    FILE *efile;
    FILE *vfile;
    FILE *pfile;
    FILE *sfile;

public:
    Anneal(const clString& id);
    virtual ~Anneal();
    virtual void execute();
};

Module* make_Anneal(const clString& id)
{
   return scinew Anneal(id);
}

static clString module_name("Anneal");

Anneal::Anneal(const clString& id)
: Module("Anneal", id, Source)
{
}

Anneal::~Anneal()
{
}

void Anneal::execute()
{
int i,j,ielectrode,i_E,i_elec;
double sb,ss,st,voltage;
//double error_best;
parameterset param;
//parameterset bestparam;

/* read in electrode coordinates */

printf("Enter 19 or 129 electrodes\n");
scanf("%d",&(nelectrodes));

if(nelectrodes==19)efile=fopen("./data/xyz19.dat","r");
if(nelectrodes==129)efile=fopen("./data/xyz129.dat","r");

for(i=1;i<=nelectrodes;i++)
  {
    fscanf(efile,"%d",&(ielectrode));
    for(j=0;j<3;j++)fscanf(efile,"%lf",&(xyz[i][j]));
  }

fclose(efile);
	
for(i=1;i<=nelectrodes;i++)
  {
    printf("%d\t%lf\t%lf\t%lf\n",i,xyz[i][0],xyz[i][1],xyz[i][2]);
  }

/* input for mock data set */

printf("Enter injected current (mA)\n");
scanf("%lf",&(I_inject));

printf("Enter brain, skull and scalp conductivity (1/Ohm*cm)\n");
scanf("%lf %lf %lf",&(sb),&(ss),&(st));

printf("Enter electrode for current injection\n");
scanf("%d %d",&(ielectrode_A),&(ielectrode_B));

param[0]=sb;
param[1]=ss;
param[2]=st;

/* generate mock data set */

vfile=fopen("./data/v.dat","w");
for(i_E=1;i_E<=nelectrodes;i_E++)
  {
	if(i_E==ielectrode_A || i_E==ielectrode_B)
	  {		
	  	voltage=0.0;
	  }
	else
	  {
		voltage=potential(param,I_inject,ielectrode_A,ielectrode_B,i_E);
	  }  
/*    printf("electrode = %d\t voltage = %lf\t mV\n",i_E,voltage); */
    fprintf(vfile,"%d\t%lf\n",i_E,voltage);
  }
fclose(vfile);

/* brute force search on skull conductivity */

vfile=fopen("./data/v.dat","r");
for(i_E=1;i_E<=nelectrodes;i_E++)
  {	
	fscanf(vfile,"%d %lf",&(i_elec),&(voltage));
	vdata[i_E]=voltage;
/*	printf("%d\t vdata = %lf mV\n",i_E,vdata[i_E]); */
  }
fclose(vfile);

//error_best=primitive(bestparam);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::potential(parameterset param, double I_inj, int ielec_A, 
	int ielec_B, int ielec_E)

{

int n,nmin,nmax;
double xA,yA,zA,xB,yB,zB,xE,yE,zE;
double rA,rB,rE,cos_A,cos_B;
double sb,ss,st,fact,pfact,sumn,sum,Vaverage;
polynomial p_A,p_B,Vpartial;

/* compute cosine of relative angles */

xA=xyz[ielec_A][0];
yA=xyz[ielec_A][1];
zA=xyz[ielec_A][2];

xB=xyz[ielec_B][0];
yB=xyz[ielec_B][1];
zB=xyz[ielec_B][2];

xE=xyz[ielec_E][0];
yE=xyz[ielec_E][1];
zE=xyz[ielec_E][2];

rA=sqrt(xA*xA+yA*yA+zA*zA);
rB=sqrt(xB*xB+yB*yB+zB*zB);
rE=sqrt(xE*xE+yE*yE+zE*zE);
cos_A=(xA*xE+yA*yE+zA*zE)/(rA*rE);
cos_B=(xB*xE+yB*yE+zB*zE)/(rB*rE);

/* printf("cos_A = %lf \t cos_B = %lf\n",cos_A,cos_B); */

/* natural parameters */

sb=param[0];
ss=param[1];
st=param[2];

fact=1.0/(2.0*pi*st*rc);

/* compute scalp potential */
/* note that factor r^n is cancelled analytically with T_n */

nmin=100;
nmax=200;

LegendreP(nmax,cos_A,p_A);
LegendreP(nmax,cos_B,p_B);

sum=0.0;

for(n=1;n<=nmax;n++)
  {
    pfact=p_A[n]-p_B[n];
    sumn=T_n(sb,ss,st,n)+W_n(sb,ss,st,n);
    sumn=I_inj*fact*sumn*pfact;
    sum=sum+sumn;
    Vpartial[n]=sum;
/*    printf("%d\t%lf\t%lf\n",n,sumn,sum); */
  }

/* compute average voltage */
  
Vaverage=0.0;

for(n=nmin;n<=nmax;n++)
  {
  	Vaverage=Vaverage+Vpartial[n];
  }
Vaverage=Vaverage/(nmax-nmin+1);

return(Vaverage);
  
}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::errorfunction(parameterset param)

{

int i;
double voltage,error;

if(param[0]<=0 || param[1]<=0 || param[2]<=0)
  {
  	error=pow(10.0,10.0);
  }

if(param[0]>0 && param[1]>0 && param[2]>0)
  {
	error=0.0;
	for(i=1;i<=nelectrodes;i++)
  	  {
  		if(i!=ielectrode_A && i!=ielectrode_B)
  		  {
  			voltage=potential(param,I_inject,ielectrode_A,ielectrode_B,i);
			error=error+pow((vdata[i]-voltage),2.0);
  		  }
  	  }
    error=error/(1.0*(nelectrodes-2));  
  }
 
return(error);
  
}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::bruteforce(parameterset bestparam)

/* computes optimal parameter set for errorfunction.
does3-parameter uniform search within specificed bounds */

{

int i,i0,i1,i2,nintervals;
double dp0,dp1,dp2;
parameterset param_best,param;
parameterset param_min,param_max;
double error_best,error;
double mean,stdev,totalpoints;

/* inits ---------------------------------------------------------- */

printf("Enter number of intervals in each dimension = ");
scanf("%d",&(nintervals));

/* enter initial search ranges and compute starting points */

param_min[0]=0.0;
param_min[1]=0.0;
param_min[2]=0.0;

param_max[0]=0.01;
param_max[1]=0.0001;
param_max[2]=0.01;

dp0=(param_max[0]-param_min[0])/nintervals;
dp1=(param_max[1]-param_min[1])/nintervals;
dp2=(param_max[2]-param_min[2])/nintervals;

/* init */

error_best=pow(10.0,10.0);
for(i=0;i<nparameters;i++)param_best[i]=param_min[i];

/* brute force search */

mean=0.0;
stdev=0.0;

for(i0=1;i0<=nintervals;i0++) 
{
  for(i1=1;i1<=nintervals;i1++)
    {
      for(i2=1;i2<=nintervals;i2++)
	{
	  param[0]=param_min[0]+i0*dp0;
	  param[1]=param_min[1]+i1*dp1;
	  param[2]=param_min[2]+i2*dp2;
	  
	  error=errorfunction(param);
/*	  printf("sb = %lf\t ss = %lf\t st = %lf\t error = %lf\n",
	  	param[0],param[1],param[2],error); */	  
	  mean=mean+error;
	  stdev=stdev+error*error;

	  if(error<error_best)
	    {
	      for(i=0;i<nparameters;i++)param_best[i]=param[i];
	      error_best=error;
	    }   
	}
    }
}
  
totalpoints=pow(1.0*nintervals,3.0);
/* printf("totalpoints = %lf",totalpoints); */
mean=mean/totalpoints;
stdev=stdev/totalpoints;
stdev=sqrt(stdev-mean*mean);

printf("mean error = %lf \t stdev = %lf\n",mean,stdev);

/* output */

for(i=0;i<nparameters;i++)bestparam[i]=param_best[i];

pfile=fopen("./data/p.dat","w");
for(i=0;i<nparameters;i++)fprintf(pfile,"%11.8f\t",param_best[i]);
fprintf(pfile,"\n");
fclose(pfile);

fprintf(stdout,"best error = %lf\n",error_best);
fprintf(stdout,"best parameter set = ");
for(i=0;i<nparameters;i++)fprintf(stdout,"%lf\t",param_best[i]);
fprintf(stdout,"\n");

return(error_best);

}


/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::primitive(parameterset bestparam)

/* computes optimal parameter set for errorfunction */

{

int i;
int iter0,niter0,iter,niter,itemp,ntemp;
parameterset param_best,param_good,param;
parameterset param_min,param_max,param_start;
parameterset temp;
double error_best,error_good,error,cool,ratio;

/* inits ---------------------------------------------------------- */

sfile=fopen("./data/s.dat","w");
pfile=fopen("./data/p.dat","w");

printf("Primitive simulated annealing:\n");

printf("\t Enter niter0 = ");
scanf("%d",&(niter0));

printf("\t Enter niter = ");
scanf("%d",&(niter));

printf("\t Enter cooling rate = ");
scanf("%lf",&(cool));

printf("\t Enter ratio stoptemp/starttemp = ");
scanf("%lf",&(ratio));

ntemp=floor(1.0+log(ratio)/log(cool)+0.5);
/* printf("ntemp = %d\n",ntemp); */

/* enter initial search ranges and compute starting points */

param_min[0]=0.0;
param_min[1]=0.0;
param_min[2]=0.0;

param_max[0]=0.01;
param_max[1]=0.0001;
param_max[2]=0.01;

for(i=0;i<nparameters;i++)
  {
    temp[i]=(param_max[i]-param_min[i])/2.0;
    param_start[i]=param_min[i]+temp[i];
  }

/* initialize flat search */

for(i=0;i<nparameters;i++)
  {
    param_best[i]=param[i]=param_start[i];
  }

/* 
error_best=error=errorfunction(param_start); 
*/

error_best=error=pow(10.0,10.0);

/*
printf("initial error = %lf\n",error_best);
fprintf(sfile,"%11.8f",error_best);
*/

/* flat search -------------------------------------------------------- */

for(iter0=1;iter0<=niter0;iter0++) 
  {
    for(i=0;i<nparameters;i++)
      {
	param[i]=param_start[i]+temp[i]*(2.0*drand48()-1.0);
      }	
      
    error=errorfunction(param);

    printf("iter0 = %d\t error = %lf\n",iter0,error);

/* don't count if parameters are intrinsically unfit */

    if(error==pow(10.0,10.0))iter0=iter0-1;

    if(error<error_best)
      {
		for(i=0;i<nparameters;i++) param_best[i]=param[i];
		error_best=error;
      }
  }

printf("best error after flat search = %lf\n",error_best);
fprintf(sfile,"%11.8f\n",error_best);
fprintf(stdout,"best parameter set = ");
for(i=0;i<nparameters;i++)fprintf(stdout,"%lf\t",param_best[i]);
fprintf(stdout,"\n");

/* initialize annealing ------------------------------------------------ */

for(i=0;i<nparameters;i++) param[i]=param_good[i]=param_best[i];
error=error_good=error_best;

/* begin annealing */

for(itemp=1;itemp<=ntemp;itemp++) /* begin outer loop */
  {

    for(iter=1;iter<=niter;iter++) /* begin inner loop */
      {

/* perturb parameter set */
	 
	for(i=0;i<nparameters;i++)
	  {
	    param[i]=param_best[i]+temp[i]*(2.0*drand48()-1.0);
	  }
	error=errorfunction(param);

/* don't count if parameters are intrinsically unfit */

	if(error==pow(10.0,10.0))iter=iter-1;

/* update good values */

	if(error<error_good)
	  {
	    for(i=0;i<nparameters;i++) param_good[i]=param[i];
	    error_good=error;
	  }
	  
    } /* close inner loop */

/* update best values */

    if(error_good<error_best)
      {
		for(i=0;i<nparameters;i++) param_best[i]=param_good[i];
		error_best=error_good;
      }

/* output at each temp */

    printf("itemp = %d/%d\t best error = %lf\n",itemp,ntemp,error_best);
    fprintf(sfile,"%f\n",error_best);
	for(i=0;i<nparameters;i++)fprintf(pfile,"%11.8f\t",param_best[i]);
	fprintf(pfile,"\n");

/* temperature reduction */
  
    temp[i]=temp[i]*cool; 
  
  } /* close outer loop */
  
fclose(sfile);  
fclose(pfile);

/* output */
  
for(i=0;i<nparameters;i++)bestparam[i]=param_best[i];

fprintf(stdout,"best error = %lf\n",error_best);
fprintf(stdout,"best parameter set = ");
for(i=0;i<nparameters;i++)fprintf(stdout,"%lf\t",param_best[i]);
fprintf(stdout,"\n");

return(error_best);

}


/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::A_n(double sb,double ss,double st,int n)

{
	
double an1,an2,an3,an4,an;

an1=((sb/ss+1.0)*n+1.0)*((ss/st+1.0)*n+1.0);
an2=(sb/ss-1.0)*(ss/st-1.0)*n*(n+1.0)*pow((ra/rb),(2.0*n+1.0));
an3=(ss/st-1.0)*(n+1.0)*((sb/ss+1.0)*n+1.0)*pow((rb/rc),(2.0*n+1.0));
an4=(sb/ss-1.0)*(n+1.0)*((ss/st+1.0)*(n+1.0)-1.0)*pow((ra/rc),(2.0*n+1.0));

an=pow((2.0*n+1.0),3.0)/(2.0*n)/(an1+an2+an3+an4);

return(an);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::S_n(double sb,double ss,double st,int n)

{

double an,sn;
	
an=A_n(sb,ss,st,n);

sn=(an/pow(rc,1.0*n))*((1.0+sb/ss)*n+1.0)/(2.0*n+1.0);

return(sn);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::U_n(double sb,double ss,double st,int n)

{
	
double an,un;

an=A_n(sb,ss,st,n);

un=(an/pow(rc,1.0*n))*(n/(2.0*n+1.0))*(1.0-sb/ss)*pow(ra,(2.0*n+1.0));

return(un);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::T_n(double sb,double ss,double st,int n)

/* note that factor rc^n has been cancelled analytically with 
term appearing in expression for potential at r=rc */

{
	
double an,tn1,tn2,tn3,tnfact,tn;

an=A_n(sb,ss,st,n);

tn1=(1.0+sb/ss)*n+1.0;
tn2=(1.0+ss/st)*n+1.0;
tn3=n*(n+1.0)*(1.0-sb/ss)*(1.0-ss/st)*pow((ra/rb),(2.0*n+1.0));
tnfact=tn1*tn2+tn3;

tn=an*tnfact/pow((2.0*n+1.0),2.0);

return(tn);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

double Anneal::W_n(double sb,double ss,double st,int n)

/* note that factor 1/rc^(n+1) has been cancelled analytically with 
term appearing in expression for potential at r=rc, as well as some
regrouping within the following expressions */

{
	
double an,wn1,wn2,wn3,wn4,wnfact,wn;

an=A_n(sb,ss,st,n);

wn1=(1.0-ss/st)*pow((rb/rc),(2.0*n+1.0));
wn2=(1.0+sb/ss)*n+1.0;
wn3=(1.0-sb/ss)*pow((ra/rc),(2.0*n+1.0));
wn4=(1.0+ss/st)*n+ss/st;
wnfact=wn1*wn2+wn3*wn4;

wn=n*an*wnfact/pow((2.0*n+1.0),2.0);

return(wn);

}

/* XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX */

void Anneal::LegendreP(int n, double x, polynomial pn)

/* 
computes Legendre polynomial of order n and argument x.
since uses recursion relation, computes all polynomials up to order n.
returns array p with nth polynomial of x in p[n]
*/

{

int j;
double c1,c2;
polynomial p;

if(n>=0)p[0]=1.0;
if(n>=1)p[1]=x;
if(n>=2)
  {
    for(j=2;j<=n;j++)
      {
	c1=(2.0*j-1.0)/j;
	c2=(j-1.0)/j;
	p[j]=c1*x*p[j-1]-c2*p[j-2];
      }
  }

for(j=0;j<=n;j++)pn[j]=p[j];
  
return;

}
} // End namespace Modules
} // End namespace Uintah


//
// $Log$
// Revision 1.1  1999/08/24 06:23:04  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:15  dmw
// Added and updated DaveW Datatypes/Modules
//
// Revision 1.1.1.1  1999/04/24 23:12:16  dav
// Import sources
//
//
