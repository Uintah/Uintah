/*
 *  Focusing.cc:
 *
 *  Written by:
 *   oleg
 *   TODAY'S DATE HERE
 *
 */

#include <sci_defs.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/share/share.h>

void focusing(double *f,double *d,double *r,double *w,double *m,
              double noise,double mu,double ml,int fcsdg,int Nd,int Nm);

namespace MatlabInterface {

using namespace SCIRun;

class MatlabInterfaceSHARE Focusing : public Module 
{
  GuiString noiseGUI;
  GuiString fcsdgGUI;
  MatrixIPort *iport1;
  MatrixIPort *iport2;
  MatrixOPort *oport1;
  MatrixOPort *oport2;

public:
  Focusing(const string& id);
  virtual ~Focusing();
  virtual void execute();
};

extern "C" MatlabInterfaceSHARE Module* make_Focusing(const string& id) {
  return scinew Focusing(id);
}

Focusing::Focusing(const string& id)
  : Module("Focusing", id, Filter, "Math", "MatlabInterface"), noiseGUI("noiseGUI",id,this),
    fcsdgGUI("fcsdgGUI", id, this)
  //  : Module("Focusing", id, Source, "Math", "MatlabInterface")
{
}

Focusing::~Focusing(){}


void Focusing::execute()
{

// DECLARATIONS

  double noise;
  int fcsdg; 
  double *F,*d,*m,*r,*w; 
  int    Nd,Nm;

  MatrixHandle mh1,mh2,mh3,mh4;
  iport1 = (MatrixIPort *)get_iport("Lead Field");
  iport2 = (MatrixIPort *)get_iport("RHS (data)");
  oport1 = (MatrixOPort *)get_oport("Sources");
  oport2 = (MatrixOPort *)get_oport("Residual");

  if (!iport1) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!iport2) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!oport1) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!oport2) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  DenseMatrix  *inp1;   // Sensitivity matrix
  ColumnMatrix *inp2;   // data (right-hand side)
  ColumnMatrix *otp1;   // model
  ColumnMatrix *otp2;   // residual
  ColumnMatrix *wwww;   // weighting matrix

// OBTAIN SCALAR PARAMETERS FROM GUI

  noise=atof(noiseGUI.get().c_str());
  fcsdg=atoi(fcsdgGUI.get().c_str());

// OBTAIN F FROM FIRST INPUT PORT

  iport1->get(mh1);
  inp1=dynamic_cast<DenseMatrix*>(mh1.get_rep()); //upcast
  if(inp1==NULL) 
  {
    fprintf(stderr,"Focusing needs DenseMatrix as first input\n");
    return;
  }
  F=(double*)&((*inp1)[0][0]);

// OBTAIN PROBLEM DIMENSIONS

  Nd=inp1->nrows();
  Nm=inp1->ncols();

// OBTAIN d FROM SECOND INPUT PORT

  iport2->get(mh2);
  inp2=dynamic_cast<ColumnMatrix*>(mh2.get_rep()); //upcast
  if(inp2==NULL) 
  {
    fprintf(stderr,"Focusing needs ColumnMatrix as second input\n");
    return;
  }
  d=(double*)&((*inp2)[0]);

// CREATE m

  otp1=scinew ColumnMatrix(Nm);
  mh3=MatrixHandle(otp1);
  m=&((*otp1)[0]);

// CREATE r
  
  otp2=scinew ColumnMatrix(Nd);
  mh4=MatrixHandle(otp2);
  r=&((*otp2)[0]);

// CREATE w
  
  wwww=scinew ColumnMatrix(Nm);
  w=&((*wwww)[0]);
//  for(int i=0;i<Nm;i++) w[i]=1.;


  /* Compute w */

  for(int i1=0;i1<Nm;i1++) 
  {
    double t;
    t=0.;
    for(int i2=0;i2<Nd;i2++) t+=F[i1+i2*Nm]*F[i1+i2*Nm]; 
    if(t==0) w[i1]=0.;
    else     w[i1]=1./sqrt(sqrt(t));
  }

// ACTUAL OPERATION 

  focusing(F,d,r,w,m,noise,1e5,-1e+5,(int)fcsdg,Nd,Nm);

// fprintf(stderr,"Noise, fcsdg= %g %g\n",noise,fcsdg); 
// for(int i=0;i<Nm;i++) m[i]=F[i];
// for(int i=0;i<Nd;i++) r[i]=d[i];
  
// SEND RESULTS DOWNSTREAM

  oport1->send(mh3);
  oport2->send(mh4);
}

} // End namespace MatlabInterface

#define in_SCIRun

#ifndef MLB_HPP
#define MLB_HPP

#include <stdio.h>
//--------- BASE CLASS AND mlb.cpp functions ------------------

class mlb 
{
 public:
  char **nms;  // Names of the variable. If NULL, temporary
  int  Nnms;   // How many variables currently

  int type;    /* type 
                  0 : to be destructed
                  1 : empty 
                  2 : double 
               */

  int ndms;    // number of dimensions
  int *dms;    // dimensions

  double *db;  // data container

  mlb(); 
  mlb(char *name);

  int sizedb();
  void operator = (mlb & t);  
  void operator = (double t);  
  void operator = (int t) { *this=(double)t; } 
  ~mlb(); 
  void thisprt(char *s);

  void set_double_ptr(int a,int b, double *c);
  void release_double_ptr();

  operator bool ();             // mlb1.cpp functions
  void mkmatrix(int n1,int n2); 
  mlb & operator + ();
  mlb & operator - ();
  void same(mlb &a);

  mlb & operator () (mlb &i);
  void assign_ind(mlb &i,mlb &what);
  void assign_ind(mlb &i,double what);

  void operator ++ ();          // for loop, prefix


};

void show(mlb &a);
mlb& a2m(char *expr);
mlb& a2m(double expr);
inline mlb& a2m(int e) {return( a2m((double)e) );}

void fatalerr(char *msg);
void chkout(mlb & a);
int  istmp(mlb & a);
char *currname(mlb &a);
mlb & assume(mlb &tmp);
void chkin(mlb &a,char *nm);
void MLBerr(bool flag,char *msg); 

//------------------------ mlb1.cpp functions ------------------
int is1d(mlb &a);
mlb & trnsp(mlb & abc);
int issamedms(mlb &a,mlb &b);
int isscalar(mlb &a);
double mlb2double(mlb & a);

mlb& operator * (mlb & b, double ml);
mlb& operator * (mlb & a, mlb & b);
mlb& operator / (mlb & a, mlb &b);   // Divide by scalar
inline mlb& operator * (double m,mlb &a){return(a*m);}
// inline mlb& operator * (mlb& a,int m){return(a*(double)m);}
inline mlb& operator * (int m,mlb& a){return(a*(double)m);}
// inline mlb& operator / (mlb& a,double b){return(a/a2m(b));}
// inline mlb& operator / (double b,mlb &a){return(a2m(b)/a);}
// inline mlb& operator / (mlb& a,int b){return(a/a2m(b));}
// inline mlb& operator / (int b,mlb &a){return(a2m(b)/a);}

int   mlb2size(mlb &b);

mlb& size(mlb &a, mlb &b);
inline mlb& size(mlb &a, int b)  {return(size(a,a2m(b)));}
mlb& zeros(mlb &a, mlb &b);
// inline mlb& zeros(int a, mlb &b) {return(zeros(a2m(a),b));}
// inline mlb& zeros(int a, int  b) {return(zeros(a2m(a),a2m(b)));}
inline mlb& zeros(mlb& a, int b) {return(zeros(a,a2m(b)));}
mlb& ones(mlb &a, mlb &b);
// inline mlb& ones(int a, mlb &b) {return(ones(a2m(a),b));}
// inline mlb& ones(int a, int  b) {return(ones(a2m(a),a2m(b)));}
inline mlb& ones(mlb& a, int b) {return(ones(a,a2m(b)));}

mlb& length(mlb &a);
mlb & sqrt(mlb & a);
mlb & abs(mlb & a);
mlb & sum(mlb & a);
mlb & min(mlb & a);
mlb & max(mlb & a);
mlb & operator << (mlb &a, mlb &b);    // [a b]
mlb & operator >> (mlb &a, mlb &b);    // [a;b]
mlb & dmlt(mlb & a, mlb & b);
void  disp(mlb &a);
mlb& operator % (mlb &a,mlb &b);      // [a:b]
// inline mlb& operator % (double a,mlb &b){return(a2m(a)%b);}
// inline mlb& operator % (mlb& a,double b){return(a%a2m(b));}
inline mlb& operator % (int a,mlb &b){return(a2m(a)%b);}
// inline mlb& operator % (mlb& a,int b){return(a%a2m(b));}

// mlb& plus(double a1,mlb&c1,double a2)        a1*c1+a2
// mlb& plus(double a1,mlb&c1,double a2,mlb&c2) a1*c1+a2*c2
mlb& operator + (mlb& a, mlb& b);
mlb& operator + (mlb& b, double ml);
// inline mlb& operator + (double m,mlb & b) {return(b+m);}
// inline mlb& operator + (int m,mlb & b) {return(b+(double)m);}
// inline mlb& operator + (mlb &b, int m) {return(b+(double)m);}
inline mlb& operator - (mlb& a, mlb& b){return(-b+a);}
// inline mlb& operator - (mlb &b, double m) {return(b+(double)(-m));}
// inline mlb& operator - (double m, mlb &b) {return(-b+m);}
inline mlb& operator - (mlb &b, int m) {return(b+(double)(-m));}
inline mlb& operator - (int m, mlb &b) {return(-b+(double)m);}

mlb & find(mlb & a);

mlb& operator <  (double a, mlb & b);
mlb& operator <  (mlb & a, double b);
mlb& operator <  (mlb & a, mlb & b);
mlb& operator <= (double a, mlb & b);
mlb& operator <= (mlb&  a, double b);
mlb& operator <= (mlb & a, mlb & b);
mlb& operator == (double a, mlb & b);
mlb& operator == (mlb & a, mlb & b);
inline mlb& operator > (mlb& a, mlb& b)  { return( b<a ); }
// inline mlb& operator > (double a, mlb& b) { return( b<a ); }
// inline mlb& operator > (mlb& a, double b) { return( b<a ); }
inline mlb& operator >=(mlb& a, mlb& b) { return( b<=a ); }
// inline mlb& operator >=(double a, mlb& b) { return( b<=a ); }
// inline mlb& operator >=(mlb& a, double b) { return( b<=a ); }
inline mlb& operator ==(mlb&b, double a) { return(a==b); }

// inline mlb& operator <  (int a, mlb & b){return((double)a<b);}
inline mlb& operator <  (mlb & a, int b){return(a<(double)b);}
// inline mlb& operator <= (int a, mlb & b){return((double)a<=b);}
inline mlb& operator <= (mlb&  a, int b){return(a<=(double)b);}
// inline mlb& operator == (int a, mlb & b){return((double)a==b);}
// inline mlb& operator > (int a, mlb& b) { return( b<(double)a );  }
inline mlb& operator > (mlb& a, int b) { return( (double)b<a );  }
// inline mlb& operator >=(int a, mlb& b) { return( b<=(double)a ); }
// inline mlb& operator >=(mlb& a, int b) { return( (double)b<=a ); }
inline mlb& operator ==(mlb& b, int a) { return( (double)a==b ); }

int  MLBempty(mlb &a);

#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef __sgi
#include <iostream.h>
#endif

/****************************************************
   Construct empty temporary object
****************************************************/

mlb::mlb()         
{
 nms=new char* [1];
 Nnms=1;
 nms[0]=NULL;

 type=1;
 ndms=0;
 dms=NULL;
 db=NULL;
 thisprt(" Construct etmp");
}

/****************************************************
   Construct empty object with name
****************************************************/

mlb::mlb(char *name)   
{
  nms=new char* [1];
  Nnms=1;
  nms[0]=name;

  type=1;
  ndms=0;
  dms=NULL;
  db=NULL;
  thisprt("Construct eobj");
}

/****************************************************
   Size of data function
****************************************************/

int mlb::sizedb()
{
  int sz= (ndms==0) ? 0 : 1;
  for(int k=0;k<ndms;k++) sz*=dms[k];
  return sz;
}

/****************************************************
   Assignment 10.4.5 p 246
****************************************************/

void mlb::operator = (mlb & t) 
{
   thisprt(" Assign to ");
   if(istmp(*this)) fatalerr("= : to a temporary");
   if(this==&t)    return; // self-assignment

   if(issamedms(*this,t))
   {
    for(int k=0;k<sizedb();k++) db[k]=t.db[k];
    if(istmp(t)) chkout(t);
    return;
   }

   ndms=t.ndms;
   type=t.type;
   if(dms!=NULL) {delete [] dms; dms=NULL;}
   if(db!=NULL)  {delete [] db; db=NULL;}

   if(istmp(t))   // if temporary, assume memory
   {
    dms=t.dms; 
    db=t.db; 
    t.ndms=0;
    t.dms=NULL;
    t.db=NULL;
    chkout(t);
   }
   else           // if permanent, create new memory and copy
   {
    this->same(t);
    for(int k=0;k<sizedb();k++) db[k]=t.db[k];
   }
}
/****************************************************
   Assignment from double
****************************************************/

void mlb::operator = (double t) 
{
   thisprt("Assign from double");
   MLBerr(istmp(*this),"= :  double to a temporary");
   if(!isscalar(*this))
   {
    if(dms==NULL) {delete [] dms; dms=NULL;}
    if(db==NULL)  {delete [] db; db=NULL;}
    this->mkmatrix(1,1);
   }
   db[0]=t;
}

/****************************************************
   Destructor
****************************************************/

mlb::~mlb()            
{
  thisprt("Destruct");
  if(dms!=NULL) delete [] dms;
  if(db !=NULL) delete [] db;
  if(nms!=NULL) delete [] nms;
}


/****************************************************
   Debug print
****************************************************/


#ifdef DeBug
void mlb::thisprt(char *s)
{
  printf("%15s  ",s);
  printf(" %10u",(int)this);
  printf(" %i", type );
  printf(" %s", currname(*this) );
  printf("\n");
}
#else
void mlb::thisprt(char *) {  return;}
#endif

/****************************************************
   operator without a semicolon (show)
****************************************************/

void show(mlb &a)
{

 printf("%s =\n\n",currname(a));

 if( abs(a.type) ==1)  // empty object
 {
  printf("     []\n\n");
 }

 if( abs(a.type)==2) 
 {
  if(a.ndms==1)
   for(int k=0;k<a.sizedb();k++) printf(" %9.4g\n", a.db[k]);

  if(a.ndms==2)
  {
   for(int k1=0;k1<a.dms[0];k1++) 
   {
    for(int k2=0;k2<a.dms[1];k2++) 
      printf(" %9.4g",a.db[k1+k2*a.dms[0]]);
    printf("\n");
   }
  }
  printf("\n");
 }

}

/****************************************************
   ascii to mlb conversion
****************************************************/

void getexprdms(int *n1,int *n2,char *expr)
{
 char *expr1=strdup(expr);
 
 // compute number of lines

 char *tk=strtok(expr1,";]");
 *n1=0;
 while(tk!=NULL) 
 {
   // printf("%s\n",tk);
   tk=strtok(NULL,";]");
   (*n1)++;
 }

 // compute total number of elements

 int n=0;
 strcpy(expr1,expr);
 tk=strtok(expr1," ,;[]");
 while(tk!=NULL) 
 {
   // printf("%s ",tk);
   tk=strtok(NULL," ,;[]");
   n++;
 }

 *n2=n/(*n1);

 free((void *)expr1);
}

void readexpr(int n1,int n2,char *expr,double *db)
{
 char *expr1=strdup(expr);
 int k1=0;
 int k2=0;
 char *tk=strtok(expr1," ,;[]");
 while(tk!=NULL) 
 {
   db[k1+k2*n1]=atof(tk);
   tk=strtok(NULL," ,;[]");
   k2++;
   if(k2==n2) {k2=0; k1++;}
 }
 free((void *)expr1);
}

mlb & a2m(char *expr)
{
 int n1,n2;
 
 getexprdms(&n1,&n2,expr);
 // printf("%i %i\n",n1,n2);

 mlb *tmp=new mlb [1];
 if(n1*n2==0) return *tmp;
 tmp->type=2; // double temporary

 tmp->ndms=2;
 tmp->dms=new int [ tmp->ndms ];
 tmp->dms[0]=n1;
 tmp->dms[1]=n2;
 tmp->db= new double [ tmp->sizedb() ];

 readexpr(n1,n2,expr,tmp->db);

 return *tmp;
}

/****************************************************
   conversion from double
****************************************************/

mlb & a2m(double expr)
{
 mlb *tmp=new mlb [1];
 tmp->mkmatrix(1,1);
 tmp->db[0]=expr;
 return(*tmp);
}

/****************************************************
   fatal error
****************************************************/

void fatalerr(char *msg)
{
  #ifdef in_SCIRun
    ASSERTFAIL(msg);
  #else
  //printf("ERROR: %s \n",msg);
  //exit(1);
  #endif
}

/****************************************************
   Checkout function
****************************************************/

void chkout(mlb & a)
{
  // check out one last name

  char **nms=a.nms;

  a.Nnms=a.Nnms-1;
  a.nms=NULL;
  if(a.Nnms>0) 
  {
   a.nms=new char* [a.Nnms] ;
   for(int k=0;k<a.Nnms;k++) a.nms[k]=nms[k];
  }
  delete [] nms;

  // destruct if temporary
  if(istmp(a)) a.~mlb();
}

/****************************************************
   is it currently temporary?
****************************************************/

int istmp(mlb & a)
{
  if(a.Nnms<=0)             return 1;
  if(a.nms==NULL)           return 1;
  if(a.nms[a.Nnms-1]==NULL) return 1;
  return 0;
}

/****************************************************
   get current variable name
****************************************************/

char *currname(mlb & a)
{
  char *tmp="Temporary";
  if(istmp(a)) return(tmp);
  return(a.nms[a.Nnms-1]);
}

/****************************************************
   assume function 
****************************************************/

mlb & assume(mlb & tmp)
{
  // make permanent object

  mlb *perm=new mlb [1];

  // inherit all data from tmp

  perm->type=tmp.type;
  perm->ndms=tmp.ndms;
  perm->dms=tmp.dms;
  perm->db=tmp.db;

  // prepare tmp for destruction
  // names will be destroyed together with tmp

  tmp.type=0;
  tmp.ndms=0;
  tmp.dms=NULL;
  tmp.db=NULL;

  return(*perm);
}

/****************************************************
   check in function
****************************************************/

void chkin(mlb & a, char *nm)
{
  char **nms=a.nms;

  a.Nnms++;
  a.nms=new char* [a.Nnms];
   for(int k=0;k<a.Nnms-1;k++) a.nms[k]=nms[k];
  if(nms!=NULL) delete [] nms;
  a.nms[a.Nnms-1]=nm;

}

/****************************************************
   Internal error check
****************************************************/

void MLBerr(bool flag,char *msg)
{
  if(flag)
  {
   #ifdef in_SCIRun
    ASSERTFAIL(msg);
   #else
   // printf("ERROR: %s \n",msg);
   // exit(1);
   #endif
  }
}

/****************************************************
   Is empty object
****************************************************/

int MLBempty(mlb &a)
{
  if((a.ndms==0)&&(a.db==NULL))  return(1);
  if(a.dms[0]==0) return(1);
  return(0);
}

/****************************************************
   Set double pointer
****************************************************/

void mlb::set_double_ptr(int n1,int n2,double *ptr)
{
   MLBerr((db!=NULL)||(dms!=NULL),"set_double for non-empty object");
   if(n2==1)
   {
    ndms=1;
    dms=new int [1];
   }
   else
   {
    ndms=2;
    dms=new int [2];
    dms[1]=n2;
   }
   type=2;
   dms[0]=n1;
   db=ptr;
}

/****************************************************
   Release double pointer
****************************************************/

void mlb::release_double_ptr()
{
   MLBerr( (db==NULL)&&(dms==NULL) ,"release_double for empty object");
   delete [] dms;
   dms=NULL;
   db=NULL;  // This is not deleted because is does not belong here
}

/****************************************************
    BLAS functions 
****************************************************/

/****************************************************
   is it one-dimensional?
****************************************************/

int is1d(mlb &a)
{
 if(a.ndms==0) return 0;
 int d=0;
 for(int k=0;k<a.ndms;k++) if(a.dms[k]>d) d=a.dms[k];
 if(d==a.sizedb()) return 1;
 return 0;
}

/****************************************************
   transpose operator
****************************************************/

mlb & trnsp(mlb & abc)
{
 if(abs(abc.type)!=2) fatalerr("Non-double type for transpose");
 if(abc.ndms>2)       fatalerr("Attempt to transpose n-d matrix");
 if(abc.ndms==0)      fatalerr("Zero-length transpose");

 mlb tmp("tmp");
 int istmpabc=istmp(abc);
 chkin(abc,"abc");

 tmp.type=2;  
 tmp.ndms=2;
 tmp.dms=new int [2];

 tmp.dms[0]= (abc.ndms==1) ? 1 : abc.dms[1] ;
 tmp.dms[1]=  abc.dms[0];
 
 if( istmpabc & is1d(abc))   // if temporary and one-dimensional
 {                           // assume data memory
   tmp.db=abc.db;
   abc.db=NULL;
 }
 else                        // new memory and copy
 {
   tmp.db= new double [ tmp.sizedb() ];
   for(int k1=0;k1<tmp.dms[0];k1++)
    for(int k2=0;k2<tmp.dms[1];k2++)
      tmp.db[ k1 + k2*tmp.dms[0] ]= abc.db[ k2 + k1*abc.dms[0] ];
 }

 if(is1d(tmp)&&(tmp.dms[1]==1)) // remove trailing singleton
 {
   int n=tmp.dms[0];
   tmp.ndms=1;
   delete [] tmp.dms;
   tmp.dms=new int [1];
   tmp.dms[0]=n;
 }

 chkout(abc);
 return (assume(tmp));
}

/****************************************************
   are dimenstions the same?
****************************************************/

int issamedms(mlb &a,mlb &b)
{
 int n;

 if(a.ndms==b.ndms) n=b.ndms;
 /*
 else               return 0;
 */
 else
 {
  if(b.ndms < a.ndms) 
  {
   n=b.ndms; 
   for(int k=n;k<a.ndms;k++) if(a.dms[k]!=1) return 0;
  }
  else
  {
   n=a.ndms;
   for(int k=n;k<b.ndms;k++) if(b.dms[k]!=1) return 0;
  }
  if(n==0) return 0;
 }

 int isum=0;
 for(int k=0;k<n; k++) isum+=a.dms[k]-b.dms[k];

 if(isum!=0) return 0;
 return 1;
}

/****************************************************
   Operator plus
****************************************************/

mlb& operator + (mlb & b, mlb & a)
{
 if((a.type!=2)||(b.type!=2)) fatalerr("+ of non-double");
 if(isscalar(a))  return( b+mlb2double(a) );
 if(isscalar(b))  return( a+mlb2double(b) );
 if(!issamedms(a,b)) fatalerr("+: dimensions must agree");

 mlb tmp;
 tmp.type=2;
 tmp.ndms=a.ndms;
 tmp.dms=new int [tmp.ndms];
 for(int k=0;k<a.ndms; k++) tmp.dms[k]=a.dms[k];

 if(istmp(a))
 {
   tmp.db= a.db;
   a.db=NULL;
   for(int k=0; k<tmp.sizedb(); k++)  tmp.db[k]+=b.db[k];
 }
 else if(istmp(b))
 {
   tmp.db= b.db;
   b.db=NULL;
   for(int k=0; k<tmp.sizedb(); k++)  tmp.db[k]+=a.db[k];
 }
 else   // none are temporary
 {
   tmp.db= new double [ tmp.sizedb() ];
   for(int k=0; k<tmp.sizedb(); k++)  tmp.db[k]=b.db[k]+a.db[k];
 }

 if(istmp(a)) chkout(a);
 if(istmp(b)) chkout(b);
 return(assume(tmp));
}

/****************************************************
   is it non-empty scalar?
****************************************************/

int isscalar(mlb &a)
{
 if(a.sizedb()==1) return 1;
 return 0;
}
/****************************************************
   multiply by double
****************************************************/

mlb & operator * (mlb & b, double ml)
{
  if(b.type!=2)
    fatalerr("Scalar multiply of non-double is not defined");

  if(b.sizedb()==0)
    fatalerr("Scalar multiply by empty is not implemented");

  mlb tmp;
  tmp.type=b.type;
  tmp.ndms=b.ndms;
  if(istmp(b))
  {
    tmp.dms=b.dms;
    tmp.db=b.db;
    b.ndms=0;
    b.dms=NULL;
    b.db=NULL;
    chkout(b);
    for(int k=0;k<tmp.sizedb();k++) tmp.db[k]*=ml;
  }
  else
  {
    int k;
    tmp.dms=new int[tmp.ndms];
    for(k=0;k<tmp.ndms;k++) tmp.dms[k]=b.dms[k];
    tmp.db=new double [tmp.sizedb()];
    for(k=0;k<tmp.sizedb();k++) tmp.db[k]=b.db[k]*ml;
  }
  return(assume(tmp));
}

/****************************************************
   mlb 2 double conversion
****************************************************/

double mlb2double(mlb &a)
{
  if((isscalar(a)==0)||(a.type!=2)) 
    fatalerr("Attempt to convert non-scalar to double");
 
  double ml=a.db[0];
  if(istmp(a)) chkout(a);
  return ml;
}

/****************************************************
   multiply operator
****************************************************/

#ifdef BLAS_ENABLED
extern "C" {
void dgemm_(char *ta,char *tb,int *m,int *n,int *k,double *alpha,
            double *A, int *lda,double *B,int *ldb,double *beta,double *c,int *ldc);
}
#endif


mlb & operator * (mlb &b, mlb &a)
{
 if(isscalar(a))  return( b*mlb2double(a) ); 
 if(isscalar(b))  return( a*mlb2double(b) ); 

 if((a.type!=2)||(b.type!=2))
    fatalerr("Multiply of non-double is undefined");

 if((a.ndms>2)||(b.ndms>2))
    fatalerr("Multiply on n-d matrix is undefined");

 if((a.sizedb()==0)||(b.sizedb()==0))
    fatalerr("Multiply by empty is not implemented");

 // MATRIX-TO-MATRIX MULTIPLY
  
 if(b.dms[1] != a.dms[0])
    fatalerr("Multiply: dimensions must agree");

 mlb tmp;
 tmp.type=2;
 tmp.ndms=2;
 tmp.dms=new int [tmp.ndms];

 tmp.dms[0]=b.dms[0];
 tmp.dms[1]=(a.ndms==1)? 1 : a.dms[1];
 tmp.db= new double [ tmp.sizedb() ];

#ifdef BLAS_ENABLED

 double *B=a.db,*A=b.db,*C=tmp.db,alpha=1.,beta=0.;
 int   m=tmp.dms[0],n=tmp.dms[1],k=a.dms[0];
 dgemm_("N","N",&m,&n,&k,&alpha,A,&m,B,&k,&beta,C,&m);

#else

//#error BlAS_ENABLED should be set

 for(int k1=0; k1<tmp.dms[0]; k1++)
  for(int k2=0; k2<tmp.dms[1]; k2++)
  {
    double sum=0;
    for(int k3=0; k3<a.dms[0]; k3++)
      sum+=b.db[k1+k3*b.dms[0]]*a.db[k3+k2*a.dms[0]];
    tmp.db[k1+k2*tmp.dms[0]]=sum;
  }

#endif

 if(istmp(b)) chkout(b);
 if(istmp(a)) chkout(a);
 return(assume(tmp));
}

/****************************************************
   operator bool (for use in if,while, for, etc)
****************************************************/

mlb::operator bool()
{
 bool res;
 if(!isscalar(*this)) fatalerr("Non-scalar bool is not implemented");
 thisprt("bool");
 res= this->db[0]!=0.;
 if(istmp(*this)) chkout(*this);
 return res;
}
/****************************************************
   allocate space for matrix 
****************************************************/

void mlb::mkmatrix(int n1,int n2)
{
 if((n1<=0)||(n2<=0)) fatalerr("mkmatrix: zero-length");

 type=2;
 if(n2==1)
 {
  ndms=1;
  dms=new int [ndms];
 }
 else
 {
  ndms=2;
  dms=new int[ ndms ];
  dms[1]=n2;
 }
 dms[0]=n1;
 db= new double [ sizedb() ];
}

/****************************************************
   unary plus
****************************************************/

mlb & mlb::operator + ()
{
 if(this->type!=2) fatalerr("Unary plus of non-double");
 return *this;
}

/****************************************************
   unary minus
****************************************************/

mlb & mlb::operator - ()
{
 if(this->type!=2) fatalerr("Unary minus of non-double");
 return (*this)*(-1.);
}

/****************************************************
   mlb 2 (scalar int size) conversion
****************************************************/

int mlb2size(mlb &b)
{
  if(!isscalar(b)) fatalerr("mlb2size: non-scalar");
  if(b.db[0]<=0) fatalerr("mlb2size: non-positive");
  int n=(int)b.db[0];
  if(n!=b.db[0]) fatalerr("mlb2size: non-integer");
  if(istmp(b)) chkout(b);
  return(n);
}


/****************************************************
   size function
****************************************************/

mlb & size(mlb & a, mlb &b)
{
  int n=mlb2size(b);
  mlb tmp;
  tmp.mkmatrix(1,1);
  tmp.db[0]=(n-1<a.ndms)? a.dms[ n-1 ] : 1;

  if(istmp(a)) chkout(a);
  return(assume(tmp));
}

/****************************************************
   length function
****************************************************/

mlb & length(mlb & a)
{
  int mx=0;
  for(int k=0;k<a.ndms;k++) if(a.dms[k]>mx) mx=a.dms[k];

  mlb tmp;
  tmp.mkmatrix(1,1);
  tmp.db[0]=mx;

  if(istmp(a)) chkout(a);
  return(assume(tmp));
}


/****************************************************
   zeros function
****************************************************/

mlb & zeros(mlb & a, mlb &b)
{
  mlb tmp;
  tmp.mkmatrix(mlb2size(a),mlb2size(b));
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=0;
  return(assume(tmp));
}

/****************************************************
   ones function
****************************************************/

mlb & ones(mlb & a, mlb &b)
{
  mlb tmp;
  tmp.mkmatrix(mlb2size(a),mlb2size(b));
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=1.;
  return(assume(tmp));
}

/****************************************************
   allocate object of the same size and type
****************************************************/

void mlb::same(mlb &a)
{
 if((dms!=NULL)||(db!=NULL)) fatalerr("same: non-empty object");

 type=a.type;
 ndms=a.ndms;
 if(ndms!=0)
 {
  dms=new int[ ndms ];
  for(int k=0;k<ndms; k++) dms[k]=a.dms[k];
  db= new double [ sizedb() ];
 }
}

/****************************************************
   operator divide
****************************************************/

mlb & operator / (mlb & a, mlb &b)
{
  double t=mlb2double(b);
  if(t==0) fatalerr("/: Divide by zero");

  if(istmp(a))
  {
    for(int k=0;k<a.sizedb();k++) a.db[k]/=t;
    return(a);
  }

  mlb tmp;
  tmp.same(a);
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=a.db[k]/t;
  return(assume(tmp));

}

/****************************************************
   function sqrt
****************************************************/

mlb & sqrt(mlb & a)
{
  if(a.type!=2) fatalerr("sqrt of non-double");
  if(istmp(a))
  {
    for(int k=0;k<a.sizedb();k++) a.db[k]=sqrt(a.db[k]);
    return(a);
  }
  mlb tmp;
  tmp.same(a);
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=sqrt(a.db[k]);
  return(assume(tmp));
}

/****************************************************
   function abs
****************************************************/

mlb & abs(mlb & a)
{
  if(a.type!=2) fatalerr("abs of non-double");
  if(istmp(a))
  {
    for(int k=0;k<a.sizedb();k++) a.db[k]=fabs(a.db[k]);
    return(a);
  }
  mlb tmp;
  tmp.same(a);
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=fabs(a.db[k]);
  return(assume(tmp));
}

/****************************************************
   function sum
****************************************************/

mlb & sum(mlb & a)
{
  if(a.type!=2) fatalerr("sum of non-double");
  if(!is1d(a))   fatalerr("sum of non-1d is not implemented");

  double sm=0.;
  for(int k=0;k<a.sizedb();k++) sm+=a.db[k];

  mlb tmp;
  tmp.mkmatrix(1,1);
  tmp.db[0]=sm;

  if(istmp(a)) chkout(a);
  return(assume(tmp));
}

/****************************************************
   function min
****************************************************/

mlb & min(mlb & a)
{
  if(a.type!=2) fatalerr("min of non-double");
  if(!is1d(a))  fatalerr("min of non-1d is not implemented");

  double sm=a.db[0];
  for(int k=0;k<a.sizedb();k++) 
     if(a.db[k]<sm) sm=a.db[k];

  mlb tmp;
  tmp.mkmatrix(1,1);
  tmp.db[0]=sm;

  if(istmp(a)) chkout(a);
  return(assume(tmp));
}

/****************************************************
   function max
****************************************************/

mlb & max(mlb & a)
{
  if(a.type!=2) fatalerr("min of non-double");
  if(!is1d(a))  fatalerr("min of non-1d is not implemented");

  double sm=a.db[0];
  for(int k=0;k<a.sizedb();k++)
     if(a.db[k]>sm) sm=a.db[k];

  mlb tmp;
  tmp.mkmatrix(1,1);
  tmp.db[0]=sm;

  if(istmp(a)) chkout(a);
  return(assume(tmp));
}


/****************************************************
   operator append horizontal [a b] = a<<b;
****************************************************/

mlb & operator << (mlb & a, mlb & b)
{
  if((a.type!=2)||(b.type!=2))  fatalerr("combin of non-doubles");
  if((a.ndms>2)||(b.ndms>2)) fatalerr("combine of n-d matrices");
  if(a.dms[0]!=b.dms[0]) fatalerr("combine: first dimensions disagree");

  int n1=a.dms[0];
  int na2=(a.ndms==1)? 1 : a.dms[1];
  int nb2=(b.ndms==1)? 1 : b.dms[1];
  int n2=na2+nb2;

  mlb tmp;
  tmp.mkmatrix(n1,n2);
  int k1,k2;

  for(k1=0;k1<n1;k1++)
   for(k2=0;k2<na2;k2++) 
     tmp.db[k1+k2*n1]=a.db[k1+k2*n1];

  for(k1=0;k1<n1;k1++)
   for(k2=0;k2<nb2;k2++) 
     tmp.db[k1+(na2+k2)*n1]=b.db[k1+k2*n1];

  if(istmp(a)) chkout(a);
  if(istmp(b)) chkout(b);
  return(assume(tmp));
}

/****************************************************
   dot-multiply
****************************************************/

mlb & dmlt(mlb & a, mlb & b)
{
  if((a.type!=2)||(b.type!=2))  fatalerr("dmlt of non-doubles");
  if(!issamedms(a,b)) fatalerr("dmlt: sizes disagree");

  if(istmp(a))
  {
    for(int k=0;k<a.sizedb();k++) a.db[k]*=b.db[k];
    if(istmp(b)) chkout(b);
    return(a);
  }
  if(istmp(b))
  {
    for(int k=0;k<a.sizedb();k++) b.db[k]*=a.db[k];
    if(istmp(a)) chkout(a);
    return(b);
  }

  mlb tmp;
  tmp.same(a);
  for(int k=0;k<tmp.sizedb();k++) tmp.db[k]=b.db[k]*a.db[k];

  if(istmp(a)) chkout(a);
  if(istmp(b)) chkout(b);
  return(assume(tmp));
}

/****************************************************
   function disp
****************************************************/

void disp(mlb &a)
{
 if( a.type ==1)  // empty object
 {
  printf("[]\n");
 }

 if( a.type==2)
 {
  if(a.ndms==1)
   for(int k=0;k<a.sizedb();k++) printf(" %9.4g\n", a.db[k]);

  if(a.ndms==2)
  {
   for(int k1=0;k1<a.dms[0];k1++)
   {
    for(int k2=0;k2<a.dms[1];k2++) printf(" %9.4g",a.db[k1+k2*a.dms[0]]);
    printf("\n");
   }
  }
  printf("\n");
 }
 if(istmp(a)) chkout(a);
}

/****************************************************
   function colon   a:b
****************************************************/

mlb & operator % (mlb &a,mlb & b)
{
 double d1=mlb2double(a); // checked out already
 double d2=mlb2double(b);

  mlb tmp;
  tmp.mkmatrix(1,(int)(d2-d1)+1);
  for(int k=0;k<tmp.sizedb();k++,d1=d1+1) tmp.db[k]=d1;

 return(assume(tmp));
}

/****************************************************
   indexing function, 1-D, no assignment
****************************************************/

mlb & mlb::operator () (mlb &i)
{
  if(i.type!=2) fatalerr("Index is non-double"); 
  if(!is1d(i))  fatalerr("Index is not 1-D"); 
  if(i.sizedb()==0)  fatalerr("Index is empty"); 
  int itmp;

  if(istmp(i))
  {
   for(int k=0;k<i.sizedb();k++) 
   {
    itmp=(int) i.db[k];
    if((double)itmp!=i.db[k]) fatalerr("Index is non-integer");
    if(itmp>sizedb()) fatalerr("Index exceeds dimenstions");
    if(itmp<1) fatalerr("Index less than 1");
    i.db[k]= db[ itmp-1 ];
   }
   return(i);
  }
  
  mlb tmp;
  tmp.mkmatrix(i.dms[0],  (i.ndms==1) ? 1 : i.dms[1] ); 

  for(int k=0;k<tmp.sizedb();k++) 
  {
    itmp=(int) i.db[k];
    if((double)itmp!=i.db[k]) fatalerr("Index is non-integer");
    if(itmp>sizedb()) fatalerr("Index exceeds dimenstions");
    if(itmp<1) fatalerr("Index less than 1");
    tmp.db[k]= db[ itmp-1 ];
  }
  return(assume(tmp));
}

/****************************************************
   assignment with indexing, from double
****************************************************/

void mlb::assign_ind(mlb &i, double what)
{
  if(MLBempty(i)) return;
  MLBerr(i.type!=2,"assign_ind(double) index is non-double");
  MLBerr(!is1d(i),"assign_ind(double) index is not 1-D");

  for(int itmp,k=0;k<i.sizedb();k++)
  {
    itmp=(int) i.db[k];
    MLBerr((double)itmp!=i.db[k],"Index is non-integer");
    MLBerr(itmp>sizedb(),"Index exceeds dimenstions");
    MLBerr(itmp<1,"Index less than 1");
    db[ itmp-1 ] = what;
  }
  if(istmp(i))    chkout(i);
}

/****************************************************
   assignment with indexing
****************************************************/

void mlb::assign_ind(mlb &i,mlb &what)
{
  if(MLBempty(i)) return;
  MLBerr(i.type!=2,"assign_ind index is non-double");
  MLBerr(!is1d(i),"assign_ind index is not 1-D");

  if(isscalar(what)) 
  {
   this->assign_ind(i,mlb2double(what));
   return;
  }

  for(int itmp,k=0;k<i.sizedb();k++)
  {
    itmp=(int) i.db[k];
    MLBerr((double)itmp!=i.db[k],"Index is non-integer");
    MLBerr(itmp>sizedb(),"Index exceeds dimenstions");
    MLBerr(itmp<1,"Index less than 1");
    db[ itmp-1 ] = what.db[k];
  }
  if(istmp(what)) chkout(what);
}

/****************************************************
   increment operator
****************************************************/

void mlb::operator ++ ()
{
  if(!isscalar(*this))  fatalerr("++ of non-scalar");
  db[0]+= 1.;
}

/****************************************************
   add double
****************************************************/

mlb & operator + (mlb & b, double ml)
{
  if(b.type!=2)
    fatalerr("Scalar plus of non-double is not defined");

  if(b.sizedb()==0)
    fatalerr("Scalar plus of empty is not implemented");

  if(istmp(b))
  {
    for(int k=0;k<b.sizedb();k++) b.db[k]+=ml;
    return(b);
  }

  int k;
  mlb tmp;
  tmp.type=b.type;
  tmp.ndms=b.ndms;
  tmp.dms=new int[tmp.ndms];
  for(k=0;k<tmp.ndms;k++) tmp.dms[k]=b.dms[k];
  tmp.db=new double [tmp.sizedb()];
  for(k=0;k<tmp.sizedb();k++) tmp.db[k]=b.db[k]+ml;
  return(assume(tmp));
}

/****************************************************
   operator append vertical [a;b] = a>>b;
****************************************************/

mlb & operator >> (mlb & a, mlb & b)
{
  if(MLBempty(a)) 
  {
   if(istmp(a)) chkout(a);
   return(b);
  }
  if(MLBempty(b)) 
  {
   if(istmp(b)) chkout(b);
   return(a);
  }
 
  MLBerr((a.type!=2)||(b.type!=2),"append of non-doubles");
  MLBerr((a.ndms>2)||(b.ndms>2),  "append of n-d matrices");

  int na1=a.dms[0];
  int na2=(a.ndms==1)? 1 : a.dms[1];
  int nb1=b.dms[0];
  int nb2=(b.ndms==1)? 1 : b.dms[1];

  if(na2!=nb2) fatalerr("append: second dimensions disagree");
  int n1=na1+nb1;
  int n2=na2;

  mlb tmp;
  tmp.mkmatrix(n1,n2);

  int k1,k2;

  for(k1=0;k1<na1;k1++)
   for(k2=0;k2<n2;k2++)
     tmp.db[k1+k2*n1]=a.db[k1+k2*na1];

  for(k1=0;k1<nb1;k1++)
   for(k2=0;k2<n2;k2++)
     tmp.db[k1+na1+k2*n1]=b.db[k1+k2*nb1];

  if(istmp(a)) chkout(a);
  if(istmp(b)) chkout(b);
  return(assume(tmp));
}

/***************************************************
  function find
****************************************************/

mlb & find(mlb &i)
{
  if(i.type!=2) fatalerr("Find of non-double"); 
  if(!is1d(i))  fatalerr("Find of non 1-D"); 
  if(i.sizedb()==0)  fatalerr("Find of empty"); 
  int itmp=0; // how many non-zeros?

  for(int k=0;k<i.sizedb();k++) if(i.db[k]!=0) itmp++;

  mlb tmp;
  if(itmp!=0)
  {
    bool n=(i.dms[0]==1);
    tmp.mkmatrix( (n)?1:itmp , (n)?itmp:1 );

    for(int k=0,j=0;k<i.sizedb();k++) 
      if(i.db[k]!=0) tmp.db[j++]=(double) k+1 ;
  }
  if(istmp(i)) chkout(i);
  return(assume(tmp));
}

/****************************************************
   less: double < mlb&
****************************************************/

mlb & operator < (double a, mlb & b)
{
 int k;
 if(istmp(b)) 
 {
  for(k=0;k<b.sizedb();k++) 
     b.db[k]= (a<b.db[k]) ? 1. : 0. ;
  return(b);
 }
 mlb tmp;
 tmp.same(b);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a<b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   less: mlb& < double
****************************************************/

mlb & operator < (mlb & b,double a)
{
 int k;
 if(istmp(b)) 
 {
  for(k=0;k<b.sizedb();k++) 
     b.db[k]= (b.db[k]<a) ? 1. : 0. ;
  return(b);
 }
 mlb tmp;
 tmp.same(b);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (b.db[k]<a) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   less mlb& < mlb&
****************************************************/

mlb & operator < (mlb & a, mlb & b)
{
 if(isscalar(a)) return(mlb2double(a)<b);
 if(isscalar(b)) return(a<mlb2double(b));

 MLBerr( !issamedms(a,b) , "< : dimensions disagree" );

 int k;
 if(istmp(a)) 
 {
  for(k=0;k<a.sizedb();k++) 
     a.db[k]= (a.db[k]<b.db[k]) ? 1. : 0. ;
  if(istmp(b)) chkout(b); 
  return(a);
 }
 if(istmp(b)) 
 {
  for(k=0;k<a.sizedb();k++) 
     b.db[k]= (a.db[k]<b.db[k]) ? 1. : 0. ;
  return(b);
 }
 
 mlb tmp;
 tmp.same(a);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a.db[k]<b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   less or equal: mlb& <= double
****************************************************/

mlb & operator <= (mlb & b,double a)
{
 int k;
 if(istmp(b)) 
 {
  for(k=0;k<b.sizedb();k++) 
     b.db[k]= (b.db[k]<=a) ? 1. : 0. ;
  return(b);
 }
 mlb tmp;
 tmp.same(b);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (b.db[k]<=a) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   less or equal: double <= mlb&
****************************************************/

mlb & operator <= (double a,mlb & b)
{
 int k;
 if(istmp(b)) 
 {
  for(k=0;k<b.sizedb();k++) 
     b.db[k]= (a<=b.db[k]) ? 1. : 0. ;
  return(b);
 }
 mlb tmp;
 tmp.same(b);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a<=b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   less or equal
****************************************************/

mlb & operator <= (mlb & a, mlb & b)
{
 if(isscalar(a)) return(mlb2double(a)<=b);
 if(isscalar(b)) return(a<=mlb2double(b));

 MLBerr( !issamedms(a,b) , "<= : dimensions disagree" );
 int k;
 if(istmp(a)) 
 {
  for(k=0;k<a.sizedb();k++) 
     a.db[k]= (a.db[k]<=b.db[k]) ? 1. : 0. ;
  if(istmp(b)) chkout(b); 
  return(a);
 }
 if(istmp(b)) 
 {
  for(k=0;k<a.sizedb();k++) 
     b.db[k]= (a.db[k]<=b.db[k]) ? 1. : 0. ;
  return(b);
 }
 
 mlb tmp;
 tmp.same(a);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a.db[k]<=b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   equal: double == mlb&
****************************************************/

mlb & operator == (double a,mlb & b)
{
 int k;
 if(istmp(b)) 
 {
  for(k=0;k<b.sizedb();k++) 
     b.db[k]= (a==b.db[k]) ? 1. : 0. ;
  return(b);
 }
 mlb tmp;
 tmp.same(b);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a==b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

/****************************************************
   equal
****************************************************/

mlb & operator == (mlb & a, mlb & b)
{
 if(isscalar(a)) return(mlb2double(a)==b);
 if(isscalar(b)) return(a==mlb2double(b));
 MLBerr( !issamedms(a,b) , "== : dimensions disagree" );
 int k;
 if(istmp(a)) 
 {
  for(k=0;k<a.sizedb();k++) 
     a.db[k]= (a.db[k]==b.db[k]) ? 1. : 0. ;
  if(istmp(b)) chkout(b); 
  return(a);
 }
 if(istmp(b)) 
 {
  for(k=0;k<a.sizedb();k++) 
     b.db[k]= (a.db[k]==b.db[k]) ? 1. : 0. ;
  return(b);
 }
 
 mlb tmp;
 tmp.same(a);
 
 for(k=0;k<tmp.sizedb();k++) 
   tmp.db[k]= (a.db[k]==b.db[k]) ? 1. : 0. ;
  
 return(assume(tmp));
}

//
// Regularized conjugate gradient 
//
//

mlb& cnjgrd1(mlb& f, mlb& w, mlb& dd, mlb& phi1, mlb& mu, mlb& ml){

chkin(f,"f");
chkin(w,"w");
chkin(dd,"dd");
chkin(phi1,"phi1");
chkin(mu,"mu");
chkin(ml,"ml");

mlb Nm("Nm");
mlb Nd("Nd");
mlb m("m");
mlb sqrtalp("sqrtalp");
mlb s2old("s2old");
mlb rold("rold");
mlb h("h");
mlb mm("mm");
mlb it("it");
mlb conjug("conjug");
mlb ii("ii");
mlb estn("estn");
mlb s("s");
mlb snorm("snorm");
mlb jj("jj");
mlb fh("fh");
mlb fh2("fh2");
mlb a("a");
mlb b("b");
mlb c("c");
mlb d("d");
mlb step("step");
mlb r("r");

/****************************************************************/

Nm=size(w,1);
Nd=size(dd,1);

m=zeros(Nm+Nd,1);

r=dd;
sqrtalp=0;
s2old=1;
rold=1e35;
h=zeros(Nm+Nd,1);

ii=trnsp(Nm+(1%Nd));
jj=trnsp(1%Nm);

for(it=1;it<=Nd;++it){

 conjug=trnsp(r)*r;
 estn=0.1*sqrtalp*sqrtalp*trnsp(m(ii))*m(ii);
 if(conjug<estn){ break; }
// disp( it << sqrt(conjug) ); 

 s=( dmlt(w,trnsp(trnsp(r)*f)) >> sqrtalp*r );
 snorm=trnsp(s)*s;
 if(snorm == 0 ) break; 
 if(conjug>= rold) break;

 h=h+(snorm/s2old-1)*h;
 h=h+s;

 s2old=snorm;
 rold=conjug;

 fh=f*dmlt(w,h(jj))+sqrtalp*h(ii);
 fh2=trnsp(fh)*fh;

 if(it==1){

  a=conjug*(1-phi1);
  b=snorm*(1-2*phi1);
  c=-phi1*fh2;
  sqrtalp=sqrt( (-b+sqrt(b*b-4*a*c) )/(2*a) );

  h.assign_ind(ii,   sqrtalp*r  );

  s2old=trnsp(h)*h;

  fh=fh+sqrtalp*h(ii);
  fh2=trnsp(fh)*fh;

 }

 step=sum(dmlt(fh,r))/fh2;
 m=m-step*h;
 r=r-step*fh;

 mm=dmlt(w,m(jj));
 if((max(mm)>mu)+(min(mm)<ml)){ break; }

}

/****************************************************************/
chkout(f);
chkout(w);
chkout(dd);
chkout(phi1);
chkout(mu);
chkout(ml);
return(assume(mm));
}

//
//
// Focusing inversion for linear problem
//
//
mlb& fcsinv1(mlb& f,mlb & wm,mlb& d,mlb& noise,mlb& mu,mlb& ml,mlb& fcsdg){

chkin(f,"f");
chkin(wm,"wm");
chkin(d,"d");
chkin(noise,"noise");
chkin(mu,"mu");
chkin(ml,"ml");
chkin(fcsdg,"fcsgd");

mlb m1("m1");

mlb Nd("Nd");
mlb Nm("Nd");
mlb w("w");
mlb m3("m3");
mlb nsf("nsf");
mlb r("r");
mlb rnorm("rnorm");
mlb it("it");
mlb i2("i2");
mlb iu("iu");
mlb il("il");
mlb tmp("tmp");
mlb mx("mx");

/******************************************************************/

//disp(f);

Nd=size(f,1);
Nm=size(f,2);

w=ones(Nm,1);
m1=zeros(Nm,1);
m3=zeros(Nm,1);

nsf=noise;
if(noise<1e-8) nsf=1e-8; 
nsf=nsf*sqrt(trnsp(d)*d);

r=-d;

for(it=1;it<=fcsdg;++it){

//disp(it);

for(;;){
 rnorm=sqrt(trnsp(r)*r);
 if(rnorm <= nsf ) break;

 i2=find(w>0);
 if(length(i2)<1) break;

 m1=cnjgrd1(f,dmlt(wm,w),r,nsf/rnorm,mu,ml);

 // disp(m1);
 
 iu=find(m1>=mu); m3.assign_ind(iu, mu );
 il=find(m1<=ml); m3.assign_ind(il, ml );

 i2=(il>>iu);
 
 if(length(i2)>0){
  tmp=m3*0.;
  tmp.assign_ind(i2, m3(i2) );
  r=r+f*tmp;
  w.assign_ind(i2, 0 );
  m1.assign_ind(i2, 0 );
 }else{

  mx=max(abs(m1));
  if(mx==0){ break; }
  w=abs(m1)/mx;

  i2=find(w<=0.001);
  if(length(i2)<=1) break;
  w.assign_ind(i2, 0 );
  m1.assign_ind(i2 , 0 );
  break;

 }
}
 
}

m1=m1+m3;

/******************************************************************/

chkout(f);
chkout(wm);
chkout(d);
chkout(noise);
chkout(mu);
chkout(ml);
chkout(fcsdg);
return(assume(m1));
}


// wrapper function
void focusing(double *f,double *d,double *r,double *w,double *m,
              double noise,double mu,double ml,int fcsdg,int Nd,int Nm)
{

mlb mm("mm"),dd("dd"),ww("ww"),ff("ff"),rr("rr");

dd.set_double_ptr(Nd,1,d);
rr.set_double_ptr(Nd,1,r);
ww.set_double_ptr(Nm,1,w);
mm.set_double_ptr(Nm,1,m);

ff.set_double_ptr(Nm,Nd,f);
mlb pp("pp");
pp=trnsp(ff);

mm=fcsinv1(pp,ww,dd,a2m(noise),a2m(mu),a2m(ml),a2m(fcsdg));
rr=pp*mm-dd;

ff.release_double_ptr();
dd.release_double_ptr();
rr.release_double_ptr();
ww.release_double_ptr();
mm.release_double_ptr();

}

void tikhonov(double *f,double *d,double *r,double *w,double *m,
              double noise,double mu,double ml,int Nd,int Nm)
{

mlb mm("mm"),dd("dd"),ww("ww"),ff("ff"),rr("rr");

dd.set_double_ptr(Nd,1,d);
rr.set_double_ptr(Nd,1,r);
ww.set_double_ptr(Nm,1,w);
mm.set_double_ptr(Nm,1,m);

ff.set_double_ptr(Nm,Nd,f);
mlb pp("pp");
pp=trnsp(ff);
rr=-dd;

mm=cnjgrd1(pp,ww,rr,a2m(noise),a2m(mu),a2m(ml));
rr=pp*mm-dd;

ff.release_double_ptr();
dd.release_double_ptr();
rr.release_double_ptr();
ww.release_double_ptr();
mm.release_double_ptr();

}
