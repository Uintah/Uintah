/*

#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

 *  MatrixSend.cc:
 *
 *  Written by:
 *   oleg@cs.utah.edu
 *   01Mar14 
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Packages/MatlabInterface/share/share.h>
#include <stdio.h>

namespace MatlabInterface
{
      char *bring(int wordy,int flag,char *hostport,int lbuf,char *buf);
      void endiswap(int lbuf, char *buf,int num);
      int  endian(void);

using namespace SCIRun;

class MatlabInterfaceSHARE MatrixSend : public Module 
{
 GuiString hpTCL;
 MatrixIPort *imat1;
 MatrixIPort *imat2;
 MatrixOPort *omat;

 public:
  MatrixSend(const string& id);
  virtual ~MatrixSend();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_MatrixSend(const string& id) 
{
  return scinew MatrixSend(id);
}

MatrixSend::MatrixSend(const string& id)
: Module("MatrixSend", id, Filter, "DataIO", "MatlabInterface"), hpTCL("hpTCL",id,this)
//  : Module("MatrixSend", id, Source, "DataIO", "MatlabInterface")
{
}

MatrixSend::~MatrixSend(){}

void MatrixSend::execute()
{

  imat1 = (MatrixIPort *)get_iport("host:port string");
  imat2 = (MatrixIPort *)get_iport("Matrix being sent");
  omat = (MatrixOPort *)get_oport("host:port string");

  if (!imat1) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imat2) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!omat) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  /* OBTAIN HOST:PORT INFORMATION */

 const char *hport=hpTCL.get().c_str();
 DenseMatrix *matr;
 ColumnMatrix *cmatr;
 SparseRowMatrix *smatr;
 MatrixHandle mIH;
 double *db;
 char cb[128];
 int  lcb=sizeof(cb);
 int  wordy=2;

 if(strcmp(hport,"")!=0)  
 {
   fprintf(stderr,"Seed matr and send downstream\n");
   matr=scinew DenseMatrix(10,1);
   strcpy((char*)&((*matr)[0][0]),hport); 
   mIH=MatrixHandle(matr);
 }

 if(strcmp(hport,"")==0)
 {
  imat1->get(mIH);
  matr=dynamic_cast<DenseMatrix*>(mIH.get_rep()); //upcast
  if(matr==NULL)
  {
    fprintf(stderr,"MatrixSend needs DenseMatrix for host:port input\n");
   return;
  }
  hport=(char*)&((*matr)[0][0]);
  fprintf(stderr,"Name: %s\n",hport);
 }

/* ACTUAL SEND OPERATION */

 MatrixHandle mh;
 imat2->get(mh);
 DenseMatrix *tmp=dynamic_cast<DenseMatrix*>(mh.get_rep()); 

 if(tmp==NULL)
 {
   cmatr=dynamic_cast<ColumnMatrix*>(mh.get_rep()); //upcast
   if(cmatr==NULL)
   {
    smatr=dynamic_cast<SparseRowMatrix*>(mh.get_rep());
    if(smatr==NULL)
    {
     fprintf(stderr,"MatrixSend needs Dense-  Column- or SparseRow- Matrix as input\n");
     return;
    }
    /* SEND SPARSE MATRIX */

    int nr=smatr->nrows();
    int nc=smatr->ncols();
    int *rows=smatr->get_row();
    int *cols=smatr->get_col();
    double *d=smatr->get_val();
    int nnz=smatr->get_nnz();


   /*
    int k;
    printf("Send sparse nr nc nnz: %i %i %i\n",nr,nc,nnz);
    printf("rows:"); for(k=0;k<nc+1;k++) printf(" %i",rows[k]); printf("\n");
    printf("cols:"); for(k=0;k<nnz;k++) printf(" %i",cols[k]); printf("\n");
    printf("d   :"); for(k=0;k<nnz;k++) printf(" %g",d[k]); printf("\n");
   */

    sprintf(cb,"%i %i %i %i %i\n",nnz,9,endian(),nr,nc);
    bring(wordy,2,(char *)hport,lcb,cb);

    if(bring(wordy,2,(char*)hport,nnz*4,(char*)cols)==NULL)
       fprintf(stderr,"Not enough memory on receiving side");
    bring(wordy,2,(char*)hport,(nc+1)*4,(char*)rows);
    bring(wordy,2,(char*)hport,nnz*8,(char*)d);

    omat->send(mIH);
    return;
   }
   db=&((*cmatr)[0]);
 }
 else db=&((*tmp)[0][0]);

 int nr=mh->nrows();
 int nc=mh->ncols();
 int  sd=8;
 int  lbuf=nc*nr*sd;




 sprintf(cb,"%i %i %i %i %i\n",lbuf,sd,endian(),nr,nc);
 bring(wordy,2,(char *)hport,lcb,cb);

 if(bring(wordy,2,(char*)hport,lbuf,(char*)db)==NULL)
      fprintf(stderr,"Not enough memory on receiving side");

// fprintf(stderr,"send double data\n");
// for(int i=0;i<nr*nc;i++) fprintf(stderr,"%g ",db[i]);

/* SEND HOST:PORT DOWNSTREAM */

 omat->send(mIH);

}

void MatrixSend::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}



/* data transport routine
   Oleg onp 00Dec21

   1) to compile this file into executable test:
      [g]cc -o bring bring.c -DTEST_EXEC

   2) to run test on one machine:

      ./bring 2 2 127.0.0.1:5505 8 300 &
      ./bring 2 1 :5505
 
   3) Prototypes to use (instead of bring.h):

      char *bring(int wordy,int flag,char *hostport,int lbuf,char *buf);
      void endiswap(int lbuf, char *buf,int num);
      int  endian(void);

  4a) Altavista search: Berkeley Socket Library
  4b) SOCK_STREAM Server/Client Example from
      http://web.cs.mun.ca/~rod/Winter98/cs4759/berkeley.html
  4c) four server/client examples:
      http://blondie.mathcs.wilkes.edu/~sullivan/sockets/
*/
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
//#include <strings.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef __sgi
#define socklen_t int
#endif 

char *bring(int wordy,int flag,char *hostport,int lbuf,char *buf)
{
   struct sockaddr_in server;
   struct hostent *hp;
   int as=0,ms,len,tmp,temp;
   time_t t1=time(NULL);
   char locbuf[64];
   int smx=1500; /* SSIZE_MAX */

   char *hname=strncpy(locbuf,hostport,sizeof(locbuf));
   char *cport=strstr(hname,":");
   int port=atoi(cport+1);

   cport[0]='\0';

   if(wordy>1)
   {
    printf("Bring %s ",(flag==1)?"from":" to "); 
    printf("%s:%i ",hname,port); 
    printf("%i bts ",lbuf);
   }

   server.sin_family = AF_INET;
   server.sin_port = htons(port);

   if(flag==1) /* receive */
   {
    as=socket(AF_INET, SOCK_STREAM, 0 );
    len=1;
    setsockopt(as,SOL_SOCKET,SO_LINGER,&len,sizeof(len));
    setsockopt(as,SOL_SOCKET,SO_REUSEADDR,&len,sizeof(len));
    server.sin_addr.s_addr = INADDR_ANY;
    bind(as, (struct sockaddr*) &server, sizeof(server));
    len=sizeof(len);
    getsockname(as,(struct sockaddr*)&server,(socklen_t *)&len);

    listen(as,5);
    ms=accept(as,0,0);
    bzero(buf,lbuf);

    /* rcv handshake */

    if(read(ms,locbuf,sizeof(locbuf))!=sizeof(locbuf))
     {printf("error rcv1\n"); exit(1);}
    sscanf(locbuf,"%i %i",&tmp,&temp);
    sprintf(locbuf,"%i %i",port,lbuf);
    write(ms,locbuf,sizeof(locbuf));       

    if(wordy>1) printf("hshake: %i %i ",tmp,temp);

    if(tmp!=port) {printf("error rcv2\n"); exit(1);}
    if(temp>lbuf) /* too much, cannot accept */
    {
      sprintf(locbuf,"too much");
      close(ms);
      shutdown(as,2);
      close(as);
      if(wordy>0) printf("rcv: asked %i have %i so refused\n",temp,lbuf);
      return(NULL);
    }
 
    len=0;
    for(;;) /* transport */
    {
     tmp=read(ms,&buf[len],((lbuf-len)>smx)?smx:temp-len);
     if(tmp==0) break;
     len+=tmp;
    }
    close(ms);

   }

   if(flag==2) /* send */
   {
    hp=gethostbyname(hname);
    bcopy( hp->h_addr, &server.sin_addr, hp->h_length);

    for(;;)  /* wait for server */
    {
     as=socket(AF_INET, SOCK_STREAM, 0 );
     len=1;
     setsockopt(as,SOL_SOCKET,SO_LINGER,&len,sizeof(len));
     setsockopt(as,SOL_SOCKET,SO_REUSEADDR,&len,sizeof(len));
     if(connect(as,(struct sockaddr *)&server,sizeof(server))==0) break;

     close(as);
    }

    /* snd handshake */

    sprintf(locbuf,"%i %i",port,lbuf);
    write(as,locbuf,sizeof(locbuf));
    read(as,locbuf,sizeof(locbuf));
    sscanf(locbuf,"%i %i",&tmp,&temp);

    if(wordy>1) printf("hshake: %i %i ",tmp,temp);

    if(tmp!=port) {printf("error snd2\n"); exit(1);}
    if(temp<lbuf) /* too much, cannot send */
    {
      shutdown(as,2);
      close(as);
      if(wordy>0) printf("snd: not enough memory on rcv\n");
      return(NULL);
    }

    /* transport */

    len=0;
    while(len<lbuf) 
     len+=write(as,&buf[len],((lbuf-len)>smx)?smx:lbuf-len);
   }

   shutdown(as,2);
   close(as);

   if(wordy>0)
   {
    printf("%s %i byts ",(flag==1)?"Rcv":"Snd",len); 
    printf("in %i sec\n",(int)(time(NULL)-t1)); 
   }

   return(buf);
}

/* Returns 1 if this computer is little-endian, 0 otherwise. */

int endian(void)
{
  int x=1;
  return( x == *((char*)&x) );
}

/* endian swap routine */

void endiswap(int lbuf, char *buf,int num)
{
  int  i,k;
  char msg[8];

  for(i=0;i<lbuf;i+=num)
  {
    memcpy(msg,(char*)&buf[i],num);
    for(k=0;k<num;k++) buf[i+k]=msg[num-1-k]; 
  }
}

/* test byte order on the machine */

void byteorder() 
{
 double a;
 int    i;
 unsigned char   *tmp=(unsigned char*)&a;

 for(a=0,i=8;i>0;i--) a=a*256.+i;
 for(i=0;i<8;i++) printf(" %i ",tmp[i]);
 printf(" %g\n",a);
}


#ifdef TEST_EXEC

char *chkbuf(int flag,int len,int type,char *msg)
{
  char   *cb=(flag==2) ? (char*)calloc(len,type) : msg ;
  double s,*db=(double*)cb;
  int    *ib=(int*)cb;
  int    k;
   
  if(flag==2) /* send */
  {
   if(type==1) strcpy(cb,msg);
   if(type==4) for(k=0;k<len;k++) ib[k]=k;  
   if(type==8) for(k=0;k<len;k++) db[k]=k;  
  }
  if(flag==1) /* receive */
  {
   if(type==1) printf("Message= %s\n",cb);
   if(type==4) for(k=0,s=0;k<len;k++) s+=ib[k]-k;  
   if(type==8) for(k=0,s=0;k<len;k++) s+=db[k]-k;  
   if(type>1) 
    printf("%s %s\n",(type==4)?"int32":"double", (s==0)?"correct":"error");
  }
  return(cb);
}

main(int ac,char *av[])
{
  int  wordy=atoi(av[1]);
  int  flag=atoi(av[2]);
  char *hport=av[3];
  int  type;
  int  len;
  int  endi;
  char *buf,cb[128];
  int  lcb=sizeof(cb);

  if(flag==2) /* send */
  {
    type=atoi(av[4]);
    len=(type==1) ? strlen(av[5])+1 : atoi(av[5]) ;
    buf=chkbuf(2,len,type,av[5]);
    sprintf(cb,"%i %i %i",len,type,endian());
    bring(wordy,2,hport,lcb,cb);
    if(bring(wordy,2,hport,len*type,buf)==NULL) 
      printf("not enough memory on rcv side");
  }

  if(flag==1) /* receive */
  {
    sscanf(bring(wordy,1,hport,lcb,cb),"%i %i %i",&len,&type,&endi);
    buf=(char*)calloc(len,type);
    if(buf==NULL) len=0;
    bring(wordy,1,hport,len*type,buf);
    if(endi!=endian()) endiswap(len*type,buf,type);
    chkbuf(1,len,type,buf);
  }
  free(buf);
}

#endif

} // End namespace MatlabInterface
