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
*/

/* data transport routine
   Oleg onp 00Dec21 oleg@cs.utah.edu
   01Aug23 - protocol repaired to prevent hanging
   01Aug27 - controlling sequences changed

   1a) to compile this file into executable test:
      [g]cc -o bring bring.c -DTEST_EXEC

   1b) to compile this file into executable test on sgi:
      [g]cc -o bring bring.c -DTEST_EXEC -DSGI64

   2a) To run this standalone test:

      ./bring 2 1 :5505
      ./bring 2 2 127.0.0.1:5505 8 300 &

   2b) test without closing the connection after each communication:

      ./bring 2 3 :5505
      ./bring 2 4 127.0.0.1:5505 8 300 &
 
   3) Prototypes to use (instead of bring.h):
*/
      char *bring(int wordy,int flag, const char *hostport,int lbuf,char *buf);
      void endiswap(int lbuf, char *buf,int num);
      int  endian(void);
/*
  4a) Altavista search: Berkeley Socket Library
  4b) SOCK_STREAM Server/Client Example from
      http://web.cs.mun.ca/~rod/Winter98/cs4759/berkeley.html
  4c) four server/client examples:
      http://blondie.mathcs.wilkes.edu/~sullivan/sockets/

  5)  ideas:
      for endiness, it is possible to use host-to-network
      endiness swap routines, such as
      ntohl htonl, etc 

*/
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <errno.h>

#ifdef __sgi
#define socklen_t int
#endif 
#ifdef __APPLE__
#define socklen_t int
#endif
  /* values of flag: */
  /* 1 - RECEIVE (OPEN SERVER, CLOSE) */
  /* 2 - SEND    (OPEN CLIENT CLOSE) */
  /* 3 - OPEN SERVER */
  /* 4 - OPEN CLIENT */
  /* 5 - CLOSE */

void fatalerr(char *msg);
void server(int wordy,const char *hostport, int *asout,int *msout);
int  client(int wordy,const char *hostport);
int  rcvsnd(int wordy,int flag,int sock, int lbuf, char *buf);
void sockclose(int wordy, int as);

char *bring(int wordy,int flag,const char *hostport,int lbuf,char *buf)
{
  time_t t1=time(NULL);
  static int as,ms,status_flag=0; 
  int err;

  switch(flag) 
  {
          case 1:   /* RECEIVE */
  
    if(status_flag==0) /* Open server, rcv, close */
    {
     server(wordy-2,hostport,&as,&ms);
     err=rcvsnd(wordy-2,1,ms,lbuf,buf);
     close(ms);
     sockclose(wordy-2,as);
    } 

    if(status_flag==1) /* Server already open, just rcv */ 
     err=rcvsnd(wordy-2,1,ms,lbuf,buf);

    if(status_flag==2) /* Client already open, just rcv */ 
     err=rcvsnd(wordy-2,1,as,lbuf,buf);

    if(err) fatalerr("not enough memory on snd side");

    if(wordy>0) 
      printf("bring: rcv from %s %i bts in %i sec\n",hostport,lbuf,(int)(time(NULL)-t1));

   break; case 2: /* SEND */

    if(status_flag==0) /* Open client, snd, close */
    {
     as=client(wordy-2,hostport);
     err=rcvsnd(wordy-2,2,as,lbuf,buf);
     sockclose(wordy-2,as);
    } 

    if(status_flag==1) /* Server already open, just snd */ 
     err=rcvsnd(wordy-2,2,ms,lbuf,buf);

    if(status_flag==2) /* Client already open, just snd */ 
     err=rcvsnd(wordy-2,2,as,lbuf,buf);

    if(err) fatalerr("not enough memory on rcv side");

    if(wordy>0) 
      printf("bring: snd  to  %s %i bts in %i sec\n",hostport,lbuf,(int)(time(NULL)-t1));

   break; case 3:      /* OPEN SERVER */

    if(status_flag) fatalerr("Socket already open\n");
    server(wordy-2,hostport,&as,&ms);
    status_flag=1;

   break; case 4:      /* OPEN CLIENT */

    if(status_flag) fatalerr("Socket already open\n");
    as=client(wordy-2,hostport);
    status_flag=2;

   break; case 5:      /* CLOSE */

    if(status_flag==0) fatalerr("bring: Socket is not open\n");
    if(status_flag==1) close(ms);
    sockclose(wordy-2,as);
    status_flag=0;

   break; default:

    fatalerr("bring: ERROR, unknown operation code");
  }

  return(buf);
}

/*******************************************************************************/


void server(int wordy,const char *hostport, int *asout,int *msout)
{
   int    as,ms;

   struct sockaddr_in server;

   int  len,tmp,err;
   char locbuf[64];
   char *cport=strstr(hostport,":");

   if(wordy>0) printf("server: OPEN\n");

   server.sin_family = AF_INET;
   server.sin_port = htons( atoi(cport+1) );
   as=socket(AF_INET, SOCK_STREAM, 0 );
   if(wordy>1) printf("socket\n");
   len=1;
   setsockopt(as,SOL_SOCKET,SO_LINGER,&len,sizeof(len));
   setsockopt(as,SOL_SOCKET,SO_REUSEADDR,&len,sizeof(len));
   server.sin_addr.s_addr = INADDR_ANY;
 
   for(;;)
   {
    err=bind(as, (struct sockaddr*) &server, sizeof(server));
    if(err==0) break;
   }
   if(wordy>1) printf("bind %i\n",err);
 
   len=sizeof(len);
   tmp=getsockname(as,(struct sockaddr*)&server,(socklen_t *)&len);
   if(wordy>1) printf("getsockname %i\n",tmp);

   tmp=listen(as,5);
   if(wordy>1) printf("listen %i\n",tmp);

   ms=accept(as,0,0);
   if(wordy>1) printf("accept %i\n",ms);

   sprintf(locbuf,"READY");
   err=write(ms,locbuf,sizeof(locbuf));       
   if(wordy>1) printf("write READY %i\n",err);
   *asout=as;
   *msout=ms;
}

/*******************************************************************/

int client(int wordy, const char *hostport)
{
   int    as;

   struct sockaddr_in server;
   struct hostent *hp;

   int  len,tmp,err;
   char locbuf[64];
   char *hname=strncpy(locbuf,hostport,sizeof(locbuf));
   char *cport=strstr(hname,":");
   cport[0]='\0'; /* cut hname for gethostbyname */

   if(wordy>0) printf("client: OPEN\n");

   server.sin_family = AF_INET;
   server.sin_port = htons( atoi(cport+1) );
   hp=gethostbyname(hname);
   bcopy( hp->h_addr, &server.sin_addr, hp->h_length);

   for(;;) /* wait for server */
   {
     as=socket(AF_INET, SOCK_STREAM, 0 );
     len=1;
     setsockopt(as,SOL_SOCKET,SO_LINGER,&len,sizeof(len));
     setsockopt(as,SOL_SOCKET,SO_REUSEADDR,&len,sizeof(len));
     tmp=connect(as,(struct sockaddr *)&server,sizeof(server));
     if(tmp==0) 
     {
       err=read(as,locbuf,sizeof(locbuf));
       if(wordy>1) printf("read READY %i %s\n",err,locbuf);
       if( (err==sizeof(locbuf))&&(strncmp(locbuf,"READY",5)==0) ) break;
     }
     close(as);
   }
   if(wordy>1) printf("connect \n");
   return(as);
}

/*******************************************************************/

void fatalerr(char *msg)
{
  printf("\n fatalerr: %s\n",msg);
  exit(1);
}

/*******************************************************************/

void sockclose(int wordy, int as)
{
   int err,tmp;
    if(wordy>0) printf("sockclose: CLOSE\n");
    err=shutdown(as,2); 
    /*This is always -1, what to do? 
    if(err!=0) if(wordy>2) printf("SHUTDOWN WARNING: %s\n",strerror(errno));
    */
    tmp=close(as);
    if(wordy>1) printf("shutdown %i close %i\n",err,tmp);
}

/*******************************************************************/

int rcvsnd(int wordy,int flag,int sock, int lbuf, char *buf)
{
   char locbuf[64];
   int  tmp,len,err;
   int  smx=1500; /* SSIZE_MAX */

   if(flag==1)
   {
    if(wordy>0) printf("rcvsnd RECEIVE \n");

    bzero(buf,lbuf);

    tmp=read(sock,locbuf,sizeof(locbuf));
    if(tmp!=sizeof(locbuf)) fatalerr("rcvsnd error rcv1");
    if(wordy>1) printf("read %i\n",tmp);

    sscanf(locbuf,"%i ",&tmp);
    bzero(locbuf,sizeof(locbuf));
    sprintf(locbuf,"%i ",lbuf);

    err=write(sock,locbuf,sizeof(locbuf));
    if(wordy>1) printf("write %i\n",err);

    if(tmp>lbuf) return(1); /* too much, cannot accept */

    len=0;
    while(len<lbuf) /* transport */
    {
     tmp=((lbuf-len)>smx)?smx:lbuf-len;
     err=read(sock,&buf[len],tmp);
     if(wordy>1) printf("read %i bts %i\n",err,tmp);
     if(err<=0) break;
     len+=err;
    }
   }
   
   if(flag==2)
   {
    if(wordy>0) printf("rcvsnd SEND\n");

    sprintf(locbuf,"%i ",lbuf);

    tmp=write(sock,locbuf,sizeof(locbuf));
    if(wordy>1) printf("write %i\n",tmp);

    tmp=read(sock,locbuf,sizeof(locbuf));
    if(wordy>1) printf("read %i\n",tmp);

    sscanf(locbuf,"%i ",&tmp);

    if(tmp<lbuf) return(1);

    len=0;
    while(len<lbuf)
    {
     tmp=((lbuf-len)>smx)?smx:lbuf-len;
     err=write(sock,&buf[len],tmp);
     if(wordy>1) printf("write: %i %i\n",err,tmp);
     if(err<=0) break;
     len+=err;
    }
   }
   return(0);
}

/*******************************************************************/
/* Returns 1 if this computer is little-endian, 0 otherwise. */
/*******************************************************************/

int endian(void)
{
  int x=1;
  return( x == *((char*)&x) );
}

/*******************************************************************/
/* endian swap routine */
/*******************************************************************/

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

/*******************************************************************/
/* test byte order on the machine COMMENTED OUT*/
/*******************************************************************/
/*
void byteorder() 
{
 double a;
 int    i;
 unsigned char   *tmp=(unsigned char*)&a;

 for(a=0,i=8;i>0;i--) a=a*256.+i;
 for(i=0;i<8;i++) printf(" %i ",tmp[i]);
 printf(" %g\n",a);
}
*/

/*******************************************************************/
/* Testing routines */
/*******************************************************************/
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
/*******************************************************************/

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

  if(flag==3)  bring(wordy,3,hport,0,NULL); /* open server */
  if(flag==4)  bring(wordy,4,hport,0,NULL); /* open client */

  if((flag==1)|(flag==3)) /* receive */
  {
    sscanf(bring(wordy,1,hport,lcb,cb),"%i %i %i",&len,&type,&endi);
    buf=(char*)calloc(len,type);
    if(buf==NULL) len=0;
    bring(wordy,1,hport,len*type,buf);
    if(endi!=endian()) endiswap(len*type,buf,type);
    chkbuf(1,len,type,buf);
  }

  if((flag==2)|(flag==4)) /* send */
  {
    type=atoi(av[4]);
    len=(type==1) ? strlen(av[5])+1 : atoi(av[5]) ;
    buf=chkbuf(2,len,type,av[5]);
    sprintf(cb,"%i %i %i",len,type,endian());
    bring(wordy,2,hport,lcb,cb);
    if(bring(wordy,2,hport,len*type,buf)==NULL) 
      printf("not enough memory on rcv side");
  }

  free(buf);
  if((flag==3)|(flag==4)) bring(wordy,5,NULL,0,NULL); /* close */
}

#endif

