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


      ideas:
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

#ifdef SGI64
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
    while(1) /* transport */
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

    while(1) /* wait for server */
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

