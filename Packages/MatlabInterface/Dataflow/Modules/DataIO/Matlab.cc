/*
 *  Matlab.cc:
 *
 *  Written by:
 *   oleg@cs.utah.edu
 *   02Jan23 
 *
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <sci_defs.h> // for SCIRUN_OBJDIR 

#include <Packages/MatlabInterface/Core/Util/transport.h>

namespace MatlabInterface {

using namespace SCIRun;

MatrixHandle transport(int wordy,int flag,char *hport,MatrixHandle mh);
void         transport(int wordy,int flag,char *hport,char *cmd);

class MatlabInterfaceSHARE Matlab : public Module 
{
  GuiString hpTCL;
  GuiString cmdTCL;
  MatrixIPort *ip[5];
  MatrixOPort *op[5];

  int wordy;            // how wordy debug output is

  static char hport[256];      // Host and port information
  static int engine_running;   // ==1 if engine runs under SCIRun
  static Mutex engine_lock_;

public:
  Matlab(const string& id);
  virtual ~Matlab();
  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_Matlab(const string& id) {
  return scinew Matlab(id);
}

Matlab::Matlab(const string& id) :
  Module("Matlab", id, Source, "DataIO", "MatlabInterface"), 
  hpTCL("hpTCL",id,this),
  cmdTCL("cmdTCL",id,this)
{
  wordy=0;
}

#include <unistd.h>

bool Matlab::engine_running = 0;
Mutex Matlab::engine_lock_("Matlab engine lock");
char Matlab::hport[256];

Matlab::~Matlab()
{
 /* MATLAB ENGINE SHUTDOWN IF NECESSARY */
 
 engine_lock_.lock();
 if(engine_running)
 {
  if(wordy>0) fprintf(stderr,"Closing matlab engine on %s\n",hport); 
  engine_running--;
  transport(2,4,hport,NULL);       // OPEN CLIENT 
  transport(2,2,NULL,"stop");      // SHUT THE ENGINE DOWN 
  transport(2,5,NULL,NULL);        // CLOSE 
 }
 engine_lock_.unlock();
}

/****************************MAIN ROUTINE*********************************************/

double mh2double(MatrixHandle mh);
void   cmdparse(int *ioflags,char *cmd);

void Matlab::execute()
{

/* CREATE I/O PORTS */

 for(int k=0;k<5;k++)
 {
  char chr[128];
  sprintf(chr,"i%1i%c",k+1,'\0');
  ip[k] = (MatrixIPort *)get_iport(chr);
  if (!ip[k]) { postMessage("Unable to initialize "+name+"'s iport\n"); return; }

  chr[0]='o';
  op[k] = (MatrixOPort *)get_oport(chr);
  if (!op[k]) { postMessage("Unable to initialize "+name+"'s oport\n"); return; }
 }

/* OBTAIN GUI STRING - COMMAND */

 cmdTCL.reset();
 string s1=cmdTCL.get();
 char *cmd=(char *)s1.c_str(); // Command is taken from interface
 cmd=scinew char[strlen(cmd)+1];
 strcpy(cmd,(char *)s1.c_str());

 if(wordy>0) fprintf(stderr,"Command is: \n%s\n",cmd); 

/* START THE MATLAB ENGINE */

 engine_lock_.lock();

 if(!engine_running)
 {
   /* OBTAIN GUI STRING - HOST:PORT  AND WORDY PARAMETER */

   hpTCL.reset();
   string s2=hpTCL.get();
   sscanf((char *)s2.c_str(),"%s %i",hport,&wordy);
   if(wordy>0) fprintf(stderr,"host and port: %s wordy %i\n",hport, wordy); 
 
   if(strncmp(hport,"127.0.0.1",9)==0)
   {
     if(wordy>0) fprintf(stderr,"Starting matlab engine on %s\n",hport); 
     char cl[1024];
     
     strcpy(cl,"echo 'path('\\''"); 
     strcat(cl,SCIRUN_OBJDIR);
     strcat(cl,"/../src/Packages/MatlabInterface/matlab/engine");
     strcat(cl,"'\\'',path); mlabengine(");
    sprintf(cl,"%s%i",cl,wordy-2);
     strcat(cl,",'\\''"); 
     strcat(cl,hport); 
     strcat(cl,"'\\'');'|matlab -nosplash &");

     if(wordy>1) fprintf(stderr,"line for system call: \n %s\n",cl); 
     system(cl);
   } else
   {
     if(wordy>0) fprintf(stderr,"Assuming that mlabengine is on %s\n",hport); 
   }
   engine_running++;
 }

 char rcv[10],snd[10];
 MatrixHandle mhi[5],mho[5],err;
 int ioflags[10];

 cmdparse(ioflags,cmd);                  /* PARSE THE STRING FOR i o NAMES, delete \n */

 for(int k=0;k<5;k++)
  if(ioflags[k]) ip[k]->get(mhi[k]);     /* GET DATA FROM RELEVANT PORTS */

 transport(wordy-2,4,hport,NULL);        /* OPEN CLIENT */

 for(int k=0;k<5;k++)                    /* SEND DATA FROM ALL PORTS */
  if(ioflags[k]) 
  {
   strcpy(rcv,"i1=rcv;");
   rcv[1]+=k;
   transport(wordy-2,2,NULL,rcv);          
   transport(wordy-2,2,NULL,mhi[k]);
   err=transport(wordy-2,1,NULL,err);     
  }

 transport(wordy-2,2,NULL,cmd);          /* COMMAND */
 err=transport(wordy-2,1,NULL,err);      /* RCV ERROR CODE */
 if(mh2double(err)>0)
 {
  fprintf(stderr,"Matlab returned error: %g\n",mh2double(err)); 
  transport(wordy-2,2,NULL,"break");      /* RELEASE SERVER */
  transport(wordy-2,5,NULL,NULL);         /* CLOSE */
  engine_lock_.unlock();
  return;
 }

 for(int k=0;k<5;k++)                    /* BRING BACK RESULTS */
  if(ioflags[k+5])
  {
   strcpy(snd,"snd(o1);");
   snd[5]+=k;
   transport(wordy-2,2,NULL,snd); 
   mho[k]=transport(wordy-2,1,NULL,mho[k]);
   err=transport(wordy-2,1,NULL,err);     
  }

 transport(wordy-2,2,NULL,"break");      // RELEASE SERVER 

 // transport(wordy-2,2,NULL,"stop");       // SHUT SERVER
 // engine_running=0;

 transport(wordy-2,5,NULL,NULL);         /* CLOSE */

 for(int k=0;k<5;k++)                    /* SEND DATA TO RELEVANT OUTPUT PORTS */
  if(ioflags[k+5]) op[k]->send(mho[k]);             

 delete [] cmd;

 engine_lock_.unlock();
}

/****************************END OF MAIN ROUTINE**************************************/

void Matlab::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

// Convert matrix handle to double

double mh2double(MatrixHandle mh)
{
  double *db;
  DenseMatrix     *dmatr;
  ColumnMatrix    *cmatr;
  // SparseRowMatrix *smatr;
  
  dmatr=dynamic_cast<DenseMatrix*>(mh.get_rep());
  cmatr=dynamic_cast<ColumnMatrix*>(mh.get_rep());
  // smatr=dynamic_cast<SparseRowMatrix*>(mh.get_rep());

  db=NULL;
  if(cmatr!=NULL) db=&((*cmatr)[0]);
  if(dmatr!=NULL) db=&((*dmatr)[0][0]);

  if(db==NULL) 
  {
   fprintf(stderr,"mh2double: Cannot convert matrix to double\n");
   return(-1.);
  }

  return(*db);
}

// Parsing of input command to recognize
// i1 i2 i3 and o1 o2 o3 names

void cmdparse(int *ioflags,char *cmd)
{
 for(int k=0;k<10;k++) ioflags[k]=0;
 if(strstr(cmd,"i1")!=NULL) ioflags[0]=1;
 if(strstr(cmd,"i2")!=NULL) ioflags[1]=1;
 if(strstr(cmd,"i3")!=NULL) ioflags[2]=1;
 if(strstr(cmd,"i4")!=NULL) ioflags[3]=1;
 if(strstr(cmd,"i5")!=NULL) ioflags[4]=1;

 if(strstr(cmd,"o1")!=NULL) ioflags[5]=1;
 if(strstr(cmd,"o2")!=NULL) ioflags[6]=1;
 if(strstr(cmd,"o3")!=NULL) ioflags[7]=1;
 if(strstr(cmd,"o4")!=NULL) ioflags[8]=1;
 if(strstr(cmd,"o5")!=NULL) ioflags[9]=1;

 for(unsigned int k=0;k<strlen(cmd);k++) if(cmd[k]=='\n') cmd[k]=' ';

}

} // End namespace MatlabInterface

