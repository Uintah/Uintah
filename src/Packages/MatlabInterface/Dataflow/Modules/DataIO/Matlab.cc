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
  int ip_generations_[5];
  string last_command_;
  string last_hport_;

  char hport[256];
  int wordy;            // how wordy debug output is

  static bool engine_running;   // ==1 if engine runs under SCIRun

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
  for (int i = 0; i < 5; i++)
  {
    ip_generations_[i] = 0;
  }

  wordy=0;
}


#include <unistd.h>

bool Matlab::engine_running = 0;


Matlab::~Matlab()
{
  /* MATLAB ENGINE SHUTDOWN IF NECESSARY */
 
  if (engine_running)
  {
    if(wordy>0) fprintf(stderr,"Closing matlab engine on %s\n",hport); 
    transport(2,4,hport,NULL);       // OPEN CLIENT 
    transport(2,2,NULL,"stop");      // SHUT THE ENGINE DOWN 
    transport(2,5,NULL,NULL);        // CLOSE 
    engine_running = false;
  }
}

/****************************MAIN ROUTINE*********************************************/

// Convert matrix handle to double

static double
mh2double(MatrixHandle mh)
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

static string
remove_newlines(const string &str)
{
  string result(str);
  for (string::size_type i = 0; i < result.size(); i++)
  {
    if (result[i] == '\n')
    {
      result[i] = ' ';
    }
  }
  return result;
}

// Parsing of input command to recognize
// i1 i2 i3 and o1 o2 o3 names

static void
cmdparse(int *ioflags, const char *cmd)
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
}


void
Matlab::execute()
{
  /* CREATE I/O PORTS */
  MatrixIPort *ip[5];
  MatrixOPort *op[5];
  for (int k=0; k<5; k++)
  {
    char chr[128];
    sprintf(chr,"i%1i%c",k+1,'\0');
    ip[k] = (MatrixIPort *)get_iport(chr);
    if (!ip[k])
    {
      postMessage("Unable to initialize "+name+"'s iport\n");
      return;
    }

    chr[0]='o';
    op[k] = (MatrixOPort *)get_oport(chr);
    if (!op[k])
    {
      postMessage("Unable to initialize "+name+"'s oport\n");
      return;
    }
  }

  /* OBTAIN GUI STRING - COMMAND */
  cmdTCL.reset();
  string command = remove_newlines(cmdTCL.get());
  if (wordy>0) fprintf(stderr,"Command is: \n%s\n",command.c_str()); 

  char rcv[10], snd[10];
  MatrixHandle mhi[5], mho[5], err;

  int ioflags[10];
  /* PARSE THE STRING FOR i o NAMES */
  cmdparse(ioflags, command.c_str());

  // Get the input ports.
  bool different_p = command != last_command_;
  last_command_ = command;
  if (last_hport_ != hpTCL.get()) { different_p = true; }
  last_hport_ = hpTCL.get();
  for(int k=0;k<5;k++)
  {
    if(ioflags[k])
    {
      ip[k]->get(mhi[k]);     /* GET DATA FROM RELEVANT PORTS */
      if (mhi[k]->generation != ip_generations_[k])
      {
	different_p = true;
	ip_generations_[k] = mhi[k]->generation;
      }
    }
  }

  // If input data and the script have not changed since last execute
  // then there is nothing to do.
  if (!different_p)
  {
    remark("No change in data or script.");
    return;
  }

  /* START THE MATLAB ENGINE */
  /* OBTAIN GUI STRING - HOST:PORT  AND WORDY PARAMETER */
  hpTCL.reset();
  string s2=hpTCL.get();
  sscanf(s2.c_str(), "%s %i", hport, &wordy);
  if(wordy>0) fprintf(stderr,"host and port: %s wordy %i\n",hport, wordy); 
  
  if(strncmp(hport,"127.0.0.1",9)==0)
  {
    if (!engine_running)
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
      engine_running = true;
    }
  }
  else
  {
    if(wordy>0) fprintf(stderr,"Assuming that mlabengine is on %s\n",hport); 
  }

  transport(wordy-2,4,hport,NULL);        /* OPEN CLIENT */

  for (int k=0;k<5;k++)                   /* SEND DATA FROM ALL PORTS */
  {
    if(ioflags[k]) 
    {
      strcpy(rcv,"i1=rcv;");
      rcv[1]+=k;
      transport(wordy-2,2,NULL,rcv);          
      transport(wordy-2,2,NULL,mhi[k]);
      err=transport(wordy-2,1,NULL,err);     
    }
  }

  // TODO:  Make transport take (const char *) instead of (char *)
  transport(wordy-2,2,NULL,(char *)(command.c_str()));  /* SEND COMMAND */
  err = transport(wordy-2, 1, NULL, err);      /* RCV ERROR CODE */
  if (mh2double(err) > 0)
  {
    fprintf(stderr,"Matlab returned error: %g\n", mh2double(err)); 
    transport(wordy-2, 2, NULL, "break");      /* RELEASE SERVER */
    return;
  }
  
  for (int k=0;k<5;k++)                    /* BRING BACK RESULTS */
  {
    if (ioflags[k+5])
    {
      strcpy(snd,"snd(o1);");
      snd[5]+=k;
      transport(wordy-2,2,NULL,snd); 
      mho[k]=transport(wordy-2,1,NULL,mho[k]);
      err=transport(wordy-2,1,NULL,err);     
    }
  }

  transport(wordy-2,2,NULL,"break");      // RELEASE SERVER 
  transport(wordy-2,5,NULL,NULL);         /* CLOSE */

  for (int k=0; k<5; k++)         /* SEND DATA TO RELEVANT OUTPUT PORTS */
  {
    if(ioflags[k+5])
    {
      op[k]->send(mho[k]);
    }
  }
}


/****************************END OF MAIN ROUTINE**************************************/

void
Matlab::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}


} // End namespace MatlabInterface

