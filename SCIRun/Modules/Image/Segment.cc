//static char *id="@(#) $Id"

/*
 *  Segment.cc:  Segment Module
 *
 *  Written by:
 *    Scott Morris
 *    July 1998
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <SCICore/Geom/GeomGrid.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <math.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;

using namespace SCICore::TclInterface;
using namespace SCICore::Math;

class Segment : public Module {
   ScalarFieldIPort *inscalarfield;
   ScalarFieldOPort *outscalarfield;
   int gen;

   ScalarFieldRG* newgrid;
   ScalarFieldRG*  rg;

   double min,max;       // Max and min values of the scalarfield

   TCLstring conn;

   int np; // number of proccesors
  
public:
   Segment(const clString& id);
   virtual ~Segment();
   virtual void execute();

//   void tcl_command( TCLArgs&, void *);

   void do_Segment(int proc);

};

Module* make_Segment(const clString& id)
{
   return scinew Segment(id);
}

static clString module_name("Segment");

Segment::Segment(const clString& id)
: Module("Segment", id, Filter), conn("conn", id, this)
{
    // Create the input ports
    // Need a scalar field
  
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);

    // Create the output port
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_oport( outscalarfield);
    newgrid=new ScalarFieldRG;
}

Segment::~Segment()
{
}

void Segment::do_Segment(int proc)    
{
  int start = (rg->grid.dim2())*proc/np;
  int end   = (proc+1)*(rg->grid.dim2())/np;

  for(int x=start; x<end; x++) {
    for(int y=0; y<rg->grid.dim1(); y++) {
      if (rg->grid(y,x,0)>0)
	newgrid->grid(0,rg->grid(y,x,0),0)++;
    }
  }
}

#define MGREY 2000000000 /* maximum label value */

int remllb(int *pc,int *remo,int *numb,int *rlab,long size)
     /*register int  *pc,*remo;
int             *numb,*rlab;
register long   size; */
{
register int    i,j;
register int  *m;

   if (!(m = (int *) malloc((unsigned int)(*numb+1)*sizeof(unsigned int))))
	return 2;

   for (i = 0; i <= *numb; i++) m[i] = i;
   for (i = 1; i <= *rlab; i++) m[remo[i]] = 0;

   for (i = 1, j = 0; i <= *numb; i++)
   {
      if (m[i]) j++;
      m[i] = j;
   }

   while (--size >= 0) {*pc = m[*pc]; pc++;}

   *numb = j;
   *rlab = 0;

   free(m);
   return 0;
}

int cblabel(int *wp_in,int sizex,int sizey,int msk,int *wp_out,int conn,
	    int *nu,int *ir)

/*
  labels a 1-bit pi, result is a 16 bit pc.   
  parameters :
  
  wp_in          : input image
  msk            : bitplane mask in input image
  wp_out         : output image
  conn           : connectivity
                   0 : 4-connected 
                   1 : 8-connected 
  numb           : last label
  rlab           : number of removed labels
*/

     /*int *wp_in, *wp_out;
int sizex,sizey;  
int *nu,*ir,conn;
register int msk; */
{
register int          *pi,*pc,*pn,labl;
register int            j,l;
int                     numb, rlab, jma;
int                   *remo, *pp, lab2;
int                     i,stopp,start,jmi,k,error;
int                     flag;

   pi = wp_in;
   pc = wp_out;
   
   if (!(remo = (int *) malloc(sizeof(int)*(sizex+1)))) return 2;
   numb = rlab = 0;
   conn = (conn >> 3) & 1;

   for (pp = pc - sizex, i = 0; i < sizey;
        i++, pp = pc, pi += sizex, pc += sizex)
   {
      stopp = -2;
     
nexrn:
      for (j = stopp+2; j < sizex; j++)
      {
         if ((pi[j] & msk )!= 0) goto newrn;
         pc[j] = 0;
      }
      continue;
     
newrn:
      pc[j] = 0;
      for (start = j, j++; j < sizex; j++)
      {
         if ((pi[j] & msk)== 0) break;
         pc[j] = 0;
      }
      pc[j] = 0;
      stopp = j-1; 
     
      if (i != 0)
      {
         jmi = Max(0,start-conn); 
         jma = Min(sizex-1,stopp+conn);
    
         for (j = jmi; j <= jma; j++)
            if (pp[j] != 0) {labl = pp[j]; goto label;}
      }
     
      if ((numb == MGREY) && (rlab == 0)) 
         {free(remo); return 12;}
      if (rlab == 0)
         {numb++; labl = numb;}
      else
         {labl = remo[rlab--];}
     
label:
      for (jma = stopp, j = start; j <= jma; j++) pc[j] = labl;
     
      if (i == 0) goto nexrn;
     
      jmi = Max(0,start-conn); 
      jma = Min(sizex-1,stopp+conn);
     
      for (j = jmi; j <= jma; j++) 
      {
         if (((lab2 = pp[j]) == 0) || (lab2 == labl)) continue; 

         if (labl == numb) numb--;
         else
         {
            if (rlab == MGREY) 
               {free(remo); return 12;}
            remo[++rlab] = labl;
         }
    
         for (pn = pc, k = i; k >= 0; k--, pn -= sizex) 
         {
            flag = 0;
            for (l = sizex; --l >= 0;)
                {if (pn[l] == labl) {flag = 1; pn[l] = lab2;}} 
            if (!flag && (k != i)) break; 
         }
         labl = lab2;
      }
     
      goto nexrn;
   }
  
   while (rlab && (numb == remo[rlab])) {rlab--; numb--;} 

   pc -= (long)sizex*sizey;
   error = remllb(pc,remo,&numb,&rlab,(long)sizex*sizey);

   *nu = numb;
   *ir = rlab;
   free(remo);
   return(error); 
} 




void Segment::execute()
{
    // get the scalar field...if you can

    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;

    rg=sfield->getRG();
    
    if(!rg){
      cerr << "Segment cannot handle this field type\n";
      return;
    }
    gen=rg->generation;
    
    if (gen!=newgrid->generation){
    //  newgrid=new ScalarFieldRGint;
      // New input
    }
    newgrid=new ScalarFieldRG(*rg);

    rg->compute_minmax();
    rg->get_minmax(min,max);

    int width = rg->grid.dim2();
    int height = rg->grid.dim1();

    newgrid->resize(rg->grid.dim1(),rg->grid.dim2(),rg->grid.dim3());

    int *in = new int[width*height];
    int *out = new int[width*height];
    int num,ir,x,y;

    clString connst(conn.get());
    int con=(connst=="Eight")*8;
    
    for (x=0;x<width;x++)
      for (y=0;y<height;y++)
	if (rg->grid(y,x,0)!=0)
	  in[y*width+x]=1;
    
    cblabel(in,width,height,1,out,con,&num,&ir);
    cerr << "Labeling with " << connst << " connectivity.\n";
    cerr << "Number of labels : " << num << "\n";
    cerr << "Number removed : " << ir << "\n";
        
    for (x=0;x<width;x++)
      for (y=0;y<height;y++)
	newgrid->grid(y,x,0)=out[y*width+x];

    outscalarfield->send( newgrid );
}

/*
void Segment::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}
*/

} // End namespace Modules
} // End namespace SCIRun

//
// $Log$
// Revision 1.5  1999/09/08 02:27:01  sparker
// Various #include cleanups
//
// Revision 1.4  1999/08/31 08:55:34  sparker
// Bring SCIRun modules up to speed
//
// Revision 1.3  1999/08/25 03:48:57  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:54  mcq
// Initial commit
//
// Revision 1.1  1999/04/29 22:26:34  dav
// Added image files to SCIRun
//
//
