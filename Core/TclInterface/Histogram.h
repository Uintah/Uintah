
/*
 *  Histogram.h: Histogram range widget
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Apr. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_Histogram_h
#define SCI_project_Histogram_h 1

#include <Containers/Array1.h>
#include <TclInterface/TCL.h>
#include <TclInterface/TCLvar.h>

namespace SCICore {
namespace TclInterface {

using SCICore::Containers::Array1;

class Histogram : public TCL {
   Array1<double> data;
   int numbuckets;
   Array1<int> freqs;
   
   double minfreq, maxfreq;
   double minval, maxval;
   clString id;

   void initfreqs();
   void FillBuckets();

   TCLdouble l, r;
   
public:
   Histogram();
   ~Histogram();
   
   void init_tcl();
   virtual void tcl_command(TCLArgs&, void*);

   void SetTitle( const clString& t ) const;
   void SetValueTitle( const clString& t ) const;
   void SetFrequencyTitle( const clString& t ) const;

   void ShowGrid() const;
   void HideGrid() const;
   
   void ShowRange() const;
   void HideRange() const;

   void GetRange( double& left, double& right );
   void SetRange( const double left, const double right );
   
   void GetMaxMin( double& left, double& right );

   int GetNumBuckets();
   void SetNumBuckets( const int nb );
   
   void SetData( const Array1<double> values );
   void ui() const;
   void update() const;
};

} // End namespace TclInterface
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:14  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:23  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:32  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:25  dav
// Import sources
//
//

#endif
