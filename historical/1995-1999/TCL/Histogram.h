
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

#include <Classlib/Array1.h>
#include <TCL/TCL.h>
#include <TCL/TCLvar.h>

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

#endif
