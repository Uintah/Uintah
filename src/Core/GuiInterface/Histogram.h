/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/GuiInterface/TCL.h>
#include <Core/GuiInterface/GuiVar.h>
#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE Histogram : public TCL {
   vector<double> data;
   int numbuckets;
   vector<int> freqs;
   
   double minfreq, maxfreq;
   double minval, maxval;
   string id;

   void initfreqs();
   void FillBuckets();

   GuiDouble l, r;
   
public:
   Histogram();
   ~Histogram();
   
   void init_tcl();
   virtual void tcl_command(TCLArgs&, void*);

   void SetTitle( const string& t ) const;
   void SetValueTitle( const string& t ) const;
   void SetFrequencyTitle( const string& t ) const;

   void ShowGrid() const;
   void HideGrid() const;
   
   void ShowRange() const;
   void HideRange() const;

   void GetRange( double& left, double& right );
   void SetRange( const double left, const double right );
   
   void GetMaxMin( double& left, double& right );

   int GetNumBuckets();
   void SetNumBuckets( const int nb );
   
   void SetData( const vector<double> &values );
   void ui() const;
   void update() const;
};

} // End namespace SCIRun


#endif
