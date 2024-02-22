/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/Materials/Contact/Contact.h>
#include <vector>

using namespace Uintah;
using namespace std;

Contact::Contact(const ProcessorGroup* myworld, MPMLabel* Mlb, MPMFlags* MFlag, ProblemSpecP ps)
  : lb(Mlb), flag(MFlag), d_matls(ps)
{
}

Contact::~Contact()
{
}

void Contact::setContactMaterialAttributes()
{
}

// find velocity from table of values
Vector Contact::findValFromProfile(double t,
                               vector<pair<double, Vector> > profile) const
{
  int smin = 0;
  int smax = (int)(profile.size())-1;
  double tmin = profile[0].first;
  double tmax = profile[smax].first;
  if(t<=tmin) {
      return profile[0].second;
  }
  else if(t>=tmax) {
      return profile[smax].second;
  }
  else {
      // bisection search on table
      // could probably speed this up by keeping copy of last successful
      // search, and looking at that point and a couple to the right
      //
      while (smax>smin+1) {
          int smid = (smin+smax)/2;
          if(d_vel_profile[smid].first<t){
            smin = smid;
          }
          else{
            smax = smid;
          }
      }
      double l  = (profile[smin+1].first-profile[smin].first);
      double xi = (t-profile[smin].first)/l;
      double vx = xi*profile[smin+1].second[0]+(1-xi)*profile[smin].second[0];
      double vy = xi*profile[smin+1].second[1]+(1-xi)*profile[smin].second[1];
      double vz = xi*profile[smin+1].second[2]+(1-xi)*profile[smin].second[2];
      return Vector(vx,vy,vz);
    }
}
