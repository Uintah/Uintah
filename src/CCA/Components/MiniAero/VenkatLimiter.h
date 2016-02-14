/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

double limit(double dumax, double dumin, double du, double deltax3){
  double beta = 1;
  double epstilde2 = deltax3*beta*beta*beta;
  double phi = 0;
  
  double denom = 0;
  double num = 0;
  
  
  if (du > 0)
  {
    num = (dumax*dumax + epstilde2)*du + 2*du*du*dumax;
    denom = du*(dumax*dumax  + 2*du*du + dumax*du + epstilde2);
    phi = num/denom;
  }
  else if (du < 0)
  {
    num  = (dumin*dumin + epstilde2)*du + 2*du*du*dumin;
    denom = du*(dumin*dumin  + 2*du*du + dumin*du + epstilde2);
    phi = num/denom;
  }
  else
  {
    phi = 1;
  }
  
  
  return phi;
}
