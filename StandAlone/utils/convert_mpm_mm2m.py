#  
#  For more information, please see: http://software.sci.utah.edu
#  
#  The MIT License
#  
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
#  
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#  
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#  
#    File   : convert_mpm_mm2m.py
#    Author : Martin Cole
#    Date   : Mon May 30 13:53:58 2005
#    
#   Convert mpm files generated in millimeters to meter units.

import sys
import re

# col index where volume lives.
vol_idx = 3

float_re = re.compile('([\d\.\-e]+)')

def convert(input) :
    out_ln = []
    for ln in input.readlines() :
        s = 0
        e = len(ln) - 1
        col = 0
        oln = ''
        while (s < e) :
            mo = float_re.search(ln, s, e)
            if (mo == None) :
                print "no match in :"
                print ln[s:e]
            else :
                if col < vol_idx :
                    oln = "%s %e" % (oln, float(mo.group(1)) * .001)
                elif col == vol_idx :
                    oln = "%s %e" % (oln,
                                     float(mo.group(1)) * .001 * .001 * .001)
                else :
                    oln = "%s %e" % (oln, float(mo.group(1)))
                s = mo.end(1)
                col = col + 1
        out_ln.append("%s\n" % oln)
    return out_ln


def show_usage() :
    print "python convert_mpm_mm2m.py <input> <output>"

if __name__ == '__main__' :
    inp = ''
    outp = ''
    try :
        inp = sys.argv[1]
        outp = sys.argv[2]
    except :
        show_usage()
        sys.exit(1);

    lns = convert(open(inp))
    o = open(outp, 'w')
    o.writelines(lns)
