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
#    File   : seg-nrrd.py
#    Author : Martin Cole
#    Date   : Fri Jun 30 11:52:38 2006

from sys import argv
import popen2
import re
import os

maxval_re = re.compile("max: ([\d\.\d]+)")
axsizes_re = re.compile("sizes: (\d+) (\d+) (\d+)")
spacing_re = re.compile("space directions: \(([\-\d\.]+),?[\-\d\.]+,?[\-\d\.]+\) \([\-\d\.]+,?([\-\d\.]+),?[\-\d\.]+\) \([\-\d\.]+,?[\-\d\.]+,?([\-\d\.]+)\)")
maxval = 0
axsizes = ()
small_ax = 99999999

def inspect_nrrd(innrrd) :
    global axsizes
    global maxval
    cmmd = "unu minmax %s" % innrrd
    t = popen2.popen2(cmmd)
    for l in t[0].readlines() :
        mo = maxval_re.match(l)
        if mo != None :
            maxval = int(mo.group(1))

    cmmd = "unu head %s" % innrrd
    print cmmd
    t = popen2.popen2(cmmd)
    for l in t[0].readlines() :
        mo = axsizes_re.match(l)
        if mo != None :
            axsizes = (int(mo.group(1)), int(mo.group(2)), int(mo.group(3)))

#there is inner data and outer data.  For all materials considered to be
# inner, zero out the beginning and ending slice, so that in the final
# set of isosurfaces to not have intersecting faces at these cut off points.
def pad_nrrd(innrrd, idx) :
    global small_ax

    if (idx != 0) :
        # zero out the first and last slice in z
        # grab a slice
        cmmd = "unu slice -a 2 -p 0 -i %s -o tmp.nrrd" % innrrd
        print cmmd
        os.system(cmmd)
        cmmd = 'unu 2op "x" tmp.nrrd 0.0 -o zero_slice.nrrd'
        print cmmd
        os.system(cmmd)
        cmmd = "unu splice -a 2 -p 0 -s zero_slice.nrrd -i %s -o %s"\
               % (innrrd, innrrd)
        print cmmd
        os.system(cmmd)
        cmmd = "unu splice -a 2 -p M -s zero_slice.nrrd -i %s -o %s"\
               % (innrrd, innrrd)
        print cmmd
        os.system(cmmd)

    cmmd = "unu pad -min -2 -2 -2 -max %d %d %d "\
           "-b pad -v 0 -i %s -o %s.pad.nhdr" %\
           (small_ax + 1, small_ax + 1, small_ax + 1, innrrd, innrrd[:-5])

    print cmmd
    os.system(cmmd)

def make_lut (idx) :
    global maxval
    hit_str = "1.0\n"
    miss_str = "0.0\n"
    if (idx == 0) :
        hit_str = "0.0\n"
        miss_str = "1.0\n"
        
    f = open("lut%d.raw" % idx, "w")
    flist = []
    
    for i in range(0, maxval + 1) :
        if i == idx :
            flist.append(hit_str)
        else :
            flist.append(miss_str)
    f.writelines(flist)
    f.close()

    cmmd = "unu make -i lut%d.raw -t float -s %d -e ascii "\
           "-o lut%d.nrrd" % (idx, maxval+1, idx)
    print cmmd
    os.system(cmmd)

def uniform_resample_nnrd(idx) :
    global small_ax
    global axsizes
    small_ax = 99999999
    if axsizes[0] < small_ax :
        small_ax = axsizes[0]
    if axsizes[1] < small_ax :
        small_ax = axsizes[1]
    if axsizes[2] < small_ax :
        small_ax = axsizes[2]
        
    # create a uniformly sampled nrrd
    cmmd = "unu resample -s %d %d %d -k cubic:0,0.5 "\
           "-i mat%d.nrrd -o mat%d.resamp.nhdr" % \
           (small_ax, small_ax, small_ax, idx, idx)
    print cmmd
    os.system(cmmd)


def make_afront_nhdr(hdr) :
    # create an afront compatible nrrd header
    fout = []
    hf = open(hdr)
    for l in hf.readlines() :
        mo = spacing_re.match(l)
        if mo != None :
            x = float(mo.group(1))
            y = float(mo.group(2))
            z = float(mo.group(3))
##             if x < 0.0 : x *= -1
##             if y < 0.0 : y *= -1
##             if z < 0.0 : z *= -1
            
            fout.append("spacings: %f %f %f\n" % (x,y,z))

        else :
            fout.append(l)

    # write an afront compatible nrrd header
    new_hdr = open("afront.%s" % hdr, "w")
    new_hdr.writelines(fout)
    

def make_solo_material(idx, innrrd) :
    make_lut(idx)
    cmmd = "unu lut -m lut%d.nrrd -t float -i %s"\
           " -o mat%d.nrrd" % (idx, innrrd, idx)
    print cmmd
    os.system(cmmd)

def afront_isosurface(f, t) :
    # the index
    idx = t[0]
    # the isoval to use at that index
    iv = t[1]
    
    exe = "/scratch/mjc/afront-claudio/afront"
    cmmd = "%s -nogui %s -failsafe f -rho 0.5 -tri %f bspline" % (exe, f, iv)
    
    print cmmd
    os.system(cmmd)
    cmmd = "mv outmesh.m mat%dmesh.m" % idx
    os.system(cmmd)
    cmmd = "mv outmesh.failsafe.txt mat%dfailsafe.txt" % idx
    os.system(cmmd)

def make_trisurf(idx) :
    exe = "/scratch/mjc/trunk/SCIRun/dbg/StandAlone/convert/MtoTriSurfField"
    cmmd = "%s mat%dmesh.m mat%d.ts.fld" % (exe, idx, idx)
    print cmmd
    os.system(cmmd)
    
if __name__ == "__main__" :
    nrrd = argv[1]
    inspect_nrrd(nrrd)
    # (index, isoval) pairs : We chose isovals such that adjacent surfaces
    # to not intersect, which would make tetgen fail.
    mats = ((0, 0.501), (8, 0.63), (9, 0.501), (11, 0.68))
    #mats = ((8, 0.63), (11, 0.68))
    for t in mats :
        idx = t[0]
        make_solo_material(idx, nrrd)
        uniform_resample_nnrd(idx)
        
        pad_nrrd("mat%d.resamp.nhdr" % idx, idx)
        make_afront_nhdr("mat%d.resamp.pad.nhdr" % idx)
        afront_isosurface("afront.mat%d.resamp.pad.nhdr" % idx, t)
        make_trisurf(idx)
