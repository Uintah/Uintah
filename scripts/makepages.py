#! /usr/local/bin/python

import sys
import os
import re

def mkdir_and_copy(src, dsttop, trim) :   
    dst = dsttop + src[trim:]
    cmmd = 'mkdir -p ' + dst
    os.system(cmmd)
    cmmd = 'cp ' + src + os.sep + '*.* ' + dst
    os.system(cmmd)
    
def copy_doc1(arg, dirname, names) :
    if dirname[-3:] == 'CVS' : return #don't work in CVS dirs!!
    #print "recurse working in: " + dirname
    webtop = arg[0]
    trim = arg[1]
    mkdir_and_copy(dirname, webtop, trim)
    
def copy_doc(arg, dirname, names) :
    if dirname[-3:] != 'doc' : return #only work in doc dirs!!
    #print "working in: " + dirname
    webtop = arg[0]
    trim = arg[1]
    os.path.walk(dirname, copy_doc1, (webtop, trim))

def deepcopy_doc_dirs(psetop, webtop) :
    os.path.walk(psetop, copy_doc, (webtop, len(psetop)) )


def create_customs_file(webtop, urltop, tmptop, l) :
    # default customization
    cust = [
        'customize\n',
        '        usetable\n',
        '        logo            /images/SCI_logo.gif\n',
        '        showprivates\n',
        '        nobacktotop\n',
        '        backcolor white\n',
        '        ignoredirective PSECORESHARE\n',
        '        ignoredirective SCICORESHARE\n',
        '	numkeycolumns 3\n',
        'end_customize\n',
        '\n',
        'webroot  ' + webtop + '\n',
        'urlroot  ' + urltop + '\n',
        '\n',
        '#\n',
        '# libary <name> <source_dir> <web_sub_dir>\n',
        '#\n']
    
    #get the dirs to skip 
    skip_fptr = open('src/scripts/cocoon.skip', 'r')
    skip_list = skip_fptr.readlines()
    skip_fptr.close()

    clist = []
    for webname in l :
        # see if this is on the skip list
        skip_p = 0
        for skip in skip_list[1:] :
            if webname[4:] == skip[:-1] :
                #print 'skipping -- ' + webname[4:]
                skip_p = 1
                break
        if skip_p : continue
        # make the libname out of the path
        libname = re.sub('/', '::', webname)
        if(libname == '') :
            libname="PSE"
            webname="."
            exit(1)

        #build list of strings to send to file
        src = psetop + os.sep + webname
        clist.append("library " + libname + ' ' + src + ' ' + webname + '\n')

        # If there is a doc/cocoon.cust file, use it.
        #Otherwise use the default that specifies no sentinels
        try:
            fname = src + os.sep + 'doc/cocoon.cust'
            cust_fptr = open(fname, 'r')
            print "using " + fname
            for ln in cust_fptr.readlines() :
                clist.append(ln)
            cust_fptr.close()
        except IOError:
            clist.append(' customize\n')
            clist.append('     nosentinels\n')
            clist.append(' end_customize\n')
            
        #add a newline bewteen library data
        clist.append('\n')

    
    #create the customization file
    customs_filename = tmptop + os.sep + 'cocoon_config'
    file_ptr = open(customs_filename, 'w')
    file_ptr.writelines(cust)
    #write data do file
    file_ptr.writelines(clist)
    file_ptr.close()
    return customs_filename

def hfile_p(arg, dirname, names) :
    for n in names :
        if n[-2:] == '.h' :
            arg[0].append(dirname[arg[1]+1:])
            break

def all_h_dirs(psetop) :
    hlist = []
    os.path.walk(psetop, hfile_p, (hlist, len(psetop)))
    return hlist


# main 
if __name__ == '__main__' :

    webtop = ''
    psetop = ''
    urltop = ''
    try:
        webtop = sys.argv[1:][0]
        psetop = sys.argv[1:][1]
        urltop = sys.argv[1:][2]
    except IndexError:
        print "usage: makepages.py WEBTOP PSETOP URLTOP"
        print "WEBTOP is the root of the web directory"
        print "PSETOP is the location of the PSE/src directory"
        print "URLTOP is the web location of the pages"
        sys.exit(1) #no need to continue

    tmptop = psetop + '/..'
    deepcopy_doc_dirs(psetop, webtop)
    l = all_h_dirs(psetop)
    cfn = create_customs_file(webtop, urltop, tmptop, l)
    # run cocoon...
    cmmd = 'time cocoon ' + cfn
    status = os.system(cmmd)
    if status != 0 :
        print "COCOON FAILED using config file: "
        fptr = open(cfn, 'r')
        for ln in fptr.readlines() :
            print ln[:-1]
        fptr.close()
        #error is in the high-byte of the 16 bit return value.
        sys.exit(status >> 8) # signal the same error
    os.remove(cfn)
