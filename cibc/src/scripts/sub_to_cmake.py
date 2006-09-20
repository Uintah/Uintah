# Python script for converting sub.mk files into CMakeLists.txt files.
#
# usage 'python sub_to_cmake.py sub.mk > CMakeLists.txt'

import sys, regex, string, regsub

rex_srcdir = regex.compile('^SRCDIR.*:= \([a-zA-Z0-9/]+\)')
rex_srcs = regex.compile('^SRCS')
rex_pselibs = regex.compile('^PSELIBS')
rex_libs = regex.compile('^LIBS')
rex_subdirs = regex.compile('^SUBDIRS')

def cont(file, line):
    while line[-2:] == '\\\n':
        next = file.readline();
        if not next: break
        line = line[:-2] + next
    return line

def gen_cmake(filename):
    file = open(filename, 'r')
    libs = ''
    srcs = ''
    dirs = ''
    while 1:
        line = file.readline()
        if not line: break

        n = rex_srcdir.match(line)
        if n >= 0:
            srcdir = rex_srcdir.group(1)
            srcdir_ = string.joinfields(string.splitfields(srcdir, '/'), '_')
            continue

        n = rex_srcs.match(line)
        if n >= 0:
            srcs = cont(file, line)
            srcs = regsub.gsub('\$(SRCDIR)/', '', srcs)
            srcs = regsub.gsub('#.*', '', srcs)
            lsrcs = string.splitfields(srcs)
            srcs = '  '+string.joinfields(lsrcs[2:], '\n  ')
            continue

        n = rex_pselibs.match(line)
        if n >= 0:
            libstmp = cont(file, line)
            libstmp = regsub.gsub('/', '_', libstmp)
            libstmp = regsub.gsub('#.*', '', libstmp)
            llibs = string.splitfields(libstmp)
            libs = libs+'  '+string.joinfields(llibs[2:], '\n  ')
            continue

        n = rex_libs.match(line)
        if n >= 0:
            libstmp = cont(file, line)
            libstmp = regsub.gsub('(', '{', libstmp)
            libstmp = regsub.gsub(')', '}', libstmp)
            libstmp = regsub.gsub('#.*', '', libstmp)
            llibs = string.splitfields(libstmp)
            libs = libs+'\n  '+string.joinfields(llibs[2:], '\n  ')

        n = rex_subdirs.match(line)
        if n >= 0:
            dirs = cont(file, line)
            dirs = regsub.gsub('\$(SRCDIR)/', '', dirs)
            dirs = regsub.gsub('#.*', '', dirs)
            ldirs = string.splitfields(dirs)
            dirs = string.joinfields(ldirs[2:], ' ')
            continue

    print '#'
    print '#  For more information, please see: http://software.sci.utah.edu'
    print '# '
    print '#  The MIT License'
    print '# '
    print '#  Copyright (c) 2004 Scientific Computing and Imaging Institute,'
    print '#  University of Utah.'
    print '# '
    print '#  License for the specific language governing rights and limitations under'
    print '#  Permission is hereby granted, free of charge, to any person obtaining a'
    print '#  copy of this software and associated documentation files (the "Software"),'
    print '#  to deal in the Software without restriction, including without limitation'
    print '#  the rights to use, copy, modify, merge, publish, distribute, sublicense,'
    print '#  and/or sell copies of the Software, and to permit persons to whom the'
    print '#  Software is furnished to do so, subject to the following conditions:'
    print '# '
    print '#  The above copyright notice and this permission notice shall be included'
    print '#  in all copies or substantial portions of the Software.'
    print '# '
    print '#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS'
    print '#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,'
    print '#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL'
    print '#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER'
    print '#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING'
    print '#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER'
    print '#  DEALINGS IN THE SOFTWARE.'
    print '#'
    print ''
    print '# CMakeLists.txt for %s' % srcdir
    print ''
    if srcs != '':
        print 'SET(%s_SRCS' % srcdir_
        print srcs
        print ')'
        print ''
        print 'ADD_LIBRARY(%s ${%s_SRCS})' % (srcdir_, srcdir_)
        print ''
        print 'TARGET_LINK_LIBRARIES(%s' % srcdir_
        print libs
        print ')'
        print ''
        print 'IF(NOT BUILD_SHARED_LIBS)'
        print '  ADD_DEFINITIONS(-DBUILD_%s)' % srcdir_
        print 'ENDIF(NOT BUILD_SHARED_LIBS)'
    if dirs != '':
        print 'SUBDIRS(%s)' % dirs
        print ''

if __name__ == '__main__':
    import sys
    gen_cmake(sys.argv[1])
        
