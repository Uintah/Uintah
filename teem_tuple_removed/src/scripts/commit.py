#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#  
#    File   : commit.py
#    Author : Martin Cole
#    Date   : Tue Sep 10 11:09:58 2002

import os
import sys
import tempfile

all_messages = []
file_msg_map = {}
all_docs = []
file_doc_map = {}
all_progress = []
file_progress_map = {}

global use_editor
use_editor = ''

if os.environ.has_key('EDITOR') :
    use_editor = os.environ['EDITOR']
    print "generating messages with %s" % use_editor
if use_editor == '' :
    use_editor = 'emacs'
    print "no EDITOR env var, using emacs"

def do_it(cmmd) :
    #print cmmd
    os.system(cmmd)

def get_user_input(pre) :
    # load editor with random filename
    filename = tempfile.mktemp(".%s.commit" % pre)
    os.system("emacs %s" % filename)
    return filename

# files are all relative to this working tree.  This file lives in
# SCIRun/src/scripts.
def get_changed_files() :
    cwd = os.getcwd()
    print 'checking build from %s' % cwd

    cmmd = 'cvs -n update' 
    f_list = os.popen(cmmd, 'r').readlines()
    return f_list;

def choose_message(l) :
    index = 0
    for msg in l :
        print '------------ message %d is in %s ------------' % (index, msg)
        try:
            fp = open(msg, 'r')
        except IOError :
            print "----EMPTY MESSAGE---"
            continue
            
        for ln in fp.readlines() :
            print ln[:-1]
        print ""
        index = index + 1
    s = '-1'
    while int(s) >= index or int(s) < 0 :
        s = raw_input("which message? ")
        try:
            int(s)
        except ValueError :
            s = '-1'
    return s

def remove_mapping(list, map, file) :
    for i in range(0, len(list)) :
        if map.has_key(i) :
            l = map[i]
            if file in l :
                l.remove(file)

# add user input to the list, and map the file to the list index
def add_text(list, prompt, map, file) :
    t = get_user_input(prompt)
    print "here is t: %s " % t
    list.append(t)
    # pull file out of any previous mappings
    remove_mapping(list, map, file)
    index = len(list) - 1
    if not map.has_key(index) : map[index] = []
    map[index].append(file)

def map_text(list, map, file) :
    if (len(list) == 0) :
        print "Error! no messages added yet"
        return
    
    # pull file out of any previous mappings
    remove_mapping(list, map, file)
    # get the msg index
    msg = choose_message(list)
    # add this filename to the list at index in map
    index = int(msg)
    if not map.has_key(index) : map[index] = []
    map[index].append(file)
        
def add_commit(f) :
    global all_messages
    global file_msg_map
    global all_docs
    global file_doc_map
    global all_progress
    global file_progress_map
    
    message = "1) add message\n2) select message for %s\n3) add documentation\n4) select documentation for %s\n5) add progress\n6) select progress for %s\n7) done mapping %s\n" % (f, f, f, f)
    done = 0
    status = 0
    while not done :
        os.system("cvs status %s" % f)
        s = raw_input(message)
        if s == '1' :
            add_text(all_messages, 'msg', file_msg_map, f)
        if s == '2' :
            map_text(all_messages, file_msg_map, f)
        if s == '3' :
            add_text(all_docs, 'doc', file_doc_map, f)
        if s == '4' :
            map_text(all_docs, file_doc_map, f)
        if s == '5' :
            add_text(all_progress, 'prog', file_progress_map, f)
        if s == '6' :
            map_text(all_progress, file_progress_map, f)
        if s == '7' :
            done = 1

    return 1


def map_and_msg_files(flist) :
    print "Add commit message."
    add_text(all_messages, 'msg', file_msg_map, flist[0])
    for f in flist[1:] :
        if not file_msg_map.has_key(0) : file_msg_map[0] = []
        file_msg_map[0].append(f)

    s = '-1'
    while s != 'y' and s != 'n' :
        s = raw_input("Add documentation message? ")

    if s == 'y' :
        add_text(all_docs, 'doc', file_doc_map, flist[0])
        for f in flist[1:] :
            if not file_doc_map.has_key(0) : file_doc_map[0] = []
            file_doc_map[0].append(f)

    s = '-1'
    while s != 'y' and s != 'n' :
        s = raw_input("Add progress message? ")

    if s == 'y' :        
        add_text(all_progress, 'prog', file_progress_map, flist[0])
        for f in flist[1:] :
            if not file_progress_map.has_key(0) :
                file_progress_map[0] = []
            file_progress_map[0].append(f)

    
def visit_files(files) :
    
    for f in files :
        # show a diff
        print f[:-1]
        cmmd = "cvs diff %s" % f[2:-1]
        print cmmd
        os.system(cmmd)

        message = "1) cvs commit\n2) cvs add\n3) cvs rm -f \n4) rm\n5) done with: %s \n" % f[2:-1]
        done = 0
        while not done :
            os.system("cvs status %s" % f[2:-1])
            s = raw_input(message)
            if s == '1' :
                done = not add_commit(f[2:-1])
            if s == '2' :
                os.system('cvs add %s' % f[2:-1])
            if s == '3' :
                os.system('cvs rm -f %s' % f[2:-1])
            if s == '4' :
                done = 1
                os.system('rm -f %s' % (f[2:-1]))
            if s == '5' :
                done = 1


def send_mail(map, list, mail_tup) :
    for k in map.keys() :
        fstr = ''
        files = map[k]
        for f in files :
            fstr = fstr + f + ' '
        l = list[int(k)]

        cmmd = 'echo "-- associated files --" >> %s' % (l)
        do_it(cmmd)
        cmmd = 'echo "%s" >> %s' % (fstr, l)
        do_it(cmmd)

        for i in range(0, len(mail_tup)) :
            mt = mail_tup[i]
            cmmd = 'mail %s -s "%s Notify" < %s' % (mt[0], mt[1], l)
            do_it(cmmd)
            
        #cmmd = 'rm -f %s' % l
        #do_it(cmmd)   

def do_commit() :
    global all_messages
    global file_msg_map
    global all_docs
    global file_doc_map
    global all_progress
    global file_progress_map


    for k in file_msg_map.keys() :
        fstr = ''
        files = file_msg_map[k]
        for f in files :
            fstr = fstr + f + ' '
        l = all_messages[int(k)]

        if fstr != '' :
            cmmd = 'cvs commit -F %s %s' % (l, fstr)
            do_it(cmmd)


    send_mail(file_doc_map, all_docs,
              (('scirun-doc@sci.utah.edu', 'Documentation'),))

    send_mail(file_progress_map, all_progress,
              (('scirun-exec@sci.utah.edu', 'Progress'),
               ('sci-media@sci.utah.edu', 'Release Notes')))

def analyze_files(files) :
    status = 'good'
    for f in files :
        if f[0] == 'U' :
            print "%s needs an update" % f[2:-1]
            files.remove(f)

        if f[0] == 'C' :
            print "%s has a conflict" % f[2:-1]
            status = 'bad'

    if status == 'bad' : return 0
    return 1


    
if __name__ == '__main__' :

    if len(sys.argv) > 1 :
        # expert mode.
        print "expert mode"
        map_and_msg_files(sys.argv[1:])
        do_commit()
    else :
        files = get_changed_files()
        if analyze_files(files) :
            visit_files(files)
            do_commit()
    
    
