/*
 *  FdStreambuf.h: A file descriptor streambuf
 *
 *  Written by:
 *   Jason V. Morgan
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *  Updated by:
 *   Jason V. Morgan
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Morgan_FdStream_h
#define Morgan_FdStream_h

#include <iostream>
#include <stdio.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace Morgan {
    using namespace std;

    class fdstreambuf : public streambuf {
    private:
        typedef char char_type;
        typedef int int_type;

    public:
        fdstreambuf(int fd) : fd(fd) {}
        ~fdstreambuf() { close(fd); }

        streamsize xsgetn(char_type* s, streamsize n) {
            streamsize sz = read(fd, s, n * sizeof(char_type)) / sizeof(char_type);
            return sz;
        }

        int_type underflow() {
            char_type c;
            if(read(fd, &c, sizeof(char_type)) != sizeof(char_type)) {
                return EOF;
            }
            return c;
        }
        
        int_type uflow() {
            return underflow();
        }

        streamsize xsputn(const char_type* s, streamsize n) {
            return write(fd, s, n * sizeof(char_type)) / sizeof(char_type);
        }

        int_type overflow(int_type c = EOF) {
            char_type c_out = c;
            if(write(fd, &c_out, sizeof(char_type)) != sizeof(char_type)) {
                perror("Can't write character");
                return EOF;
            }
            return c;
        }

    private:
        int fd;
    };
}

#endif // FdStreambuf_h
