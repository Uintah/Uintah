
/*
 *  Persistent.h: Base class for persistent objects...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Persistent_h
#define SCI_project_Persistent_h 1

class clString;
class ifstream;
class ofstream;
class Point;
class Vector;

class Piostream {
public:
    enum Direction {
	Read,
	Write,
    };
protected:
    Piostream(Direction, int);
    Direction dir;
    int version;
    int err;
public:
    virtual ~Piostream();
    virtual void begin_class(const clString& name)=0;
    virtual void end_class()=0;

    virtual void begin_cheap_delim()=0;
    virtual void end_cheap_delim()=0;

    virtual void io(char&)=0;
    virtual void io(unsigned char&)=0;
    virtual void io(short&)=0;
    virtual void io(unsigned short&)=0;
    virtual void io(int&)=0;
    virtual void io(unsigned int&)=0;
    virtual void io(long&)=0;
    virtual void io(unsigned long&)=0;
    virtual void io(double&)=0;
    virtual void io(float&)=0;
    virtual void io(clString& string)=0;

    void io(Vector&);
    void io(Point&);

    int reading();
    int writing();
    int error();
};

class BinaryPiostream : public Piostream {
    ifstream* istr;
public:
    BinaryPiostream(const clString& filename, Direction dir);
    BinaryPiostream(ifstream*, int);

    virtual ~BinaryPiostream();
    virtual void begin_class(const clString& name);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
};

class TextPiostream : public Piostream {
    ifstream* istr;
    ofstream* ostr;
    void expect(char);
public:
    TextPiostream(const clString& filename, Direction dir);
    TextPiostream(ifstream*, int);
    virtual ~TextPiostream();
    virtual void begin_class(const clString& name);
    virtual void end_class();

    virtual void begin_cheap_delim();
    virtual void end_cheap_delim();

    virtual void io(char&);
    virtual void io(unsigned char&);
    virtual void io(short&);
    virtual void io(unsigned short&);
    virtual void io(int&);
    virtual void io(unsigned int&);
    virtual void io(long&);
    virtual void io(unsigned long&);
    virtual void io(double&);
    virtual void io(float&);
    virtual void io(clString& string);
};

Piostream* auto_istream(const clString& filename);

class Persistent {
public:
    virtual ~Persistent();
    virtual void io(Piostream&)=0;
};

#endif
