#ifndef __GEOMETRY_PIECE_H__
#define __GEOMETRY_PIECE_H__

class GeomPiece {
private:

        double          geomBounds[7];
        int             pieceType;
        int             piecePosNeg;
        int             pieceMatNum;
	int		pieceVelFieldNum;
	double		initVel[4];

public:

        GeomPiece();
        ~GeomPiece();

        void    setPieceType(int pt);
        void    setPosNeg(int pn);
        void    setMaterialNum(int mt);
        void    setVelFieldNum(int vf_num);
        int     getPosNeg();
        int     getPieceType();
        int     getMaterialNum();
	int	getVFNum();
        double  getGeomBounds(int j);
        void    setGeomBounds(double bnds[7]);
	void	setInitialConditions(double icv[4]);
	double	getInitVel(int i);

};

#endif // __GEOEMTRY_PIECE_H__

// $Log$
// Revision 1.1  2000/02/24 06:11:57  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:51  sparker
// Stuff may actually work someday...
//
// Revision 1.1  1999/06/14 06:23:41  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.4  1999/02/10 20:53:10  guilkey
// Updated to release 2-0
//
// Revision 1.3  1999/01/26 21:53:34  campbell
// Added logging capabilities
//
