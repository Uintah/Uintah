/******************************************************************************
 * File: labelmaps.h
 *
 * Description: C header source class definitions to provide an API for
 *		Visible Human segmentation information:
 *		* The Master Anatomy Label Map
 *		* Spatial Adjacency relations for the anatomical structures
 *		and
 *		* Bounding Boxes for each anatomical entity.
 *
 * Author: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *	   <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#ifndef LABELMAPS_H
#define LABELMAPS_H

#define VH_LM_NUM_NAMES 512

/******************************************************************************
 * class VH_MasterAnatomy
 *
 * description: Two parallel arrays -- names <-> indices
 *
 *		The map from anatomical volume (e.g., C-T, MRI) tags to
 *		anatomical names.
 ******************************************************************************/

class VH_MasterAnatomy {
public:
	VH_MasterAnatomy();
	~VH_MasterAnatomy();
	void readFile(char *infilename);
	void readFile(FILE *infileptr);
	char *get_anatomyname(int labelindex);
	int get_labelindex(char *anatomyname);
	int get_num_names() { return num_names; };
private:
	char **anatomyname;
	int *labelindex;
	int num_names;
};

/******************************************************************************
 * class VH_AdjacencyMapping
 *
 * description: An array of integers for each anatomical entity -- the
 *		relation describes every entity spatially adjacent to the
 *		indexed entity.
 ******************************************************************************/

#define VH_FILE_MAXLINE 2048

class VH_AdjacencyMapping {
public:
	VH_AdjacencyMapping();
	~VH_AdjacencyMapping();
	void readFile(char *infilename);
        void readFile(FILE *infileptr);
	int get_num_rel(int index);
	int get_num_names() { return num_names; };
	int *adjacent_to(int index);
private:
	int **rellist;
	int *numrel;
	int num_names;
};

/******************************************************************************
 * class VH_AnatomyBoundingBox
 ******************************************************************************/
class VH_AnatomyBoundingBox {
public:
        VH_AnatomyBoundingBox() { flink = (VH_AnatomyBoundingBox *)0; };
        void readFile(FILE *infileptr);
	char *anatomyname;
	int minX, maxX, minY, maxY, minZ, maxZ, minSlice, maxSlice;
	VH_AnatomyBoundingBox *flink;
private:
};

/******************************************************************************
 * Read an ASCII AnatomyBoundingBox file into a linked list
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_readBoundingBox_File(char *infilename);

/******************************************************************************
 * Find the boundingBOx of a named anatomical entity
 ******************************************************************************/
VH_AnatomyBoundingBox *
VH_Anatomy_findBoundingBox(VH_AnatomyBoundingBox *list, char *anatomyname);

#endif
