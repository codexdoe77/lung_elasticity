
// CUDA runtime
#include <cuda_runtime.h>
#include <mcheck.h>

#define MAX_INPUT_FILES 562
#define FILE_TYPE_SIZE 3
#define ACQUISITION_DATE_SIZE 8
#define MAX_CHAR_LENGTH 1024

#define SUCCESS 1
#define FAILURE 0

#define PI 3.14159f
#define CTR_RES 0.2f
#define DOWN_SAMP 2

#define LM_COUNT 100

// #define NUM_SPRINGS 26
#define SPRING_SEARCH_SIZE 8

#define WIDTH 400
#define HEIGHT 400


/////////////////////////////// Structure Definitions ////////////////////////////////////////////////
typedef struct
{
    char name[MAX_CHAR_LENGTH];             // char array to hold filename
} FILE_NAME;

typedef struct
{
    FILE_NAME slice[MAX_INPUT_FILES];       // array of chars to hold filenames
} FILE_LIST;

typedef struct
{
    int ROInumber;                              // the id number of this contour
    int *CTRpoints;                             // array of ints containing the number of points in each sub-contour
    int TOTpoints;                              // total number of points in all sub-contours
    int subCntrs;                               // number of sub-contours that comprise contour
    float avgHU;                                // average HU of contour
    int3 rgb;                                   // color associated with contour in rgb as (x,y,z)
    char ROIname[MAX_CHAR_LENGTH];              // contour name
    float3 contourSize;                         // a float3 vector holding the dimensions of the sub-volume around a contour extent
    float3 *matrix;                             // matrix of float3 vectors, containing the points that draw the contour
    unsigned int *lm_index;                     // array to hold the voxel id of each landmark within the contour
} CNTR_SPECS;

typedef struct
{
    char *dir;                                  // dicom directory
    FILE_NAME file;                             // file name
    char type[FILE_TYPE_SIZE];                   // filetype, should be dcm
    int CTRnumber;                               // total number of contours in the dicomrt file
    CNTR_SPECS *contour;                        // struct containing details about loaded contours
    int *cntrParticles;                           // an int array to hold the number of particles in each contour
    int *cntrSegData;                           // an int array to hold all differen enumerator values for types of tissue - used to sort particles later
    int cntr_count;                             // number of contours loaded + generic bony anatomy + generic soft tissue
    unsigned int *include;                  // boolean array to signal which contours to load
} DICOM_STRUCT;


typedef struct
{
    float3 refPoint;                        // reference point of 3D data volume used to localize contours
    float3 voxelSize;                       // size of voxels in mm (x,y,z)
    int3 arraySize;                         // size of 3D data volume, in voxels (x,y,z)
    float max;                              // maximum value in 3D data volume
    float min;                              // minimum value in 3D data volume
} DATA_PARAMS;

typedef struct
{
    DATA_PARAMS params;                      // struct holding the parameters of the 3D data volume
    unsigned char *array3D;                         // 3D data volume
    unsigned char max;
    unsigned char min;
} CHARDATA_VOLUME;


typedef struct
{
    DATA_PARAMS params;                      // struct holding the parameters of the 3D data volume
    float *array3D;                         // 3D data volume
} DATA_VOLUME;

typedef struct
{
    char *dir;                              // dicom directory
    FILE_LIST files;                        // list of dicom files within dicom directory
    DATA_VOLUME dataset;                    // information about the 3D dicom dataset
    char type[FILE_TYPE_SIZE];              // filetype, should be dcm
    char date[ACQUISITION_DATE_SIZE];        // date of dicom acquisition
    int *segment;                           // int array used to enumerate each voxel as structures from dicomrt file
    //unsigned int *sortedIDs;                         //uint array to hold the original voxel ids of the particles after sorting
} DICOM_CT;

#define GRID_VALUE(GRID_ptr, i, j, k)\
    ((GRID_ptr)->array3D[(i) + (GRID_ptr)->params.arraySize.x * ((j) + ((k) * (GRID_ptr)->params.arraySize.y))])

void LoadImageAsText(float *data, uint3 dim, const char *path);
void writeToTextFile(const char *path, int3 size, float *data);

int load_files( char *, FILE_LIST *, DATA_VOLUME *, char *, char *, int );
int load_data( char *, FILE_LIST *, DATA_VOLUME *, char *, int, bool );
int save_data( const char *, const char *, FILE_LIST *, DATA_VOLUME *, float *, char * );


