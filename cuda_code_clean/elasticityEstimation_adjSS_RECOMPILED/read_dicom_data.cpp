#define HAVE_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <cuda_runtime_api.h>

#include "read_dicom_data.h"

/*
import_text_data can import the pixel data from a directory of text files. The data is loaded into a DATA_VOLUME structure.
images are assumed to be square
*/
bool import_text_data( char *path, FILE_NAME *file, DATA_VOLUME *dataVol, int slice, int downSamp)
{
    char filename[255];
    sprintf(filename,"%s/%s",path,file->name);
    FILE *fp;

    int position = 0;
    fp = fopen(filename, "r");
    if (fp != NULL)
    {
        for (int j = 0; j < dataVol->params.arraySize.y / downSamp; j++)
            for (int i = 0; i < dataVol->params.arraySize.x / downSamp; i++)
            {
                float temp;
                fscanf(fp,"%f", &temp);
                if (temp > dataVol->params.max) dataVol->params.max = temp;
                if (temp < dataVol->params.min) dataVol->params.min = temp;
                position = i + (dataVol->params.arraySize.x / downSamp)*(j + slice*(dataVol->params.arraySize.y / downSamp));
                dataVol->array3D[position] = temp;
            }
    }
    else
    {
        printf(" LOAD DATA ERROR \n");
        return false;
    }
    fclose(fp);

    return true;
}

/*
sort_files uses the hash value in the int array sort to re-order the filenames
in the dfiles FILE_LIST into increasing numerical order
*/
int sort_files(FILE_LIST *dfiles, int *sort, int count)
{
    int count2 = count;
    for ( int i=0; i<2*count; i++){
        for( int j=1; j<count2; j++){
            if ( sort[j-1] > sort[j] ){

                int tempint = sort[j];
                sort[j] = sort[j-1];
                sort[j-1] = tempint;

                char tempchar[MAX_CHAR_LENGTH];
                strcpy(tempchar,dfiles->slice[j].name);
                strcpy(dfiles->slice[j].name,dfiles->slice[j-1].name);
                strcpy(dfiles->slice[j-1].name,tempchar);
            }
        }
        count2--;
    }
    return SUCCESS;
}

/*
load_files opens the given directory and loads the filenames of all the files within it.
The parameters of the data volume are found by opening the first file.
All files are sorted into numerical order.
*/
int load_files( char *path, FILE_LIST *dfiles, DATA_VOLUME *dataVol, char *filetype, char *date, int downSamp )
{
    
    bool ytrue = false;
    struct dirent *dp = NULL;
    DIR *dfd = NULL;
    int count = 0;
    
    if ((dfd = opendir(path)) == NULL) {
           printf("\n dirwalk: can't open %s\n", path);
           return 0;
    }

    while ((dp = readdir(dfd)) != NULL) {
       if ( strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0 )
            continue;
       else {
            strcpy(dfiles->slice[count].name, dp->d_name);
            char *pch;
            pch=strrchr(dfiles->slice[count].name,'.');
            if( strcmp(pch,".dcm") == 0 || strcmp(pch,".txt") == 0 )
                count++;
       }
    }
    printf("\n Total Files Read: %d\n",count);
    closedir(dfd);

    int length = strlen(dfiles->slice[0].name);
    for (int s=0; s<3; s++)
        filetype[s] = dfiles->slice[0].name[length - 3 + s];
    printf("\n Filetype is %s",filetype); fflush(stdout);

    int *sort;
    sort = (int *)malloc(count*sizeof(int));
    memset(sort,0,count*sizeof(int));

    dataVol->params.arraySize.z = count;
    dataVol->params.min = 1e5;
    dataVol->params.max = -1e5;

    if ( strncmp(filetype,"txt",3) == 0 )
    {

        float temp;
        int pixels = 0;
        char filename[MAX_CHAR_LENGTH];
        sprintf(filename,"%s/%s",path,dfiles->slice[0].name);
        FILE *slice0 = fopen(filename,"r");
        if (slice0 != NULL){
            while (fscanf(slice0,"%f",&temp) != EOF) {
                pixels++;
            }
            fclose(slice0);
        }
        else { printf("!!! Failed to open file !!!"); }

        pixels = sqrt( pixels );
        dataVol->params.arraySize.x = pixels;
        dataVol->params.arraySize.y = pixels;
        printf("\n Data Dimensions: %d x %d x %d (Down Sample: %d)\n\n",dataVol->params.arraySize.x,dataVol->params.arraySize.y,dataVol->params.arraySize.z,downSamp);

        int pos = 0;
        bool diff = 1;
        do {
            for (int k = 1; k<count; k++){
                diff = (dfiles->slice[0].name[pos] == dfiles->slice[k].name[pos]);
                if (!diff) break;
            }
            if (diff) pos++;
        } while (diff);

        for (int k=0; k<count; k++){
            sort[k] = atoi( &dfiles->slice[k].name[pos] );
        }
    }
    else { printf("\n ERROR! Data Files Must Be Text Images \n"); return 0; }

    sort_files( dfiles, sort, count);

return SUCCESS;
}

/*
load_data opens the sorted files and loads the pixel data into the dataVol->array3D float array
*/
int load_data( char *path, FILE_LIST *dfiles, DATA_VOLUME *dataVol, char *filetype, int downSamp, bool zdown )
{
    int slices = dataVol->params.arraySize.z;
    if (zdown) slices/=downSamp;

    dataVol->array3D = (float *)malloc( (dataVol->params.arraySize.x / downSamp)*(dataVol->params.arraySize.y / downSamp)*slices*sizeof(float) );
    memset(dataVol->array3D,0,(dataVol->params.arraySize.x / downSamp)*(dataVol->params.arraySize.y / downSamp)*slices*sizeof(float));
    //printf("\n Loading data..."); fflush(stdout);
    int f = strncmp(filetype,"txt",3);
    bool ftrue = false;
    if (f == 0)
    {
        ftrue = true;
    }
    //printf("%s",ftrue ? "true" : "false");

    if ( ftrue )
    {
        //printf("\n Filetype is text");
        dataVol->params.max = -1e5;
        dataVol->params.min = 1e5;
        for (int k=0; k<dataVol->params.arraySize.z; k++)
        {
            int index = dataVol->params.arraySize.z - 1 - k;
            import_text_data( path, &dfiles->slice[index], dataVol, index, downSamp );
        }

        printf("\n Data Dimensions: %d x %d x %d \n Voxel Dimensions: %2.3f x %2.3f x %2.3f \n",
                (dataVol->params.arraySize.x / downSamp),(dataVol->params.arraySize.y / downSamp),slices,(dataVol->params.voxelSize.x * downSamp),(dataVol->params.voxelSize.y * downSamp),dataVol->params.voxelSize.z);
    }

    printf("\n Data Min: %4.1f\n Data Max: %4.1f\n\n",dataVol->params.min,dataVol->params.max); fflush(stdout);

return 1;
}


/*
save_data allows the float array newData to be saved as a series of DICOM files in the directory indicated by outpath.
newData is assumed to have the same parameters as the dataVol DATA_VOLUME.
*/
int save_data( const char *outpath, const char *inpath, FILE_LIST *dfiles, DATA_VOLUME *dataVol, float *newData, char *filetype )
{
    //char outpath[512];
    //sprintf(outpath,"%s/%s/testDICOM",CONVO_OUTPUT,patient);
    printf("\n%s",outpath);
    if (0 == mkdir((const char *)outpath,S_IRWXU|S_IRWXG|S_IRWXO) )
    {
        printf(" ...directory created successfully."); fflush(stdout);
    }
    else
    {
        printf("\n SourceCT already exists.\n"); fflush(stdout);
    }

    // for (int k=0; k<dataVol->params.arraySize.z; k++)
    //     export_dicom_data( outpath, inpath, &dfiles->slice[k], dataVol, newData, filetype, k );

    printf("\n DICOM data created successfully.\n\n");

return SUCCESS;
}




/* Debugging functions to write out array contents for closer inspection
*/
void LoadImageAsText(float *data, uint3 dim, const char *path)
{
    FILE *fp;
    char name[255];
    printf("\n loading data.....\n"); fflush(stdout);
    memset(data, 0.0, dim.x * dim.y * dim.z * sizeof(float) );
    int position = 0;
    for (uint k = 0; k < dim.z; k++)
    {
        //printf(" loading frame %sFile%04d.txt \n", path, k);
        sprintf(name, "%sFile%04d.txt", path, k);
        fp = fopen(name, "r");
        if (fp != NULL)
        {
            for (uint j = 0; j < dim.y; j++)
                for (uint i = 0; i < dim.x; i++)
                {
                    float temp;
                    fscanf(fp,"%f", &temp);
                    data[position] = temp;
                    position = position + 1;
                }
            fclose(fp);
            //printf("%s is loaded\n", name);
        }
        else
        {
            printf(" LOAD DATA ERROR \n");
        }
    }
}
void writeToTextFile(const char *path, int3 size, float *data)
{
    FILE *fp;
    char name[200];
    char bmpname[200];
    int position = 0;
    //printf ( " fileDimensions %d %d %d \n", Data->count.x, Data->count.y, Data->count.z);
    for (int k = 0; k < size.z; k++)
    {
        sprintf(name, "%sFile%04d.txt", path, k);
        sprintf(bmpname, "%sFile%04d.bmp", path, k);
        fp = fopen(name, "w");
        if (fp == NULL)
        {
            printf("Could not save flow to \"%s\"\n", name);
            return;
        }

        for (int i = 0; i < size.y; ++i)
        {
            for (int j = 0; j < size.x; ++j)
            {
                fprintf(fp, "%f ", data[position]);
                position = position + 1;
            }
            fprintf(fp, "\n");

        }
        fclose(fp);
        }
    //printf(" data files saved.\n");
}
