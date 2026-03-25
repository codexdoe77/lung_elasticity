#define HAVE_CONFIG_H

#include "dcmtk/config/osconfig.h"
#include "dcmtk/dcmdata/dctk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "read_dicom_data.h"


/*
import_structure_number loads the DICOMRT file and finds the total number of contours, allowing the array of structures to be dynamically allocated
*/
int
import_structure_number( char *path, char *file, DICOM_STRUCT *structures )
{
    DcmFileFormat format_number;
    char filename[1024];
    sprintf(filename,"%s%s",path,file);
    printf("%s%s\n",path,file);
    fflush(stdout);

    OFCondition status = format_number.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s\n", filename ); return 0; }

    DcmDataset *dataset = format_number.getDataset();

    DcmSequenceOfItems *ROIsequence = NULL;
    dataset->findAndGetSequence(DCM_StructureSetROISequence, ROIsequence).good();

    structures->CTRnumber = (int)ROIsequence->card();
    printf(" Number of Contours: %d\n",structures->CTRnumber); fflush(stdout);

    return SUCCESS;
}

/*
import_structure_info loads the DICOMRT file and finds the contour id, contour name, and rgb color associated with the contour.
this information is displayed through the standard output for the user to select specific contours to load
*/
int
import_structure_info( char *path, char *file, DICOM_STRUCT *structures )
{
    DcmFileFormat format_info;
    char filename[1024];
    sprintf(filename,"%s%s",path,file);

    OFCondition status = format_info.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s\n", filename ); return 0; }

    DcmDataset *dataset = format_info.getDataset();

    DcmSequenceOfItems *ROIsequence = NULL;
    dataset->findAndGetSequence(DCM_StructureSetROISequence, ROIsequence).good();

    DcmSequenceOfItems *ContourSequence = NULL;
    dataset->findAndGetSequence(DCM_ROIContourSequence, ContourSequence).good();

    Sint32 roi[50];
    for (int s = 0; s < structures->CTRnumber; s++)
        {
            DcmItem *item1 = ROIsequence->getItem(s);

            item1->findAndGetSint32(DCM_ROINumber, roi[s]).good();
            structures->contour[s].ROInumber = (int)roi[s] - 1;

            OFString strName;
            item1->findAndGetOFString(DCM_ROIName, strName).good();
            strcpy( structures->contour[s].ROIname , strName.data() );

            DcmItem *item2 = ContourSequence->getItem(s);

            OFString strColor;
            item2->findAndGetOFStringArray(DCM_ROIDisplayColor, strColor).good();
            printf("color: "); fflush(stdout);

            sscanf(strColor.c_str(), "%d\\%d\\%d", &structures->contour[roi[s]-1].rgb.x, &structures->contour[roi[s]-1].rgb.y, &structures->contour[roi[s]-1].rgb.z);
            //printf("(%d,%d,%d)...",structures->contour[roi[s]-1].rgb.x,structures->contour[roi[s]-1].rgb.y,structures->contour[roi[s]-1].rgb.z); fflush(stdout);
        }
    return SUCCESS;
}

/*
import_structure_data loads the DICOMRT file and for the user selected contours, it opens the embedded sequences
to find the number of sub-contours, and the list of points that define each sub-contour.
*/
int
import_structure_data( char *path, char *file, DICOM_STRUCT *structures, int s )
{
    DcmFileFormat format_data;
    char filename[1024];
    sprintf(filename,"%s%s",path,file);

    OFCondition status = format_data.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s\n", filename ); return 0; }

    DcmDataset *dataset = format_data.getDataset();

    DcmSequenceOfItems *ContourSequence = NULL;
    dataset->findAndGetSequence(DCM_ROIContourSequence, ContourSequence).good();

    Sint32 roi;
    DcmItem *item = ContourSequence->getItem(s);

    printf("\n %d. Loading %s...\n\n",s,structures->contour[s].ROIname);
    fflush(stdout);

    DcmSequenceOfItems *sequence = NULL;
    item->findAndGetSequence(DCM_ContourSequence, sequence).good();

    roi = (Sint32)structures->contour[s].ROInumber;
    printf("ROI %d...",roi);
    fflush(stdout);

    structures->contour[s].subCntrs = sequence->card();
    printf("%d sub-parts,",structures->contour[s].subCntrs);
    fflush(stdout);

    structures->contour[s].TOTpoints = 0;
    structures->contour[s].CTRpoints = (int*)malloc(sequence->card() * sizeof(int) );

    for (int c = 0; c < (int) sequence->card(); c++)
    {
        DcmItem *sItem = sequence->getItem(c);

        Sint32 ContourPoints;
        sItem->findAndGetSint32(DCM_NumberOfContourPoints, ContourPoints).good();
        structures->contour[s].CTRpoints[c] = (int) ContourPoints;
        structures->contour[s].TOTpoints += (int) ContourPoints;
    }
    structures->contour[s].matrix = (float3*)malloc( structures->contour[s].TOTpoints * sizeof(float3) );
    memset(structures->contour[s].matrix,0.0f,structures->contour[s].TOTpoints * sizeof(float3));

    printf(" %d data points...",(int)structures->contour[s].TOTpoints);
    fflush(stdout);

    int idx = 0;
    for (int c = 0; c < (int) sequence->card(); c++)
    {
        DcmItem *sItem = sequence->getItem(c);

        Sint32 ContourPoints;
        sItem->findAndGetSint32(DCM_NumberOfContourPoints, ContourPoints).good();

        Float64 v = 0.0;
        unsigned long k = 0;
        for (int p = 0; p < ContourPoints; p++)
        {
            sItem->findAndGetFloat64(DCM_ContourData, v, k++).good();
            structures->contour[s].matrix[idx].x = v;

            sItem->findAndGetFloat64(DCM_ContourData, v, k++).good();
            structures->contour[s].matrix[idx].y = v;

            sItem->findAndGetFloat64(DCM_ContourData, v, k++).good();
            structures->contour[s].matrix[idx].z = v;

            idx++;
        }
    }

    printf("loaded."); fflush(stdout);

 return 1;
}

/*
import_dicom_parameters loads the first file from the DICOM directory, and extracts all the parameters of the 3D dataset
include the acquisition date, the image size in voxels, the voxel size in mm, and point of reference in the scanner coordinate system
*/
int
import_dicom_parameters( char *path, FILE_NAME *file, DATA_VOLUME *dataVol, char *date )
{
    DcmFileFormat format;
    char filename[MAX_CHAR_LENGTH];
    sprintf(filename,"%s/%s",path,file->name);

    OFCondition status = format.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s\n", filename ); return 0; }

    Uint16 width,height;
    format.getDataset()->findAndGetUint16(DCM_Columns,width).good();
    format.getDataset()->findAndGetUint16(DCM_Rows,height).good();
    dataVol->params.arraySize.x = width;
    dataVol->params.arraySize.y = height;

    Float64 v;
    format.getDataset()->findAndGetFloat64(DCM_PixelSpacing,v,0).good();
        dataVol->params.voxelSize.x = v;
    format.getDataset()->findAndGetFloat64(DCM_PixelSpacing,v,0).good();
        dataVol->params.voxelSize.y = v;
    format.getDataset()->findAndGetFloat64(DCM_SliceThickness,v).good();
        dataVol->params.voxelSize.z = v;

    format.getDataset()->findAndGetFloat64(DCM_ImagePositionPatient, v, 0).good();
        dataVol->params.refPoint.x = v;
    format.getDataset()->findAndGetFloat64(DCM_ImagePositionPatient, v, 1).good();
        dataVol->params.refPoint.y = v;
    format.getDataset()->findAndGetFloat64(DCM_ImagePositionPatient, v, 2).good();
        dataVol->params.refPoint.z = v;

    OFString dicom_date;
    format.getDataset()->findAndGetOFString(DCM_AcquisitionDate,dicom_date).good();
    strcpy(date,dicom_date.data());

    return 1;
}

/*
import_dicom_index is launched in a loop, once for each file in the DICOM directory. It loads each file and extracts the instance number
if each slice and writes this value into an int array, so that the files can be properly sorted.
*/
int
import_dicom_index( char *path, FILE_NAME *file, int *slice )
{

    DcmFileFormat format;
    char filename[MAX_CHAR_LENGTH];
    sprintf(filename,"%s/%s",path,file->name);


    OFCondition status = format.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s\n", filename ); return 0; }

    const char *instance = NULL;
    format.getDataset()->findAndGetString(DCM_InstanceNumber,instance).good();
    *slice = atoi( instance );
    return SUCCESS;
}

/*
import_dicom_data loads each file in the DICOM directory and reads the pixel data. It will also apply a down-sampling
in the x and y directions if one is specified. It converts the unsigned int data to HU units by subtracting 1024, and converts these values to floats
in the array3D within the dataVol struct.
*/
int
import_dicom_data( char *path, FILE_NAME *file, DATA_VOLUME *dataVol, int slice, int downSamp )
{
    DcmFileFormat format;
    char filename[MAX_CHAR_LENGTH];
    sprintf(filename,"%s/%s",path,file->name);

    OFCondition status = format.loadFile( filename );
    if (status.bad())
        { printf("\n Error reading DICOM file:\n\t%s", filename ); return 0; }

    const Uint16 *pixelData;
    unsigned long *count = NULL;
    format.getDataset()->findAndGetUint16Array(DCM_PixelData,pixelData,count,0).good();

    for (int j=0; j<(dataVol->params.arraySize.y / downSamp); j++)
        for (int i=0; i<(dataVol->params.arraySize.x / downSamp); i++)
        {
            float temp = 0.f;
            for (int n=j*downSamp; n<(j+1)*downSamp; n++)
                for (int m=i*downSamp; m<(i+1)*downSamp; m++)
                {
                    temp += (float)pixelData[m + n*dataVol->params.arraySize.x] - 1024.0f;
                }
            temp /= (float)(downSamp * downSamp);
            dataVol->array3D[i + (dataVol->params.arraySize.x / downSamp)*(j + slice*(dataVol->params.arraySize.y / downSamp))] += temp;
            if (temp < dataVol->params.min) { dataVol->params.min = temp; }
            if (temp > dataVol->params.max) { dataVol->params.max = temp; }
        }

    return SUCCESS;
}


/*
anonymize ensures no patient data is included when exporting DICOM data in the export_dicom_data function
*/
void
anonymize( DcmDataset *dataset )
{
    dataset->findAndDeleteElement(DCM_AccessionNumber, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianAddress, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianTelephoneNumbers, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianIdentificationSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PhysiciansOfRecord, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionAddress, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_IssuerOfPatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_TypeOfPatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_IssuerOfPatientIDQualifiersSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthDate, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthTime, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSex, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientInsurancePlanCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientPrimaryLanguageCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientPrimaryLanguageModifierCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_OtherPatientIDs, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_OtherPatientNames, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_OtherPatientIDsSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientAge, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSize, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSizeCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientWeight, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientAddress, OFTrue, OFTrue);
    //dataset->findAndDeleteElement(DCM_ACR_NEMA_InsurancePlanIdentification, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientMotherBirthName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_MilitaryRank, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_BranchOfService, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_MedicalRecordLocator, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_MedicalAlerts, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_Allergies, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_CountryOfResidence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_RegionOfResidence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientTelephoneNumbers, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_EthnicGroup, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_Occupation, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_SmokingStatus, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_AdditionalPatientHistory, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PregnancyStatus, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_LastMenstrualDate, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientReligiousPreference, OFTrue, OFTrue);
}

/*
export_dicom_data is launched once for each slice to written. It creates a DICOM file in the indicated output path, load a template header from the incidated input path,
applies an anonymization to the header, writes the 3D volume parameters from the dataVol.params structure to the header file, and writes the pixel data from the newData source.
The file is then saved to the DICOM directory.
*/
int
export_dicom_data( const char *outpath, const char *inpath, FILE_NAME *file, DATA_VOLUME *dataVol, float *newData, char *filetype, int slice )
{
    DcmFileFormat format;
    OFCondition status;
    if( strcmp(filetype,"dcm") == 0 )
    {
        char infilename[MAX_CHAR_LENGTH];
        sprintf(infilename,"%s/%s",inpath,file->name);
        status = format.loadFile( infilename );
    }

    DcmDataset *dataset = format.getDataset();
    anonymize( dataset );

    Uint16 width,height;
    width = dataVol->params.arraySize.x;
    height = dataVol->params.arraySize.y;
    dataset->putAndInsertUint16(DCM_Columns,width).good();
    dataset->putAndInsertUint16(DCM_Rows,height).good();
    //printf("  Data size written...\n");
    //fflush(stdout);

    Float64 vz;
    vz = dataVol->params.voxelSize.z;
    dataset->putAndInsertFloat64(DCM_SliceThickness,vz).good();

    OFString vxy;
    char pixelSpacing[32];
    sprintf(pixelSpacing,"%3.3f\\%3.3f",dataVol->params.voxelSize.x,dataVol->params.voxelSize.y);
    vxy.assign( (const char*)pixelSpacing );
    dataset->putAndInsertOFStringArray(DCM_PixelSpacing,vxy,OFTrue).good();
    //printf("  Voxel size written...\n");
    //fflush(stdout);

    char instance[64];
    sprintf(instance,"%d",slice+1);
    dataset->putAndInsertString(DCM_InstanceNumber,instance).good();
    //printf("  Slice instance written...\n");
    //fflush(stdout);

    time_t rawtime;
    struct tm * timeinfo;
    char buffer[64];
    time (&rawtime);
    timeinfo = localtime (&rawtime);
    strftime (buffer,32,"%G%m%d",timeinfo);
    dataset->putAndInsertString(DCM_AcquisitionDate,buffer).good();
    dataset->putAndInsertString(DCM_SeriesDate,buffer).good();
    //printf("  Creation date written...\n");
    //fflush(stdout);

    sprintf(buffer,"DeformedBreast");
    dataset->putAndInsertString(DCM_StudyDescription,buffer).good();
    dataset->putAndInsertString(DCM_SeriesDescription,buffer).good();

    Uint16 *pixelDataOut;
    pixelDataOut = (Uint16*)malloc(width*height*sizeof(Uint16));
    //printf("  Pixel data allocated...\n");
    //fflush(stdout);

    for (int j=0; j<dataVol->params.arraySize.y; j++)
        for (int i=0; i<dataVol->params.arraySize.x; i++){
            Uint16 temp = (uint)abs(floor(1024.5f + newData[i + j*dataVol->params.arraySize.x + slice*dataVol->params.arraySize.x*dataVol->params.arraySize.y]));
            pixelDataOut[i + j*dataVol->params.arraySize.x] = temp;
        }
    //printf("  Pixel data recorded...\n");
    //fflush(stdout);
    dataset->putAndInsertUint16Array(DCM_PixelData,pixelDataOut,width*height,OFTrue).good();
    //printf("  Pixel data written...\n");
    //fflush(stdout);

    char outfilename[512];
    sprintf(outfilename,"%s/File%04d.dcm",outpath,slice);
    status = format.saveFile( outfilename, EXS_LittleEndianExplicit );
    if (status.bad())
        printf("Error: cannot write DICOM file ( %s )",status.text() );

    return SUCCESS;
}

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


