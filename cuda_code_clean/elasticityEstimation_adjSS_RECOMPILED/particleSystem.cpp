/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "particleSystem.h"
#include "particleSystem.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <string>
#include <random>


#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

#ifndef E_F
#define E_F                 2.718281828459045
#endif


float3 initalComPos = make_float3(0.f,0.f,0.f);
char confile[255];

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

ParticleSystem::ParticleSystem(uint numParticles, uint staticParticles, uint dynamicParticles, uint3 dataSize,
                               uint3 gridSize, float3 voxelSize, bool bUseOpenGL, bool bUseInput, bool bUseContour,
                               int *segmentInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *structures, float airThresh, float boneThresh, float headLevel, bool geom, char *geom_file, bool m_bAnchor) :

    m_bInitialized(false),
    m_bUseInput(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_staticParticles(staticParticles),
    m_dynamicParticles(dynamicParticles),
    m_hCon(0),
    m_hRestLengths(0),
    m_hPos(0),
    m_hInitialPos(0),
    m_hSourcePos(0),
    m_hVel(0),
    m_hForceS(0),
    m_hColor(0),
    m_dCon(0),
    m_dRestLengths(0),
    m_dVel(0),
    m_dForceS(0),
    m_dStretchS(0),
    m_dInitPos(0),
    m_dFinalPos(0),
    m_dStaticTrue(0),
    m_dForceT(0),
    m_dColor(0),
    m_dCntrScale(0),
    m_dataSize(dataSize),
    m_gridSize(gridSize),
    m_voxelSize(voxelSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_bUseInput = bUseInput;
    m_bUseContour = bUseContour;
    if (staticParticles > 0) m_bStatic = true;
    else m_bStatic = false;
    if (dynamicParticles > 0) m_bDynamic = true;
    else m_bDynamic = false;
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    m_numContours = 0;
    m_convergeCounter = 0;
    if (m_bUseContour)
        m_numContours = structures->CTRnumber;
    m_worldSize = make_float3( (float)m_gridSize.x*m_voxelSize.x, (float)m_gridSize.y*m_voxelSize.y, (float)m_gridSize.z*m_voxelSize.z);
    printf("\nWorld Size (m): ( %3.3f, %3.3f, %3.3f )",m_worldSize.x,m_worldSize.y,m_worldSize.z);

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.voxelSize = m_voxelSize;
    m_params.worldSize = m_worldSize;
    m_params.numCells = m_numGridCells;

    //set particle radius and rest spacing between particles
    if(bUseInput)
    {
        m_params.particleRadius = 0.5f * m_voxelSize.x;
        m_params.particleSpacing = sqrtf(m_voxelSize.x*m_voxelSize.x + m_voxelSize.y*m_voxelSize.y + m_voxelSize.z*m_voxelSize.z) / 1.7f;
    }
    else
    {
        float mr = std::max( std::max( m_voxelSize.x, m_voxelSize.y), m_voxelSize.z);
        m_params.particleRadius = 0.5f * mr;
        m_params.particleSpacing = m_params.particleRadius*2.f;
    }
    m_params.particleMass = 1000.f * m_params.voxelSize.x * m_params.voxelSize.y * m_params.voxelSize.z; //assumes water equivalent density: 1000 kg/m^3

    m_params.energyModulus     = 0.0f;

    m_params.MINmod             = MIN_MOD;
    m_params.MAXmod             = MAX_MOD;
    m_params.elasticModulus     = ELASTIC_MOD;
    m_params.shearModulus       = SHEAR_MOD;

    m_params.minPotential       = 0.f;
    m_params.stretchiness.x     = 0.f;
    m_params.stretchiness.y     = 0.f;

    m_params.springStrength     = 0.2f;
    m_params.springDamping      = 0.2f;

    m_params.colorSwitch        = 0;
    m_params.contourSwitch      = 0;
    m_params.restLengthMult     = 1.f;

    m_params.comPos = make_float3( 0.0f, 0.0f, 0.0f);
    m_params.colliderVel = make_float3( 0.0f, 0.0f, 0.0f);
    m_params.colliderOuterRadius[0] = 0.1f * std::max( std::max( m_worldSize.x, m_worldSize.y), m_worldSize.z);
    m_params.colliderInnerRadius[0] = 0.01f * std::max( std::max( m_worldSize.x, m_worldSize.y), m_worldSize.z);
    m_params.colliderPos = make_float3(-.5f * m_worldSize.x - 2.f * m_params.colliderOuterRadius[0], 0.0f, 0.0f);
    m_params.colliderMass = 1000.f * 4.f * CUDART_PI_F * powf(m_params.colliderOuterRadius[0],3.f) / 3.f;

    printf("\nParticle -\n Radius: %3.4g m\n Spacing: %3.4f m\n Mass: %3.4g kg", m_params.particleRadius,m_params.particleSpacing,m_params.particleMass);
//   printf("\nCollider (%3.2g, %3.2g, %3.2g)-\n Radius: %3.2g m\n Mass: %3.2g kg",m_params.colliderPos.x,m_params.colliderPos.y,m_params.colliderPos.z,m_params.colliderOuterRadius,*m_params.colliderMass);

    m_params.worldOrigin = make_float3(-0.5f*m_worldSize.x, -0.5f*m_worldSize.y, -0.5f*m_worldSize.z);

    m_params.spring = 0.f;
    m_params.damping = 0.99f;
    m_params.shear = 0.f;
    m_params.attraction = 0.f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, 0.0f, 1.0f);
    m_params.globalDamping = 1.0f;

    m_params.headLevel = headLevel;
    m_params.headSwitch = 0;
    m_params.headAngle = make_float3( 0.f, 0.f, 0.f );

    m_params.simulationTime = 0.f;

    printf("\n initializing..."); fflush(stdout);
    _initialize(numParticles,staticParticles,dynamicParticles,segmentInput,inputDataset,structures,bUseInput, geom, geom_file,m_bAnchor);
}

ParticleSystem::~ParticleSystem()
{
    m_numParticles = 0;
    m_staticParticles = 0;
    m_dynamicParticles = 0;
}

void ParticleSystem::_initializeMemberVariablesMemory(uint numParticles, uint staticParticles, uint dynamicParticles)
{
    m_numParticles = numParticles;
    m_staticParticles = staticParticles;
    m_dynamicParticles = dynamicParticles;

  // allocate host storage
    m_hInput = new uint[m_numParticles*3];
    m_hLast = new float[m_numParticles*4];
    m_hPos = new float[m_numParticles*4];
    m_hInitialPos = new float[m_numParticles*4];
    m_hPreMagPos = new float[m_numParticles*4];
    m_hTempPos = new float[m_numParticles*4];
    m_hSourcePos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    m_hForceS = new float[m_numParticles*4];
    m_hRefColor = new float[m_numParticles*4];
    m_hColor = new float[m_numParticles*4];
    m_hSegment = new int[m_numParticles];
    m_hSorted = new uint[m_numParticles];

    condenseArray_3D = new float[m_numParticles];
    convergedParticles = new float[m_dynamicParticles];
    dispErr = new float[m_dynamicParticles];
    groundTruth = new float[m_numParticles*4];
    groundTruthPosition = new float[4*m_numParticles];
    groundTruthDisplacement = new float[m_numParticles*4];
    positionArray = new float[4*m_numParticles];
    sourceArray = new float[4*m_numParticles];

     //Arrays for FSA
    last_guess_elasticity = new float[m_numParticles];
    last_guess_error = new float[m_numParticles];
    step = new float[m_numParticles];
    searchDirection = new float[m_numParticles];

    disp = new float[m_numParticles*4];
    dispMag2 = new float[m_numParticles];
    UVW_data = new float[m_numParticles*4];
    groundTruthmag = new float[m_numParticles];
    initialRL = new float[4*m_dynamicParticles*NUM_SPRINGS];
    deformRL = new float[4*m_dynamicParticles*NUM_SPRINGS];
    m_hStaticTrue = new int[m_dynamicParticles];
    m_hSurface = new int[m_dynamicParticles];

    memset(m_hInput, 0, m_numParticles*3*sizeof(uint));
    memset(m_hLast, 0, m_numParticles*4*sizeof(float));
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hInitialPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hPreMagPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hTempPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hSourcePos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
    memset(m_hForceS, 0, m_numParticles*4*sizeof(float));
    memset(m_hRefColor, 0, m_numParticles*4*sizeof(float));
    memset(m_hColor, 0, m_numParticles*4*sizeof(float));
    memset(m_hSegment, 0, m_numParticles*sizeof(uint));
    memset(m_hSorted, 0, m_numParticles*sizeof(uint));

    memset(condenseArray_3D, 0, m_numParticles*sizeof(float));
    memset(convergedParticles, 0, m_dynamicParticles*sizeof(float));
    memset(dispErr, 0, m_dynamicParticles*sizeof(float));
    memset(groundTruth, 0, m_numParticles*4*sizeof(float));
    memset(groundTruthPosition, 0, 4*m_numParticles*sizeof(float));
    memset(groundTruthDisplacement, 0, m_numParticles*4*sizeof(float));
    memset(positionArray, 0, m_numParticles*4*sizeof(float));
    memset(sourceArray, 0, m_numParticles*4*sizeof(float));


    //Arrays for FSA
    memset(last_guess_elasticity, 0, m_numParticles*sizeof(float));
    memset(last_guess_error, 0, m_numParticles*sizeof(float));
    memset(step, 0, m_numParticles*sizeof(float));
    memset(searchDirection, 0, m_numParticles*sizeof(float));

    memset(disp, 0, m_numParticles*4*sizeof(float));
    memset(dispMag2, 0, m_numParticles*sizeof(float));
    memset(UVW_data, 0, m_numParticles*4*sizeof(float));
    memset(groundTruthmag, 0, m_numParticles*sizeof(float));
    memset(initialRL, 0, 4*m_dynamicParticles*NUM_SPRINGS*sizeof(float));
    memset(deformRL, 0, 4*m_dynamicParticles*NUM_SPRINGS*sizeof(float));
    memset(m_hStaticTrue, 0, m_dynamicParticles*sizeof(int));
    memset(m_hSurface, 0, m_dynamicParticles*sizeof(int));


    m_hCntrScale = new float[(1+m_numContours)];
    memset(m_hCntrScale,0,(1+m_numContours)*sizeof(float));
    m_hSourcePos_isSet = 0;
    if (m_bDynamic)
    {
        m_hElasticity = new float[m_dynamicParticles];
        convergenceElasticity = new float[m_dynamicParticles];
        m_hDispMag = new float [m_dynamicParticles];
        m_hMINmod = new float[m_dynamicParticles];
        m_hMAXmod = new float[m_dynamicParticles];
        m_hVolume = new float[m_dynamicParticles];

        memset(m_hVolume,0,m_dynamicParticles*sizeof(float));
    }

    size_t free, total;
    checkCudaErrors(cudaMemGetInfo(&free,&total));
    firstUpdateElasticity = 1;
    printf("\n Free Memory: %lu / %lu",free,total);
    fflush(stdout);

}
void ParticleSystem::_AllocateAndSetMemberVariables(uint numParticles, uint staticParticles, uint dynamicParticles)
{
// allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;
    unsigned int staticSize = sizeof(float) * 4 * m_staticParticles;
    unsigned int dynamicSize = sizeof(float) * 4 * m_dynamicParticles;

    if (m_bDynamic)
    {
        allocateArray((void **)&tempPos, dynamicSize);
        setArrayToZero(tempPos,dynamicSize);
        allocateArray((void **)&m_dPos, dynamicSize);
        setArrayToZero(m_dPos,dynamicSize);
        allocateArray((void**)&m_dColor, dynamicSize);
        setArrayToZero(m_dColor,dynamicSize);

        allocateArray((void **)&m_dVel, dynamicSize);
        allocateArray((void **)&m_dForceS, dynamicSize);
        allocateArray((void **)&m_dForceT, dynamicSize);

        setArrayToZero(m_dVel,dynamicSize);
        setArrayToZero(m_dForceS,dynamicSize);
        setArrayToZero(m_dForceT,dynamicSize);

        allocateArray((void **)&m_dSegment, m_dynamicParticles*sizeof(int));

        allocateArray((void **)&m_dElasticity, m_dynamicParticles*sizeof(float));
        allocateArray((void **)&m_dVolume, m_dynamicParticles*sizeof(float));
        allocateArray((void **)&m_dRefVolume, m_dynamicParticles*sizeof(float));
        allocateArray((void **)&m_dSurface, m_dynamicParticles*sizeof(int));
        allocateArray((void **)&m_dStaticTrue,m_dynamicParticles*sizeof(int));

        allocateArray((void **)&m_dCntrScale, (1+m_numContours)*sizeof(float));
        setArrayToZero(m_dCntrScale,(1+m_numContours)*sizeof(float));
    }
    if (m_bStatic)
    {
        allocateArray((void **)&m_dStatic, staticSize);
        allocateArray((void **)&m_dSegmentStatic, m_staticParticles*sizeof(int));
        setArrayToZero(m_dStatic,staticSize);
    }

    size_t free, total;
    checkCudaErrors(cudaMemGetInfo(&free,&total));
    printf("\n Free Memory: %lu / %lu",free,total);


}

void ParticleSystem::_initialize_m_hInput( int *segmentInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *structures, bool m_bAnchor)
{
    if (m_bUseInput)
    {
        uint p=0, n=0;
        int c_counter = 0;
        if (m_bUseContour)
        {
            for (int i=0; i<structures->cntr_count; i++)
            {
                for (uint z=0; z<m_dataSize.z; z++)
                for (uint y=0; y<m_dataSize.y; y++)
                for (uint x=0; x<m_dataSize.x; x++)
                {
                    int par = segmentInput[x+m_dataSize.x*(y+m_dataSize.y*z)];
                    if (par == structures->cntrSegData[i])
                    {
                        m_hSorted[n] = x + m_dataSize.x * (y + m_dataSize.y * z);
                        m_hSegment[n++] = par;
                        m_hInput[p++] = x;
                        m_hInput[p++] = y;
                        m_hInput[p++] = z;
                        c_counter++;
                    }
                }
                printf("\n COMPARING CNTR %d: %d Expected | %d Found",i,structures->cntrParticles[i],c_counter);
                if (c_counter != structures->cntrParticles[i])
                    structures->cntrParticles[i] = c_counter;
                c_counter = 0;
            }
        }
        else
        {
            for (uint z=0; z<m_dataSize.z; z++)
            for (uint y=0; y<m_dataSize.y; y++)
            for (uint x=0; x<m_dataSize.x; x++)
            {
                int par = segmentInput[x+m_dataSize.x*(y+m_dataSize.y*z)];
                if (par != 0)
                {
                    m_hSorted[n] = x + m_dataSize.x * (y + m_dataSize.y * z);
                    m_hSegment[n++] = par;
                    m_hInput[p++] = x;
                    m_hInput[p++] = y;
                    m_hInput[p++] = z;

                }
            }
        }

    }
    else
    {
        uint p = 0;
        for (uint i=0; i<m_numParticles; i++)
            m_hSegment[i] = 1;
        for (uint z=0; z<m_dataSize.z; z++)
        for (uint y=0; y<m_dataSize.y; y++)
        for (uint x=0; x<m_dataSize.x; x++)
            {
                    m_hInput[p++] = x;
                    m_hInput[p++] = y;
                    m_hInput[p++] = z;
            }
    }
    printf("\n Segment data assigned..."); fflush(stdout);

}

void ParticleSystem::loadFromGeomFile(char *geom_file, float *x,float *y,float *z,int *diameter,float *sphereElas)
{
  {
            char geoname[255];
            sprintf(geoname, "%s", geom_file);
            FILE *fg;
            fg = fopen(geoname, "r");
            printf(" opening filename %s \n", geoname); exit(0);

            if (fg != NULL)
            {

                fscanf(fg,"%f %f %f %d %f ",x,y,z,diameter,sphereElas);
                printf("\n Center: %4.3f, %4.3f, %4.3f  Diameter: %d   Sphere Elasticity: %f  \n", *x,*y,*z,*diameter,*sphereElas); fflush(stdout);
            }
            else
            {
                printf("\n LOAD ELASTICITY DISTRIBUTION FILE ERROR \n %s", geoname);fflush(stdout);
            }
            fclose(fg);
    }

}

void ParticleSystem::computeCentroid(float *cx, float *cy, float *cz)
{
    // printf("HERE CENTROID\n");
    cx = cy = cz = 0;
    for (uint i=0; i<m_dynamicParticles; i++)
        {

            int x = m_hInput[3*i];
            int y = m_hInput[3*i+1];
            int z = m_hInput[3*i+2];
            cx += x;
            cy += y;
            cz += z;

        }
        // printf("DYNAMIC PARTICLES %d\n", m_dynamicParticles);
        // printf("Cx %4.3f\n",cx );  fflush(stdout);
        // printf("Cy %4.3f\n",cy ); fflush(stdout);
        // printf("Cz %4.3f\n",cz ); fflush(stdout);
        *cx /= (int)m_dynamicParticles;
        *cy /= (int)m_dynamicParticles;
        *cz /= (int)m_dynamicParticles;

}

float ParticleSystem::computeDistance(float a, float b, float c, float d, float e, float f)
{
    return sqrt( pow(a-d,2) + pow(b-e,2) + pow(c-f,2));

}


void ParticleSystem::setElasticityAndAssociatedParameters(int i, float value, float minMod = MIN_MOD, float maxMod = MAX_MOD)
{

    m_hElasticity[i] = value;
    convergenceElasticity[i] = m_hElasticity[i];
    m_hMINmod[i] = minMod;
    m_hMAXmod[i] = maxMod;
    m_hElasticity[i] *= ELAS_SCALE;


}

void ParticleSystem::setCondenseArrayFromInputDataset(DATA_VOLUME *inputDataset)
{
    uint b = 0;
    for (uint n = 0; n < m_dataSize.x*m_dataSize.y*m_dataSize.z; n++)
    {
        if (inputDataset->array3D[n] != 0.f)
        {
            condenseArray_3D[b] = inputDataset->array3D[n];
            b++;
        }

    }

}
void ParticleSystem::_initializeSystematicElasticity(  bool geom, char *geom_file)
{

    if (m_bDynamic)
    {

        float3 cent = make_float3(0.f, 0.f, 0.f);
        int diameter;
        float sphereElas;

        //Loads elasticity distribution file if one is provided in command line with -geometry flag
        if (geom) loadFromGeomFile(geom_file, &cent.x, &cent.y, &cent.z, &diameter, &sphereElas);
        //Sets ground-truth elasticity values based upon DICOM CT values
;
        for (uint i=0; i<m_dynamicParticles; i++)
        {
            float value = 20.0*(condenseArray_3D[i] - (-1100))/500.0;
            setElasticityAndAssociatedParameters(i, value, MIN_MOD , MAX_MOD);
        }

        checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));

    }


}

void
ParticleSystem::_initialize(uint numParticles, uint staticParticles, uint dynamicParticles, int *segmentInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *structures, bool bUseInput,  bool geom, char *geom_file, bool m_bAnchor)
{
    assert(!m_bInitialized);
    _initializeMemberVariablesMemory(numParticles, staticParticles, dynamicParticles);
    _AllocateAndSetMemberVariables(numParticles, staticParticles, dynamicParticles);
    _initialize_m_hInput( segmentInput, inputDataset, structures,  m_bAnchor);
    setCondenseArrayFromInputDataset(inputDataset);
    _initializeSystematicElasticity(geom, geom_file);
    setParameters(&m_params);

    //Sets ground truth elasticity values
    sdkCreateTimer(&m_timer);
    size_t free, total;
    checkCudaErrors(cudaMemGetInfo(&free,&total));
    printf("\n Free Memory: %lu / %lu\n",free,total); fflush(stdout);

    printf("\n setting parameters...");
    m_bInitialized = true;
}

void ParticleSystem::finalFree()
{
    _finalize();
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hInput;
    delete [] m_hLast;
    delete [] m_hPos;
    delete [] m_hInitialPos;
    delete [] m_hTempPos;
    delete [] m_hSourcePos;
    delete [] m_hVel;
    delete [] m_hForceS;
    delete [] m_hRefColor;
    delete [] m_hColor;
    delete [] m_hSegment;
    delete [] m_hSorted;
    delete [] m_hCntrScale;
    delete [] groundTruth;
    delete [] groundTruthPosition;
    delete [] groundTruthDisplacement;
    delete [] sourceArray;
    delete [] positionArray;
    delete [] last_guess_elasticity;
    delete [] step;
    delete [] searchDirection;

    delete [] UVW_data;
    delete [] groundTruthmag;
    delete [] disp;
    delete [] dispMag2;
    delete [] condenseArray_3D;
    delete [] convergedParticles;
    delete [] dispErr;
    delete [] initialRL;
    delete [] deformRL;


    if (m_bDynamic)
    {
        delete [] m_hCon;
        delete [] m_hConCount;
        delete [] m_hConStart;
        delete [] m_hConStatic;
        delete [] m_hConCountStatic;
        delete [] m_hConStartStatic;
        delete [] m_hRestLengths;
        delete [] m_hRestLengthsStatic;
        //delete [] m_hStaticTrue;
        //delete [] m_hSurface;

        delete [] m_hMINmod;
        delete [] m_hMAXmod;
        delete [] m_hElasticity;
        delete [] convergenceElasticity;
        delete [] m_hDispMag;
        delete [] m_hVolume;
    }
    printf("\n Host memory freed."); fflush(stdout);

    if (m_bStatic)
    {
        freeArray(m_dStatic);
        freeArray(m_dSegmentStatic);
    }

    printf("\n Device memory freed.\n"); fflush(stdout);
}


void ParticleSystem::resetRestLength()
{
    if (m_bUseContour)
    {
        assert(m_bInitialized);
        memset(m_hCntrScale,0,(1+m_numContours)*sizeof(float));
        checkCudaErrors(cudaMemset(m_dCntrScale,0,(1+m_numContours)*sizeof(float)));

    }
}


/**
 * @brief Steps the simulation
 * @details Updates constants and connections in model in order to step through simulation.
 *
 * @param deltaTime float. Set in particles.cpp, used in integrateConnect and collide functions to update simulation.
 */
float ParticleSystem::update(float deltaTime, bool magnify, int iteration_counter)
{
    // std::cout <<" updating the particles system ---> "<<iteration_counter<<std::endl;
    float simTime = m_params.simulationTime;
    if (m_dynamicParticles > 0)
    {
        assert(m_bInitialized);

        // update constants
        setParameters(&m_params);
        m_hCntrScale[m_params.contourSwitch] = m_params.restLengthMult - 1.f;
        checkCudaErrors(cudaMemcpy(m_dCntrScale,m_hCntrScale,(1+m_numContours)*sizeof(float),cudaMemcpyHostToDevice));

        if (m_bStatic)
        {
          staticConnections(
            m_dPos,
            m_dVel,
            m_dForceS,
            m_dStatic,
            m_dElasticity,
            m_dConStatic,
            m_dConCountStatic,
            m_dConStartStatic,
            m_dRestLengthsStatic,
            m_dynamicParticles,
            m_staticParticles);
        }

        connections(
            tempPos,
            m_dPos,
            m_dVel,
            m_dForceS,
            m_dStretchS,
            m_dForceT,
            m_dElasticity,
            m_dCon,
            m_dConCount,
            m_dConStart,
            m_dRestLengths,
            m_dSegment,
            m_dCntrScale,
            m_dynamicParticles,
            m_dColor,
            m_dataSize,
            m_dInitPos,
            m_dFinalPos,
            m_dStaticTrue,
            m_bAnchor,
            m_dSurface);

        checkCudaErrors(cudaMemcpy(m_hForceS,m_dForceS,m_numParticles*4*sizeof(float),cudaMemcpyDeviceToHost));

        // integrate

        checkCudaErrors(cudaMemcpy(m_dPos,m_hPos,m_dynamicParticles*4*sizeof(float),cudaMemcpyHostToDevice));

        integrateConnect(
            m_dPos,
            m_dVel,
            m_dForceS,
            m_dForceT,
            deltaTime,
            m_dynamicParticles);

        // collide(
        //     m_dPos,
        //     m_dVel,
        //     m_dForceS,
        //     m_dSegment,
        //     m_dynamicParticles,
        //     deltaTime);

        checkCudaErrors(cudaMemcpy(m_hPos+4*m_staticParticles,m_dPos,4*m_dynamicParticles*sizeof(float),cudaMemcpyDeviceToHost));

        m_convergeCounter++;
    }

return simTime;


}
void ParticleSystem::resetElasticityNeighbors(DATA_VOLUME *inputDataset)
{
    checkCudaErrors(cudaMemcpy(m_hElasticity, m_dElasticity, m_dynamicParticles*sizeof(float), cudaMemcpyDeviceToHost));
    float *elasNeighbors = new float [m_dynamicParticles];
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        int startCount = m_hConStart[i];
        int conCount = m_hConCount[i];
        for (int j = 0; j < conCount; j++)
        {
            int index = m_hCon[startCount + j];

            if (m_hStaticTrue[index]) continue;

            elasNeighbors[i] +=  (1 - ALPHA) * m_hElasticity[index];
        }

        if (conCount < 1)
            elasNeighbors[i] = 0;
        else
            elasNeighbors[i] /= conCount;

        m_hElasticity[i] = ALPHA * m_hElasticity[i] + elasNeighbors[i];
    }
    checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));
}

void ParticleSystem::resetElasticityforMagnify( float modulus)
{
    for (uint i=0; i<m_dynamicParticles; i++)
    {
        m_hElasticity[i] = modulus;
        m_hElasticity[i] *= ELAS_SCALE;

    }
    checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));
}
/**
 * Updates elastic modulus.
 * Resets elastic modulus after each iteration.
 *
 * @param modulus float. Elastic modulus value to reset system to.
 * @param reset bool. If true, reset system.
 */
void ParticleSystem::resetElasticityforOptimization(bool total)
{
    if (m_bDynamic)
    {
        for (uint i=0; i<m_dynamicParticles; i++)
        {
            m_hElasticity[i] = convergenceElasticity[i];
            m_hElasticity[i] *= ELAS_SCALE;

            if (total)
            {
                m_hMAXmod[i] = MAX_MOD;
                m_hMINmod[i] = MIN_MOD;
            }

        }

        checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));
    }


}

/**
 * Updates rest lengths to current length of spring
 *
 */
void ParticleSystem::recordRestLengths(bool deform, bool init)
{
    uint dSprings=0;
    if (m_bUseInput)
    {

        if (!deform)
        {
            if (init)
            {
                for (uint i = 0; i < m_dynamicParticles; i++)
                {

                    int conCount = m_hConCount[i];
                    for (int j = 0; j < conCount; j++)
                    {

                        initialRL[4*dSprings+0] = m_hRestLengths[4*dSprings + 0];
                        initialRL[4*dSprings+1] = m_hRestLengths[4*dSprings + 1];
                        initialRL[4*dSprings+2] = m_hRestLengths[4*dSprings + 2];

                        dSprings++;
                    }
                }
            }
            else
            {

                for (uint i = 0; i < m_dynamicParticles; i++)
                {

                    int conCount = m_hConCount[i];
                    for (int j = 0; j < conCount; j++)
                    {
                        m_hRestLengths[4*dSprings+0] = initialRL[4*dSprings + 0];
                        m_hRestLengths[4*dSprings+1] = initialRL[4*dSprings + 1];
                        m_hRestLengths[4*dSprings+2] = initialRL[4*dSprings + 2];

                        dSprings++;
                    }
                }

            }
        }
        else
        {
            if (init)
            {
                for (uint i = 0; i < m_dynamicParticles; i++)
                {

                    int startCount = m_hConStart[i];
                    int conCount = m_hConCount[i];
                    for (int j = 0; j < conCount; j++)
                    {
                        int index = m_hCon[startCount + j];
                        deformRL[4*dSprings+0] = (m_hPos[4*index+0] - m_hPos[4*i+0]);
                        deformRL[4*dSprings+1] = (m_hPos[4*index+1] - m_hPos[4*i+1]);
                        deformRL[4*dSprings+2] = (m_hPos[4*index+2] - m_hPos[4*i+2]);

                        m_hRestLengths[4*dSprings+0] = deformRL[4*dSprings + 0];
                        m_hRestLengths[4*dSprings+1] = deformRL[4*dSprings + 1];
                        m_hRestLengths[4*dSprings+2] = deformRL[4*dSprings + 2];

                        dSprings++;
                    }
                }

            }
            else
            {

                for (uint i = 0; i < m_dynamicParticles; i++)
                {

                    int conCount = m_hConCount[i];
                    for (int j = 0; j < conCount; j++)
                    {
                        m_hRestLengths[4*dSprings+0] = deformRL[4*dSprings + 0];
                        m_hRestLengths[4*dSprings+1] = deformRL[4*dSprings + 1];
                        m_hRestLengths[4*dSprings+2] = deformRL[4*dSprings + 2];

                        dSprings++;
                    }
                }

            }

        }

    }

    checkCudaErrors(cudaMemcpy(m_dRestLengths,m_hRestLengths,dSprings*4*sizeof(float),cudaMemcpyHostToDevice));

}

void ParticleSystem::populateDataFromFile(DATA_VOLUME *inputDataset, char* filename)
{
    std::cout << "\n Getting data from: " << filename << std::endl;
    std::ifstream infile;
    infile.open(filename);


    if(infile.fail()) // checks to see if file opened
    {
      std::cout << "\n error with filename" <<filename << std::endl;
      exit(0);
    }
    unsigned int lineCount = 0;
    unsigned int rowIndex = 0;
    unsigned int maxreadindex = 1000000;
    while(!infile.eof() )
    {
        char buf[1000];
        float val[15];
        int tokenCount = 0;
        infile.getline(buf, 1000);
        if (lineCount >= 0)
        {
            char *token;
            token = strtok(buf,",");
            while(token != NULL)
            {
                val[tokenCount++] = atof(token);


                token = strtok(NULL,",");
            }

            m_hStaticTrue[rowIndex] = val[1];
            sourceArray[4*rowIndex + 0] = val[2];
            sourceArray[4*rowIndex + 1] = val[3];
            sourceArray[4*rowIndex + 2] = val[4];

            positionArray[4*rowIndex + 0] = val[5];
            positionArray[4*rowIndex + 1] = val[6];
            positionArray[4*rowIndex + 2] = val[7];

            groundTruthPosition[4*rowIndex + 0] = val[8];
            groundTruthPosition[4*rowIndex + 1] = val[9];
            groundTruthPosition[4*rowIndex + 2] = val[10];
        }

        lineCount++;
        rowIndex++;
    }
    infile.close();

    //Record UVW Data as differences between position and source array
    //Record ground truth as differences between x'' and x
    for (int i = 0; i < m_dynamicParticles*4; i++)
    {
        UVW_data[i] = sourceArray[i] - positionArray[i];
        groundTruth[i] = UVW_data[i];//positionArray[i] - groundTruthPosition[i];
    }
    //Sets ground truth displacement magnitude values
    float maxGroundTruthMag = -32767.0;
    for (int i = 0; i < m_dynamicParticles; i++)
    {
        groundTruthmag[i] = sqrt(groundTruth[4*i+0]*groundTruth[4*i+0] + groundTruth[4*i+1]*groundTruth[4*i+1] + groundTruth[4*i+2]*groundTruth[4*i+2]);
        groundTruthmag[i] *= DISP_SCALE;
        if (maxGroundTruthMag < groundTruthmag[i]) maxGroundTruthMag = groundTruthmag[i];
    }

    std::cout <<" Max Ground truth displacement is "<<maxGroundTruthMag<<std::endl;
    //Liver YM values
    // float max_elas = -32767;
    // float min_elas = 32767;
    // for (int i = 0; i < m_dynamicParticles; i++)
    // {
    //     float minMod = MIN_MOD;
    //     float maxMod = MAX_MOD;
    //     float value1 = (maxMod - minMod) * pow((/*(0.0 + maxGroundTruthMag) - */groundTruthmag[i])/maxGroundTruthMag,2);
    //     float value2 = (maxMod - minMod) * (condenseArray_3D[i])/329.0;
    //     float value = minMod + (value1 + value2)/2.f;
    //     setElasticityAndAssociatedParameters(i,value, minMod, maxMod);
    //     if (max_elas < value) max_elas = value;
    //     if (min_elas > value) min_elas = value;
    // }
    // Lung YM values
    float max_elas = -32767.0;
    float min_elas = 32767.0;
    for (int i = 0; i < m_dynamicParticles; i++)
    {
      float minMod = MIN_MOD;
      float maxMod = MAX_MOD;
      if (condenseArray_3D[i] <=-500 && condenseArray_3D[i] >= -600)
      {
        minMod = 5.0;
        maxMod = 12.0;
      }
      else
      {
        if (condenseArray_3D[i] <-600 && condenseArray_3D[i] >= -800)
        {
            minMod = 4.0;
            maxMod = 8.0;
        }
        else
        {
          if (condenseArray_3D[i] <= 0 && condenseArray_3D[i] >= -500)
          {
              minMod = 10.0;
              maxMod = 100.0;
          }
          else
          {
                minMod = 1.0;
                maxMod = 5.0;
          }
         }

        }
      float value1 = (maxMod - minMod) * pow((/*(0.0 + maxGroundTruthMag) - */groundTruthmag[i])/maxGroundTruthMag,2);
      float value2 = (maxMod - minMod) * (condenseArray_3D[i] - (-1100))/500.0;
      float value = minMod + (value1 + value1)/2.f;
      setElasticityAndAssociatedParameters(i,value, minMod, maxMod);
      if (max_elas < value) max_elas = value;
      if (min_elas > value) min_elas = value;
    }

    float *elasOut = new float[m_dynamicParticles];
    for (int i = 0; i < m_dynamicParticles; i++)
    {
        elasOut[i] = m_hElasticity[i];
        elasOut[i] *= 1/ELAS_SCALE;
    }
    dumpParticles(inputDataset, elasOut, "INITIALELAS");
    dumpParticles(inputDataset, groundTruthmag, "GROUNDTRUTHDISP");
    std::cout <<" max_elas is "<<max_elas<<" min_elas is "<<min_elas << std::endl;
    checkCudaErrors(cudaMemcpy(m_dStaticTrue,m_hStaticTrue,m_dynamicParticles*sizeof(int),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));

    printf("\n Data successfully populated\n");

}


/**
 * [ParticleSystem::inputStatic Anchors specific particles to be static]
 * @param static_in [char   File name input into command line to pull static/anchored particles from]
 * @param m_bAnchor [bool   Describes whether or not static array is input into command line]
 */
void ParticleSystem::inputStatic(const char *static_in, bool m_bAnchor)
{
    checkCudaErrors(cudaMemcpy(m_hPos+4*m_staticParticles,m_dPos,4*m_dynamicParticles*sizeof(float),cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hStaticTrue,m_dStaticTrue,m_dynamicParticles*sizeof(int),cudaMemcpyDeviceToHost));


    uint *InS;
    InS = new uint[m_dynamicParticles];

    std::ifstream infile;
    int counterS = 0;

    infile.open(static_in);// file containing numbers

     if(infile.fail()) // checks to see if file opened
    {
      std::cout << "\n error with Static" << std::endl;
    }

    while  (infile >> InS[counterS])
    {
        counterS++;
    }

    infile.close();

    printf("\n Done reading static file");
    for (uint i = 0; i < m_dynamicParticles; ++i)
    {
        m_hStaticTrue[i] = 0;
        for (uint j = 0; j < counterS; j++)
        {
            if (i == InS[j])
            {
                m_hStaticTrue[i] = 1;

            }

        }
    }

    checkCudaErrors(cudaMemcpy(m_dStaticTrue,m_hStaticTrue,m_dynamicParticles*sizeof(int),cudaMemcpyHostToDevice));
    delete [] InS;
    printf("\n Particles to be held static: %d",counterS);

}

float* ParticleSystem::recordDisplacement(DATA_VOLUME *inputDataset)
{
    float* displacement = new float[4*m_dynamicParticles];
    float* dispMag = new float[m_dynamicParticles];

    checkCudaErrors(cudaMemcpy(m_hPos+4*m_staticParticles,m_dPos,4*m_dynamicParticles*sizeof(float),cudaMemcpyDeviceToHost));
    //Record displacement of each particle
    int p=0;
    for (uint i=0; i<m_dynamicParticles; i++)
    {
        displacement[p] = (m_hPreMagPos[p] - sourceArray[p]);
        p++;
        displacement[p] = (m_hPreMagPos[p] - sourceArray[p]);
        p++;
        displacement[p] = (m_hPreMagPos[p] - sourceArray[p]);
        p++;
        p++;

    }
    //Record magnitude of displacement
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        dispMag[i] = sqrt(displacement[4*i+0]*displacement[4*i+0] + displacement[4*i+1]*displacement[4*i+1] + displacement[4*i+2]*displacement[4*i+2]);
        dispMag[i] *= DISP_SCALE; //Scales displacement to be mm
        m_hDispMag[i] = dispMag[i];
    }
    delete [] displacement;
    return dispMag;

}
float* ParticleSystem::recordDisplacementError(DATA_VOLUME *inputDataset)
{

    float* dispMag = recordDisplacement(inputDataset);
    float* dispNorm = new float[m_dynamicParticles];
    float* error = new float[m_dynamicParticles];
    //Record error
    float maxDispMag = -32767.0;
    float maxError = -32767.0;
    int p = 0;
    printf(" inside recordDisplacementError \n"); fflush(stdout);
    float temp[3];
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
     //   for (int j = 0; j < 4; j++)
        {
            temp[0] = (groundTruthPosition[p] - m_hPos[p]);     p++;
            temp[1] = (groundTruthPosition[p] - m_hPos[p]);     p++;
            temp[2] = (groundTruthPosition[p] - m_hPos[p]);     p++;
            p++;
        }
        dispErr[i] = sqrt(pow(temp[0],2)+ pow(temp[1],2)+ pow(temp[2],2));

         if (maxDispMag < dispMag[i])
            maxDispMag = dispMag[i];

        if (maxError < fabs(dispErr[i]))
            maxError = fabs(dispErr[i]);

        if (dispMag[i] > groundTruthmag[i])
            dispErr[i] *= -1;

        error[i] = dispErr[i];
        if (error[i] != error[i])
            error[i] = 0;
       // std::cout <<" i "<<i<<" error "<<error[i]<<std::endl;
        /*if (error[i] > 2.0f)
        {
            std::cout <<"\t\t "<<groundTruthPosition[p-3]<<" "<<groundTruthPosition[p-2]<<" "<<groundTruthPosition[p-1]<<std::endl;
            std::cout <<"\t\t "<<m_hPos[p-3]<<" "<<m_hPos[p-2]<<" "<<m_hPos[p-1]<<std::endl;
            std::cout <<" p "<<p<<" m_dynamicParticles "<<m_dynamicParticles<<std::endl;
            { int x; std::cin>>x; }
        }*/

    }
/*    for (uint i = 0; i < m_dynamicParticles; ++i)
    {
        dispErr[i] = (dispMag[i] - groundTruthmag[i]);
        dispNorm[i] = dispErr[i] /  groundTruthmag[i];

        if (groundTruthmag[i] < 0.00001f)
            dispNorm[i] = 0.f;

        if (maxDispMag < dispMag[i])
            maxDispMag = dispMag[i];

        if (maxError < fabs(dispErr[i]))
            maxError = fabs(dispErr[i]);

        error[i]= fabs(dispErr[i]);
    }
*/
    std::cout <<" maxDispMag is -----> "<<maxDispMag<<std::endl;
    std::cout <<" maxError is -----> "<<maxError<<std::endl;
    delete [] dispNorm;
    delete [] dispMag;
 //   dumpParticles(inputDataset, dispErr, "DISPERROR");
 //   dumpResults(inputDataset); exit(0);

    return error;
}
float *ParticleSystem::recordNeighborError(DATA_VOLUME *inputDataset)
{
    float *error = recordDisplacementError(inputDataset);
    float *dispErrNeighbor = new float[m_dynamicParticles];
    int nanCount = 0;
    memset(dispErrNeighbor, 0, sizeof(float)*m_dynamicParticles);

    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        // int startCount = m_hConStart[i];
        // int conCount = m_hConCount[i];
        // for (int j = 0; j < conCount; j++)
        // {
        //     int index = m_hCon[startCount + j];

        //     if (m_hStaticTrue[index]) continue;

        //     dispErrNeighbor[i] +=  (1 - ALPHA) * error[index];
        // }

        // if (conCount < 1)
        //     dispErrNeighbor[i] = 0;
        // else
        //     dispErrNeighbor[i] /= conCount;

        // if (dispErrNeighbor[i] != dispErrNeighbor[i])
        // {
        //     printf("\nNaN error... Manually setting error to 0.\n");
        //     nanCount++;
        //     dispErrNeighbor[i] = 0;
        // }

        // // error[i] = ALPHA * error[i] + dispErrNeighbor[i];
        // dispErrNeighbor[i] +=  ALPHA * error[i];

       //Without neighbor error
        dispErrNeighbor[i] = error[i];
    }
    if (nanCount != 0)
    {
        printf("Number of Voxels with NaN error: %d\n",nanCount );
    }
    delete [] error;

    return dispErrNeighbor;

}
/**
 * Analyzes deformation of particle system, especially current displacement and Jacobian values.
 * Using current guess YM to get displacement and Jacobian values. Compares to ground truth values to get error, which
 * is then used to update YM value
 *
 * @param iteration int. Current iteration in loop.
 * @param tolerance float. Individual particle tolerance.
 * @param array pointer float. Contains HU values.
 * @param m_banchor bool. True if particle to be anchored as called in static file, if static file is input.
 *
 * @return averages float4. Contains average error and average YM and % particles converged for the system to be printed in terminal.
 */

float4 ParticleSystem::updateElasticity(int iteration, float tolerance, float *array_3D, bool m_bAnchor, DATA_VOLUME *inputDataset, float &mmPercent)
{
    float4 averages = make_float4( 0.f, 0.f, 0.f, 0.f );
    checkCudaErrors(cudaMemcpy(m_hElasticity, m_dElasticity, m_dynamicParticles*sizeof(float), cudaMemcpyDeviceToHost));

    // float* error = recordDisplacementError(inputDataset);
    float* error = recordNeighborError(inputDataset);
    // dumpParticles(inputDataset,error,"ERROR");
    float avBefore = 0.0f;
    float maxErr = 0.0f;
    float c1 =0.3f;
    int r = 0;
    int mm = 0;
    //Record Total and Maximum Error
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        averages.x += error[i];
        if (fabs(error[i]) > maxErr)
            maxErr = fabs(error[i]);
    }
    //Update Elasticity
    if (iteration == 2) {

    for (uint i = 0; i < m_dynamicParticles; i++)
        {
            m_hElasticity[i] *= (1/ELAS_SCALE); //scales elasticity back to kPa after being copied from GPU

            avBefore += m_hElasticity[i];

            step[i] = fabs(error[i]) / maxErr;

            if (fabs(error[i]) > PARTICLE_TOL)
            {
            if (firstUpdateElasticity)
            {

                 last_guess_elasticity[i] = m_hElasticity[i];
                 last_guess_error[i] = error[i];

                if (error[i] > 0)
                    searchDirection[i] = m_hMAXmod[i] - m_hElasticity[i];
                if (error[i] < 0)
                    searchDirection[i] = m_hMINmod[i] - m_hElasticity[i];

                m_hElasticity[i] = m_hElasticity[i] + step[i] * searchDirection[i] * c1;
                if (m_hElasticity[i] > m_hMAXmod[i])
                    m_hElasticity[i] = m_hMAXmod[i];
                if (m_hElasticity[i] < m_hMINmod[i])
                    m_hElasticity[i] = m_hMINmod[i];

                //firstUpdateElasticity = 0;

            }
            else
            {
                float X = m_hElasticity[i] - last_guess_elasticity[i];
                float Y = fabs(error[i]) - fabs(last_guess_error[i]);

                last_guess_elasticity[i] = m_hElasticity[i];
                last_guess_error[i] = error[i];

                if (fabs(X) < 0.00001) X = 1;

                if ((Y > 0.00001 && X > 0.00001) || (Y < -0.00001 && X < -0.00001))
                    searchDirection[i] = m_hMINmod[i] - m_hElasticity[i];
                else
                    searchDirection[i] = m_hMAXmod[i] - m_hElasticity[i];

                m_hElasticity[i] = m_hElasticity[i] + step[i] * searchDirection[i] * c1;
                if (m_hElasticity[i] > m_hMAXmod[i])
                    m_hElasticity[i] = m_hMAXmod[i];
                if (m_hElasticity[i] < m_hMINmod[i])
                    m_hElasticity[i] = m_hMINmod[i];




            }
            }

            if (fabs(error[i]) < PARTICLE_TOL)
            {
                convergedParticles[i] = 1;
                r++;
            }
            if (fabs(error[i]) < MM_TOL)
            {
                mm++;
            }

        }

    }

    else {

        for (uint i = 0; i < m_dynamicParticles; i++)
        {
            m_hElasticity[i] *= (1/ELAS_SCALE); //scales elasticity back to kPa after being copied from GPU

            avBefore += m_hElasticity[i];

            step[i] = fabs(error[i]) / maxErr;

            if (fabs(error[i]) > PARTICLE_TOL)
            {
            if (firstUpdateElasticity)
            {

                 last_guess_elasticity[i] = m_hElasticity[i];
                 last_guess_error[i] = error[i];

                if (error[i] > 0)
                    searchDirection[i] = m_hMAXmod[i] - m_hElasticity[i];
                if (error[i] < 0)
                    searchDirection[i] = m_hMINmod[i] - m_hElasticity[i];

                m_hElasticity[i] = m_hElasticity[i] + step[i] * searchDirection[i] * c1;
                if (m_hElasticity[i] > m_hMAXmod[i])
                    m_hElasticity[i] = m_hMAXmod[i];
                if (m_hElasticity[i] < m_hMINmod[i])
                    m_hElasticity[i] = m_hMINmod[i];

                //firstUpdateElasticity = 0;

            }
            else
            {
                float X = m_hElasticity[i] - last_guess_elasticity[i];
                float Y = fabs(error[i]) - fabs(last_guess_error[i]);

                last_guess_elasticity[i] = m_hElasticity[i];
                last_guess_error[i] = error[i];

                if (fabs(X) < 0.00001) X = 1;

                if ((Y > 0.00001 && X > 0.00001) || (Y < -0.00001 && X < -0.00001))
                    searchDirection[i] = m_hMINmod[i] - m_hElasticity[i];
                else
                    searchDirection[i] = m_hMAXmod[i] - m_hElasticity[i];

                m_hElasticity[i] = m_hElasticity[i] + step[i] * searchDirection[i] * c1;
                if (m_hElasticity[i] > m_hMAXmod[i])
                    m_hElasticity[i] = m_hMAXmod[i];
                if (m_hElasticity[i] < m_hMINmod[i])
                    m_hElasticity[i] = m_hMINmod[i];




            }
            }

            if (fabs(error[i]) < PARTICLE_TOL)
            {
                convergedParticles[i] = 1;
                r++;
            }
            if (fabs(error[i]) < MM_TOL)
            {
                mm++;
            }




        }
    }
   // if (firstUpdateElasticity) firstUpdateElasticity = 0;
    averages.z = r;
    mmPercent = mm;

    for (uint i=0; i<m_dynamicParticles; i++)
    {
        //addGauss[i] =  distribution(generator);
        m_hElasticity[i] = m_hElasticity[i]; //+ addGauss[i];
        //printf("\n    Added elasticity value: %4.9f kPa",addGauss);
        averages.y += m_hElasticity[i];
        convergenceElasticity[i] = m_hElasticity[i];
        m_hElasticity[i] *= ELAS_SCALE; //Scale elasticity to be copied to GPU for force calculations
    }
    checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));
    averages.y /= (float)(m_dynamicParticles);// - static_counter);
    avBefore /= (float)(m_dynamicParticles);// - static_counter);

    averages.z /= (float)(m_dynamicParticles);// - static_counter);
    averages.z *= 100.f; //scales decimal to percent

    mmPercent /= (float)(m_dynamicParticles);
    mmPercent *= 100.f;

    printf("\n Iteration %d:",iteration);
    printf("\n    Previous Average Elasticity: %4.9f kPa",avBefore);
    printf("\n    Cumulative Error: %4.6f m",averages.x);
    printf("\n    Current Average Elasticity: %4.9f kPa",averages.y);
    printf("\n    %d out of %d particles converged : %4.3f %% ",r,m_dynamicParticles, averages.z);
    printf("\n    %d out of %d particles converged within 1 mm: %4.3f %% \n", mm, m_dynamicParticles, mmPercent);

    delete [] error;
    return averages;
}

float4 ParticleSystem::updateElasticityOld(int iteration, float tolerance, float *array_3D, bool m_bAnchor, DATA_VOLUME *inputDataset)
{
    float4 averages = make_float4( 0.f, 0.f, 0.f, 0.f );
    checkCudaErrors(cudaMemcpy(m_hElasticity, m_dElasticity, m_dynamicParticles*sizeof(float), cudaMemcpyDeviceToHost));

    // float* error = recordDisplacementError(inputDataset);
    float* error = recordNeighborError(inputDataset);

    float avBefore = 0.0f;
    float maxErr = 0.0f;
    float c1 = 1.f;
    int r = 0;
    int mm = 0;
    float mmPercent = 0;
    //Record Total and Maximum Error
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        averages.x += error[i];
        if (fabs(error[i]) > maxErr)
            maxErr = fabs(error[i]);
    }
    std::cout <<" maxErr is "<<maxErr<<std::endl;
    //Update Elasticity
    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        m_hElasticity[i] *= (1/ELAS_SCALE);
        last_guess_elasticity[i] = m_hElasticity[i];
        avBefore += m_hElasticity[i];

        if (fabs(error[i]) > PARTICLE_TOL)
        {
            if (error[i] > 0.f)
            {
                m_hMINmod[i] = m_hElasticity[i];
                m_hElasticity[i] = fabs(m_hMAXmod[i] + m_hMINmod[i]) / 2.f;
            }

            else if (error[i] < 0.f)
            {
                m_hMAXmod[i] = m_hElasticity[i];
                m_hElasticity[i] = fabs(m_hMAXmod[i] + m_hMINmod[i]) / 2.f;
            }

        }

        if (fabs(error[i]) < PARTICLE_TOL)
        {
            convergedParticles[i] = 1;
            r++;
        }
        if (fabs(error[i]) < MM_TOL)
            mm++;

        mmPercent /= m_dynamicParticles;


    }
    averages.z = r;

    for (uint i=0; i<m_dynamicParticles; i++)
    {
        averages.y += m_hElasticity[i];
        convergenceElasticity[i] = m_hElasticity[i];
        m_hElasticity[i] *= ELAS_SCALE; //Scale elasticity to be copied to GPU for force calculations
    }
    checkCudaErrors(cudaMemcpy(m_dElasticity,m_hElasticity,m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));
    averages.y /= (float)(m_dynamicParticles);// - static_counter);
    avBefore /= (float)(m_dynamicParticles);// - static_counter);

    averages.z /= (float)(m_dynamicParticles);// - static_counter);
    averages.z *= 100.f; //scales decimal to percent


    printf("\n Iteration %d:",iteration);
    printf("\n    Previous Average Elasticity: %4.9f kPa",avBefore);
    printf("\n    Cumulative Error: %4.6f m",averages.x);
    printf("\n    Current Average Elasticity: %4.9f kPa",averages.y);
    printf("\n    %d out of %d particles converged : %4.3f %% \n",r,m_dynamicParticles, averages.z);
    printf("\n    %d out of %d particles converged within 1 mm: %4.3f %% \n", mm, m_dynamicParticles, mmPercent);



    delete [] error;
    return averages;
}


float* ParticleSystem::normalize(float *dispMag)
{
    float minErr = 100000;
    float maxErr = 0;
    float* normErr = new float[m_numParticles];

    //Find min and max values
    for (uint i = 0; i < m_numParticles; i++)
    {

        if (!m_hStaticTrue[i])
        {
            if (fabs(dispMag[i]) > maxErr)
            {
                maxErr = fabs(dispMag[i]);
            }
            if (fabs(dispMag[i]) < minErr)
            {
                minErr = fabs(dispMag[i]);
            }
        }

    }
    float den = maxErr - minErr;

    //Normalize between 0 and 1

    for (uint i = 0; i < m_numParticles; i++)
    {
        if (m_hStaticTrue[i])
            normErr[i] = 0;
        else
            normErr[i] = fabs(dispErr[i] - minErr) / den;
    }
    printf("\nden %4.6f, Min %4.6f, Max %4.6f\n",den, minErr, maxErr );
    return normErr;
}


void ParticleSystem::setOscillatorOrigin(uint oscillator)
{
    m_params.osc_origin.x = m_hPos[4*oscillator + 0];
    m_params.osc_origin.y = m_hPos[4*oscillator + 1];
    m_params.osc_origin.z = m_hPos[4*oscillator + 2];
}


uint staticSprings = 0, dynamicSprings = 0;

void
ParticleSystem::setConnections(bool initialIteration, bool reset)
{
    staticSprings = dynamicSprings = 0;
    if (m_bDynamic)
    {
        if (reset)
        {
            freeArray(m_dCon);
            freeArray(m_dConCount);
            freeArray(m_dConStart);
            freeArray(m_dConStatic);
            freeArray(m_dConCountStatic);
            freeArray(m_dConStartStatic);
            freeArray(m_dRestLengths);
            freeArray(m_dRestLengthsStatic);
        }
        // memcpy(m_hColor,m_hRefColor,4*m_numParticles*sizeof(float));

        float *dPos;
        uint   *tempCon;
        float  *tempRestLengths;
        float *m_dSortedPos;
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell


        allocateArray((void **)&tempCon, NUM_SPRINGS*m_dynamicParticles*sizeof(uint));
        allocateArray((void **)&tempRestLengths, NUM_SPRINGS*m_dynamicParticles*4*sizeof(float));

        allocateArray((void **)&m_dSortedPos, sizeof(float)*4*m_numParticles);
        allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
        allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));
        allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
        allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

        allocateArray((void **)&dPos, sizeof(float)*4*m_numParticles);
        if (m_staticParticles > 0)
            checkCudaErrors( cudaMemcpy( dPos, m_dStatic, sizeof(float)*4*m_staticParticles, cudaMemcpyDeviceToDevice));
        checkCudaErrors( cudaMemcpy( dPos+4*m_staticParticles, m_dPos, sizeof(float)*4*m_dynamicParticles, cudaMemcpyDeviceToDevice));
        // }
        // calculate grid hash
        calcHash(
            m_dGridParticleHash,
            m_dGridParticleIndex,
            dPos,
            m_numParticles);


        // sort particles based on hash
        sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

        // reorder particle arrays into sorted order and
        // find start and end of each cell
        reorderDataAndFindCellStart(
            m_dCellStart,
            m_dCellEnd,
            m_dSortedPos,
            m_dGridParticleHash,
            m_dGridParticleIndex,
            dPos,
            m_numParticles,
            m_numGridCells);

        // determine spring connections between particles
        deviceSetConnect(
            m_dPos,
            tempCon,
            tempRestLengths,
            m_dSortedPos,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_dynamicParticles,
            m_staticParticles,
            m_numGridCells);

        // if (m_bUseOpenGL)
        //     unmapGLBufferObject(m_cuda_posvbo_resource);
        // else
        freeArray(dPos);

        uint *hTempCon;
        hTempCon = new uint[NUM_SPRINGS*m_dynamicParticles];
        m_hCon = new uint[NUM_SPRINGS*m_dynamicParticles];
        m_hConStatic = new uint[NUM_SPRINGS*m_dynamicParticles];

        float *hTempRestLengths;
        hTempRestLengths = new float[NUM_SPRINGS*4*m_dynamicParticles];
        m_hRestLengths = new float[NUM_SPRINGS*4*m_dynamicParticles];
        m_hRestLengthsStatic = new float[NUM_SPRINGS*4*m_dynamicParticles];


        m_hConCount = new uint[m_dynamicParticles*sizeof(uint)];
        memset(m_hConCount, 0, m_dynamicParticles*sizeof(uint));

        m_hConStart = new uint[m_dynamicParticles*sizeof(uint)];
        memset(m_hConStart, 0, m_dynamicParticles*sizeof(uint));

        m_hConCountStatic = new uint[m_dynamicParticles*sizeof(uint)];
        memset(m_hConCountStatic, 0, m_dynamicParticles*sizeof(uint));

        m_hConStartStatic = new uint[m_dynamicParticles*sizeof(uint)];
        memset(m_hConStartStatic, 0, m_dynamicParticles*sizeof(uint));

        checkCudaErrors(cudaMemcpy(hTempCon, tempCon, sizeof(uint)*NUM_SPRINGS*m_dynamicParticles, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hTempRestLengths, tempRestLengths, sizeof(float)*NUM_SPRINGS*4*m_dynamicParticles, cudaMemcpyDeviceToHost));


        // Free memory associated with grid-hash sorting
        freeArray(tempCon);
        freeArray(tempRestLengths);

        freeArray(m_dSortedPos);
        freeArray(m_dGridParticleHash);
        freeArray(m_dGridParticleIndex);
        freeArray(m_dCellStart);
        freeArray(m_dCellEnd);


        uint springCount = 0, centerCount = 0, countSurf = 0;

        for (uint p=0; p<m_dynamicParticles; p++)
        {
            m_hConStart[p] = dynamicSprings;
            m_hConStartStatic[p] = staticSprings;

            float3 kdir = make_float3( 0.f, 0.f, 0.f );
            float kcount = 0.f;

            for (int s=0; s<NUM_SPRINGS; s++)
            {
                if(hTempCon[p*NUM_SPRINGS + s] != 0xffffffff)
                {
    // Find surface particles
                    float3 len = make_float3( hTempRestLengths[4*(NUM_SPRINGS*p + s) + 0],
                                              hTempRestLengths[4*(NUM_SPRINGS*p + s) + 1],
                                              hTempRestLengths[4*(NUM_SPRINGS*p + s) + 2] );

                    float norm = sqrt( len.x*len.x + len.y*len.y + len.z*len.z );
                    kdir.x += (len.x / norm);
                    kdir.y += (len.y / norm);
                    kdir.z += (len.z / norm);
                    kcount++;

                    springCount++;
                    float sdSwitch = hTempRestLengths[4*(NUM_SPRINGS*p + s) + 3];
                    if (sdSwitch < 0)
                    {
                        m_hConStatic[staticSprings] = hTempCon[p*NUM_SPRINGS + s];
                        m_hConCountStatic[p]++;
                        m_hRestLengthsStatic[4*staticSprings + 0] = len.x;
                        m_hRestLengthsStatic[4*staticSprings + 1] = len.y;
                        m_hRestLengthsStatic[4*staticSprings + 2] = len.z;
                        m_hRestLengthsStatic[4*staticSprings + 3] = sdSwitch;
                        staticSprings++;
                    }
                    else
                    {
                        m_hCon[dynamicSprings] = hTempCon[p*NUM_SPRINGS + s];
                        m_hConCount[p]++;
                        m_hRestLengths[4*dynamicSprings + 0] = len.x;
                        m_hRestLengths[4*dynamicSprings + 1] = len.y;
                        m_hRestLengths[4*dynamicSprings + 2] = len.z;
                        m_hRestLengths[4*dynamicSprings + 3] = sdSwitch;
                        dynamicSprings++;
                    }
                }
            }

    // Set surface particle color to white
            float ksize = sqrt( kdir.x*kdir.x + kdir.y*kdir.y + kdir.z*kdir.z ) / kcount;
            // int x = m_hInput[3*p];
            // int y = m_hInput[3*p+1];
            int z = m_hInput[3*p+2];

            //uint cube = (int) floorf( powf((float) m_numParticles, 1.0f / 3.0f) + 0.5f);
            if (ksize > 0.25f)
            {
                m_hColor[4*(p + m_staticParticles) + 0] = 1.f;
                m_hColor[4*(p + m_staticParticles) + 1] = 1.f;
                m_hColor[4*(p + m_staticParticles) + 2] = 1.f;
                m_hSurface[p] = 0;
                //Only count sides as surface
                //**CHANGE with cube/data size**
                // // if (z > 0) //anchors top only **CHANGED FOR BREAST DATA
               if (!m_bAnchor)
               {
                    if (z < 1 || z > 8 ) //anchors all sides minus top/bottom
                   {
                        m_hSurface[p] = 0;
                   }
               }

                //No input static, makes surface minus top and bottom rows static.
                if (!m_bAnchor)
                {
                    if (m_hSurface[p])
                    {
                        m_hStaticTrue[p] = 1;
                        countSurf ++;
                    }
                }

            }
        }

        checkCudaErrors(cudaMemcpy(m_dSurface,m_hSurface,m_dynamicParticles*sizeof(int),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_dStaticTrue,m_hStaticTrue,m_dynamicParticles*sizeof(int),cudaMemcpyHostToDevice));


       delete [] hTempCon;
       delete [] hTempRestLengths;


        if (initialIteration)
        {
            printf("\n Total Springs: %d (%d per particle)\n %d for central element\n %d Static Springs\n %d Dynamic Springs",springCount,springCount/m_dynamicParticles,centerCount,staticSprings,dynamicSprings);
            printf("\n Surface Particles: %d\n", countSurf);
            fflush(stdout);

        }

        fflush(stdout);
    // Dynamic Connections
        allocateArray((void **)&m_dCon, dynamicSprings*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dCon,m_hCon,dynamicSprings*sizeof(uint),cudaMemcpyHostToDevice));


    // Number of Dynamic Springs for each particle
        allocateArray((void **)&m_dConCount, m_dynamicParticles*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dConCount,m_hConCount,m_dynamicParticles*sizeof(uint),cudaMemcpyHostToDevice));
    // Beginning index of the dynamic connections array for each particle
        allocateArray((void **)&m_dConStart, m_dynamicParticles*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dConStart,m_hConStart,m_dynamicParticles*sizeof(uint),cudaMemcpyHostToDevice));
    // Static connections
        allocateArray((void **)&m_dConStatic, staticSprings*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dConStatic,m_hConStatic,staticSprings*sizeof(uint),cudaMemcpyHostToDevice));
    // Number of Static Springs for each particle
        allocateArray((void **)&m_dConCountStatic, m_dynamicParticles*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dConCountStatic,m_hConCountStatic,m_dynamicParticles*sizeof(uint),cudaMemcpyHostToDevice));
    // Beginning index of the static connections array for each particle
        allocateArray((void **)&m_dConStartStatic, m_dynamicParticles*sizeof(uint));
        checkCudaErrors(cudaMemcpy(m_dConStartStatic,m_hConStartStatic,m_dynamicParticles*sizeof(uint),cudaMemcpyHostToDevice));

        allocateArray((void **)&m_dInitPos,m_dynamicParticles*sizeof(float)*4);
        allocateArray((void **)&m_dFinalPos,dynamicSprings*sizeof(float)*4);


    // Rest length vectors for all dynamic connections
        allocateArray((void **)&m_dRestLengths, dynamicSprings*4*sizeof(float));
        checkCudaErrors(cudaMemcpy(m_dRestLengths,m_hRestLengths,dynamicSprings*4*sizeof(float),cudaMemcpyHostToDevice));
    // Vectors for spring stretch
        allocateArray((void **)&m_dStretchS, m_dynamicParticles*sizeof(float));
    // Rest length vectors for all static connections
        allocateArray((void **)&m_dRestLengthsStatic, staticSprings*4*sizeof(float));
        checkCudaErrors(cudaMemcpy(m_dRestLengthsStatic,m_hRestLengthsStatic,staticSprings*4*sizeof(float),cudaMemcpyHostToDevice));

    }

    size_t free, total;
    checkCudaErrors(cudaMemGetInfo(&free,&total));
    if (initialIteration)
        printf("\n Free Memory: %lu / %lu \n",free,total);
}
void ParticleSystem::dumpResults(DATA_VOLUME *grid)
{
    float *elasOut = new float[m_dynamicParticles];
    for (int i = 0; i < m_dynamicParticles; i++)
    {
        elasOut[i] = m_hElasticity[i];
        elasOut[i] *= 1/ELAS_SCALE;
    }
    dumpParticles(grid, elasOut, "ELASTICITY");
    dumpParticles(grid, m_hDispMag, "DISPMAG");
    dumpParticles_absDiff(grid, m_hDispMag,groundTruthmag, "DISPERR");
}

void
ParticleSystem::dumpParticles(DATA_VOLUME *grid, float *input, const char* fileName)
{
  if (m_hInput!=NULL) // check for ct input
  {
      // create output folder for sim CT
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[32];
        time (&rawtime);
        timeinfo = localtime (&rawtime);
        strftime (buffer,32,"%m.%d.%y-%H.%M.%S",timeinfo);
        printf(" DUMPED: %s %s\n",fileName,buffer );
        fflush(stdout);

        char outpath[255];
        sprintf(outpath,"%s/%s-%s/",FILE_ELASTICITY_OUT,fileName,buffer);
        mkdir((const char *)outpath,S_IRWXU|S_IRWXG|S_IRWXO);

        float *output;
        output = (float*)malloc(grid->params.arraySize.x*grid->params.arraySize.y*grid->params.arraySize.z*sizeof(float));
        memset(output,0,grid->params.arraySize.x*grid->params.arraySize.y*grid->params.arraySize.z*sizeof(float));

        int counter = 0;

    // get particle positions and connections
        checkCudaErrors(cudaMemcpy(m_hPos+4*m_staticParticles, m_dPos, sizeof(float)*4*m_dynamicParticles, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_hRestLengths, m_dRestLengths, 4*dynamicSprings*sizeof(float), cudaMemcpyDeviceToHost));

        for (uint i=0; i<m_numParticles; i++)
        {

        // find original voxel addres of particle, to retrieve CT value
            int x = m_hInput[3*i];
            int y = m_hInput[3*i+1];
            int z = m_hInput[3*i+2];
            int loc = x + grid->params.arraySize.x * (y + grid->params.arraySize.y * z);

            output[loc] = input[counter++];

        }

        // save_data(outpath,path,files,grid,output,filetype);
        int3 size = make_int3(m_dataSize.x,m_dataSize.y,m_dataSize.z);
        writeToTextFile(outpath, size, output);

        free(output);

  }
}

void
ParticleSystem::dumpParticles_absDiff(DATA_VOLUME *grid, float *input1, float* input2, const char* fileName)
{
  if (m_hInput!=NULL) // check for ct input
  {
      // create output folder for sim CT
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[32];
        time (&rawtime);
        timeinfo = localtime (&rawtime);
        strftime (buffer,32,"%m.%d.%y-%H.%M.%S",timeinfo);
        printf(" DUMPED: %s %s\n",fileName,buffer );
        fflush(stdout);

        char outpath[255];
        sprintf(outpath,"%s/%s-%s/",FILE_ELASTICITY_OUT,fileName,buffer);
        mkdir((const char *)outpath,S_IRWXU|S_IRWXG|S_IRWXO);

        float *output;
        output = (float*)malloc(grid->params.arraySize.x*grid->params.arraySize.y*grid->params.arraySize.z*sizeof(float));
        memset(output,0,grid->params.arraySize.x*grid->params.arraySize.y*grid->params.arraySize.z*sizeof(float));

        int counter = 0;

    // get particle positions and connections
        checkCudaErrors(cudaMemcpy(m_hPos+4*m_staticParticles, m_dPos, sizeof(float)*4*m_dynamicParticles, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(m_hRestLengths, m_dRestLengths, 4*dynamicSprings*sizeof(float), cudaMemcpyDeviceToHost));

        for (uint i=0; i<m_numParticles; i++)
        {

        // find original voxel addres of particle, to retrieve CT value
            int x = m_hInput[3*i];
            int y = m_hInput[3*i+1];
            int z = m_hInput[3*i+2];
            int loc = x + grid->params.arraySize.x * (y + grid->params.arraySize.y * z);

            output[loc] = fabs(input1[counter] - input2[counter]);
            counter++;
        }

        // save_data(outpath,path,files,grid,output,filetype);
        int3 size = make_int3(m_dataSize.x,m_dataSize.y,m_dataSize.z);
        writeToTextFile(outpath, size, output);

        free(output);

  }
}


void ParticleSystem::setPreMagPos()
{
    int v = 0;
    for (uint i=0; i < m_dynamicParticles; i++)
    {
        for(int j = 0; j < 4; j++)
        {
                        //    m_hPos[v] = positionArray[v];
            m_hPreMagPos[v] = m_hPos[v];
            v++;
        }
    }


}
void
ParticleSystem::systemReset(ParticleConfig config, const char *path, bool reset, bool setCon, uint userSize, bool m_bUseInput, bool initialIteration)
{

    switch (config)
    {
        default:
            case CONFIG_INPUT:
            {
                config = CONFIG_INPUT; setCon  = 1;
                memset(m_hForceS, 0, 4*m_dynamicParticles*sizeof(float));
                memset(m_hVel, 0, 4*m_dynamicParticles*sizeof(float));
                memset(m_hPos, 0, 4*m_dynamicParticles*sizeof(float));

                if (m_bUseInput)
                {
                    uint p=0, n=0;
                    uint contourCount = 0;
                    initalComPos = make_float3(0.f,0.f,0.f);
                    m_params.headStart = -1;
                    for (uint i=0; i < m_numParticles; i++)
                    {
                        float point[3];
                        point[0] = (float) (m_hInput[n++] + 10.f) / (float)m_gridSize.x;
                        point[1] = (float) (m_hInput[n++] + 10.f) / (float)m_gridSize.y;
                        point[2] = (float) (m_hInput[n++] + 10.f) / (float)m_gridSize.z;

                        if(!m_hSourcePos_isSet)
                        {
                            m_hPos[p++] = m_worldSize.x * (point[0] - 0.5f);
                            m_hPos[p++] = m_worldSize.y * (point[1] - 0.5f);
                            m_hPos[p++] = m_worldSize.z * (point[2] - 0.5f);
                            m_hPos[p++] = -1.f;


                        }
                        else
                        {

                            for(int j = 0; j < 4;j++)
                                m_hPos[p] = m_hSourcePos[p++];

                        }

                        if ( (m_hSegment[i] < 0) && (m_worldSize.z * (point[2] - 0.5f) < m_params.headLevel * m_params.worldSize.z) )
                        {
                            m_params.headStart = i;
                        }


                        if (m_hSegment[i] == m_params.contourSwitch && !m_hSourcePos_isSet)
                        {
                            contourCount++;
                            initalComPos.x += m_worldSize.x * (point[0] - 0.5f);
                            initalComPos.y += m_worldSize.y * (point[1] - 0.5f);
                            initalComPos.z += m_worldSize.z * (point[2] - 0.5f);
                        }
                    }




                    initalComPos.x *= 1000.f / (float)contourCount;
                    initalComPos.y *= 1000.f / (float)contourCount;
                    initalComPos.z *= 1000.f / (float)contourCount;
                }

                if (m_bStatic)
                    checkCudaErrors(cudaMemcpy(m_dSegmentStatic, m_hSegment, m_staticParticles*sizeof(uint),cudaMemcpyHostToDevice));
                if (m_bDynamic)
                    checkCudaErrors(cudaMemcpy(m_dSegment, m_hSegment+m_staticParticles, m_dynamicParticles*sizeof(uint),cudaMemcpyHostToDevice));
                m_params.headAngle = make_float3( 0.f, 0.f, 0.f );

            }
            break;

            case CONFIG_POS_RESET:
            {
                config = CONFIG_POS_RESET;
                memset(m_hForceS, 0, 4*m_dynamicParticles*sizeof(float));
                memset(m_hVel, 0, 4*m_dynamicParticles*sizeof(float));
                if (m_bUseInput)
                {
                    uint v=0;
                    uint contourCount = 0;
                    initalComPos = make_float3(0.f,0.f,0.f);
                    for (uint i=0; i < m_dynamicParticles; i++)
                    {

                        for(int j = 0; j < 4; j++)
                        {
                        //    m_hPos[v] = positionArray[v];
                            m_hInitialPos[v] = m_hPos[v];
                            v++;
                        }


                    }
                   int dSprings = 0;
                    for (uint i = 0; i < m_dynamicParticles; i++)
                    {
                        int startCount = m_hConStart[i];
                        int conCount = m_hConCount[i];
                      //  std::cout <<"----> startCount " <<startCount <<" conCount " <<conCount <<" i "<<i<<std::endl;
                        for (int j = 0; j < conCount; j++)
                        {
                            int index = m_hCon[startCount + j];
                       //     printf(" index = %d \n", index);
                      //      deformRL[4*dSprings+0] = fabs(m_hPos[4*index+0] - m_hPos[4*i+0]);
                      //      deformRL[4*dSprings+1] = fabs(m_hPos[4*index+1] - m_hPos[4*i+1]);
                      //      deformRL[4*dSprings+2] = fabs(m_hPos[4*index+2] - m_hPos[4*i+2]);
                      //      deformRL[4*dSprings+3] = 1;
                            m_hRestLengths[4*dSprings+0] = (m_hPos[4*index+0] - m_hPos[4*i+0]);//deformRL[4*dSprings + 0];
                            m_hRestLengths[4*dSprings+1] = (m_hPos[4*index+1] - m_hPos[4*i+1]);//deformRL[4*dSprings + 1];
                            m_hRestLengths[4*dSprings+2] = (m_hPos[4*index+2] - m_hPos[4*i+2]);//deformRL[4*dSprings + 2];
                            m_hRestLengths[4*dSprings+3] = 1;//deformRL[4*dSprings + 3];
                            dSprings++;
                        }
                    }


                    checkCudaErrors(cudaMemcpy(m_dRestLengths,m_hRestLengths,dSprings*4*sizeof(float),cudaMemcpyHostToDevice));
                }

            }
            break;
    }

    memcpy(m_hInitialPos,m_hPos,4*m_numParticles*sizeof(float));
    if(!m_hSourcePos_isSet)
    {
        memcpy(m_hSourcePos,m_hPos,4*m_numParticles*sizeof(float));
       // m_hSourcePos_isSet=true; // 5-15-25
    }

    if (m_bStatic)
        checkCudaErrors(cudaMemcpy(m_dStatic, m_hPos, 4*m_staticParticles*sizeof(float), cudaMemcpyHostToDevice));

    if (m_bDynamic)
    {
        checkCudaErrors(cudaMemcpy(m_dPos, m_hPos+4*m_staticParticles, 4*m_dynamicParticles*sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(tempPos, m_hPos+4*m_staticParticles, 4*m_dynamicParticles*sizeof(float), cudaMemcpyHostToDevice));
    }
    memcpy(m_hLast, m_hPos, m_numParticles*4*sizeof(float) );

    if (m_bDynamic)
        checkCudaErrors(cudaMemcpy(m_dVel, m_hVel+4*m_staticParticles, 4*m_dynamicParticles*sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_dForceS, m_hForceS, 4*m_dynamicParticles*sizeof(float), cudaMemcpyHostToDevice));

    if (setCon)
        setConnections(initialIteration,reset);
}


void ParticleSystem::moveBoundaryToTarget(DATA_VOLUME *inputDataset, int iteration)
{
    checkCudaErrors(cudaMemcpy(m_hPos,m_dPos,4*m_dynamicParticles*sizeof(float),cudaMemcpyDeviceToHost));

    float *boundScaleUVW;
    boundScaleUVW = new float[4*m_dynamicParticles];
    memset(boundScaleUVW, 0, sizeof(float)*4*m_dynamicParticles);

    float *tempPosition;
    tempPosition = new float[4*m_dynamicParticles];
    memset(tempPosition, 0, sizeof(float)*4*m_dynamicParticles);

    float itNum = ITERATION_BC;
    float boundScale = 1 / itNum;

    for (uint i = 0; i < m_dynamicParticles; i++)
    {
        tempPosition[4*i+0] = m_hPos[4*i+0];
        tempPosition[4*i+1] = m_hPos[4*i+1];
        tempPosition[4*i+2] = m_hPos[4*i+2];


        if (m_hStaticTrue[i]) //Scales boundary to be pulled out over number of boundary pull iterations
        {
            boundScaleUVW[4*i + 0] = UVW_data[4*i + 0] * boundScale;
            boundScaleUVW[4*i + 1] = UVW_data[4*i + 1] * boundScale;
            boundScaleUVW[4*i + 2] = UVW_data[4*i + 2] * boundScale;

            m_hPos[4*i+0] = tempPosition[4*i+0] - boundScaleUVW[4*i+0];
            m_hPos[4*i+1] = tempPosition[4*i+1] - boundScaleUVW[4*i+1];
            m_hPos[4*i+2] = tempPosition[4*i+2] - boundScaleUVW[4*i+2];

        }

    }


    checkCudaErrors(cudaMemcpy(m_dPos,m_hPos+4*m_staticParticles,4*m_dynamicParticles*sizeof(float),cudaMemcpyHostToDevice));

    delete [] boundScaleUVW;
    delete [] tempPosition;

}