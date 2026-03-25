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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID      0
#define DO_TIMING       0


#include <helper_functions.h>
#include <cuda_runtime.h>
#include "particles_kernel.cuh"

#include "vector_functions.h"
#include "read_dicom_data.h"
#include "constants.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>


using namespace ParticleConstants;


// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, uint staticParticles, uint dynamicParticles, uint3 dataSize, uint3 gridSize, float3 voxelSize, bool bUseOpenGL, bool bUseInput, bool bUseContour, int *dataInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *contours, float airThresh, float boneThresh, float headLevel,  bool geom, char *geom_file, bool m_bAnchor);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            CONFIG_PYRAMID,
            CONFIG_INPUT,
            CONFIG_POS_RESET,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
        };

        float update(float deltaTime, int iteration_counter);
        void systemReset(ParticleConfig config, const char *path, bool reset, bool setCon, uint userSize, bool useInput, bool initialIteration);
        void resetRestLength();
        void finalFree();
        void loadUVW(const char* file_name, float *array);
        void populateDataFromFile(DATA_VOLUME *grid, char* filename);
        void record_initialGT_deformation(DATA_VOLUME *grid);
        void record_ground_truth(DATA_VOLUME *grid);
        float* recordDisplacement(DATA_VOLUME *grid);
        float* recordDisplacementError(DATA_VOLUME *grid);
        float* recordNeighborError(DATA_VOLUME *grid);
        float4 updateElasticity(int, float, float *array, bool, DATA_VOLUME *inputDataset);
        float* normalize(float *dispErr);
        void inputStatic(const char* file_name, bool m_bAnchor);
        void render(float invViewMatrix[12], dim3 cudaGrid, dim3 cudaBlock, DATA_VOLUME *inputData);
        void resetElasticModuli(float modulus,bool reset, int iteration, bool total);
        void recordRestLengths(bool,bool);
        void initialGuessElastic(bool);
        void loadFromGeomFile(char *geom_file, float *x,float *y,float *z,int *diameter,float *sphereElas);
        void computeCentroid(float *cx, float *cy, float *cz);
        float computeDistance(float a, float b, float c, float d, float e, float h);
        void setElasticityAndAssociatedParameters(int i, float value);
        void setCondenseArrayFromInputDataset(DATA_VOLUME *inputDataset);
        char *FILE_PREWARP_IN;
        char *FILE_PREWARP_OUT;
        char *DUMP_FOLDER;

        int    getNumParticles() const
        {
            return m_numParticles;
        }
        int    getStaticParticles() const
        {
            return m_staticParticles;
        }
        int    getDynamicParticles() const
        {
            return m_dynamicParticles;
        }


        unsigned int getPosBuffer() const
        {
            return m_posVbo;
        }
        unsigned int getColorBuffer()       const
        {
            return m_colorVBO;
        }

        float *getInitialPositions() const
        {
            return m_hInitialPos;
        }
        float *getCurrentPositions() const
        {
            return m_hPos;
        }
        float *getCurrentColors() const
        {
            return m_hColor;
        }
        uint *getCurrentConnections() const
        {
            return m_hCon;
        }
        uint *getCurrentConnectionStarts() const
        {
            return m_hConStart;
        }
        uint *getCurrentConnectionCounts() const
        {
            return m_hConCount;
        }


        void switchHeadAngle(uint switcher)
        {
            m_params.headSwitch = switcher;
        }
        void setHeadLevel(float x)
        {
            m_params.headLevel = x;
        }

        void setOscillator(uint x)
        {
            m_params.oscillator = x;
        }
        void setOscillatorAxis(float x, float y, float z)
        {
            m_params.osc_axis.x = x;
            m_params.osc_axis.y = y;
            m_params.osc_axis.z = z;
        }
        void setOscillatorOrigin(uint x);
        void setAmplitude(float x)
        {
            m_params.amplitude = x;
        }
        void accumulateTime(float x)
        {
            m_params.simulationTime += x;
        }
        void resetTime()
        {
            m_params.simulationTime = 0.f;
        }

        void dumpDispMag(const char *path, DATA_VOLUME *grid, DICOM_STRUCT *contours, FILE_LIST *files, char *filetype, float *array, bool UVW);
        void dumpTarget();
        void dumpParticles( DATA_VOLUME *grid, float *array, const char* fileName);
        void dumpParticlePosition(int iteration, DATA_VOLUME *grid);
        void dumpElasticity(const char *path, DATA_VOLUME *grid, DICOM_STRUCT *contours, FILE_LIST *files, char *filetype, float *array);
        void dumpUVW(DATA_VOLUME *grid);

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x, float y, float z)
        {
            m_params.gravity = make_float3( -x, -y, z);
        }

        void setSpringConstant(float x)
        {
            m_params.springStrength = x;
        }
        void setSpringDamping(float x)
        {
            m_params.springDamping = x;
        }
        void setEnergyModulus(float x)
        {
            m_params.energyModulus = x;
        }

        void setMinimumModulus(float x)
        {
            m_params.MINmod = x;
        }
        void setMaximumModulus(float x)
        {
            m_params.MAXmod = x;
        }
        void setElasticModulus(float x)
        {
            m_params.elasticModulus = x;
        }
        void setShearModulus(float x)
        {
            m_params.shearModulus = x;
        }

        void setMinimumPotential(float x)
        {
            m_params.minPotential = x;
        }
        void setStretchiness(float x, float y)
        {
            m_params.stretchiness.x = x;
            m_params.stretchiness.y = y;
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }
        /*
        void setElasticity(float x)
        {
            m_params.elastRatio = x;
        }
        void setRigidity(float x)
        {
            m_params.rigidRatio = x;
        }
        */
        void setRestLength(float x)
        {
            m_params.restLengthMult = x;
        }
        void setColliderOuterRadius(float x[10], float w)
        {
            for (int c=0; c<10; c++)
                m_params.colliderOuterRadius[c] = x[c]*w;
        }
        void setColliderInnerRadius(float x[10], float w)
        {
            for (int c=0; c<10; c++)
                m_params.colliderInnerRadius[c] = 0.1f*x[c]*w;
        }
        void setColliderLength(uint x)
        {
            m_params.colliderLength = x;
        }
        void setColliderControl(uint x)
        {
            m_params.colliderControl = x;
        }
        void setColliderPos(float3 x)
        {
            m_params.colliderPos = x;
        }
        void setColliderVel(float3 x)
        {
            m_params.colliderVel = x;
        }
        void setColliderOrientation(float3 x)
        {
            m_params.colliderOrientation = x;
        }

        void setColorSwitcher(uint switcher)
        {
            m_params.colorSwitch = switcher;
        }
        void resetDefaultColor()
        {
            memcpy(m_hColor,m_hRefColor,4*m_numParticles*sizeof(float));
        }
        void setContourSwitch(uint switcher)
        {
            m_params.contourSwitch = switcher;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        float getParticleSpacing()
        {
            return m_params.particleSpacing;
        }
        float3 getComPos()
        {
            return m_params.comPos;
        }
        float3 getColliderPos()
        {
            return m_params.colliderPos;
        }
        float3 getColliderVel()
        {
            return m_params.colliderVel;
        }
        float getColliderOuterRadius(uint p)
        {
            return m_params.colliderOuterRadius[p];
        }
        float getColliderInnerRadius(uint p)
        {
            return m_params.colliderInnerRadius[p];
        }
        float getColliderMass()
        {
            return m_params.colliderMass;
        }
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getWorldOrigin()
        {
            return m_params.worldOrigin;
        }
        float3 getVoxelSize()
        {
            return m_params.voxelSize;
        }
        void moveBoundaryToTarget(DATA_VOLUME *inputDataset, int);
        void moveInnerVoxelsToTarget(DATA_VOLUME *inputDataset, int);
        void record_innerVoxels(DATA_VOLUME *inputDataset, int);
        void addSphere(int index, float *pos, float *vel, int r, float spacing);

    protected: // methods
        ParticleSystem() {}

        void _initializeMemberVariablesMemory(uint numParticles, uint staticParticles, uint dynamicParticles);
        void _AllocateAndSetMemberVariables(uint numParticles, uint staticParticles, uint dynamicParticles);
        void _initialize_m_hInput(int *segmentInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *structures, bool m_bAnchor);
        void _initializeColor( DICOM_STRUCT *structures, bool bUseInput);
        void _initializeSystematicElasticity( bool geom, char *geom_file);
        void _initializeRenderParams(uint numParticles, uint staticParticles, uint dynamicParticles);
        void _initialize(uint numParticles, uint staticParticles, uint dynamicParticles, int *segmentInput, DATA_VOLUME *inputDataset, DICOM_STRUCT *structures, bool bUseInput,  bool geom, char *geom_file, bool m_bAnchor);
        void _finalize();

        void initGrid(uint3 size, float spacing, float jitter, uint numParticles);
        void initPyramid(float spacing, float jitter, uint numParticles);
        void initInput(uint3 size, uint numParticles);
        void setConnections(bool initialIteration, bool reset);

    protected: // data
        bool m_bInitialized, m_bUseOpenGL, m_bUseInput, m_bUseContour, m_bDynamic, m_bStatic;
        uint m_numParticles, m_staticParticles, m_dynamicParticles, m_numContours, m_convergeCounter;
        bool m_bAnchor;

        // CPU data
        uint  *m_hInput;            // Refernce positions in voxel coordinates of original CT
        int  *m_hSegment;           // enumerator value assigning each particle to a contour/tissue type
        uint  *m_hSorted;              // original voxel location of the segment data after sorting
        uint  *m_hCon;              // particle id's for all spring dynamic connections
        uint  *m_hConCount;         // number of dynamic connectiosn for each particle
        uint  *m_hConStart;         // beinning index of dynamic connections for each particle
        uint  *m_hConStatic;              // particle id's for all spring dynamic connections
        uint  *m_hConCountStatic;           // number of static connections for each particles
        uint  *m_hConStartStatic;           // beginning index of static connections for each particle
        float *m_hRestLengths;             // rest length vectors for all spring connections
        float *m_hRestLengthsStatic;             // rest length vectors for all spring connections
        float *m_hPos;             // particle positions
        float *m_hInitialPos;
        float *m_hTempPos;
        float *m_hSourcePos;
        float *m_hVel;             // particle velocities
        float *m_hForceS;
        float *m_hRefColor;         // particle color scale initially established for contours
        float *m_hColor;            // particle color scale
        float *m_hCntrScale;        // array with length = number of contours, holds rest length multiplier
        float *m_hLast;             // array that holds previous iteration's particle positions
        int *m_hStaticTrue;
        int *m_hSurface;

        // GPU data
        int   *m_dSegment;
        int   *m_dSegmentStatic;
        uint   *m_dCon;
        uint   *m_dConCount;
        uint   *m_dConStart;
        uint   *m_dConStatic;
        uint   *m_dConStartStatic;
        uint   *m_dConCountStatic;
        float *m_dRestLengths;
        float *m_dRestLengthsStatic;
        float *m_dVel;
        float *m_dInterVel;
        float *m_dForceS;            // summation of all external and internal forces - possibly split for future rigid body interactions
        float *m_dStretchS;          // Amount of stretch for each spring for each particle
        float *m_dInitPos;
        float *m_dFinalPos;
        int *m_dStaticTrue;
        float *m_dForceT;            // summation of all external and internal forces - possibly split for future rigid body interactions
        float *m_dCntrScale;
        float *m_dColor;

        float *tempPos;
        float *m_dPos;          // position data for dynamic particles
        float *m_dStatic;       // position data for static particles

        uint  *m_dStaticConCount;      // grid hash value for each particle

        // New data arrays for particle specific properties and local volume constraints
        float *m_hDispMag;
        float *m_hMINmod;
        float *m_hMAXmod;
        float *m_hElasticity;
        float *convergenceElasticity;
        float *elasVecIn;
        float *m_hVolume;

        float *condenseArray_3D;
        float *convergedParticles;
        float *dispErr;

        float *groundTruth;
        float *groundTruthPosition;
        float *groundTruthDisplacement;
         //Arrays for FSA
        float *last_guess_elasticity;
        float *step;
        float *searchDirection;

        float *disp;
        float *dispMag2;
        float *UVW_data;
        float *groundTruthmag;
        float *initialRL;
        float *deformRL;
        int *staticTrue;


        float *m_dElasticity;
        float *m_dVolume;
        float *m_dRefVolume;
        int *m_dSurface;

        float U_Multiplier;
        float V_Multiplier;
        float W_Multiplier;

        uint   m_posVbo;            // vertex buffer object for particle positions
        uint   m_colorVBO;          // vertex buffer object for colors
//        uint   m_renderPBO;          // vertex buffer object for colors

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint3 m_dataSize;
        float3 m_voxelSize;
        float3 m_worldSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
        bool m_hSourcePos_isSet;
};

#endif // __PARTICLESYSTEM_H__
