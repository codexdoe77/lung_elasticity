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

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, int size);
    void setArrayToZero(void *devPtr, int size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void registerGLPBO(uint pbo, struct cudaGraphicsResource **cuda_vbo_resource);
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);


    void setParameters(SimParams *hostParams);

    void integrateShrink(float *pos,
                         float *vel,
                         float *forces,
                         int   *segment,
                         float deltaTime,
                         uint dynParticles);

    void integrateConnect(float *pos,
                         float *vel,
                         float *forceS,
                         float *forceT,
                         float deltaTime,
                         uint dynParticles);

    void deviceSetConnect(  float *pos,
                            uint  *cons,
                            float *restLengths,
                            float *sortedPos,
                            uint  *gridParticleIndex,
                            uint  *cellStart,
                            uint  *cellEnd,
                            uint   dynParticles,
                            uint   statParticles,
                            uint   numCells);

    void shrinkStructure(float *pos,
                        float *vel,
                        float *forces,
                        uint  *cons,
                        float *restLengths,
                        int   *segment,
                        float *cntrScale,
                        uint  dynParticles,
                        uint  statParticles,
                        float range,
                        float deltaTime,
                        float *color);

    void connections(float *temp,
                     float *pos,
                     float *vel,
                     float *forceS,
                     float *stretchS,
                     float *forceT,
                     float *elasticity,
                     uint  *cons,
                     uint  *conCount,
                     uint  *conStart,
                     float *restLengths,
                     int   *segment,
                     float *cntrScale,
                     uint  dynParticles,
                     float *color,
                     uint3 dataSize,
                     float *initPos,
                     float *finalPos,
                     int *staticTrue,
                     bool  m_bAnchor,
                     int *surface);

    void staticConnections(float *pos,
                            float *vel,
                            float *forces,
                            float *staticPos,
                            float *elasticity,
                            uint  *cons,
                            uint  *conCount,
                            uint  *conStart,
                            float *restLengths,
                            uint  dynParticles,
                            uint  statParticles);

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(float *newPos,
                 float *newVel,
                 float *newForce,
                 int   *segment,
                 uint   dynParticles,
                 float  deltaTime);


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

    void center_of_mass(float *pos, float3 *comPos, uint numParticles);
}
