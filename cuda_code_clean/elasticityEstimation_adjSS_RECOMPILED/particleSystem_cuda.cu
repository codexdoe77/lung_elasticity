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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/fill.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"
#include "particles_kernel_setConnect.cuh"


extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
        cudaDeviceReset();
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void setArrayToZero(void *devPtr, size_t size)
    {
        checkCudaErrors(cudaMemset(devPtr,0.0f,size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }




    void integrateShrink(float *pos,
                         float *vel,
                         float *forces,
                         int   *segment,
                         float deltaTime,
                         uint dynaParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);
        thrust::device_ptr<float4> d_for4((float4 *)forces);
        thrust::device_ptr<int> d_seg((int *)segment);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_for4, d_seg)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+dynaParticles, d_vel4+dynaParticles, d_for4+dynaParticles, d_seg+dynaParticles)),
            intShrink_functor(deltaTime));
    }




    void integrateConnect(float *pos,
                         float *vel,
                         float *forceS,
                         float *forceT,
                         float deltaTime,
                         uint dynaParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);
        thrust::device_ptr<float4> d_forS4((float4 *)forceS);
        thrust::device_ptr<float4> d_forT4((float4 *)forceT);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4, d_vel4, d_forS4, d_forT4)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos4+dynaParticles, d_vel4+dynaParticles, d_forS4+dynaParticles, d_forT4+dynaParticles)),
            intConnect_functor(deltaTime));

        checkCudaErrors(cudaMemset(forceS,0.f,dynaParticles*4*sizeof(float)));
        checkCudaErrors(cudaMemset(forceT,0.f,dynaParticles*4*sizeof(float)));
    }



    void deviceSetConnect(  float *dynaPos,
                            uint  *cons,
                            float *restLengths,
                            float *sortedPos,
                            uint  *gridParticleIndex,
                            uint  *cellStart,
                            uint  *cellEnd,
                            uint   dynaParticles,
                            uint   statParticles,
                            uint   numCells)
    {
        checkCudaErrors(cudaMemset(cons, 0xffffffff, NUM_SPRINGS*dynaParticles*sizeof(uint)));
        checkCudaErrors(cudaMemset(restLengths, 0, NUM_SPRINGS*dynaParticles*sizeof(float4)));

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(dynaParticles, 64, numBlocks, numThreads);

        // execute the kernel
        setConnectD<<< numBlocks, numThreads >>>((float4 *)dynaPos,
                                                 (uint   *)cons,
                                                 (float4 *)restLengths,
                                                 (float4 *)sortedPos,
                                                 (uint   *)gridParticleIndex,
                                                 (uint   *)cellStart,
                                                 (uint   *)cellEnd,
                                                 dynaParticles,
                                                 statParticles );

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }




    void shrinkStructure(float *dynaPos,
                        float *vel,
                        float *forces,
                        uint  *cons,
                        float *restLengths,
                        int   *segment,
                        float *cntrScale,
                        uint   dynaParticles,
                        uint   statParticles,
                        float  range,
                        float  deltaTime,
                        float *color)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(dynaParticles, 64, numBlocks, numThreads);

        // execute the kernel
        shrinkD<<< numBlocks, numThreads >>>((float4 *)dynaPos,
                                              (float4 *)forces,
                                              (float4 *)vel,
                                              (uint   *)cons,
                                              (float4 *)restLengths,
                                              (int    *)segment,
                                              (float  *)cntrScale,
                                              dynaParticles, statParticles, range, deltaTime,
                                              (float4 *)color );

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }




    void staticConnections(float *dynaPos,
                            float *vel,
                            float *forces,
                            float *staticPos,
                            float *elasticity,
                            uint  *cons,
                            uint  *conCount,
                            uint  *conStart,
                            float *restLengths,
                            uint   dynaParticles,
                            uint   statParticles)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(dynaParticles, 64, numBlocks, numThreads);

        // execute the kernel
        staticD<<< numBlocks, numThreads >>>((float4 *)dynaPos,
                                              (float4 *)vel,
                                              (float4 *)forces,
                                              (float4 *)staticPos,
                                              (float  *)elasticity,
                                              (uint   *)cons,
                                              (uint   *)conCount,
                                              (uint   *)conStart,
                                              (float4 *)restLengths,
                                              dynaParticles, statParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void connections(float *tempPos,
                     float *dynaPos,
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
                     uint   dynaParticles,
                     float *color,
                     uint3 dataSize,
                     float *initPos,
                     float *finalPos,
                     int *staticTrue,
                     bool m_bAnchor,
                     int *m_dSurface)
    {
    // tex pos vel

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(dynaParticles, 64, numBlocks, numThreads);

        // execute the kernel
        connectD<<< numBlocks, numThreads >>>((float4 *)tempPos,
                                              (float4 *)dynaPos,
                                              (float4 *)vel,
                                              (float4 *)forceS,
                                              (float *)stretchS,
                                              (float4 *)forceT,
                                              (float  *)elasticity,
                                              (uint   *)cons,
                                              (uint   *)conCount,
                                              (uint   *)conStart,
                                              (float4 *)restLengths,
                                              (int    *)segment,
                                              (float  *)cntrScale,
                                              dynaParticles,
                                              (float4 *)color,
                                              (uint3  )dataSize,
                                              (float4 *)initPos,
                                              (float4 *)finalPos,
                                              (int  *)staticTrue,
                                              m_bAnchor,
                                              (int  *)m_dSurface);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }



    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float4 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }




    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float4 *) sortedPos,
            gridParticleHash,
            gridParticleIndex,
            (float4 *) oldPos,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    }





    void collide(float *newPos,
                 float *newVel,
                 float *newForce,
                 int   *segment,
                 uint   dynaParticles,
                 float  deltaTime)
    {
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(dynaParticles, 64, numBlocks, numThreads);

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((float4 *)newPos,
                                              (float4 *)newVel,
                                              (float4 *)newForce,
                                              segment,
                                              dynaParticles,
                                              deltaTime);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }



    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }


    void center_of_mass(float  *pos,
                        float3 *comPos,
                        uint    numParticles)
    {
        thrust::device_ptr<float4> tpos4( (float4 *)pos );
        float4 sumPos = thrust::reduce( tpos4, tpos4+numParticles, make_float4( 0.f ), thrust::plus<float4>() );
        float num = (float)numParticles;
        comPos[0] = make_float3( sumPos.x/num, sumPos.y/num, sumPos.z/num );
        //printf("\n comPos: ( %3.2f, %3.2f, %3.2f ) \n", comPos[0].x, comPos[0].y, comPos[0].z );
    }


}   // extern "C"
